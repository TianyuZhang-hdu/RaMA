"""
SAM 修复模块 - 使用 SAM3、MedSAM2、SAM-Med2D 三模型逐像素投票生成伪标签
- 任意器官评分 < 40 (DISCARD)：整张图丢弃，不保存掩码，不参与 Dice 计算
- 全部器官评分 >= 90 (KEEP)：直接使用伪标签，跳过修复
- 无评分 / ERROR 的图片：使用原伪标签
- 其余 (所有器官 >= 40，且有器官 < 90)：三模型分别修复，逐像素投票生成最终掩码
"""

# --- RaMA: configurable workspace root (replaces the old hardcoded prefix) ---
import os as _os
def _rama_ws():
    v = _os.environ.get("RAMA_WORKSPACE_ROOT")
    if v:
        return v.rstrip("/")
    try:
        import yaml as _yaml  # type: ignore
        for _p in (_os.environ.get("RAMA_CONFIG"),
                   _os.path.join(_os.path.dirname(__file__), "..", "..", "configs", "rama_config.yaml")):
            if _p and _os.path.isfile(_p):
                with open(_p, encoding="utf-8") as _fh:
                    _d = _yaml.safe_load(_fh) or {}
                _w = (_d or {}).get("workspace_root")
                if _w:
                    return str(_w).rstrip("/")
    except Exception:
        pass
    return "/path/to/workspace"
_WS = _rama_ws()
# --- end RaMA workspace root ---
import sys
import os

# 确保导入 heart 目录下的模块
HEART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HEART_DIR not in sys.path:
    sys.path.insert(0, HEART_DIR)

# 清除可能存在的其他项目路径，避免混淆
sys.path = [p for p in sys.path if 'Fundus' not in p or p == HEART_DIR]

from PIL import Image
import numpy as np
import json
import config
from refinement.sam3_refine import SAM3Refiner
from refinement.medsam2_refine import MedSAM2Refiner
from refinement.sammed2d_refine import SAMMed2DRefiner

# 导入工具函数
from common.visualization_utils import visualize_clusters, visualize_bbox
from common.mask_utils import (
    extract_single_organ_mask, overlay_single_mask, 
    get_organ_mask_arr, apply_refined_mask,
    compute_all_dice, compute_all_assd
)
from common.clustering_utils import dbscan_crop
from common.plot_utils import plot_debug_results, save_final_mask

# 配置常量
VENDOR = "vendorD"
SPLIT = "train"
IMAGE_DIR = f"{_WS}/datas/mnms/{VENDOR}/{SPLIT}/image"
MASK_DIR = f"{_WS}/asga/heart/iplc/infer_mask/{VENDOR}/{SPLIT}"
GT_DIR = f"{_WS}/datas/mnms/{VENDOR}/{SPLIT}/mask"
TARGET_SIZE = 256

# 模型名称（可手动指定，或从 config 自动获取）
# MODEL_NAME = config.OPENAI_MODEL.replace("/", "_").replace(":", "_")
MODEL_NAME = "qwen-vl-max"

# LLM 评分结果目录（带模型名）
LLM_SCORE_DIR = f"{_WS}/asga/tests/llm_scores/{MODEL_NAME}/{VENDOR}_{SPLIT}"

# SAM 修复结果目录（带模型名）
SAM_RESULT_DIR = f"{_WS}/asga/tests/sam_results/{MODEL_NAME}/{VENDOR}_{SPLIT}"
SAM3_MASK_DIR = f"{SAM_RESULT_DIR}/masks_sam3"
MEDSAM2_MASK_DIR = f"{SAM_RESULT_DIR}/masks_medsam2"
SAMMED2D_MASK_DIR = f"{SAM_RESULT_DIR}/masks_sammed2d"
VOTE_MASK_DIR = f"{SAM_RESULT_DIR}/masks_vote"
DEBUG_DIR = f"{_WS}/asga/tests/debug_output_sam/{MODEL_NAME}/{VENDOR}_{SPLIT}"

# 最终输出目录：原图 + 最终 mask
RESULT_IMAGE_DIR = f"{_WS}/asga/result/{VENDOR}/image"
RESULT_MASK_DIR = f"{_WS}/asga/result/{VENDOR}/mask"

# KEEP 阈值
KEEP_THRESHOLD = 90
# DISCARD 阈值：评分低于此值的器官直接丢弃（清零），不参与 Dice 计算
DISCARD_THRESHOLD = 40


def load_llm_score(img_id: str) -> dict:
    """加载 LLM 评分结果"""
    score_path = f"{LLM_SCORE_DIR}/{img_id}.json"
    if not os.path.exists(score_path):
        return None
    
    with open(score_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_all_keep(llm_score: dict) -> bool:
    """检查是否所有器官都不需要修复（全部 >= 90）"""
    organs = llm_score.get("organs", {})
    for organ_key in ["LV", "MYO", "RV"]:
        organ_data = organs.get(organ_key, {})
        total = organ_data.get("scores", {}).get("total", 0)
        if total < KEEP_THRESHOLD:
            return False
    return True


def has_any_low_score(llm_score: dict) -> bool:
    """检查是否有任意一个器官评分低于 DISCARD_THRESHOLD（整张图丢弃）"""
    organs = llm_score.get("organs", {})
    for organ_key in ["LV", "MYO", "RV"]:
        organ_data = organs.get(organ_key, {})
        total = organ_data.get("scores", {}).get("total", 0)
        if total < DISCARD_THRESHOLD:
            return True
    return False


def has_error(llm_score: dict) -> bool:
    """检查是否有 ERROR"""
    organs = llm_score.get("organs", {})
    for organ_data in organs.values():
        decision = organ_data.get("decision", "")
        if decision.startswith("ERROR"):
            return True
    return False


def majority_vote_masks(
    mask_sam3: Image.Image,
    mask_medsam2: Image.Image,
    mask_sammed2d: Image.Image,
) -> Image.Image:
    """
    逐像素对三个模型的 RGB 掩码做多数投票。
    每个通道（LV=R, MYO=G, RV=B）独立投票：
      >= 2/3 的模型认为该像素属于该器官 → 最终为前景。
    """
    a3 = np.array(mask_sam3.convert("RGB"))
    a2 = np.array(mask_medsam2.convert("RGB"))
    ad = np.array(mask_sammed2d.convert("RGB"))

    vote = np.zeros_like(a3, dtype=np.uint8)
    for ch in range(3):
        b3 = (a3[:, :, ch] > 127).astype(np.uint8)
        b2 = (a2[:, :, ch] > 127).astype(np.uint8)
        bd = (ad[:, :, ch] > 127).astype(np.uint8)
        vote[:, :, ch] = ((b3 + b2 + bd) >= 2).astype(np.uint8) * 255

    return Image.fromarray(vote)


def refine_single_image(
    img_id: str,
    sam3_refiner: SAM3Refiner,
    medsam2_refiner: MedSAM2Refiner,
    sammed2d_refiner: SAMMed2DRefiner,
    llm_score: dict,
    save_detail: bool = False,
) -> dict:
    """处理单张图像的 SAM 修复流程"""
    img_path = f"{IMAGE_DIR}/{img_id}.png"
    mask_path = f"{MASK_DIR}/{img_id}.png"
    gt_path = f"{GT_DIR}/{img_id}.png"
    
    if not os.path.exists(mask_path):
        print(f"  [SKIP] mask不存在: {mask_path}")
        return None
    if not os.path.exists(gt_path):
        print(f"  [SKIP] gt不存在: {gt_path}")
        return None
    
    original = Image.open(img_path).resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
    mask = Image.open(mask_path).resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
    gt = Image.open(gt_path).resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
    
    # 从 LLM 评分结果中获取 bbox
    bbox = tuple(llm_score["bbox"])
    
    cropped_original = original.crop(bbox)
    cropped_mask = mask.crop(bbox)
    cropped_gt = gt.crop(bbox)
    
    if save_detail:
        cropped_original.save(f"{DEBUG_DIR}/{img_id}_1_cropped_original.png")
        cropped_mask.save(f"{DEBUG_DIR}/{img_id}_2_cropped_mask.png")
        cropped_gt.save(f"{DEBUG_DIR}/{img_id}_3_cropped_gt.png")
    
    # 设置图像状态
    state_sam3 = sam3_refiner.set_image(cropped_original)
    medsam2_refiner.set_image(cropped_original)
    sammed2d_refiner.set_image(cropped_original)
    
    # 准备掩码副本
    refined_mask_sam3 = cropped_mask.copy()
    refined_mask_medsam2 = cropped_mask.copy()
    refined_mask_sammed2d = cropped_mask.copy()
    keep_organs = []
    
    mask_weights = {"LV": 1.5, "MYO": 2.0, "RV": 2.0}
    organ_scores = llm_score.get("organs", {})
    
    for organ_key in ["LV", "MYO", "RV"]:
        organ_result = organ_scores.get(organ_key, {})
        total_score = organ_result.get("scores", {}).get("total", 0)
        decision = organ_result.get("decision", "UNKNOWN")
        
        # 评分 >= KEEP_THRESHOLD：跳过修复
        if total_score >= KEEP_THRESHOLD:
            keep_organs.append(organ_key)
            print(f"  {organ_key}: KEEP (score={total_score} >= {KEEP_THRESHOLD})，保留伪标签")
            continue
        
        fp_pts = organ_result.get("fp_points", [])
        fn_pts = organ_result.get("fn_points", [])
        pos_points = np.array(fn_pts, dtype=np.float32) if fn_pts else None
        neg_points = np.array(fp_pts, dtype=np.float32) if fp_pts else None
        
        print(f"  {organ_key}: {decision} (score={total_score}), pos={fn_pts}, neg={fp_pts}")
        
        # SAM3 修复
        organ_mask_arr_sam3 = get_organ_mask_arr(refined_mask_sam3, organ_key)
        refined_sam3 = sam3_refiner.refine(
            state_sam3, organ_mask_arr_sam3, 
            pos_points=pos_points, neg_points=neg_points, 
            mask_weight=mask_weights[organ_key]
        )
        refined_mask_sam3 = apply_refined_mask(refined_mask_sam3, refined_sam3, organ_key)
        
        # MedSAM2 修复
        organ_mask_arr_medsam2 = get_organ_mask_arr(refined_mask_medsam2, organ_key)
        refined_medsam2 = medsam2_refiner.refine(
            state=None,
            y_init=organ_mask_arr_medsam2, 
            pos_points=pos_points, 
            neg_points=neg_points, 
            mask_weight=mask_weights[organ_key]
        )
        refined_mask_medsam2 = apply_refined_mask(refined_mask_medsam2, refined_medsam2, organ_key)
        
        # SAM-Med2D 修复
        organ_mask_arr_sammed2d = get_organ_mask_arr(refined_mask_sammed2d, organ_key)
        refined_sammed2d = sammed2d_refiner.refine(
            state=None,
            y_init=organ_mask_arr_sammed2d,
            pos_points=pos_points,
            neg_points=neg_points,
            mask_weight=mask_weights[organ_key],
        )
        refined_mask_sammed2d = apply_refined_mask(refined_mask_sammed2d, refined_sammed2d, organ_key)
        
        if save_detail:
            idx = {"LV": 4, "MYO": 5, "RV": 6}[organ_key]
            refined_overlay_sam3 = overlay_single_mask(cropped_original, extract_single_organ_mask(refined_mask_sam3, organ_key), organ_key)
            refined_overlay_sam3.save(f"{DEBUG_DIR}/{img_id}_{idx}_{organ_key}_refined_sam3.png")
            refined_overlay_medsam2 = overlay_single_mask(cropped_original, extract_single_organ_mask(refined_mask_medsam2, organ_key), organ_key)
            refined_overlay_medsam2.save(f"{DEBUG_DIR}/{img_id}_{idx}_{organ_key}_refined_medsam2.png")
            refined_overlay_sammed2d = overlay_single_mask(cropped_original, extract_single_organ_mask(refined_mask_sammed2d, organ_key), organ_key)
            refined_overlay_sammed2d.save(f"{DEBUG_DIR}/{img_id}_{idx}_{organ_key}_refined_sammed2d.png")
    
    # 伪 Dice 阈值：修复结果与伪标签的 Dice >= 0.5 才保留，否则回退到伪标签
    PSEUDO_DICE_THRESHOLD = 0.5

    def _eval_model(name, refined_mask):
        dice_vs_pseudo = compute_all_dice(refined_mask, cropped_mask)
        use_pseudo = bool(dice_vs_pseudo["mean"] < PSEUDO_DICE_THRESHOLD)
        if use_pseudo:
            print(f"  [{name} FALLBACK] 伪Dice={dice_vs_pseudo['mean']:.3f} < {PSEUDO_DICE_THRESHOLD}")
        else:
            print(f"  [{name} ACCEPT] 伪Dice={dice_vs_pseudo['mean']:.3f} (LV:{dice_vs_pseudo['LV']:.2f}, MYO:{dice_vs_pseudo['MYO']:.2f}, RV:{dice_vs_pseudo['RV']:.2f})")
        final = cropped_mask if use_pseudo else refined_mask
        gt_dice = compute_all_dice(final, cropped_gt)
        gt_assd = compute_all_assd(final, cropped_gt)
        print(f"  [{name} GT] Dice={gt_dice['mean']:.3f} (LV:{gt_dice['LV']:.2f}, MYO:{gt_dice['MYO']:.2f}, RV:{gt_dice['RV']:.2f})")
        print(f"  [{name} GT] ASSD={gt_assd['mean']:.2f} (LV:{gt_assd['LV']:.2f}, MYO:{gt_assd['MYO']:.2f}, RV:{gt_assd['RV']:.2f})")
        return final, {
            "pseudo_dice": dice_vs_pseudo, "use_pseudo": use_pseudo,
            "gt_dice": gt_dice, "gt_assd": gt_assd,
        }

    final_mask_sam3, res_sam3 = _eval_model("SAM3", refined_mask_sam3)
    final_mask_medsam2, res_medsam2 = _eval_model("MedSAM2", refined_mask_medsam2)
    final_mask_sammed2d, res_sammed2d = _eval_model("SAM-Med2D", refined_mask_sammed2d)

    # ===== 三模型逐像素多数投票 =====
    voted_mask = majority_vote_masks(final_mask_sam3, final_mask_medsam2, final_mask_sammed2d)
    gt_dice_vote = compute_all_dice(voted_mask, cropped_gt)
    gt_assd_vote = compute_all_assd(voted_mask, cropped_gt)
    print(f"  [VOTE GT] Dice={gt_dice_vote['mean']:.3f} (LV:{gt_dice_vote['LV']:.2f}, MYO:{gt_dice_vote['MYO']:.2f}, RV:{gt_dice_vote['RV']:.2f})")
    print(f"  [VOTE GT] ASSD={gt_assd_vote['mean']:.2f} (LV:{gt_assd_vote['LV']:.2f}, MYO:{gt_assd_vote['MYO']:.2f}, RV:{gt_assd_vote['RV']:.2f})")

    # 各模型 vs 投票的 Dice 对比
    all_means = {
        "SAM3": res_sam3["gt_dice"]["mean"],
        "MedSAM2": res_medsam2["gt_dice"]["mean"],
        "SAM-Med2D": res_sammed2d["gt_dice"]["mean"],
        "VOTE": gt_dice_vote["mean"],
    }
    best_method = max(all_means, key=all_means.get)
    print(f"  [对比] best={best_method} " + " ".join(f"{k}={v:.3f}" for k, v in all_means.items()))

    # 保存各模型 + 投票掩码
    save_final_mask(final_mask_sam3, bbox, TARGET_SIZE, f"{SAM3_MASK_DIR}/{img_id}.png")
    save_final_mask(final_mask_medsam2, bbox, TARGET_SIZE, f"{MEDSAM2_MASK_DIR}/{img_id}.png")
    save_final_mask(final_mask_sammed2d, bbox, TARGET_SIZE, f"{SAMMED2D_MASK_DIR}/{img_id}.png")
    save_final_mask(voted_mask, bbox, TARGET_SIZE, f"{VOTE_MASK_DIR}/{img_id}.png")

    # 最终结果目录使用投票结果
    ref_arr = np.array(voted_mask.convert("RGB"))
    out = np.zeros_like(ref_arr)
    out[ref_arr[:, :, 1] > 127] = [0, 255, 0]   # MYO
    out[ref_arr[:, :, 0] > 127] = [255, 0, 0]    # LV
    out[ref_arr[:, :, 2] > 127] = [0, 0, 255]    # RV
    full_mask = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    full_mask.paste(Image.fromarray(out), (bbox[0], bbox[1]))
    _save_to_result(img_id, full_mask)

    # 保存 JSON 结果
    result_json = {
        "img_id": img_id,
        "keep_organs": keep_organs,
        "sam3": res_sam3,
        "medsam2": res_medsam2,
        "sammed2d": res_sammed2d,
        "vote": {"gt_dice": gt_dice_vote, "gt_assd": gt_assd_vote},
        "best_method": best_method,
    }

    with open(f"{SAM_RESULT_DIR}/{img_id}.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    return {
        "img_id": img_id,
        "keep_organs": keep_organs,
        "sam3": {"use_pseudo": res_sam3["use_pseudo"], "gt_dice": res_sam3["gt_dice"], "gt_assd": res_sam3["gt_assd"]},
        "medsam2": {"use_pseudo": res_medsam2["use_pseudo"], "gt_dice": res_medsam2["gt_dice"], "gt_assd": res_medsam2["gt_assd"]},
        "sammed2d": {"use_pseudo": res_sammed2d["use_pseudo"], "gt_dice": res_sammed2d["gt_dice"], "gt_assd": res_sammed2d["gt_assd"]},
        "vote": {"gt_dice": gt_dice_vote, "gt_assd": gt_assd_vote},
        "best_method": best_method,
    }



def _crop_and_clean_pseudo_label(img_id: str, bbox: tuple) -> Image.Image:
    """
    对原伪标签做聚类区域裁剪预处理：
      1. 加载原伪标签 + resize 到 TARGET_SIZE
      2. 使用 bbox 裁剪心脏区域
      3. 清洗 RGB 值（确保纯色）
      4. 粘贴回 256x256 黑色画布

    这与 llm_score.py 中先 dbscan_crop 再处理的逻辑一致，
    也与 refine_single_image 中 SAM 修复后 paste 回全图的逻辑一致。
    消除心脏区域以外的伪标签噪声。

    Args:
        img_id: 图片 ID
        bbox: 裁剪框 (x_min, y_min, x_max, y_max)，来自 llm_score["bbox"]

    Returns:
        PIL Image (RGB, TARGET_SIZE x TARGET_SIZE) 或 None
    """
    mask_path = f"{MASK_DIR}/{img_id}.png"
    if not os.path.exists(mask_path):
        return None

    mask = Image.open(mask_path).convert("RGB")
    if mask.size != (TARGET_SIZE, TARGET_SIZE):
        mask = mask.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)

    # 裁剪心脏区域
    cropped = mask.crop(bbox)
    cropped_arr = np.array(cropped)

    # 清洗 RGB：确保是纯净的 R/G/B 编码（去除混合色/噪声像素）
    out = np.zeros_like(cropped_arr)
    out[cropped_arr[:, :, 1] > 127] = [0, 255, 0]    # MYO (Green)
    out[cropped_arr[:, :, 0] > 127] = [255, 0, 0]     # LV  (Red)
    out[cropped_arr[:, :, 2] > 127] = [0, 0, 255]     # RV  (Blue)

    # 粘贴回全尺寸黑色画布
    full_mask = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
    full_mask.paste(Image.fromarray(out), (bbox[0], bbox[1]))
    return full_mask


def _save_to_result(img_id: str, mask_img: Image.Image = None, bbox: tuple = None):
    """
    将原图和 mask 保存到最终输出目录 result/{VENDOR}/image 和 result/{VENDOR}/mask。

    Args:
        img_id: 图片 ID
        mask_img: 已处理的 mask (PIL Image, RGB, TARGET_SIZE x TARGET_SIZE)。
                  如果为 None，则从 MASK_DIR 加载原伪标签并做聚类裁剪预处理。
        bbox: 裁剪框，当 mask_img 为 None 时必须提供（用于裁剪预处理）。
    """
    os.makedirs(RESULT_IMAGE_DIR, exist_ok=True)
    os.makedirs(RESULT_MASK_DIR, exist_ok=True)

    # 保存原图（resize 到 TARGET_SIZE）
    img_path = f"{IMAGE_DIR}/{img_id}.png"
    if not os.path.exists(img_path):
        return False
    original = Image.open(img_path).resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
    original.save(f"{RESULT_IMAGE_DIR}/{img_id}.png")

    # 保存 mask
    if mask_img is not None:
        mask_img.save(f"{RESULT_MASK_DIR}/{img_id}.png")
    else:
        # 使用原伪标签 + 聚类裁剪预处理（去除心脏区域外的噪声）
        if bbox is None:
            print(f"  [WARNING] {img_id}: 无 bbox，跳过 mask 保存")
            return False
        cleaned_mask = _crop_and_clean_pseudo_label(img_id, bbox)
        if cleaned_mask is None:
            return False
        cleaned_mask.save(f"{RESULT_MASK_DIR}/{img_id}.png")
    return True


def _copy_pseudo_label(img_id: str):
    """将伪标签直接复制到所有模型 + 投票结果目录（统一 resize 到 TARGET_SIZE）"""
    mask_path = f"{MASK_DIR}/{img_id}.png"
    if not os.path.exists(mask_path):
        print(f"  [SKIP] 伪标签不存在: {mask_path}")
        return False
    mask = Image.open(mask_path).convert("RGB")
    if mask.size != (TARGET_SIZE, TARGET_SIZE):
        mask = mask.resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
    for dest_dir in [SAM3_MASK_DIR, MEDSAM2_MASK_DIR, SAMMED2D_MASK_DIR, VOTE_MASK_DIR]:
        os.makedirs(dest_dir, exist_ok=True)
        mask.save(f"{dest_dir}/{img_id}.png")
    return True


def run_sam_refine(save_detail: bool = False):
    """运行 SAM 修复流程（SAM3 + MedSAM2 + SAM-Med2D 三模型投票）"""
    for d in [SAM_RESULT_DIR, SAM3_MASK_DIR, MEDSAM2_MASK_DIR,
              SAMMED2D_MASK_DIR, VOTE_MASK_DIR, DEBUG_DIR,
              RESULT_IMAGE_DIR, RESULT_MASK_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # 扫描所有 LLM 评分 JSON
    if not os.path.exists(LLM_SCORE_DIR):
        print(f"错误: LLM 评分目录不存在: {LLM_SCORE_DIR}")
        return
    
    all_json_files = sorted([f for f in os.listdir(LLM_SCORE_DIR) if f.endswith('.json') and f != 'summary.json'])
    print(f"\n{'='*60}")
    print(f"模型: {MODEL_NAME}")
    print(f"评分目录: {LLM_SCORE_DIR}")
    print(f"评分文件数: {len(all_json_files)}")
    print(f"SAM3 掩码保存: {SAM3_MASK_DIR}")
    print(f"MedSAM2 掩码保存: {MEDSAM2_MASK_DIR}")
    print(f"SAM-Med2D 掩码保存: {SAMMED2D_MASK_DIR}")
    print(f"投票掩码保存: {VOTE_MASK_DIR}")
    print(f"{'='*60}")
    
    # 筛选：分类所有 JSON 图片
    to_refine = []
    skip_no_score = 0
    error_ids = []
    keep_ids = []
    discard_ids = []   # 任意器官 < DISCARD_THRESHOLD，整张图丢弃
    
    for json_file in all_json_files:
        img_id = os.path.splitext(json_file)[0]
        llm_score = load_llm_score(img_id)
        
        if llm_score is None:
            skip_no_score += 1
            continue
        
        if has_error(llm_score):
            error_ids.append((img_id, llm_score))
            continue
        
        # 任意器官评分 < 40 → 整张图丢弃，不保存掩码，不计入 Dice
        if has_any_low_score(llm_score):
            # 记录哪些器官触发了丢弃
            low_organs = []
            for o in ["LV", "MYO", "RV"]:
                s = llm_score.get("organs", {}).get(o, {}).get("scores", {}).get("total", 0)
                if s < DISCARD_THRESHOLD:
                    low_organs.append(f"{o}={s}")
            discard_ids.append((img_id, llm_score, low_organs))
            continue
        
        if check_all_keep(llm_score):
            keep_ids.append((img_id, llm_score))
            continue
        
        to_refine.append((img_id, llm_score))
    
    print(f"\n筛选结果:")
    print(f"  需要修复: {len(to_refine)}")
    print(f"  无需修复 (全部KEEP): {len(keep_ids)}")
    print(f"  DISCARD  (任一器官<{DISCARD_THRESHOLD}，整图丢弃): {len(discard_ids)}")
    print(f"  ERROR (使用原伪标签): {len(error_ids)}")
    print(f"  跳过 (无评分): {skip_no_score}")
    if discard_ids:
        print(f"\n  被丢弃的图片 (前 10):")
        for img_id, _, low_organs in discard_ids[:10]:
            print(f"    {img_id}: {', '.join(low_organs)}")
    print(f"{'='*60}\n")
    
    # ========== KEEP 图片：保存原伪标签（聚类裁剪预处理）+ 计算 Dice ==========
    keep_metrics = []
    keep_saved = 0
    result_saved = 0
    print(f"保存 KEEP 图片伪标签（聚类裁剪预处理）...")
    for img_id, llm_score in keep_ids:
        # 保存原伪标签到 SAM3/MedSAM2 输出目录
        if _copy_pseudo_label(img_id):
            keep_saved += 1
        # 保存到最终结果目录（原图 + 裁剪预处理后的伪标签）
        bbox = tuple(llm_score["bbox"])
        if _save_to_result(img_id, bbox=bbox):
            result_saved += 1
        
        mask_path = f"{MASK_DIR}/{img_id}.png"
        gt_path = f"{GT_DIR}/{img_id}.png"
        if not os.path.exists(mask_path) or not os.path.exists(gt_path):
            continue
        mask = Image.open(mask_path).resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
        gt = Image.open(gt_path).resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
        bbox = tuple(llm_score["bbox"])
        cropped_mask = mask.crop(bbox)
        cropped_gt = gt.crop(bbox)
        gt_dice = compute_all_dice(cropped_mask, cropped_gt)
        gt_assd = compute_all_assd(cropped_mask, cropped_gt)
        keep_metrics.append({
            "img_id": img_id,
            "keep_organs": ["LV", "MYO", "RV"],
            "sam3": {"use_pseudo": False, "gt_dice": gt_dice, "gt_assd": gt_assd},
            "medsam2": {"use_pseudo": False, "gt_dice": gt_dice, "gt_assd": gt_assd},
            "sammed2d": {"use_pseudo": False, "gt_dice": gt_dice, "gt_assd": gt_assd},
            "vote": {"gt_dice": gt_dice, "gt_assd": gt_assd},
            "best_method": "TIE",
        })
    
    print(f"KEEP 图片: {keep_saved}/{len(keep_ids)} 张伪标签已保存")
    
    # ========== ERROR 图片：保存原伪标签（聚类裁剪预处理） ==========
    error_saved = 0
    if error_ids:
        print(f"\n保存 ERROR 图片伪标签 (聚类裁剪预处理)...")
        for img_id, llm_score in error_ids:
            if _copy_pseudo_label(img_id):
                error_saved += 1
            # 保存到最终结果目录（原图 + 裁剪预处理后的伪标签）
            bbox = tuple(llm_score.get("bbox", (0, 0, TARGET_SIZE, TARGET_SIZE)))
            if _save_to_result(img_id, bbox=bbox):
                result_saved += 1
        print(f"ERROR 图片: {error_saved}/{len(error_ids)} 张伪标签已保存")
    
    print(f"\nKEEP 图片伪标签 Dice 计算完成: {len(keep_metrics)} 张")
    
    # ========== 修复需要修复的图片 ==========
    refine_metrics = []
    
    if to_refine:
        print("\n初始化 SAM3...")
        sam3_refiner = SAM3Refiner()
        
        print("初始化 MedSAM2...")
        medsam2_refiner = MedSAM2Refiner()
        
        print("初始化 SAM-Med2D...")
        sammed2d_refiner = SAMMed2DRefiner()
        
        for i, (img_id, llm_score) in enumerate(to_refine):
            print(f"\n[{i+1}/{len(to_refine)}] SAM 三模型修复+投票 {img_id}...")
            result = refine_single_image(
                img_id, sam3_refiner, medsam2_refiner, sammed2d_refiner,
                llm_score, save_detail=save_detail,
            )
            if result:
                refine_metrics.append(result)
    
    # ========== 汇总统计 ==========
    all_metrics = refine_metrics + keep_metrics
    
    if all_metrics:
        print("\n" + "="*70)
        print(f"汇总统计 - SAM3 / MedSAM2 / SAM-Med2D / 投票 (模型: {MODEL_NAME})")
        print("="*70)
        
        total_masks_saved = keep_saved + error_saved + len(refine_metrics)
        total_result_saved = result_saved + len(refine_metrics)
        print(f"\n【图片分类统计】")
        print(f"  全部KEEP (原伪标签): {len(keep_metrics)} 张 (已保存 {keep_saved})")
        print(f"  DISCARD  (任一器官<{DISCARD_THRESHOLD}，整图丢弃): {len(discard_ids)} 张 (不保存, 不计Dice)")
        print(f"  ERROR   (原伪标签): {len(error_ids)} 张 (已保存 {error_saved})")
        print(f"  需要修复 (已修复):   {len(refine_metrics)} 张")
        print(f"  总计已保存掩码: {total_masks_saved} 张")
        print(f"  总计参与Dice统计: {len(all_metrics)} 张")
        print(f"  最终结果目录输出: ~{total_result_saved} 张 (KEEP+ERROR+修复)")
        print(f"  跳过 (无评分): {skip_no_score}")
        
        def _stats(values):
            vals = [v for v in values if v != float('inf')]
            if not vals:
                return float('inf'), float('inf')
            return np.mean(vals), np.std(vals)

        def _fmt(mean, std):
            if mean == float('inf'):
                return "N/A"
            return f"{mean:.4f}±{std:.4f}"

        def _organ_stats(metrics, method, metric_type):
            result = {}
            for key in ["mean", "LV", "MYO", "RV"]:
                vals = [m[method][metric_type][key] for m in metrics]
                result[key] = _stats(vals)
            return result

        def _print_metric(label, organ_stats):
            parts = []
            for key in ["LV", "MYO", "RV"]:
                m, s = organ_stats[key]
                parts.append(f"{key}:{_fmt(m, s)}")
            mm, ms = organ_stats["mean"]
            print(f"{label}: {_fmt(mm, ms)} ({', '.join(parts)})")

        if keep_metrics:
            keep_dice = _organ_stats(keep_metrics, "sam3", "gt_dice")
            keep_assd = _organ_stats(keep_metrics, "sam3", "gt_assd")
            print(f"\n【KEEP 图片伪标签 vs GT】")
            _print_metric("  Dice", keep_dice)
            _print_metric("  ASSD", keep_assd)
        
        if refine_metrics:
            print(f"\n【仅修复图片】")
            for mname in ["sam3", "medsam2", "sammed2d", "vote"]:
                rd = _organ_stats(refine_metrics, mname, "gt_dice")
                ra = _organ_stats(refine_metrics, mname, "gt_assd")
                label = {"sam3": "SAM3", "medsam2": "MedSAM2", "sammed2d": "SAM-Med2D", "vote": "VOTE"}[mname]
                _print_metric(f"  {label} Dice", rd)
                _print_metric(f"  {label} ASSD", ra)

        # 各模型总体统计（修复 + KEEP）
        method_dice_stats = {}
        method_assd_stats = {}
        method_pseudo_counts = {}
        for mname in ["sam3", "medsam2", "sammed2d"]:
            label = {"sam3": "SAM3", "medsam2": "MedSAM2", "sammed2d": "SAM-Med2D"}[mname]
            print(f"\n【{label} 总体结果 (修复+KEEP)】")
            pc = sum(1 for m in all_metrics if m[mname].get("use_pseudo", False))
            method_pseudo_counts[mname] = pc
            print(f"使用伪标签(FALLBACK): {pc}/{len(all_metrics)} ({100*pc/len(all_metrics):.1f}%)")
            ds = _organ_stats(all_metrics, mname, "gt_dice")
            aa = _organ_stats(all_metrics, mname, "gt_assd")
            method_dice_stats[mname] = ds
            method_assd_stats[mname] = aa
            _print_metric("Dice", ds)
            _print_metric("ASSD", aa)

        # 投票总体统计
        print(f"\n【三模型投票 总体结果 (修复+KEEP)】")
        vote_dice_stats = _organ_stats(all_metrics, "vote", "gt_dice")
        vote_assd_stats = _organ_stats(all_metrics, "vote", "gt_assd")
        method_dice_stats["vote"] = vote_dice_stats
        method_assd_stats["vote"] = vote_assd_stats
        _print_metric("Dice", vote_dice_stats)
        _print_metric("ASSD", vote_assd_stats)

        # best_method 统计
        print("\n【最佳方法统计 (逐图)】")
        best_counts = {}
        for m in all_metrics:
            bm = m.get("best_method", "TIE")
            best_counts[bm] = best_counts.get(bm, 0) + 1
        for bm in ["SAM3", "MedSAM2", "SAM-Med2D", "VOTE", "TIE"]:
            cnt = best_counts.get(bm, 0)
            if cnt > 0:
                print(f"  {bm}: {cnt}/{len(all_metrics)} ({100*cnt/len(all_metrics):.1f}%)")

        # 总体 Dice 对比
        print("\n【总体 Dice 对比】")
        for mname, label in [("sam3", "SAM3"), ("medsam2", "MedSAM2"),
                             ("sammed2d", "SAM-Med2D"), ("vote", "VOTE")]:
            mm, ms = method_dice_stats[mname]["mean"]
            print(f"  {label}: {_fmt(mm, ms)}")
        overall_means = {k: method_dice_stats[k]["mean"][0] for k in method_dice_stats}
        overall_best = max(overall_means, key=overall_means.get)
        overall_best_label = {"sam3": "SAM3", "medsam2": "MedSAM2",
                              "sammed2d": "SAM-Med2D", "vote": "VOTE"}[overall_best]
        print(f"  总体最佳: {overall_best_label} (mean Dice = {overall_means[overall_best]:.4f})")

        def _stats_to_json(organ_stats):
            d = {}
            for key in ["mean", "LV", "MYO", "RV"]:
                m, s = organ_stats[key]
                if m == float('inf'):
                    d[key] = None
                else:
                    d[key] = {"mean": float(m), "std": float(s)}
            return d

        summary = {
            "model": MODEL_NAME,
            "total_scored": len(all_json_files),
            "skip_no_score": skip_no_score,
            "total_error": len(error_ids),
            "error_saved": error_saved,
            "total_keep": len(keep_metrics),
            "keep_saved": keep_saved,
            "total_refined": len(refine_metrics),
            "total_masks_saved": total_masks_saved,
            "total_evaluated": len(all_metrics),
            "keep_threshold": KEEP_THRESHOLD,
            "discard_threshold": DISCARD_THRESHOLD,
            "total_discarded_images": len(discard_ids),
        }
        for mname in ["sam3", "medsam2", "sammed2d"]:
            summary[mname] = {
                "pseudo_count": method_pseudo_counts[mname],
                "dice": _stats_to_json(method_dice_stats[mname]),
                "assd": _stats_to_json(method_assd_stats[mname]),
            }
        summary["vote"] = {
            "dice": _stats_to_json(vote_dice_stats),
            "assd": _stats_to_json(vote_assd_stats),
        }
        summary["best_method_counts"] = best_counts
        summary["overall_best"] = overall_best_label
        
        with open(f"{SAM_RESULT_DIR}/summary_comparison.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n汇总结果已保存到: {SAM_RESULT_DIR}/summary_comparison.json")
    
    # 统计最终结果目录
    result_img_count = len([f for f in os.listdir(RESULT_IMAGE_DIR) if f.endswith('.png')])
    result_mask_count = len([f for f in os.listdir(RESULT_MASK_DIR) if f.endswith('.png')])
    
    print(f"\n完成！")
    print(f"SAM3 掩码: {SAM3_MASK_DIR}/ ({len(os.listdir(SAM3_MASK_DIR))} 张)")
    print(f"MedSAM2 掩码: {MEDSAM2_MASK_DIR}/ ({len(os.listdir(MEDSAM2_MASK_DIR))} 张)")
    print(f"SAM-Med2D 掩码: {SAMMED2D_MASK_DIR}/ ({len(os.listdir(SAMMED2D_MASK_DIR))} 张)")
    print(f"投票掩码: {VOTE_MASK_DIR}/ ({len(os.listdir(VOTE_MASK_DIR))} 张)")
    print(f"\n最终结果目录 (使用投票结果):")
    print(f"  原图: {RESULT_IMAGE_DIR}/ ({result_img_count} 张)")
    print(f"  掩码: {RESULT_MASK_DIR}/ ({result_mask_count} 张)")


if __name__ == "__main__":
    run_sam_refine(save_detail=False)
