"""
LLM 正负点评分模块 - 使用 LLM 对分割掩码进行评分并生成正负点
支持多进程并行调用大模型
"""
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
import multiprocessing as mp
from functools import partial
from llm.client_llm import send_generate_request
import config

# 导入工具函数
from common.image_utils import encode_image, load_prompt_template
from common.visualization_utils import (
    add_axis_labels, draw_points_on_image, visualize_clusters, visualize_bbox
)
from common.mask_utils import (
    ORGANS, extract_single_organ_mask, overlay_single_mask
)
from common.clustering_utils import dbscan_crop
from common.json_utils import parse_llm_response

# 配置常量
VENDOR = "vendorB"
SPLIT = "train"  # 数据集划分: "train" 或 "test"

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

IMAGE_DIR = f"{_WS}/datas/mnms/{VENDOR}/{SPLIT}/image"
# 伪标签目录 - 需要先运行 IPLC 生成对应的伪标签
MASK_DIR = f"{_WS}/asga/heart/iplc/infer_mask/{VENDOR}/{SPLIT}"
GT_DIR = f"{_WS}/datas/mnms/{VENDOR}/{SPLIT}/mask"
# Prompt template ships alongside this module under llm/system_prompts/.
PROMPT_PATH = _os.path.join(
    _os.path.dirname(__file__), "system_prompts",
    "system_prompt_single_organ.txt",
)
TARGET_SIZE = 256

# 模型名称（用于目录命名，去除特殊字符）
MODEL_NAME = config.OPENAI_MODEL.replace("/", "_").replace(":", "_")

# LLM 评分结果保存目录（包含模型名称）
LLM_SCORE_DIR = f"{_WS}/asga/tests/llm_scores/{MODEL_NAME}/{VENDOR}_{SPLIT}"
DEBUG_DIR = f"{_WS}/asga/tests/debug_output_llm/{MODEL_NAME}/{VENDOR}_{SPLIT}"

# 生成范围控制
START_INDEX = 0     # 从第1张开始
END_INDEX = 1400    # 覆盖完整 vendor split 的安全上限


def score_single_organ(original_img: Image.Image, overlay_img: Image.Image, organ_key: str, prompt_template: str) -> dict:
    """
    评分单个器官 - 使用 LLM 调用获取正负点
    
    Args:
        original_img: 原始图像
        overlay_img: 叠加掩码后的图像
        organ_key: 器官键值 ("LV", "MYO", "RV")
        prompt_template: 系统提示词模板
        
    Returns:
        dict: 包含评分、决策和正负点的字典
    """
    organ_info = ORGANS[organ_key]
    system_prompt = prompt_template.replace("{organ_name}", organ_info["name"]).replace("{organ_color}", organ_info["display_color"])
    
    original_with_axis = add_axis_labels(original_img)
    overlay_with_axis = add_axis_labels(overlay_img)
    
    b64_original = encode_image(original_with_axis)
    b64_overlay = encode_image(overlay_with_axis)
    user_prompt = (
        f"请评估 {organ_info['name']} 的分割掩码质量。\n"
        f"图1是原始心脏MRI图像，图2是叠加了{organ_info['display_color']}掩码的图像。\n\n"
        f"【强制要求】严格按照系统提示词中的JSON格式输出，必须包含以下字段：\n"
        f"- scores: {{\"semantic\": <分数>, \"morphology\": <分数>, \"boundary\": <分数>, \"total\": <分数>}}\n"
        f"- decision: \"KEEP\" 或 \"REFINE\" 或 \"REGENERATE\"\n"
        f"- fp_points: [[x, y]] （负点坐标，必填，严格只能1个点）\n"
        f"- fn_points: [[x, y]] （正点坐标，必填，严格只能1个点）\n\n"
        f"【禁止】不要输出任何解释文字，直接返回JSON对象，以 {{ 开头，以 }} 结尾。"
    )
    
    # 根据配置决定是否将 system prompt 合并到用户提示中（Gemini 等模型需要）
    if config.MERGE_SYSTEM_PROMPT:
        combined_prompt = f"【系统指令】\n{system_prompt}\n\n【用户请求】\n{user_prompt}"
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_original}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_overlay}", "detail": "high"}},
                {"type": "text", "text": combined_prompt}
            ]}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_original}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_overlay}", "detail": "high"}},
                {"type": "text", "text": user_prompt}
            ]}
        ]
    
    resp = send_generate_request(
        messages=messages,
        server_url=config.OPENAI_BASE_URL,
        model=config.OPENAI_MODEL,
        api_key=config.OPENAI_API_KEY
    )
    
    return parse_llm_response(resp)


def process_single_image_llm(img_id: str, prompt_template: str, save_detail: bool = False) -> dict:
    """
    处理单张图像的 LLM 评分流程
    
    Args:
        img_id: 图像 ID
        prompt_template: 系统提示词模板
        save_detail: 是否保存详细调试图像
        
    Returns:
        dict: 包含所有器官评分结果的字典，如果处理失败返回 None
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(LLM_SCORE_DIR, exist_ok=True)
    
    img_path = f"{IMAGE_DIR}/{img_id}.png"
    mask_path = f"{MASK_DIR}/{img_id}.png"
    
    # 检查原图是否存在
    if not os.path.exists(img_path):
        print(f"  [SKIP] 原图不存在: {img_path}")
        return None
    
    # 检查伪标签是否存在
    if not os.path.exists(mask_path):
        print(f"  [SKIP] 伪标签不存在: {mask_path}")
        return None
    
    original = Image.open(img_path).resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
    mask = Image.open(mask_path).resize((TARGET_SIZE, TARGET_SIZE), Image.NEAREST)
    
    mask_arr = np.array(mask)
    if mask_arr.ndim == 3:
        mask_arr = np.any(mask_arr > 0, axis=2).astype(np.uint8)
    
    bbox, labels, debug_info = dbscan_crop(mask_arr)
    if bbox is None:
        print(f"  [SKIP] 无有效区域")
        return None
    
    if save_detail:
        points = np.argwhere(mask_arr > 0)
        visualize_clusters(mask_arr, points, labels, f"{DEBUG_DIR}/{img_id}_1_clusters.png")
        visualize_bbox(original, bbox, f"{DEBUG_DIR}/{img_id}_2_bbox.png")
    
    cropped_original = original.crop(bbox)
    cropped_mask = mask.crop(bbox)
    
    if save_detail:
        cropped_original.save(f"{DEBUG_DIR}/{img_id}_3_cropped_original.png")
        cropped_mask.save(f"{DEBUG_DIR}/{img_id}_4_cropped_mask.png")
    
    # 对每个器官进行评分
    all_results = {
        "img_id": img_id,
        "bbox": [int(x) for x in bbox],  # 保存 bbox 供后续使用，转换为原生 int
        "organs": {}
    }
    
    for organ_key in ["LV", "MYO", "RV"]:
        single_mask = extract_single_organ_mask(cropped_mask, organ_key)
        overlay_img = overlay_single_mask(cropped_original, single_mask, organ_key)
        
        if save_detail:
            idx = {"LV": 5, "MYO": 6, "RV": 7}[organ_key]
            overlay_img.save(f"{DEBUG_DIR}/{img_id}_{idx}_{organ_key}_overlay.png")
            add_axis_labels(cropped_original).save(f"{DEBUG_DIR}/{img_id}_{idx}_{organ_key}_original_axis.png")
            add_axis_labels(overlay_img).save(f"{DEBUG_DIR}/{img_id}_{idx}_{organ_key}_overlay_axis.png")
        
        try:
            result = score_single_organ(cropped_original, overlay_img, organ_key, prompt_template)
        except Exception as e:
            err_msg = str(e)
            if "data_inspection_failed" in err_msg:
                print(f"  [{organ_key} SKIP] 图片被内容审核拦截，跳过")
                result = {"scores": {"semantic": 0, "morphology": 0, "boundary": 0, "total": 0}, "decision": "ERROR_CONTENT_FILTER", "fp_points": [], "fn_points": []}
            else:
                print(f"  [{organ_key} ERROR] {err_msg}")
                result = {"scores": {"semantic": 0, "morphology": 0, "boundary": 0, "total": 0}, "decision": "ERROR", "fp_points": [], "fn_points": []}
        all_results["organs"][organ_key] = result
        decision = result.get("decision")
        total_score = result.get('scores', {}).get('total', 0)
        print(f"  {organ_key}: {decision} (score={total_score})")
        
        if save_detail:
            draw_points_on_image(overlay_img, result, organ_key, f"{DEBUG_DIR}/{img_id}_{idx+3}_{organ_key}_error_points.png")
    
    # 保存评分结果到 JSON 文件
    output_path = f"{LLM_SCORE_DIR}/{img_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"  评分结果已保存到: {output_path}")
    
    return all_results


# 多进程配置
NUM_WORKERS = 5


def _worker_process_image(args):
    """
    单个 worker 处理一张图像（用于多进程 map）
    
    Args:
        args: (worker_id, task_index, total_tasks, img_id, save_detail)
    
    Returns:
        img_id 或 None（失败时）
    """
    worker_id, task_index, total_tasks, img_id, save_detail = args
    prompt_template = load_prompt_template(PROMPT_PATH)
    
    print(f"  [Worker-{worker_id}] [{task_index+1}/{total_tasks}] LLM 评分 {img_id}...")
    
    try:
        result = process_single_image_llm(img_id, prompt_template, save_detail=save_detail)
        if result:
            return img_id
    except Exception as e:
        print(f"  [Worker-{worker_id}] [{img_id} FATAL] {e}")
    return None


def run_llm_scoring(save_detail: bool = True, num_workers: int = NUM_WORKERS):
    """
    运行 LLM 评分流程（多进程版本）
    
    Args:
        save_detail: 是否保存详细调试图像
        num_workers: 并行进程数
    """
    os.makedirs(LLM_SCORE_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # 获取原图和伪标签的文件列表
    all_img_files = set(f for f in os.listdir(IMAGE_DIR) if f.endswith('.png'))
    all_mask_files = set(f for f in os.listdir(MASK_DIR) if f.endswith('.png'))
    
    # 计算交集 - 只处理同时存在原图和伪标签的文件
    matched_files = sorted(all_img_files & all_mask_files)
    only_in_img = all_img_files - all_mask_files
    only_in_mask = all_mask_files - all_img_files
    
    print(f"\n{'='*60}")
    print(f"文件匹配检测:")
    print(f"  原图目录: {IMAGE_DIR}")
    print(f"  伪标签目录: {MASK_DIR}")
    print(f"  原图数量: {len(all_img_files)}")
    print(f"  伪标签数量: {len(all_mask_files)}")
    print(f"  匹配数量: {len(matched_files)}")
    if only_in_img:
        print(f"  [警告] {len(only_in_img)} 张原图缺少伪标签")
    if only_in_mask:
        print(f"  [警告] {len(only_in_mask)} 张伪标签缺少原图")
    print(f"{'='*60}\n")
    
    # 按范围截取匹配的文件
    img_files = matched_files[START_INDEX:END_INDEX]
    
    # 跳过已完成的文件
    todo_files = []
    skip_done = 0
    for f in img_files:
        img_id = os.path.splitext(f)[0]
        json_path = f"{LLM_SCORE_DIR}/{img_id}.json"
        if os.path.exists(json_path):
            skip_done += 1
        else:
            todo_files.append(f)
    
    total_tasks = len(todo_files)
    print(f"总匹配图像数: {len(matched_files)}, 处理范围: [{START_INDEX}, {END_INDEX})")
    print(f"已完成(跳过): {skip_done}, 待处理: {total_tasks}")
    print(f"并行进程数: {num_workers}")
    print(f"模型: {config.OPENAI_MODEL}")
    print(f"深度思考: {getattr(config, 'ENABLE_THINKING', False)}")
    
    if total_tasks == 0:
        print("\n所有图像已处理完毕，无需重复处理。")
        return
    
    # 构造任务列表: (worker_id, task_index, total_tasks, img_id, save_detail)
    task_args = []
    for i, img_file in enumerate(todo_files):
        img_id = os.path.splitext(img_file)[0]
        worker_id = i % num_workers
        task_args.append((worker_id, i, total_tasks, img_id, save_detail))
    
    # 多进程执行
    print(f"\n启动 {num_workers} 个进程开始评分...\n")
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_worker_process_image, task_args)
    
    # 统计结果
    success_ids = [r for r in results if r is not None]
    fail_count = len(results) - len(success_ids)
    
    # 保存汇总信息（扫描目录获取全部已完成的文件）
    all_done = sorted([
        os.path.splitext(f)[0] for f in os.listdir(LLM_SCORE_DIR)
        if f.endswith('.json') and f != 'summary.json'
    ])
    
    summary = {
        "total_images": len(all_done),
        "this_run_success": len(success_ids),
        "this_run_fail": fail_count,
        "num_workers": num_workers,
        "model": config.OPENAI_MODEL,
        "processed_ids": all_done
    }
    
    with open(f"{LLM_SCORE_DIR}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"完成！本次成功: {len(success_ids)}, 失败: {fail_count}")
    print(f"目录内总计: {len(all_done)} 张")
    print(f"评分结果保存在: {LLM_SCORE_DIR}/")
    if save_detail:
        print(f"调试图像保存在: {DEBUG_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_llm_scoring(save_detail=False, num_workers=NUM_WORKERS)
