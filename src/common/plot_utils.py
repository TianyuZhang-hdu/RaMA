"""
绘图相关工具函数
用于生成调试和结果可视化图表
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .mask_utils import ORGANS, extract_single_organ_mask, overlay_single_mask, overlay_all_masks


def plot_debug_results(
    cropped_original: Image.Image,
    cropped_mask: Image.Image,
    cropped_gt: Image.Image,
    refined_mask_all: Image.Image,
    all_results: dict,
    save_path: str
):
    """
    绘制调试结果的 2x4 网格图
    
    Args:
        cropped_original: 裁剪后的原图
        cropped_mask: 裁剪后的预测掩码
        cropped_gt: 裁剪后的 GT 掩码
        refined_mask_all: 修复后的掩码
        all_results: 所有器官的 LLM 评分结果
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 第一行：原图、Mask、GT、SAM3
    axes[0, 0].imshow(cropped_original, cmap='gray')
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(overlay_all_masks(cropped_original, cropped_mask))
    axes[0, 1].set_title("Original + Mask")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(overlay_all_masks(cropped_original, cropped_gt))
    axes[0, 2].set_title("Original + GT")
    axes[0, 2].axis("off")
    axes[0, 3].imshow(overlay_all_masks(cropped_original, refined_mask_all))
    axes[0, 3].set_title("Original + SAM3")
    axes[0, 3].axis("off")
    
    # 第二行：LV、MYO、RV 的正负点
    for idx, organ_key in enumerate(["LV", "MYO", "RV"]):
        ax = axes[1, idx]
        single_mask = extract_single_organ_mask(cropped_mask, organ_key)
        overlay_img = overlay_single_mask(cropped_original, single_mask, organ_key)
        ax.imshow(overlay_img)
        
        result = all_results.get(organ_key, {})
        fp_points = result.get("fp_points", [])
        fn_points = result.get("fn_points", [])
        
        for pt in fp_points:
            if len(pt) == 2:
                ax.scatter(pt[0], pt[1], c='yellow', marker='x', s=200, linewidths=3, label='FP' if pt == fp_points[0] else '')
        for pt in fn_points:
            if len(pt) == 2:
                ax.scatter(pt[0], pt[1], c='cyan', marker='o', s=200, facecolors='none', linewidths=3, label='FN' if pt == fn_points[0] else '')
        
        decision = result.get("decision", "?")
        score = result.get("scores", {}).get("total", 0)
        ax.set_title(f"{organ_key}: {decision}({score}) X=FP O=FN")
        ax.axis("off")
    
    # 第二行第四格：显示所有点的汇总
    axes[1, 3].imshow(overlay_all_masks(cropped_original, cropped_mask))
    for organ_key in ["LV", "MYO", "RV"]:
        result = all_results.get(organ_key, {})
        for pt in result.get("fp_points", []):
            if len(pt) == 2:
                axes[1, 3].scatter(pt[0], pt[1], c='yellow', marker='x', s=150, linewidths=2)
        for pt in result.get("fn_points", []):
            if len(pt) == 2:
                axes[1, 3].scatter(pt[0], pt[1], c='cyan', marker='o', s=150, facecolors='none', linewidths=2)
    axes[1, 3].set_title("All Points (X=FP, O=FN)")
    axes[1, 3].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_final_mask(
    refined_mask_all: Image.Image,
    bbox: tuple,
    target_size: int,
    save_path: str
):
    """
    保存最终的完整尺寸掩码
    
    Args:
        refined_mask_all: 修复后的掩码（裁剪区域）
        bbox: 裁剪边界框 (x_min, y_min, x_max, y_max)
        target_size: 目标尺寸
        save_path: 保存路径
    """
    ref_arr = np.array(refined_mask_all.convert("RGB"))
    myo = ref_arr[:, :, 1] > 127
    lv = ref_arr[:, :, 0] > 127
    rv = ref_arr[:, :, 2] > 127
    out = np.zeros_like(ref_arr)
    out[myo] = [0, 255, 0]
    out[lv] = [255, 0, 0]
    out[rv] = [0, 0, 255]
    full_mask = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    full_mask.paste(Image.fromarray(out), (bbox[0], bbox[1]))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    full_mask.save(save_path)

