"""
掩码处理相关工具函数
包括提取、叠加、应用掩码等操作
"""

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial.distance import cdist


# 器官配置常量
ORGANS = {
    "LV": {"color": (255, 0, 0), "name": "左心室 (Left Ventricle)", "display_color": "红色"},
    "MYO": {"color": (0, 255, 0), "name": "心肌 (Myocardium)", "display_color": "绿色"},
    "RV": {"color": (0, 0, 255), "name": "右心室 (Right Ventricle)", "display_color": "蓝色"},
}

# 掩码透明度
MASK_ALPHA = 0.45


def extract_single_organ_mask(mask: Image.Image, organ_key: str) -> Image.Image:
    """
    从完整掩码中提取单个器官的掩码
    
    Args:
        mask: 完整的 RGB 掩码图像
        organ_key: 器官键值（"LV", "MYO", "RV"）
    
    Returns:
        单个器官的掩码图像
    """
    mask_arr = np.array(mask.convert("RGB"))
    organ_color = ORGANS[organ_key]["color"]
    h, w = mask_arr.shape[:2]
    result = Image.new("RGB", (w, h), (0, 0, 0))
    result_arr = np.array(result)
    
    if organ_key == "LV":
        mask_region = mask_arr[:, :, 0] > 127
    elif organ_key == "MYO":
        mask_region = mask_arr[:, :, 1] > 127
    else:
        mask_region = mask_arr[:, :, 2] > 127
    
    result_arr[mask_region] = organ_color
    return Image.fromarray(result_arr)


def overlay_single_mask(image: Image.Image, mask: Image.Image, organ_key: str) -> Image.Image:
    """
    将单个器官掩码叠加到图像上
    
    Args:
        image: 原始图像
        mask: 单个器官的掩码
        organ_key: 器官键值（"LV", "MYO", "RV"）
    
    Returns:
        叠加了掩码的图像
    """
    img = image.convert("RGBA")
    w, h = img.size
    mask_resized = mask.convert("RGB").resize((w, h), Image.NEAREST)
    mask_arr = np.array(mask_resized)
    mask_rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_rgba)
    alpha_int = int(255 * MASK_ALPHA)
    organ_color = ORGANS[organ_key]["color"]
    
    for y in range(h):
        for x in range(w):
            r, g, b = mask_arr[y, x]
            if r > 127 or g > 127 or b > 127:
                mask_draw.point((x, y), fill=(*organ_color, alpha_int))
    
    img = Image.alpha_composite(img, mask_rgba)
    return img.convert("RGB")


def get_organ_mask_arr(mask: Image.Image, organ_key: str) -> np.ndarray:
    """
    获取单个器官的掩码数组
    
    Args:
        mask: 完整的 RGB 掩码图像
        organ_key: 器官键值（"LV", "MYO", "RV"）
    
    Returns:
        单个器官的二值掩码数组
    """
    mask_arr = np.array(mask.convert("RGB"))
    if organ_key == "LV":
        return (mask_arr[:, :, 0] > 127).astype(np.uint8)
    elif organ_key == "MYO":
        return (mask_arr[:, :, 1] > 127).astype(np.uint8)
    else:
        return (mask_arr[:, :, 2] > 127).astype(np.uint8)


def apply_refined_mask(full_mask: Image.Image, refined_arr: np.ndarray, organ_key: str) -> Image.Image:
    """
    将修复后的掩码应用到完整掩码中
    
    Args:
        full_mask: 完整的 RGB 掩码图像
        refined_arr: 修复后的掩码数组
        organ_key: 器官键值（"LV", "MYO", "RV"）
    
    Returns:
        更新后的完整掩码图像
    """
    mask_arr = np.array(full_mask.convert("RGB"))
    channel = {"LV": 0, "MYO": 1, "RV": 2}[organ_key]
    mask_arr[:, :, channel] = (refined_arr > 0).astype(np.uint8) * 255
    return Image.fromarray(mask_arr)


def overlay_all_masks(img: Image.Image, mask: Image.Image) -> Image.Image:
    """
    将所有器官掩码叠加到图像上
    
    Args:
        img: 原始图像
        mask: 完整的 RGB 掩码图像
    
    Returns:
        叠加了所有掩码的图像
    """
    result = img.convert("RGBA")
    mask_arr = np.array(mask.convert("RGB"))
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    alpha = int(255 * MASK_ALPHA)
    myo = mask_arr[:, :, 1] > 127
    lv = mask_arr[:, :, 0] > 127
    rv = mask_arr[:, :, 2] > 127
    overlay_arr[myo] = (*ORGANS["MYO"]["color"], alpha)
    overlay_arr[lv] = (*ORGANS["LV"]["color"], alpha)
    overlay_arr[rv] = (*ORGANS["RV"]["color"], alpha)
    overlay = Image.fromarray(overlay_arr, "RGBA")
    return Image.alpha_composite(result, overlay).convert("RGB")


def compute_dice(mask1: Image.Image, mask2: Image.Image, organ_key: str = None) -> float:
    """
    计算两个掩码之间的 Dice 系数
    
    Args:
        mask1: 第一个掩码图像
        mask2: 第二个掩码图像
        organ_key: 如果指定，只计算该器官的 Dice；否则计算整体 Dice
    
    Returns:
        Dice 系数 (0-1)
    """
    arr1 = np.array(mask1.convert("RGB"))
    arr2 = np.array(mask2.convert("RGB"))
    
    if organ_key:
        # 计算单个器官的 Dice
        channel = {"LV": 0, "MYO": 1, "RV": 2}[organ_key]
        m1 = arr1[:, :, channel] > 127
        m2 = arr2[:, :, channel] > 127
    else:
        # 计算整体 Dice（任意通道有值即为前景）
        m1 = np.any(arr1 > 127, axis=2)
        m2 = np.any(arr2 > 127, axis=2)
    
    intersection = np.sum(m1 & m2)
    union = np.sum(m1) + np.sum(m2)
    
    if union == 0:
        return 1.0  # 两个都是空的，认为完全一致
    
    return 2.0 * intersection / union


def compute_all_dice(mask1: Image.Image, mask2: Image.Image) -> dict:
    """
    计算两个掩码之间所有器官的 Dice 系数
    
    Args:
        mask1: 第一个掩码图像
        mask2: 第二个掩码图像
    
    Returns:
        dict: {"LV": dice_lv, "MYO": dice_myo, "RV": dice_rv, "mean": mean_dice}
    """
    dice_scores = {}
    for organ_key in ["LV", "MYO", "RV"]:
        dice_scores[organ_key] = compute_dice(mask1, mask2, organ_key)
    
    dice_scores["mean"] = np.mean([dice_scores["LV"], dice_scores["MYO"], dice_scores["RV"]])
    return dice_scores


def get_surface_points(mask: np.ndarray) -> np.ndarray:
    """
    获取二值掩码的表面点（边界点）
    
    Args:
        mask: 二值掩码数组
    
    Returns:
        表面点坐标数组 (N, 2)
    """
    if mask.sum() == 0:
        return np.array([]).reshape(0, 2)
    
    # 使用形态学操作获取边界
    eroded = ndimage.binary_erosion(mask)
    surface = mask.astype(bool) & ~eroded
    
    # 获取边界点坐标
    points = np.argwhere(surface)  # (y, x) 格式
    return points


def compute_assd(mask1: Image.Image, mask2: Image.Image, organ_key: str = None) -> float:
    """
    计算两个掩码之间的 ASSD (Average Symmetric Surface Distance)
    
    Args:
        mask1: 第一个掩码图像
        mask2: 第二个掩码图像
        organ_key: 如果指定，只计算该器官的 ASSD；否则计算整体 ASSD
    
    Returns:
        ASSD 值（像素单位），如果无法计算返回 float('inf')
    """
    arr1 = np.array(mask1.convert("RGB"))
    arr2 = np.array(mask2.convert("RGB"))
    
    if organ_key:
        # 计算单个器官的 ASSD
        channel = {"LV": 0, "MYO": 1, "RV": 2}[organ_key]
        m1 = arr1[:, :, channel] > 127
        m2 = arr2[:, :, channel] > 127
    else:
        # 计算整体 ASSD
        m1 = np.any(arr1 > 127, axis=2)
        m2 = np.any(arr2 > 127, axis=2)
    
    # 获取表面点
    surface1 = get_surface_points(m1)
    surface2 = get_surface_points(m2)
    
    # 处理空掩码的情况
    if len(surface1) == 0 and len(surface2) == 0:
        return 0.0  # 两个都是空的
    if len(surface1) == 0 or len(surface2) == 0:
        return float('inf')  # 一个空一个不空
    
    # 计算双向表面距离
    dist_1_to_2 = cdist(surface1, surface2, 'euclidean').min(axis=1)
    dist_2_to_1 = cdist(surface2, surface1, 'euclidean').min(axis=1)
    
    # ASSD = (mean(d1->d2) + mean(d2->d1)) / 2
    assd = (np.mean(dist_1_to_2) + np.mean(dist_2_to_1)) / 2.0
    return assd


def compute_all_assd(mask1: Image.Image, mask2: Image.Image) -> dict:
    """
    计算两个掩码之间所有器官的 ASSD
    
    Args:
        mask1: 第一个掩码图像
        mask2: 第二个掩码图像
    
    Returns:
        dict: {"LV": assd_lv, "MYO": assd_myo, "RV": assd_rv, "mean": mean_assd}
    """
    assd_scores = {}
    valid_scores = []
    
    for organ_key in ["LV", "MYO", "RV"]:
        assd = compute_assd(mask1, mask2, organ_key)
        assd_scores[organ_key] = assd
        if assd != float('inf'):
            valid_scores.append(assd)
    
    # 计算平均值（排除 inf）
    assd_scores["mean"] = np.mean(valid_scores) if valid_scores else float('inf')
    return assd_scores


def compute_all_metrics(pred_mask: Image.Image, gt_mask: Image.Image) -> dict:
    """
    计算预测掩码与真实标签之间的所有评估指标
    
    Args:
        pred_mask: 预测掩码图像
        gt_mask: 真实标签图像
    
    Returns:
        dict: {
            "dice": {"LV": ..., "MYO": ..., "RV": ..., "mean": ...},
            "assd": {"LV": ..., "MYO": ..., "RV": ..., "mean": ...}
        }
    """
    return {
        "dice": compute_all_dice(pred_mask, gt_mask),
        "assd": compute_all_assd(pred_mask, gt_mask)
    }

