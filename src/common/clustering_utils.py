"""
DBSCAN 聚类相关工具函数
用于掩码区域的聚类和裁剪
"""

import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image
from typing import Tuple, Optional, Dict


def get_organ_bbox(mask: Image.Image, organ_key: str, padding: int = 5) -> Optional[np.ndarray]:
    """
    从掩码中提取单个器官的边界框
    
    Args:
        mask: 掩码图像（RGB，LV=红, MYO=绿, RV=蓝）
        organ_key: 器官键名 ("LV", "MYO", "RV")
        padding: 边界框填充像素数
    
    Returns:
        np.ndarray: 边界框 [x_min, y_min, x_max, y_max]，如果无效则返回 None
    """
    mask_arr = np.array(mask.convert("RGB"))
    
    # 提取对应器官的掩码
    channel_map = {"LV": 0, "MYO": 1, "RV": 2}  # R, G, B
    channel = channel_map.get(organ_key)
    if channel is None:
        return None
    
    organ_mask = mask_arr[:, :, channel] > 127
    
    if organ_mask.sum() == 0:
        return None
    
    # 找到非零区域的边界
    coords = np.argwhere(organ_mask)  # (y, x) 格式
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 添加填充
    h, w = organ_mask.shape
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w - 1, x_max + padding)
    y_max = min(h - 1, y_max + padding)
    
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def get_all_organ_bboxes(mask: Image.Image, padding: int = 5) -> Dict[str, Optional[np.ndarray]]:
    """
    从掩码中提取所有器官的边界框
    
    Args:
        mask: 掩码图像
        padding: 边界框填充像素数
    
    Returns:
        dict: {器官名: 边界框} 的字典
    """
    return {
        organ_key: get_organ_bbox(mask, organ_key, padding)
        for organ_key in ["LV", "MYO", "RV"]
    }


def dbscan_crop(mask_arr: np.ndarray, eps: int = 30, min_samples: int = 10, padding_ratio: float = 0.3):
    """
    使用 DBSCAN 聚类算法找到掩码的主要区域并返回裁剪边界框
    
    Args:
        mask_arr: 二值掩码数组
        eps: DBSCAN 的邻域半径参数
        min_samples: DBSCAN 的最小样本数参数
        padding_ratio: 边界框的填充比例
    
    Returns:
        tuple: (bbox, labels, debug_info)
            - bbox: 裁剪边界框 (x_min, y_min, x_max, y_max)，如果失败则为 None
            - labels: DBSCAN 聚类标签数组，如果失败则为 None
            - debug_info: 包含调试信息的字典
    """
    points = np.argwhere(mask_arr > 0)
    if len(points) == 0:
        return None, None, {}
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    
    unique_labels = set(labels) - {-1}
    if not unique_labels:
        return None, None, {}
    
    cluster_sizes = {l: np.sum(labels == l) for l in unique_labels}
    main_label = max(cluster_sizes, key=cluster_sizes.get)
    
    selected_mask = labels == main_label
    selected_points = points[selected_mask]
    
    y_min, x_min = selected_points.min(axis=0)
    y_max, x_max = selected_points.max(axis=0)
    
    h, w = mask_arr.shape
    pad_y = int((y_max - y_min) * padding_ratio)
    pad_x = int((x_max - x_min) * padding_ratio)
    
    y_min = max(0, y_min - pad_y)
    y_max = min(h, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(w, x_max + pad_x)
    
    bbox = (x_min, y_min, x_max, y_max)
    
    debug_info = {
        'total_points': len(points),
        'n_clusters': len(unique_labels),
        'noise_points': np.sum(labels == -1),
        'cluster_sizes': cluster_sizes,
        'main_label': main_label,
        'bbox': bbox
    }
    
    return bbox, labels, debug_info

