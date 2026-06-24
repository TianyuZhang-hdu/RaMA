"""
可视化相关工具函数
包括添加坐标轴、绘制点、可视化聚类结果等
"""

import os
import io
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def add_axis_labels(image: Image.Image) -> Image.Image:
    """
    为图像添加坐标轴标签
    
    Args:
        image: 输入的 PIL Image 对象
    
    Returns:
        添加了坐标轴标签的 PIL Image 对象
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    h, w = image.size[1], image.size[0]
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')
    ax.set_xticks(np.linspace(0, w-1, 5).astype(int))
    ax.set_yticks(np.linspace(0, h-1, 5).astype(int))
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', dpi=150)
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def draw_points_on_image(image: Image.Image, result: dict, organ_key: str, save_path: str):
    """
    在图像上绘制错误点（FP 和 FN）
    
    Args:
        image: 输入图像
        result: 包含 fp_points 和 fn_points 的字典
        organ_key: 器官键值（"LV", "MYO", "RV"）
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    
    color = {'LV': 'red', 'MYO': 'green', 'RV': 'blue'}[organ_key]
    fp_points = result.get("fp_points", [])
    fn_points = result.get("fn_points", [])
    
    for pt in fp_points:
        if len(pt) == 2:
            ax.scatter(pt[0], pt[1], c=color, marker='x', s=150, linewidths=3)
    for pt in fn_points:
        if len(pt) == 2:
            ax.scatter(pt[0], pt[1], c=color, marker='o', s=150, facecolors='none', linewidths=3)
    
    ax.set_title(f'{organ_key} Error Points: X=FP({len(fp_points)}), O=FN({len(fn_points)})')
    ax.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def draw_all_points(img: Image.Image, all_points: dict) -> Image.Image:
    """
    在图像上绘制所有器官的错误点
    
    Args:
        img: 输入图像
        all_points: 包含所有器官错误点的字典
    
    Returns:
        绘制了错误点的图像
    """
    fig, ax = plt.subplots(figsize=(img.size[0]/100, img.size[1]/100), dpi=100)
    ax.imshow(img)
    for organ_key, pts in all_points.items():
        for pt in pts.get("fp", []):
            ax.scatter(pt[0], pt[1], c='yellow', marker='x', s=40, linewidths=1.5, label='FP' if organ_key == 'LV' else '')
        for pt in pts.get("fn", []):
            ax.scatter(pt[0], pt[1], c='cyan', marker='o', s=40, facecolors='none', linewidths=1.5, label='FN' if organ_key == 'LV' else '')
    ax.legend(loc='upper right', fontsize=6, framealpha=0.7)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize(img.size)


def visualize_clusters(mask_arr: np.ndarray, points: np.ndarray, labels: np.ndarray, save_path: str):
    """
    可视化 DBSCAN 聚类结果
    
    Args:
        mask_arr: 掩码数组
        points: 点坐标数组
        labels: 聚类标签数组
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(mask_arr, cmap='gray', alpha=0.3)
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(sorted(unique_labels), colors):
        mask = labels == label
        cluster_points = points[mask]
        label_name = f'Cluster {label}' if label != -1 else 'Noise'
        ax.scatter(cluster_points[:, 1], cluster_points[:, 0], c=[color], s=1, label=f'{label_name} ({len(cluster_points)})')
    ax.legend(loc='upper right')
    ax.set_title('DBSCAN Clustering Result')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_bbox(original: Image.Image, bbox: tuple, save_path: str):
    """
    可视化裁剪边界框
    
    Args:
        original: 原始图像
        bbox: 边界框 (x_min, y_min, x_max, y_max)
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original)
    x_min, y_min, x_max, y_max = bbox
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='lime', linewidth=3)
    ax.add_patch(rect)
    ax.set_title(f'Crop Region: ({x_min}, {y_min}) -> ({x_max}, {y_max})')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

