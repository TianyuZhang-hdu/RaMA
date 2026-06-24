"""
图像处理和编码相关工具函数
"""

import base64
import io
import cv2
import numpy as np
from PIL import Image


def resize_image_for_sam3(img: np.ndarray, target: int = 1024) -> np.ndarray:
    """
    调整图像大小以适配 SAM3 模型
    
    Args:
        img: 输入图像数组
        target: 目标尺寸（默认1024）
    
    Returns:
        调整大小后的图像数组
    """
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target, target, 3), dtype=img.dtype)
    canvas[:new_h, :new_w] = resized
    return canvas


def resize_mask_for_sam3(mask: np.ndarray, target: int = 256) -> np.ndarray:
    """
    调整掩码大小以适配 SAM3 模型
    
    Args:
        mask: 输入掩码数组
        target: 目标尺寸（默认256）
    
    Returns:
        调整大小后的掩码数组
    """
    h, w = mask.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target, target), dtype=mask.dtype)
    canvas[:new_h, :new_w] = resized
    return canvas


def encode_image(image: Image.Image) -> str:
    """
    将 PIL Image 编码为 base64 字符串
    
    Args:
        image: PIL Image 对象
    
    Returns:
        base64 编码的字符串
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_prompt_template(prompt_path: str) -> str:
    """
    加载系统提示词模板
    
    Args:
        prompt_path: 提示词文件路径
    
    Returns:
        提示词模板内容
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

