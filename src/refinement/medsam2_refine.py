"""
MedSAM2 Refiner - 使用框提示和掩码提示进行心脏 MRI 图像分割
支持 LV、MYO、RV 三个器官的分割优化
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
                   _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..", "configs", "rama_config.yaml")):
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
import numpy as np
import torch
import cv2
import sys
from typing import Optional
from PIL import Image

# 添加 MedSAM2 路径
MEDSAM2_PATH = f"{_WS}/MedSAM2"
if MEDSAM2_PATH not in sys.path:
    sys.path.insert(0, MEDSAM2_PATH)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 默认配置
MEDSAM2_CONFIG = "configs/sam2.1_hiera_t512.yaml"
MEDSAM2_CHECKPOINT = f"{_WS}/MedSAM2/checkpoints/MedSAM2_latest.pt"

# 心脏器官的默认掩码权重
ORGAN_MASK_WEIGHTS = {
    "LV": 1.5,   # 左心室 - 中等约束
    "MYO": 2.5,  # 心肌 - 较强约束（形状较薄）
    "RV": 2.0,   # 右心室 - 中等约束
}


class MedSAM2Refiner:
    """MedSAM2 图像分割器，支持框提示、点提示和掩码提示"""
    
    def __init__(
        self, 
        config_file: str = MEDSAM2_CONFIG,
        checkpoint_path: str = MEDSAM2_CHECKPOINT,
        device: str = "cuda:0"
    ) -> None:
        """
        初始化 MedSAM2 模型
        
        Args:
            config_file: 模型配置文件路径
            checkpoint_path: 模型权重路径
            device: 运行设备
        """
        print(f"Loading MedSAM2 from {checkpoint_path}...")
        
        # 构建模型
        sam2_model = build_sam2(
            config_file=config_file,
            ckpt_path=checkpoint_path,
            device=device,
            mode="eval"
        )
        
        # 创建图像预测器
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.device = device
        print("MedSAM2 loaded successfully!")
    
    def set_image(self, img: Image.Image):
        """
        设置待分割的图像
        
        Args:
            img: PIL Image 对象
            
        Returns:
            None (状态存储在 predictor 内部)
        """
        # 转换为 numpy 数组
        if isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"))
        else:
            img_np = img
        
        self.predictor.set_image(img_np)
        return None  # 返回 None，状态在 predictor 内部
    
    def refine(
        self,
        state,  # 保留参数以兼容 SAM3Refiner 接口，但不使用
        y_init: np.ndarray,
        pos_points: Optional[np.ndarray] = None,
        neg_points: Optional[np.ndarray] = None,
        box_xyxy: Optional[np.ndarray] = None,
        mask_weight: float = 2.0,
        use_mask_input: bool = True,
    ) -> np.ndarray:
        """
        使用提示进行分割优化
        
        Args:
            state: 兼容参数（不使用）
            y_init: 初始掩码，用于生成 mask_input
            pos_points: 正样本点 [[x, y], ...]
            neg_points: 负样本点 [[x, y], ...]
            box_xyxy: 边界框 [x1, y1, x2, y2]
            mask_weight: 掩码权重（用于控制 logits 强度）
            use_mask_input: 是否使用掩码输入
            
        Returns:
            np.ndarray: 分割掩码 (H, W)，值为 True/False
        """
        # 准备点提示
        point_coords = None
        point_labels = None
        
        if pos_points is not None or neg_points is not None:
            pts = []
            labs = []
            if pos_points is not None:
                for p in pos_points:
                    pts.append(p)
                    labs.append(1)
            if neg_points is not None:
                for p in neg_points:
                    pts.append(p)
                    labs.append(0)
            if len(pts) > 0:
                point_coords = np.array(pts, dtype=np.float32)
                point_labels = np.array(labs, dtype=np.int32)
        
        # 准备掩码输入
        # MedSAM2 期望 mask_input 形状为 (1, 1, 256, 256)
        mask_input = None
        if use_mask_input and y_init is not None and y_init.sum() > 0:
            mask_input = self._mask_to_logits(y_init, mask_weight=mask_weight)
        
        # 执行预测
        masks, iou_predictions, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_xyxy,
            mask_input=mask_input,
            multimask_output=False,
            return_logits=False,
        )
        
        # 返回第一个掩码
        return masks[0]
    
    def _mask_to_logits(
        self, 
        mask: np.ndarray, 
        target_size: int = 128,  # MedSAM2: 4 * (image_size/backbone_stride) = 4 * (512/16) = 128
        mask_weight: float = 2.0
    ) -> np.ndarray:
        """
        将二值掩码转换为低分辨率 logits
        
        MedSAM2 期望 mask_input 形状为 (1, 1, 128, 128)
        计算方式: mask_input_size = 4 * image_embedding_size
                 image_embedding_size = image_size / backbone_stride = 512 / 16 = 32
                 mask_input_size = 4 * 32 = 128
        
        Args:
            mask: 输入掩码 (H, W)
            target_size: 目标尺寸（MedSAM2 固定为 128）
            mask_weight: 掩码权重（控制 logits 的强度）
            
        Returns:
            np.ndarray: logits (1, 1, 128, 128)
        """
        # 转换为浮点数
        m = (mask > 0).astype(np.float32)
        
        # 调整大小到 128x128（MedSAM2 要求的固定尺寸）
        m = cv2.resize(m, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
        # 转换为 logits
        # mask_weight 控制置信度：1-3弱, 3-5中, 5-10强
        pos_logit = mask_weight * 3.0  # 放大效果
        neg_logit = -mask_weight * 3.0
        logits = np.where(m > 0, pos_logit, neg_logit).astype(np.float32)
        
        # MedSAM2 期望形状为 (1, 1, 128, 128)
        return logits[np.newaxis, np.newaxis, ...]  # 添加 batch 和 channel 维度
    
    def refine_organ(
        self,
        y_init: np.ndarray,
        organ_key: str,
        pos_points: Optional[np.ndarray] = None,
        neg_points: Optional[np.ndarray] = None,
        box_xyxy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        针对特定器官进行分割优化（使用预设的权重）
        
        Args:
            y_init: 初始掩码
            organ_key: 器官键值 ("LV", "MYO", "RV")
            pos_points: 正样本点
            neg_points: 负样本点
            box_xyxy: 边界框
            
        Returns:
            np.ndarray: 优化后的分割掩码
        """
        mask_weight = ORGAN_MASK_WEIGHTS.get(organ_key, 2.0)
        return self.refine(
            state=None,
            y_init=y_init,
            pos_points=pos_points,
            neg_points=neg_points,
            box_xyxy=box_xyxy,
            mask_weight=mask_weight,
            use_mask_input=True,
        )
    
    def predict_with_box(
        self,
        image: Image.Image,
        box_xyxy: np.ndarray,
        initial_mask: Optional[np.ndarray] = None,
        mask_weight: float = 2.0,
    ) -> np.ndarray:
        """
        便捷方法：使用框提示进行分割
        
        Args:
            image: 输入图像
            box_xyxy: 边界框 [x1, y1, x2, y2]
            initial_mask: 初始掩码（可选）
            mask_weight: 掩码权重
            
        Returns:
            np.ndarray: 分割掩码
        """
        self.set_image(image)
        
        # 如果有初始掩码，使用其中心点作为正样本点
        pos_points = None
        if initial_mask is not None and initial_mask.sum() > 0:
            coords = np.argwhere(initial_mask > 0)
            center_y, center_x = coords.mean(axis=0).astype(int)
            pos_points = np.array([[center_x, center_y]], dtype=np.float32)
        
        return self.refine(
            state=None,
            y_init=initial_mask,
            pos_points=pos_points,
            neg_points=None,
            box_xyxy=box_xyxy,
            mask_weight=mask_weight,
            use_mask_input=(initial_mask is not None),
        )


def create_refiner(model_type: str = "medsam2", **kwargs):
    """
    工厂函数：创建指定类型的 refiner
    
    Args:
        model_type: 模型类型 ("sam3" 或 "medsam2")
        **kwargs: 传递给 refiner 构造函数的参数
        
    Returns:
        SAM3Refiner 或 MedSAM2Refiner 实例
    """
    if model_type.lower() == "sam3":
        from refinement.sam3_refine import SAM3Refiner
        return SAM3Refiner()
    elif model_type.lower() == "medsam2":
        return MedSAM2Refiner(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'sam3' or 'medsam2'")

