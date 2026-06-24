"""
SAM-Med2D Refiner - 使用点提示和掩码提示进行心脏 MRI 图像分割
基于 SAM-Med2D (ViT-B, image_size=256, encoder_adapter=True)
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
import sys
import os
import numpy as np
import torch
import cv2
from typing import Optional
from PIL import Image

SAMMED2D_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "SAM-Med2D"
)

# 临时添加到 sys.path 末尾以导入 segment_anything，
# 导入后立即移除，避免 SAM-Med2D/utils.py 污染全局命名空间
_added = SAMMED2D_PATH not in sys.path
if _added:
    sys.path.append(SAMMED2D_PATH)

from types import SimpleNamespace
from segment_anything import sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor

if _added:
    sys.path.remove(SAMMED2D_PATH)

SAMMED2D_CHECKPOINT = f"{_WS}/SFDA_SAM/heart/sam-med2d_b.pth"

ORGAN_MASK_WEIGHTS = {
    "LV": 1.5,
    "MYO": 2.0,
    "RV": 2.0,
}


class SAMMed2DRefiner:
    """SAM-Med2D 图像分割器，与 SAM3Refiner / MedSAM2Refiner 保持相同接口"""

    def __init__(
        self,
        checkpoint_path: str = SAMMED2D_CHECKPOINT,
        image_size: int = 256,
        device: str = "cuda:0",
    ) -> None:
        print(f"Loading SAM-Med2D from {checkpoint_path}...")

        sam_args = SimpleNamespace(
            image_size=image_size,
            sam_checkpoint=checkpoint_path,
            encoder_adapter=True,
        )
        sam_model = sam_model_registry["vit_b"](sam_args).to(device)
        sam_model.eval()
        self.predictor = SammedPredictor(sam_model)
        self.device = device
        self.image_size = image_size
        print("SAM-Med2D loaded successfully!")

    def set_image(self, img: Image.Image):
        """
        设置待分割的图像。

        Args:
            img: PIL Image (RGB)
        Returns:
            None (状态存储在 predictor 内部)
        """
        if isinstance(img, Image.Image):
            img_np = np.array(img.convert("RGB"))
        else:
            img_np = img
        self.predictor.set_image(img_np)
        return None

    def refine(
        self,
        state,
        y_init: np.ndarray,
        pos_points: Optional[np.ndarray] = None,
        neg_points: Optional[np.ndarray] = None,
        box_xyxy: Optional[np.ndarray] = None,
        mask_weight: float = 2.0,
    ) -> np.ndarray:
        """
        使用提示进行分割优化，接口与 SAM3Refiner.refine 对齐。

        Args:
            state: 兼容参数（不使用）
            y_init: 初始二值掩码 (H, W)，用于生成 mask logits
            pos_points: 正样本点 [[x, y], ...]
            neg_points: 负样本点 [[x, y], ...]
            box_xyxy: 边界框 [x1, y1, x2, y2]（可选）
            mask_weight: 掩码权重（控制 logits 强度，参照 SAM3 的用法）

        Returns:
            np.ndarray: 分割掩码 (H, W)，bool/uint8
        """
        point_coords = None
        point_labels = None

        if pos_points is not None or neg_points is not None:
            pts, labs = [], []
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
                point_labels = np.array(labs, dtype=np.int64)

        mask_input = None
        if y_init is not None and mask_weight > 0:
            mask_input = self._mask_to_logits(y_init, mask_weight=mask_weight)

        masks, iou_predictions, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_xyxy,
            mask_input=mask_input,
            multimask_output=False,
            return_logits=False,
        )
        return masks[0]

    @staticmethod
    def _mask_to_logits(
        mask: np.ndarray,
        target_size: int = 64,
        mask_weight: float = 2.0,
    ) -> np.ndarray:
        """
        将二值掩码转为低分辨率 logits。

        SAM-Med2D (image_size=256, vit_patch_size=16)
        → image_embedding_size = 256/16 = 16
        → mask_input_size = 4 * 16 = 64

        predictor.predict 内部会再加一个 batch 维度 → (1, 1, 64, 64)

        Args:
            mask: (H, W) 二值掩码
            target_size: 低分辨率目标尺寸 (64)
            mask_weight: logit 幅度

        Returns:
            np.ndarray: (1, 64, 64) logits
        """
        m = (mask > 0).astype(np.float32)
        m = cv2.resize(m, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        logits = np.where(m > 0, mask_weight, -mask_weight).astype(np.float32)
        return logits[np.newaxis, ...]
