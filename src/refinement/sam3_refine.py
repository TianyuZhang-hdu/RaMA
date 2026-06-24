import os
import numpy as np
import torch
from typing import Optional, Tuple
from PIL import Image

from sam3.sam3.model_builder import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor

import config


def _downsample_mask_to_logits(mask: np.ndarray, target: int = 288, 
                                pos_logit: float = None, neg_logit: float = None) -> np.ndarray:
    import cv2
    
    if pos_logit is None:
        pos_logit = config.CONF_POS_LOGIT
    if neg_logit is None:
        neg_logit = config.CONF_NEG_LOGIT

    m = (mask > 0).astype(np.float32)
    m = cv2.resize(m, (target, target), interpolation=cv2.INTER_NEAREST)
    logits = (m * pos_logit) + ((1.0 - m) * neg_logit)
    return logits[None, ...]


# 三个器官的默认 mask_weight
ORGAN_MASK_WEIGHTS = {
    "LV": 1.1,   # 左心室
    "MYO": 2.0,  # 心肌
    "RV": 2.0,   # 右心室
}


class SAM3Refiner:
    def __init__(self) -> None:
        # 显式指定 bpe_path，避免 pkg_resources 查找失败
        bpe_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
        self.model = build_sam3_image_model(
            enable_segmentation=config.SAM3_ENABLE_SEG,
            enable_inst_interactivity=config.SAM3_ENABLE_INST,
            checkpoint_path=config.SAM3_CKPT_PATH,
            load_from_HF=False,
            bpe_path=bpe_path,
        )
        self.processor = Sam3Processor(self.model)

    def set_image(self, img: Image.Image):
        return self.processor.set_image(img)

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
        Args:
            mask_weight: 掩码权重 (0=不用原掩码, 1-3弱引导, 3-5中等, 5-10强引导)
        """
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
                point_labels = np.array(labs, dtype=np.int64)
        
        # 处理掩码输入
        mask_input = None
        if y_init is not None and mask_weight > 0:
            mask_input = _downsample_mask_to_logits(
                y_init, target=288,
                pos_logit=mask_weight,
                neg_logit=-mask_weight
            )
        
        masks, ious, low_res = self.model.predict_inst(
            state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_xyxy,
            mask_input=mask_input,
            multimask_output=False,
        )
        return masks[0]


def low_res_from_init(mask_logits_256: np.ndarray) -> np.ndarray:
    return mask_logits_256.astype(np.float32)

