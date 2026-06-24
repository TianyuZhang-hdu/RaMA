from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial.distance import cdist


ORGANS = ("LV", "MYO", "RV")
CHANNELS = {"LV": 0, "MYO": 1, "RV": 2}


@dataclass(frozen=True)
class MetricSummary:
    dice: dict[str, float]
    assd: dict[str, float | None]

    def to_json(self) -> dict[str, dict[str, float | None]]:
        return {"dice": self.dice, "assd": self.assd}


def load_rgb_mask(path: str | Path, size: tuple[int, int] | None = None) -> Image.Image:
    mask = Image.open(path).convert("RGB")
    if size is not None and mask.size != size:
        mask = mask.resize(size, Image.NEAREST)
    return mask


def organ_mask(mask: Image.Image, organ: str) -> np.ndarray:
    arr = np.asarray(mask.convert("RGB"))
    return arr[:, :, CHANNELS[organ]] > 127


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum(dtype=np.float64)
    denom = pred.sum(dtype=np.float64) + gt.sum(dtype=np.float64)
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def _surface_points(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.empty((0, 2), dtype=np.int32)
    eroded = ndimage.binary_erosion(mask)
    surface = mask.astype(bool) & ~eroded
    return np.argwhere(surface)


def assd_score(pred: np.ndarray, gt: np.ndarray) -> float | None:
    pred_surface = _surface_points(pred)
    gt_surface = _surface_points(gt)
    if len(pred_surface) == 0 and len(gt_surface) == 0:
        return 0.0
    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return None
    pred_to_gt = cdist(pred_surface, gt_surface, "euclidean").min(axis=1)
    gt_to_pred = cdist(gt_surface, pred_surface, "euclidean").min(axis=1)
    return float((pred_to_gt.mean() + gt_to_pred.mean()) / 2.0)


def evaluate_pair(pred_mask: Image.Image, gt_mask: Image.Image) -> MetricSummary:
    dice: dict[str, float] = {}
    assd: dict[str, float | None] = {}
    for organ in ORGANS:
        pred = organ_mask(pred_mask, organ)
        gt = organ_mask(gt_mask, organ)
        dice[organ] = dice_score(pred, gt)
        assd[organ] = assd_score(pred, gt)
    dice["mean"] = float(np.mean([dice[o] for o in ORGANS]))
    finite_assd = [assd[o] for o in ORGANS if assd[o] is not None]
    assd["mean"] = float(np.mean(finite_assd)) if finite_assd else None
    return MetricSummary(dice=dice, assd=assd)


def aggregate_summaries(items: list[MetricSummary]) -> MetricSummary:
    if not items:
        raise ValueError("No metric summaries to aggregate")
    dice: dict[str, float] = {}
    assd: dict[str, float | None] = {}
    for key in (*ORGANS, "mean"):
        dice[key] = float(np.mean([item.dice[key] for item in items]))
        values = [item.assd[key] for item in items if item.assd[key] is not None]
        assd[key] = float(np.mean(values)) if values else None
    return MetricSummary(dice=dice, assd=assd)
