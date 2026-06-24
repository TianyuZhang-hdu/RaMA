#!/usr/bin/env python3
"""Source-model inference: generate initial pseudo-labels on a target split.

Usage:
    python scripts/source_infer.py \
        --vendor vendorB --split train \
        --model-path repro_training/source_models/vendorA/last-Res_Unet.pth \
        --dataset-root /path/to/processed/mnms_png \
        --out-dir repro_training/source_preds/vendorB
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from networks.ResUnet import ResUnet  # noqa: E402
from dataloaders.normalize import normalize_image_to_0_1_3D  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vendor", required=True, choices=["vendorA", "vendorB", "vendorC", "vendorD"])
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="Processed M&Ms PNG root (output of prepare_mnms_png.py)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = args.dataset_root / args.vendor / args.split / "image"
    assert img_dir.is_dir(), f"missing {img_dir}"

    model = ResUnet(resnet="resnet34", num_classes=3, pretrained=False, in_ch=1).to(args.device)
    state = torch.load(args.model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    files = sorted(img_dir.glob("*.png"))
    print(f"[{args.vendor}/{args.split}] inferring {len(files)} images -> {args.out_dir}")
    with torch.no_grad():
        for f in files:
            im = np.array(Image.open(f).resize((args.size, args.size)))[None].astype(np.float32)
            im = normalize_image_to_0_1_3D(im)
            x = torch.from_numpy(im).float().unsqueeze(0).to(args.device)
            pr = torch.sigmoid(model(x)[0])[0].cpu().numpy()
            pb = (pr >= 0.5).astype(np.uint8)
            rgb = np.zeros((args.size, args.size, 3), dtype=np.uint8)
            rgb[pb[0] == 1] = [255, 0, 0]   # LV
            rgb[pb[1] == 1] = [0, 255, 0]   # MYO
            rgb[pb[2] == 1] = [0, 0, 255]   # RV
            Image.fromarray(rgb).save(args.out_dir / f.name)
    print("done.")


if __name__ == "__main__":
    main()
