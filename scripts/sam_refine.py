#!/usr/bin/env python3
"""Multi-SAM consensus refinement (paper Sec. 2.2).

Given source-model pseudo-labels and LLM scoring outputs, runs three SAM
agents (SAM 3, MedSAM 2, SAM-Med2D) with LLM-derived FP/FN point prompts,
then majority-votes the three masks. Falls back to the original pseudo-label
when an agent's refined mask drifts too far (Dice < 0.5).

Usage:
    export RAMA_SAM3_CKPT=/path/to/sam3.pt
    python scripts/sam_refine.py \
        --vendor vendorB --split train \
        --dataset-root /path/to/processed/mnms_png \
        --pseudo-root repro_training/source_preds/vendorB \
        --llm-score-root repro_training/llm_scores \
        --output-root repro_training/sam_results
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vendor", required=True, choices=["vendorA", "vendorB", "vendorC", "vendorD"])
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--pseudo-root", type=Path, required=True,
                   help="Source-model pseudo-labels (same as fed to llm_score.py)")
    p.add_argument("--llm-score-root", type=Path, required=True,
                   help="Root passed to llm_score.py --output-root")
    p.add_argument("--gt-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--model-name", default=None,
                   help="LLM model dir name (defaults to config.OPENAI_MODEL)")
    p.add_argument("--save-detail", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import refinement.sam_refine as refmod
    import config as cfg

    model_dir = (args.model_name or cfg.OPENAI_MODEL).replace("/", "_").replace(":", "_")

    refmod.VENDOR = args.vendor
    refmod.SPLIT = args.split
    refmod.IMAGE_DIR = str(args.dataset_root / args.vendor / args.split / "image")
    refmod.MASK_DIR = str(args.pseudo_root)
    refmod.GT_DIR = str(args.gt_root or args.dataset_root / args.vendor / args.split / "mask")
    refmod.LLM_SCORE_DIR = str(args.llm_score_root / model_dir / f"{args.vendor}_{args.split}")

    base = args.output_root / model_dir / f"{args.vendor}_{args.split}"
    refmod.SAM_RESULT_DIR = str(base)
    refmod.SAM3_MASK_DIR = str(base / "masks_sam3")
    refmod.MEDSAM2_MASK_DIR = str(base / "masks_medsam2")
    refmod.SAMMED2D_MASK_DIR = str(base / "masks_sammed2d")
    refmod.VOTE_MASK_DIR = str(base / "masks_vote")
    refmod.DEBUG_DIR = str(args.output_root / "debug" / model_dir / f"{args.vendor}_{args.split}")
    refmod.RESULT_IMAGE_DIR = str(args.output_root / "result" / args.vendor / "image")
    refmod.RESULT_MASK_DIR = str(args.output_root / "result" / args.vendor / "mask")
    for d in (refmod.SAM3_MASK_DIR, refmod.MEDSAM2_MASK_DIR, refmod.SAMMED2D_MASK_DIR,
              refmod.VOTE_MASK_DIR, refmod.RESULT_IMAGE_DIR, refmod.RESULT_MASK_DIR):
        os.makedirs(d, exist_ok=True)
    if args.save_detail:
        os.makedirs(refmod.DEBUG_DIR, exist_ok=True)

    refmod.run_sam_refine(save_detail=args.save_detail)


if __name__ == "__main__":
    main()
