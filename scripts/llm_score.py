#!/usr/bin/env python3
"""LLM-driven pseudo-label scoring (paper Sec. 2.1).

For each image, the Multimodal LLM (Qwen-VL by default) reads the cardiac
overlay and outputs per-organ quality scores and FP/FN point prompts.

Usage:
    export RAMA_LLM_API_KEY=sk-xxxx
    python scripts/llm_score.py \
        --vendor vendorB --split train \
        --dataset-root /path/to/processed/mnms_png \
        --pseudo-root repro_training/source_preds/vendorB \
        --output-root repro_training/llm_scores
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
                   help="Directory of source-model pseudo-labels for this vendor/split")
    p.add_argument("--gt-root", type=Path, default=None,
                   help="GT mask root (defaults to <dataset-root>/<vendor>/<split>/mask)")
    p.add_argument("--output-root", type=Path, required=True,
                   help="Where {model}/{vendor}_{split}/*.json is written")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--save-detail", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("RAMA_LLM_API_KEY"):
        print("[WARN] RAMA_LLM_API_KEY is not set; live LLM calls will fail.", file=sys.stderr)

    import llm.score as scoremod
    import config as cfg

    scoremod.VENDOR = args.vendor
    scoremod.SPLIT = args.split
    scoremod.IMAGE_DIR = str(args.dataset_root / args.vendor / args.split / "image")
    scoremod.MASK_DIR = str(args.pseudo_root)
    scoremod.GT_DIR = str(args.gt_root or args.dataset_root / args.vendor / args.split / "mask")
    model_dir = cfg.OPENAI_MODEL.replace("/", "_").replace(":", "_")
    scoremod.LLM_SCORE_DIR = str(args.output_root / model_dir / f"{args.vendor}_{args.split}")
    scoremod.DEBUG_DIR = str(args.output_root / "debug" / model_dir / f"{args.vendor}_{args.split}")
    os.makedirs(scoremod.LLM_SCORE_DIR, exist_ok=True)
    if args.save_detail:
        os.makedirs(scoremod.DEBUG_DIR, exist_ok=True)

    scoremod.run_llm_scoring(save_detail=args.save_detail, num_workers=args.workers)


if __name__ == "__main__":
    main()
