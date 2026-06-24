#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from metrics import aggregate_summaries, evaluate_pair, load_rgb_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RGB cardiac masks against GT masks.")
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suffix", default=".png")
    args = parser.parse_args()

    pred_files = sorted(args.pred_dir.glob(f"*{args.suffix}"))
    if not pred_files:
        raise SystemExit(f"No prediction files found in {args.pred_dir}")

    per_image = []
    summaries = []
    missing_gt = []
    for pred_path in pred_files:
        gt_path = args.gt_dir / pred_path.name
        if not gt_path.exists():
            missing_gt.append(pred_path.name)
            continue
        pred = load_rgb_mask(pred_path)
        gt = load_rgb_mask(gt_path, size=pred.size)
        metrics = evaluate_pair(pred, gt)
        summaries.append(metrics)
        per_image.append({"id": pred_path.stem, **metrics.to_json()})

    if not summaries:
        raise SystemExit("No matched prediction/GT pairs were evaluated")

    aggregate = aggregate_summaries(summaries)
    result = {
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir),
        "num_predictions": len(pred_files),
        "num_evaluated": len(summaries),
        "num_missing_gt": len(missing_gt),
        "missing_gt": missing_gt[:50],
        "aggregate": aggregate.to_json(),
        "per_image": per_image,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
