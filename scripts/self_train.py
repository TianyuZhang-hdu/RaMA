#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


TARGET_SIZE = 256


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def import_original_modules(original_code_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(original_code_root))
    from dataloaders.normalize import normalize_image_to_0_1_3D  # noqa: PLC0415
    from networks.ResUnet import ResUnet  # noqa: PLC0415

    return {
        "ResUnet": ResUnet,
        "normalize_image_to_0_1_3D": normalize_image_to_0_1_3D,
    }


def import_dbscan_crop(clustering_utils_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("rama_clustering_utils", clustering_utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import DBSCAN helper from {clustering_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dbscan_crop


class InferenceDataset(Dataset):
    def __init__(self, image_dir: Path, normalize_fn: Any, target_size: int = TARGET_SIZE) -> None:
        self.image_dir = image_dir
        self.normalize_fn = normalize_fn
        self.target_size = (target_size, target_size)
        self.filenames = sorted(p.name for p in image_dir.glob("*.png"))
        print(f"  [InferenceDataset] {image_dir}: {len(self.filenames)} images")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        filename = self.filenames[idx]
        image = Image.open(self.image_dir / filename)
        image = image.resize(self.target_size)
        image_npy = np.array(image)[np.newaxis, ...].astype(np.float32)
        image_npy = self.normalize_fn(image_npy)
        return image_npy, filename


def prepare_symlink(link_path: Path, real_path: Path) -> None:
    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(real_path)


def find_latest_model(log_root: Path, exp_name: str, target_dataset: str, model_name: str = "best-Res_Unet.pth") -> Path | None:
    candidates = sorted((log_root / exp_name / f"vendorA_to_{target_dataset}").glob(f"*/model/{model_name}"))
    return candidates[-1] if candidates else None


def find_round0_model(round0_log_root: Path, target_dataset: str) -> Path | None:
    candidates = sorted((round0_log_root / f"vendorA_to_{target_dataset}").glob("*/model/best-Res_Unet.pth"))
    return candidates[-1] if candidates else None


def find_prev_round_model(args: argparse.Namespace) -> Path | None:
    if args.prev_model:
        return args.prev_model
    if args.round <= 1:
        if args.round0_log_root is None:
            return None
        return find_round0_model(args.round0_log_root, args.target_dataset)
    return find_latest_model(args.self_train_root / "logs", f"self_train_round{args.round - 1}", args.target_dataset)


def generate_pseudo_labels(args: argparse.Namespace, mods: dict[str, Any], dbscan_crop: Any, prev_model: Path) -> dict[str, Any]:
    device = torch.device(args.device)
    target = args.target_dataset
    round_dir = args.self_train_root / target / f"round{args.round}"
    image_out = round_dir / "image"
    mask_out = round_dir / "mask"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    train_image_dir = args.dataset_root / target / "train" / "image"
    dataset = InferenceDataset(train_image_dir, mods["normalize_image_to_0_1_3D"], args.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = mods["ResUnet"](resnet=args.backbone, num_classes=3, pretrained=False, in_ch=1).to(device)
    checkpoint = torch.load(prev_model, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    total = 0
    saved = 0
    skipped_confidence = 0
    skipped_dbscan = 0

    print("\n" + "=" * 60)
    print(f"  Self-Training Round {args.round}: generating pseudo-labels")
    print(f"  Target: {target}")
    print(f"  Model: {prev_model}")
    print(f"  Train images: {len(dataset)}")
    print(f"  Output: {round_dir}")
    print(f"  Confidence threshold: {args.confidence_threshold}")
    print(f"  DBSCAN crop: {not args.no_dbscan}")
    print("=" * 60 + "\n")

    with torch.no_grad():
        for image_npy, filenames in tqdm(loader, desc="Generating pseudo-labels"):
            filename = filenames[0]
            total += 1

            x = torch.from_numpy(np.array(image_npy)).float().to(device)
            pred_logit, _ = model(x)
            pred = torch.sigmoid(pred_logit)[0].cpu().numpy()

            max_fg = np.max(pred, axis=0)
            bg_conf = 1.0 - max_fg
            pixel_conf = np.maximum(max_fg, bg_conf)
            mean_conf = float(pixel_conf.mean())
            if mean_conf < args.confidence_threshold:
                skipped_confidence += 1
                continue

            binary = (pred >= args.mask_threshold).astype(np.uint8)
            h, w = binary.shape[1], binary.shape[2]
            rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
            rgb_mask[binary[0] == 1] = [255, 0, 0]
            rgb_mask[binary[1] == 1] = [0, 255, 0]
            rgb_mask[binary[2] == 1] = [0, 0, 255]
            mask_pil = Image.fromarray(rgb_mask)

            if not args.no_dbscan:
                mask_binary = np.any(rgb_mask > 0, axis=2).astype(np.uint8)
                bbox, _, _ = dbscan_crop(mask_binary)
                if bbox is None:
                    skipped_dbscan += 1
                    continue

                cropped = mask_pil.crop(bbox)
                cropped_arr = np.array(cropped)
                cleaned = np.zeros_like(cropped_arr)
                cleaned[cropped_arr[:, :, 1] > 127] = [0, 255, 0]
                cleaned[cropped_arr[:, :, 0] > 127] = [255, 0, 0]
                cleaned[cropped_arr[:, :, 2] > 127] = [0, 0, 255]

                mask_pil = Image.new("RGB", (args.image_size, args.image_size), (0, 0, 0))
                mask_pil.paste(Image.fromarray(cleaned), (bbox[0], bbox[1]))

            source_image = Image.open(train_image_dir / filename).resize((args.image_size, args.image_size), Image.BILINEAR)
            source_image.save(image_out / filename)
            mask_pil.save(mask_out / filename)
            saved += 1

    report = {
        "target_dataset": target,
        "round": args.round,
        "prev_model": str(prev_model),
        "total": total,
        "saved": saved,
        "skipped_confidence": skipped_confidence,
        "skipped_dbscan": skipped_dbscan,
        "confidence_threshold": args.confidence_threshold,
        "mask_threshold": args.mask_threshold,
        "use_dbscan": not args.no_dbscan,
        "round_dir": str(round_dir),
    }
    (round_dir / "pseudo_label_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def build_compat_result_root(args: argparse.Namespace) -> Path:
    round_dir = args.self_train_root / args.target_dataset / f"round{args.round}"
    compat_root = args.self_train_root / f"_compat_round{args.round}"
    compat_vendor = compat_root / args.target_dataset
    compat_vendor.mkdir(parents=True, exist_ok=True)
    prepare_symlink(compat_vendor / "image", round_dir / "image")
    prepare_symlink(compat_vendor / "mask", round_dir / "mask")
    return compat_root


def run_finetune(args: argparse.Namespace, prev_model: Path, compat_result_root: Path) -> Path | None:
    log_root = args.self_train_root / "logs"
    exp_name = f"self_train_round{args.round}"
    command = [
        str(args.python),
        "finetune_target.py",
        "--Source_Dataset",
        "vendorA",
        "--Target_Dataset",
        args.target_dataset,
        "--exp_name",
        exp_name,
        "--model_path",
        str(prev_model),
        "--dataset_root",
        str(args.dataset_root),
        "--result_root",
        str(compat_result_root),
        "--path_save_log",
        str(log_root),
        "--lr",
        str(args.lr),
        "--num_epochs",
        str(args.num_epochs),
        "--warmup_epochs",
        str(args.warmup_epochs),
        "--freeze_encoder_layers",
        str(args.freeze_encoder_layers),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--curve_loss_weight",
        str(args.curve_loss_weight),
        "--device",
        args.device,
        "--no_score_weight",
    ]

    print("\n" + "#" * 60)
    print(f"  Self-Training Round {args.round}: fine-tuning")
    print(f"  Initial model: {prev_model}")
    print(f"  Result root: {compat_result_root}")
    print(f"  LR: {args.lr} | Epochs: {args.num_epochs}")
    print("#" * 60 + "\n")
    print("[CMD] " + " ".join(command) + "\n")

    subprocess.run(command, cwd=args.original_code_root, check=True)
    return find_latest_model(log_root, exp_name, args.target_dataset)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run RaMA/IPLC self-training with the original logic and configurable paths.")
    parser.add_argument("--original-code-root", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--self-train-root", type=Path, required=True)
    parser.add_argument("--target-dataset", dest="target_dataset", required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--prev-model", type=Path, default=None)
    parser.add_argument("--round0-log-root", type=Path, default=None)
    parser.add_argument("--clustering-utils-path", type=Path, default=repo_root / "src/common/clustering_utils.py")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))

    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--no-dbscan", action="store_true", default=False)
    parser.add_argument("--infer-only", action="store_true", default=False)
    parser.add_argument("--train-only", action="store_true", default=False)

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--freeze-encoder-layers", type=int, default=2)
    parser.add_argument("--curve-loss-weight", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    prev_model = find_prev_round_model(args)
    if prev_model is None:
        raise FileNotFoundError("Previous-round model was not found. Pass --prev-model or --round0-log-root.")
    if not prev_model.exists():
        raise FileNotFoundError(prev_model)

    print("\n" + "#" * 60)
    print("  Self-Training Pipeline")
    print(f"  Target: {args.target_dataset} | Round: {args.round}")
    print(f"  Previous model: {prev_model}")
    print("#" * 60 + "\n")

    mods = import_original_modules(args.original_code_root)
    dbscan_crop = import_dbscan_crop(args.clustering_utils_path)

    round_dir = args.self_train_root / args.target_dataset / f"round{args.round}"
    if not args.train_only:
        generate_pseudo_labels(args, mods, dbscan_crop, prev_model)
    else:
        mask_dir = round_dir / "mask"
        n_masks = len(list(mask_dir.glob("*.png"))) if mask_dir.exists() else 0
        if n_masks == 0:
            raise FileNotFoundError(f"No pseudo-label masks found in {mask_dir}")
        print(f"[INFO] Using existing pseudo-labels: {mask_dir} ({n_masks} masks)")

    compat_result_root = build_compat_result_root(args)
    if args.infer_only:
        print(f"[INFO] --infer-only set; pseudo-labels are in {round_dir}")
        return

    best_model = run_finetune(args, prev_model, compat_result_root)
    if best_model is None:
        raise FileNotFoundError(f"Round {args.round} best model was not found.")

    print("\n" + "#" * 60)
    print(f"  Self-Training Round {args.round} complete")
    print(f"  Best model: {best_model}")
    print(f"  Pseudo-labels: {round_dir}")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
