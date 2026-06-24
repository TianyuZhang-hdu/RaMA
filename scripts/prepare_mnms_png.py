#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


RAW_SPLITS = ("Training/Labeled", "Training/Unlabeled", "Validation", "Testing")
VENDOR_NAMES = {"A": "vendorA", "B": "vendorB", "C": "vendorC", "D": "vendorD"}
TRAIN_CASES = {"A": 76, "B": 100, "C": 40, "D": 40}
LABEL_COLORS = {
    1: (255, 0, 0),   # LV
    2: (0, 255, 0),   # MYO
    3: (0, 0, 255),   # RV
}


@dataclass(frozen=True)
class CaseInfo:
    code: str
    vendor: str
    split_dir: Path
    ed: int
    es: int


def read_cases(raw_root: Path) -> dict[str, list[CaseInfo]]:
    metadata_path = raw_root / "211230_MnMs_Dataset_information_diagnosis_opendataset.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_path}")

    code_to_split: dict[str, Path] = {}
    for split in RAW_SPLITS:
        split_dir = raw_root / split
        if not split_dir.exists():
            continue
        for case_dir in split_dir.iterdir():
            if case_dir.is_dir():
                code_to_split[case_dir.name] = case_dir

    cases: dict[str, list[CaseInfo]] = {v: [] for v in VENDOR_NAMES}
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            code = row["External code"]
            vendor = row["Vendor"]
            split_dir = code_to_split.get(code)
            if vendor not in cases or split_dir is None:
                continue
            if vendor == "C" and "Training/Unlabeled" in str(split_dir):
                continue
            cases[vendor].append(
                CaseInfo(
                    code=code,
                    vendor=vendor,
                    split_dir=split_dir,
                    ed=int(row["ED"]),
                    es=int(row["ES"]),
                )
            )

    for vendor in cases:
        cases[vendor].sort(key=lambda item: item.code)
    return cases


def normalize_image(slice_arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(slice_arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - lo) / (hi - lo)
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def rgb_mask(label_slice: np.ndarray) -> np.ndarray:
    labels = np.rint(label_slice).astype(np.uint8)
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        out[labels == label] = color
    return out


def complete_class_slices(label_volume: np.ndarray, times: tuple[int, int]) -> list[tuple[int, int, int]]:
    z_dim = label_volume.shape[2]
    selected: list[tuple[int, int, int]] = []
    for t_idx in times:
        for z_idx in range(z_dim):
            label_slice = np.rint(label_volume[:, :, z_idx, t_idx]).astype(np.uint8)
            if all(np.any(label_slice == label) for label in LABEL_COLORS):
                slice_id = t_idx * z_dim + z_idx
                selected.append((slice_id, t_idx, z_idx))
    return sorted(selected)


def write_case(
    case: CaseInfo,
    case_index: int,
    split_name: str,
    vendor_dir: Path,
) -> list[tuple[str, str]]:
    img_path = case.split_dir / f"{case.code}_sa.nii"
    gt_path = case.split_dir / f"{case.code}_sa_gt.nii"
    image_volume = np.asarray(nib.load(str(img_path)).dataobj)
    label_volume = np.asarray(nib.load(str(gt_path)).dataobj)
    selected = complete_class_slices(label_volume, (case.ed, case.es))

    image_dir = vendor_dir / split_name / "image"
    mask_dir = vendor_dir / split_name / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str]] = []
    vendor_name = vendor_dir.name
    for slice_id, t_idx, z_idx in selected:
        filename = f"{case_index:03d}{slice_id:03d}.png"
        image_rel = f"{vendor_name}/{split_name}/image/{filename}"
        mask_rel = f"{vendor_name}/{split_name}/mask/{filename}"
        Image.fromarray(normalize_image(image_volume[:, :, z_idx, t_idx]), mode="L").save(image_dir / filename)
        Image.fromarray(rgb_mask(label_volume[:, :, z_idx, t_idx]), mode="RGB").save(mask_dir / filename)
        rows.append((image_rel, mask_rel))
    return rows


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "mask"])
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [(row["image"], row["mask"]) for row in csv.DictReader(handle)]


def compare_reference(csv_path: Path, reference_root: Path | None) -> None:
    if reference_root is None:
        return
    reference_path = reference_root / csv_path.name
    if not reference_path.exists():
        print(f"[WARN] reference CSV missing: {reference_path}")
        return
    actual = read_csv_rows(csv_path)
    expected = read_csv_rows(reference_path)
    if actual != expected:
        first_diff = next(
            (idx for idx, (a, b) in enumerate(zip(actual, expected)) if a != b),
            min(len(actual), len(expected)),
        )
        raise RuntimeError(
            f"{csv_path.name} does not match reference {reference_path}; "
            f"actual={len(actual)} expected={len(expected)} first_diff={first_diff}"
        )
    print(f"[OK] {csv_path.name} matches reference ({len(actual)} rows)")


def prepare_dataset(raw_root: Path, out_root: Path, reference_csv_root: Path | None) -> None:
    cases_by_vendor = read_cases(raw_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for vendor, cases in cases_by_vendor.items():
        train_case_count = TRAIN_CASES[vendor]
        if len(cases) < train_case_count:
            raise RuntimeError(f"{vendor} has {len(cases)} cases, expected at least {train_case_count}")
        vendor_name = VENDOR_NAMES[vendor]
        vendor_dir = out_root / vendor_name
        all_rows: dict[str, list[tuple[str, str]]] = {"train": [], "test": []}
        for split_name, split_cases in {
            "train": cases[:train_case_count],
            "test": cases[train_case_count:],
        }.items():
            for local_index, case in enumerate(split_cases, start=0 if split_name == "train" else train_case_count):
                all_rows[split_name].extend(write_case(case, local_index, split_name, vendor_dir))

            csv_path = out_root / f"{vendor_name}_{split_name}.csv"
            write_csv(csv_path, all_rows[split_name])
            compare_reference(csv_path, reference_csv_root)
            print(f"{vendor_name} {split_name}: {len(split_cases)} cases, {len(all_rows[split_name])} slices")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert M&Ms 4D NIfTI files into the PNG layout used by RaMA.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Path to the unpacked MnM directory.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output directory for vendor PNG folders and CSVs.")
    parser.add_argument("--reference-csv-root", type=Path, help="Optional old CSV root used for exact index validation.")
    args = parser.parse_args()
    prepare_dataset(args.raw_root, args.out_root, args.reference_csv_root)


if __name__ == "__main__":
    main()
