"""Subject-wise split + dataset stats for State Farm Distracted Driver.

Reads `driver_imgs_list.csv`, partitions rows by subject (5 drivers held
out for val), writes `splits/train.csv` and `splits/val.csv`, and
computes per-channel RGB mean/std over the training partition, writing
`splits/stats.json`.

Run on Kaggle:
    python data_prep.py \\
        --data-root /kaggle/input/competitions/state-farm-distracted-driver-detection \\
        --out-dir   /kaggle/working/splits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

HELD_OUT_SUBJECTS = ["p022", "p035", "p047", "p056", "p075"]
CLASSES = [f"c{i}" for i in range(10)]


class _RawImageDataset(Dataset):
    """Yields tensors in [0, 1] without normalization — for stats computation."""

    def __init__(self, df: pd.DataFrame, img_root: Path, size: int = 224):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.tx = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        path = self.img_root / row["classname"] / row["img"]
        with Image.open(path) as im:
            im = im.convert("RGB")
            return self.tx(im)


def build_splits(driver_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(driver_csv)
    expected = {"subject", "classname", "img"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"driver_imgs_list.csv missing columns: {missing}")

    held = set(HELD_OUT_SUBJECTS)
    val_df = df[df["subject"].isin(held)].copy()
    train_df = df[~df["subject"].isin(held)].copy()

    found = set(val_df["subject"].unique())
    if found != held:
        raise ValueError(f"Held-out subjects not all present. Expected {held}, found {found}")

    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        counts = split_df["classname"].value_counts().sort_index()
        if set(counts.index) != set(CLASSES):
            missing_cls = set(CLASSES) - set(counts.index)
            raise ValueError(f"{split_name} split missing classes: {missing_cls}")

    return train_df, val_df


def compute_stats(
    train_df: pd.DataFrame,
    img_root: Path,
    *,
    size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, list[float]]:
    """Compute per-channel mean and std over the train partition.

    Two-pass algorithm: pass 1 computes the global mean, pass 2 uses that
    mean to compute std. More numerically stable than the
    sum/sum-of-squares trick for image data.
    """
    ds = _RawImageDataset(train_df, img_root, size=size)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Pass 1: mean
    pixel_sum = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    for batch in tqdm(loader, desc="stats pass 1 (mean)"):
        b, _, h, w = batch.shape
        pixel_sum += batch.to(torch.float64).sum(dim=(0, 2, 3))
        n_pixels += b * h * w
    mean = (pixel_sum / n_pixels).to(torch.float32)

    # Pass 2: std
    sq_diff = torch.zeros(3, dtype=torch.float64)
    for batch in tqdm(loader, desc="stats pass 2 (std)"):
        diff = batch - mean.view(1, 3, 1, 1)
        sq_diff += (diff.to(torch.float64) ** 2).sum(dim=(0, 2, 3))
    std = torch.sqrt(sq_diff / n_pixels).to(torch.float32)

    return {"mean": mean.tolist(), "std": std.tolist()}


def per_subject_class_report(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index="subject", columns="classname", values="img", aggfunc="count", fill_value=0
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path,
                    help="Kaggle dataset root (contains imgs/ and driver_imgs_list.csv)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--skip-stats", action="store_true",
                    help="Build splits only; do not run the (slow) stats pass.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    driver_csv = args.data_root / "driver_imgs_list.csv"
    img_root = args.data_root / "imgs" / "train"

    train_df, val_df = build_splits(driver_csv)
    train_df.to_csv(args.out_dir / "train.csv", index=False)
    val_df.to_csv(args.out_dir / "val.csv", index=False)

    print(f"Train: {len(train_df)} imgs from {train_df['subject'].nunique()} subjects")
    print(f"Val  : {len(val_df)} imgs from {val_df['subject'].nunique()} subjects")
    print("\nClass distribution (train):")
    print(train_df["classname"].value_counts().sort_index().to_string())
    print("\nClass distribution (val):")
    print(val_df["classname"].value_counts().sort_index().to_string())
    print("\nVal subjects x classes:")
    print(per_subject_class_report(val_df).to_string())

    if args.skip_stats:
        print("\nSkipping stats computation.")
        return

    stats = compute_stats(
        train_df, img_root,
        size=args.size, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    stats["size"] = args.size
    stats["held_out_subjects"] = HELD_OUT_SUBJECTS
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nDataset stats: mean={stats['mean']} std={stats['std']}")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
