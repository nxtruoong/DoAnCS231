"""Two-stream augmentation + dataset for Run 7.

Each sample produces three tensors:
    (full_tensor, face_tensor, label)

`face_tensor` is derived by cropping the top fraction (default 45%) of
the *raw* PIL image, then running its own aug pipeline. Both streams
use independent stochastic aug — except no horizontal flip is used in
either pipeline (project-wide rule, see docs/adr/0002-no-hflip.md), so
the "share flip decision between streams" constraint from the Run 7
plan is automatically satisfied.
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from augment import CLASS_TO_IDX

POSE_DIM = 8


class TopCrop:
    """Crop the top `frac` of a PIL image. Used as the face-stream input.

    State Farm dashcam is fixed (right-side mount, fixed angle), so the
    driver's head sits in the upper portion of every frame. A static
    top crop captures the head/gaze area without needing a face detector.
    """

    def __init__(self, frac: float = 0.50):
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"frac must be in (0, 1], got {frac}")
        self.frac = frac

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.crop((0, 0, w, int(h * self.frac)))

    def __repr__(self) -> str:
        return f"TopCrop(frac={self.frac})"


class RegionCrop:
    """Generic rectangular crop in normalised coordinates.

    Used for the Run 8 hand/lap stream: bottom-center region of the
    cabin where phone/cup/radio interactions live. Static crop because
    State Farm dashcam is fixed-mount.
    """

    def __init__(self, top_frac: float = 0.0, bottom_frac: float = 1.0,
                 left_frac: float = 0.0, right_frac: float = 1.0):
        if not (0.0 <= top_frac < bottom_frac <= 1.0):
            raise ValueError(f"bad vertical: top={top_frac}, bottom={bottom_frac}")
        if not (0.0 <= left_frac < right_frac <= 1.0):
            raise ValueError(f"bad horizontal: left={left_frac}, right={right_frac}")
        self.top, self.bottom = top_frac, bottom_frac
        self.left, self.right = left_frac, right_frac

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.crop((
            int(w * self.left), int(h * self.top),
            int(w * self.right), int(h * self.bottom),
        ))

    def __repr__(self) -> str:
        return (f"RegionCrop(top={self.top}, bottom={self.bottom}, "
                f"left={self.left}, right={self.right})")


def build_full_train_transform(mean, std, size: int = 384) -> transforms.Compose:
    """Run 6 aug stack, applied to the full frame."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
    ])


def build_face_train_transform(mean, std, size: int = 224,
                               top_frac: float = 0.50) -> transforms.Compose:
    """Top-crop the head region, then run lighter aug at smaller resolution.

    Smaller size (224 default) because the cropped region is already
    smaller than the full frame; resizing up to 384 would blur it.
    RandomErasing kept light — erasing too much of a small face crop
    removes the only signal the stream has.
    """
    return transforms.Compose([
        TopCrop(frac=top_frac),
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.06), ratio=(0.3, 3.3), value="random"),
    ])


def build_full_eval_transform(mean, std, size: int = 384) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_face_eval_transform(mean, std, size: int = 224,
                              top_frac: float = 0.50) -> transforms.Compose:
    return transforms.Compose([
        TopCrop(frac=top_frac),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# --- Run 8 hand-stream transforms ------------------------------------------

def build_hand_train_transform(mean, std, size: int = 224,
                               top_frac: float = 0.45,
                               bottom_frac: float = 1.0,
                               left_frac: float = 0.20,
                               right_frac: float = 0.85) -> transforms.Compose:
    """Crop driver's lap + lower wheel region, then standard aug.

    Defaults: bottom 55% (top=0.45..bottom=1.0), center 65% width
    (left=0.20..right=0.85). Cuts dashboard above wheel and right-side
    passenger window glare. Captures phone-in-hand (c3, c4), radio
    (c5), drink (c6), and the "empty" lap region defining c0/c9.
    """
    return transforms.Compose([
        RegionCrop(top_frac=top_frac, bottom_frac=bottom_frac,
                   left_frac=left_frac, right_frac=right_frac),
        transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.06), ratio=(0.3, 3.3), value="random"),
    ])


def build_hand_eval_transform(mean, std, size: int = 224,
                              top_frac: float = 0.45,
                              bottom_frac: float = 1.0,
                              left_frac: float = 0.20,
                              right_frac: float = 0.85) -> transforms.Compose:
    return transforms.Compose([
        RegionCrop(top_frac=top_frac, bottom_frac=bottom_frac,
                   left_frac=left_frac, right_frac=right_frac),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# --- Run 8 pose lookup -----------------------------------------------------

def load_pose_lookup(parquet_path: Path) -> dict[str, "np.ndarray"]:
    """Load filename -> 8-d pose vector dict from precomputed parquet.

    Missing filenames at lookup time get zero-imputed (gating bit p7=0
    in zeroed vector → model learns "pose unavailable").
    """
    import numpy as np
    import polars as pl
    df = pl.read_parquet(parquet_path)
    cols = [f"p{i}" for i in range(POSE_DIM)]
    arr = df.select(cols).to_numpy().astype("float32")
    names = df["filename"].to_list()
    return {n: arr[i] for i, n in enumerate(names)}


class TwoStreamDataset(Dataset):
    """Returns (full_tensor, face_tensor, label) per item."""

    def __init__(self, csv_path: Path, img_root: Path,
                 full_transform: transforms.Compose,
                 face_transform: transforms.Compose):
        import pandas as pd
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.img_root = Path(img_root)
        self.tx_full = full_transform
        self.tx_face = face_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = self.img_root / row["classname"] / row["img"]
        with Image.open(path) as im:
            img = im.convert("RGB")
            full = self.tx_full(img)
            face = self.tx_face(img)
        return full, face, CLASS_TO_IDX[row["classname"]]


class ThreeStreamDataset(Dataset):
    """Returns (full_tensor, hand_tensor, pose_vec, label) per item.

    Pose vector is looked up from a precomputed dict by image filename.
    Missing entries are zero-imputed (gating bit p7=0 by construction).
    """

    def __init__(self, csv_path: Path, img_root: Path,
                 full_transform: transforms.Compose,
                 hand_transform: transforms.Compose,
                 pose_lookup: dict[str, "np.ndarray"]):
        import numpy as np
        import pandas as pd
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.img_root = Path(img_root)
        self.tx_full = full_transform
        self.tx_hand = hand_transform
        self.pose_lookup = pose_lookup
        self._zero_pose = np.zeros(POSE_DIM, dtype="float32")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.img_root / row["classname"] / row["img"]
        with Image.open(path) as im:
            img = im.convert("RGB")
            full = self.tx_full(img)
            hand = self.tx_hand(img)
        pose_np = self.pose_lookup.get(row["img"], self._zero_pose)
        pose = torch.from_numpy(pose_np.copy())
        return full, hand, pose, CLASS_TO_IDX[row["classname"]]


def cutmix_twostream(
    full: torch.Tensor,
    face: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 0.5,
    p: float = 0.20,
):
    """CutMix applied to BOTH streams with the **same** permutation + box.

    Crucial: if streams get independent CutMix permutations, the model
    sees two different label-mix targets and cannot learn the fusion.
    Same `perm` and same `(y1:y2, x1:x2)` (scaled per-stream resolution).
    """
    import numpy as np
    if np.random.rand() > p:
        return full, face, labels, labels, 1.0

    b = full.size(0)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(b, device=full.device)

    def cut(t: torch.Tensor, lam: float) -> tuple[torch.Tensor, float]:
        _, _, h, w = t.shape
        cut_ratio = (1.0 - lam) ** 0.5
        cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
        cy, cx = int(torch.randint(h, (1,))), int(torch.randint(w, (1,)))
        y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
        x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
        t = t.clone()
        t[:, :, y1:y2, x1:x2] = t[perm, :, y1:y2, x1:x2]
        actual_lam = 1.0 - ((y2 - y1) * (x2 - x1) / (h * w))
        return t, actual_lam

    full, lam_full = cut(full, lam)
    face, _        = cut(face, lam)  # same perm; box geometry computed per-stream
    # Use full-stream lam as the loss-weighting target (dominant signal).
    return full, face, labels, labels[perm], lam_full
