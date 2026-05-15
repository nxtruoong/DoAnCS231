"""Eval + figures for a trained two-stream ResNet-18+CBAM checkpoint.

Mirrors eval.py but feeds two tensors per sample. Outputs same artifact
set: metrics.json, classification_report.txt, confusion_matrix.png,
per_driver_accuracy.{png,csv}, training_curves.png, attention_grid.png
(full stream SAM), failures.png.

Run on Kaggle:
    python eval_twostream.py --ckpt /kaggle/working/run7/best.pt \\
        --data-root /kaggle/input/competitions/state-farm-distracted-driver-detection \\
        --splits-dir /kaggle/working/splits \\
        --out-dir /kaggle/working/run7/eval \\
        --history-json /kaggle/working/run7/history.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from augment import CLASSES, load_stats
from augment_twostream import (
    TwoStreamDataset, build_face_eval_transform, build_full_eval_transform,
)
from eval import (
    CLASS_NAMES, attention_grid as _attn_single, failure_cases as _fail_single,
    per_driver_breakdown, plot_confusion_matrix, plot_training_curves,
    write_classification_report,
)
from model_twostream import build_twostream


def load_twostream(ckpt_path: Path, use_ema: bool = True) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    use_cbam = not saved_args.get("no_cbam", False)
    model = build_twostream(num_classes=10, use_cbam=use_cbam)
    state = ckpt["ema" if use_ema else "model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    model.to(device)
    for full, face, labels in loader:
        full = full.to(device, non_blocking=True)
        face = face.to(device, non_blocking=True)
        logits = model(full, face)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return (np.concatenate(all_labels), np.concatenate(all_preds),
            np.concatenate(all_probs))


@torch.no_grad()
def attention_grid_twostream(model, loader, mean, std, device, out_path: Path,
                             n_per_class: int = 1) -> None:
    """Use SAM from the full stream. Wrap two-stream forward into a single
    callable so we can reuse eval.attention_grid (which expects a single
    image input)."""
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from eval import _denorm_to_uint8, overlay_sam

    picked: dict[int, list[tuple[np.ndarray, np.ndarray, int]]] = {i: [] for i in range(10)}
    model.to(device)
    for full, face, labels in loader:
        full = full.to(device, non_blocking=True)
        face = face.to(device, non_blocking=True)
        logits = model(full, face)
        preds = logits.argmax(dim=1)
        sam = model.last_spatial_attention()  # full-stream SAM
        sam_up = F.interpolate(sam, size=full.shape[-2:], mode="bilinear", align_corners=False)
        for i in range(full.size(0)):
            cls = int(labels[i])
            if len(picked[cls]) >= n_per_class:
                continue
            rgb = _denorm_to_uint8(full[i], mean, std)
            picked[cls].append((rgb, sam_up[i, 0].cpu().numpy(), int(preds[i])))
        if all(len(v) >= n_per_class for v in picked.values()):
            break

    cols, rows = 10, n_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    if rows == 1:
        axes = np.array([axes])
    for c in range(10):
        for r in range(n_per_class):
            ax = axes[r, c]
            if r < len(picked[c]):
                rgb, sam_map, pred = picked[c][r]
                ax.imshow(overlay_sam(rgb, sam_map))
                color = "green" if pred == c else "red"
                ax.set_title(f"{CLASSES[c]} -> {CLASSES[pred]}", fontsize=9, color=color)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def failure_cases_twostream(model, loader, mean, std, device, out_path: Path,
                            n: int = 12) -> None:
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from eval import _denorm_to_uint8, overlay_sam

    model.to(device)
    failures: list[tuple[np.ndarray, np.ndarray, int, int, float]] = []
    for full, face, labels in loader:
        full = full.to(device, non_blocking=True)
        face = face.to(device, non_blocking=True)
        logits = model(full, face)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        sam = model.last_spatial_attention()
        sam_up = F.interpolate(sam, size=full.shape[-2:], mode="bilinear", align_corners=False)
        wrong = (preds.cpu() != labels).nonzero(as_tuple=True)[0]
        for idx in wrong:
            t = int(labels[idx]); p = int(preds[idx])
            conf = float(probs[idx, p])
            rgb = _denorm_to_uint8(full[idx], mean, std)
            heat = sam_up[idx, 0].cpu().numpy()
            failures.append((rgb, heat, t, p, conf))
        if len(failures) >= n * 4:
            break

    failures.sort(key=lambda t: -t[4])
    failures = failures[:n]
    if not failures:
        print("No failures — suspect bug."); return

    cols = 4
    rows = (len(failures) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    for i, (rgb, heat, t, p, conf) in enumerate(failures):
        r, c = divmod(i, cols); ax = axes[r, c]
        ax.imshow(overlay_sam(rgb, heat))
        ax.set_title(f"true {CLASSES[t]} | pred {CLASSES[p]} ({conf:.2f})", fontsize=8, color="red")
        ax.axis("off")
    for i in range(len(failures), rows * cols):
        r, c = divmod(i, cols); axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--history-json", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="use_ema", action="store_false")
    ap.add_argument("--full-size", type=int, default=384)
    ap.add_argument("--face-size", type=int, default=224)
    ap.add_argument("--top-frac", type=float, default=0.45)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = load_stats(args.splits_dir / "stats.json")
    tx_full = build_full_eval_transform(mean, std, size=args.full_size)
    tx_face = build_face_eval_transform(mean, std, size=args.face_size,
                                        top_frac=args.top_frac)
    val_df = pd.read_csv(args.splits_dir / "val.csv")
    val_ds = TwoStreamDataset(args.splits_dir / "val.csv",
                              args.data_root / "imgs" / "train",
                              tx_full, tx_face)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = load_twostream(args.ckpt, use_ema=args.use_ema)
    y_true, y_pred, _ = collect_predictions(model, val_loader, device)

    write_classification_report(y_true, y_pred, args.out_dir / "classification_report.txt")
    plot_confusion_matrix(y_true, y_pred, args.out_dir / "confusion_matrix.png")
    per_driver_breakdown(y_true, y_pred, val_df,
                         args.out_dir / "per_driver_accuracy.csv",
                         args.out_dir / "per_driver_accuracy.png")
    if args.history_json and args.history_json.exists():
        plot_training_curves(args.history_json, args.out_dir / "training_curves.png")
    attention_grid_twostream(model, val_loader, mean, std, device,
                             args.out_dir / "attention_grid.png", n_per_class=1)
    failure_cases_twostream(model, val_loader, mean, std, device,
                            args.out_dir / "failures.png", n=12)
    print(f"\nAll artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
