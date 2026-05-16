"""Gradio demo (Run 6, single-stream ResNet-18 + CBAM).

Local run from the demo/ folder:

    cd demo
    python app.py --ckpt checkpoints/best.pt --stats splits/stats.json

For HuggingFace Spaces deployment, set env vars CKPT_PATH and
STATS_PATH, then `python app.py` with no flags.

Preloaded examples expected under `demo/examples/` (mix of dataset
images + hand-filtered OOD images).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow importing project modules (augment, eval, model) from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: F401  (kept; used transitively by eval.overlay_sam)
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from augment import CLASSES, load_stats
from eval import CLASS_NAMES, load_model, overlay_sam

UNSURE_THRESHOLD = 0.40


def build_predict_fn(model: torch.nn.Module, mean, std, device: torch.device):
    eval_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    model.to(device).eval()

    @torch.no_grad()
    def predict(image: Image.Image):
        if image is None:
            return None, None, "Upload an image."
        pil = image.convert("RGB")
        x = eval_tx(pil).unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        top = int(probs.argmax())
        top_conf = float(probs[top])
        confidences = {f"{CLASSES[i]} ({CLASS_NAMES[CLASSES[i]]})": float(probs[i])
                       for i in range(10)}

        sam = model.last_spatial_attention()
        if sam is not None:
            sam_up = F.interpolate(sam, size=(224, 224), mode="bilinear", align_corners=False)
            sam_map = sam_up[0, 0].cpu().numpy()
            rgb = np.array(pil.resize((256, 256)).crop((16, 16, 240, 240)))
            overlay = overlay_sam(rgb, sam_map, alpha=0.45)
            overlay_img = Image.fromarray(overlay)
        else:
            overlay_img = None

        verdict = (f"Prediction: **{CLASSES[top]} — {CLASS_NAMES[CLASSES[top]]}** "
                   f"(confidence {top_conf:.2%})")
        if top_conf < UNSURE_THRESHOLD:
            verdict += f"\n\n_Model unsure_ (confidence < {UNSURE_THRESHOLD:.0%}) — " \
                       f"likely out-of-distribution input."
        return overlay_img, confidences, verdict

    return predict


_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Human-readable subfolder name per class. Order = display order in gallery.
CATALOG_FOLDER_NAMES: list[tuple[str, str]] = [
    ("c0", "safe_driving"),
    ("c1", "text_right"),
    ("c2", "phone_right"),
    ("c3", "text_left"),
    ("c4", "phone_left"),
    ("c5", "radio"),
    ("c6", "drinking"),
    ("c7", "reach_behind"),
    ("c8", "hair_makeup"),
    ("c9", "talk_passenger"),
]


def collect_catalog(catalog_dir: Path) -> list[str]:
    """List catalog images grouped by class subfolder, ordered c0..c9 then filename.

    Accepts both descriptive subfolder names (safe_driving, text_right, ...) and
    legacy c0..c9 codes for backward compat.
    """
    if not catalog_dir.exists():
        return []
    out: list[str] = []
    for code, name in CATALOG_FOLDER_NAMES:
        cls_dir = catalog_dir / name
        if not cls_dir.exists():
            cls_dir = catalog_dir / code  # fallback
            if not cls_dir.exists():
                continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in _IMG_EXT:
                out.append(str(p))
    return out


def collect_flat(folder: Path) -> list[str]:
    """Flat list of images in one folder, sorted by filename."""
    if not folder.exists():
        return []
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in _IMG_EXT)


def build_ui(predict_fn, catalog: list[str], google: list[str]):
    with gr.Blocks(title="Driver distraction classifier (Run 6)") as demo:
        gr.Markdown(
            "# Distracted-driver classifier — Run 6\n"
            "ResNet-18 + CBAM, trained from scratch on State Farm with "
            "subject-wise split. Upload your own image, or pick from the "
            "**State Farm test catalog** (3/class, held-out drivers) or "
            "the **Google side-view OOD gallery**. The heatmap shows where "
            "the model is looking."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input image")
                btn = gr.Button("Classify", variant="primary")
                if catalog:
                    gr.Examples(
                        examples=catalog,
                        inputs=inp,
                        label=f"State Farm test catalog ({len(catalog)} images, 3 per class)",
                        examples_per_page=15,
                    )
                if google:
                    gr.Examples(
                        examples=google,
                        inputs=inp,
                        label=f"Google side-view OOD ({len(google)} images)",
                        examples_per_page=10,
                    )
                if not catalog and not google:
                    gr.Markdown(
                        "_No example galleries found. Drop images under_ "
                        "`demo/examples/catalog/c{0..9}/` _and_ `demo/examples/google/`."
                    )
            with gr.Column():
                overlay = gr.Image(label="CBAM spatial attention", type="pil")
                probs = gr.Label(num_top_classes=5, label="Class probabilities")
                verdict = gr.Markdown()
        btn.click(predict_fn, inputs=inp, outputs=[overlay, probs, verdict])
        inp.change(predict_fn, inputs=inp, outputs=[overlay, probs, verdict])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path(os.environ.get("CKPT_PATH", "checkpoints/best.pt")))
    ap.add_argument("--stats", type=Path,
                    default=Path(os.environ.get("STATS_PATH", "splits/stats.json")))
    ap.add_argument("--catalog-dir", type=Path,
                    default=Path(os.environ.get("CATALOG_DIR", "examples/catalog")),
                    help="State Farm test catalog, subfolders c0..c9")
    ap.add_argument("--google-dir", type=Path,
                    default=Path(os.environ.get("GOOGLE_DIR", "examples/google")),
                    help="Flat folder of OOD side-view images")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--server-port", type=int, default=7860)
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    if not args.stats.exists():
        raise SystemExit(f"Stats file not found: {args.stats}")

    mean, std = load_stats(args.stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, use_ema=True)
    predict_fn = build_predict_fn(model, mean, std, device)

    catalog = collect_catalog(args.catalog_dir)
    google = collect_flat(args.google_dir)
    print(f"[examples] catalog: {len(catalog)} images from {args.catalog_dir}")
    print(f"[examples] google:  {len(google)} images from {args.google_dir}")

    demo = build_ui(predict_fn, catalog, google)
    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
