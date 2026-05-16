"""Gradio demo (Run 7, two-stream ResNet-18 + CBAM).

Full stream (384) + face stream (TopCrop frac=0.50, 224) fused at GAP.

Local run from the demo/ folder:

    cd demo
    python app_twostream.py --ckpt checkpoints/run7_best.pt --stats splits/stats.json

Env vars CKPT_PATH and STATS_PATH supported for HuggingFace Spaces.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2  # noqa: F401
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from augment import CLASSES, load_stats
from augment_twostream import (
    build_face_eval_transform, build_full_eval_transform,
)
from eval import CLASS_NAMES, overlay_sam
from eval_twostream import load_twostream

UNSURE_THRESHOLD = 0.40


def build_predict_fn(model: torch.nn.Module, mean, std, device: torch.device,
                     full_size: int, face_size: int, top_frac: float):
    tx_full = build_full_eval_transform(mean, std, size=full_size)
    tx_face = build_face_eval_transform(mean, std, size=face_size, top_frac=top_frac)
    model.to(device).eval()

    @torch.no_grad()
    def predict(image: Image.Image):
        if image is None:
            return None, None, None, "Upload an image."
        pil = image.convert("RGB")
        full = tx_full(pil).unsqueeze(0).to(device)
        face = tx_face(pil).unsqueeze(0).to(device)
        logits = model(full, face)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        top = int(probs.argmax())
        top_conf = float(probs[top])
        confidences = {f"{CLASSES[i]} ({CLASS_NAMES[CLASSES[i]]})": float(probs[i])
                       for i in range(10)}

        sam = model.last_spatial_attention()  # full-stream SAM
        if sam is not None:
            sam_up = F.interpolate(sam, size=(full_size, full_size),
                                   mode="bilinear", align_corners=False)
            sam_map = sam_up[0, 0].cpu().numpy()
            rgb_full = np.array(pil.resize((full_size, full_size)))
            overlay_full = Image.fromarray(overlay_sam(rgb_full, sam_map, alpha=0.45))
        else:
            overlay_full = None

        # Face-stream preview: visualize the top-crop region the model sees.
        w, h = pil.size
        face_crop = pil.crop((0, 0, w, int(h * top_frac))).resize((face_size, face_size))

        verdict = (f"Prediction: **{CLASSES[top]} — {CLASS_NAMES[CLASSES[top]]}** "
                   f"(confidence {top_conf:.2%})")
        if top_conf < UNSURE_THRESHOLD:
            verdict += f"\n\n_Model unsure_ (confidence < {UNSURE_THRESHOLD:.0%}) — " \
                       f"likely out-of-distribution input."
        return overlay_full, face_crop, confidences, verdict

    return predict


def build_ui(predict_fn, examples: list[str]):
    with gr.Blocks(title="Driver distraction classifier (Run 7)") as demo:
        gr.Markdown(
            "# Distracted-driver classifier — Run 7 (two-stream)\n"
            "Full-frame stream (384) + TopCrop face stream (224, frac=0.50), "
            "ResNet-18 + CBAM each, fused at GAP. Heatmap is the full-stream "
            "spatial attention; face panel is the top-crop region the face "
            "branch sees."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input image")
                btn = gr.Button("Classify", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=inp, label="Samples")
            with gr.Column():
                overlay = gr.Image(label="Full-stream CBAM attention", type="pil")
                face = gr.Image(label="Face-stream input (TopCrop)", type="pil")
                probs = gr.Label(num_top_classes=5, label="Class probabilities")
                verdict = gr.Markdown()
        btn.click(predict_fn, inputs=inp, outputs=[overlay, face, probs, verdict])
        inp.change(predict_fn, inputs=inp, outputs=[overlay, face, probs, verdict])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path(os.environ.get("CKPT_PATH", "checkpoints/run7_best.pt")))
    ap.add_argument("--stats", type=Path,
                    default=Path(os.environ.get("STATS_PATH", "splits/stats.json")))
    ap.add_argument("--examples-dir", type=Path,
                    default=Path(os.environ.get("EXAMPLES_DIR", "examples")))
    ap.add_argument("--full-size", type=int, default=384)
    ap.add_argument("--face-size", type=int, default=224)
    ap.add_argument("--top-frac", type=float, default=0.50)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--server-port", type=int, default=7860)
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    if not args.stats.exists():
        raise SystemExit(f"Stats file not found: {args.stats}")

    mean, std = load_stats(args.stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_twostream(args.ckpt, use_ema=True)
    predict_fn = build_predict_fn(model, mean, std, device,
                                  full_size=args.full_size,
                                  face_size=args.face_size,
                                  top_frac=args.top_frac)

    examples: list[str] = []
    if args.examples_dir.exists():
        examples = sorted(str(p) for p in args.examples_dir.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    demo = build_ui(predict_fn, examples)
    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
