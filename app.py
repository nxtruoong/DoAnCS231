"""Gradio demo: upload image -> top-1 prediction + class probabilities
   + CBAM spatial-attention heatmap overlay.

Local run:
    python app.py --ckpt checkpoints/best.pt --stats splits/stats.json

For HuggingFace Spaces deployment, set env vars CKPT_PATH and
STATS_PATH, then `python app.py` with no flags.

Preloaded examples expected under `examples/` (mix of dataset images +
your hand-filtered Google OOD images).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
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

        # Attention overlay (SAM from layer4)
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


def build_ui(predict_fn, examples: list[str]):
    with gr.Blocks(title="Driver distraction classifier") as demo:
        gr.Markdown(
            "# Distracted-driver classifier\n"
            "ResNet-18 + CBAM, trained from scratch on State Farm with "
            "subject-wise split. Upload a driver image (side view, full "
            "torso visible) or pick an example. The heatmap shows where "
            "the model is looking."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input image")
                btn = gr.Button("Classify", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=inp, label="Samples (dataset + OOD)")
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
    ap.add_argument("--examples-dir", type=Path,
                    default=Path(os.environ.get("EXAMPLES_DIR", "examples")))
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    if not args.stats.exists():
        raise SystemExit(f"Stats file not found: {args.stats}")

    mean, std = load_stats(args.stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, use_ema=True)
    predict_fn = build_predict_fn(model, mean, std, device)

    examples: list[str] = []
    if args.examples_dir.exists():
        examples = sorted(str(p) for p in args.examples_dir.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    demo = build_ui(predict_fn, examples)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
