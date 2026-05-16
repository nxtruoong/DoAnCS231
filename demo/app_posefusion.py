"""Gradio demo (Run 8, pose-fusion ResNet-18 + CBAM + MediaPipe pose).

Single ResNet18+CBAM stream over the full image at 384 fused with a
36-d MediaPipe Pose feature vector (head + wrists + elbows + fingers +
hips + derived signals + visibility gates) extracted on-the-fly from
the uploaded image.

Local run from the demo/ folder:

    cd demo
    python app_posefusion.py --ckpt checkpoints/run8_best.pt --stats splits/stats.json

Env vars CKPT_PATH and STATS_PATH supported for HuggingFace Spaces.

Dependencies beyond the Run 6/7 demos: mediapipe (pinned to 0.10.13 on
Kaggle / Python 3.12; see RUN8_HOWTO.md). Pose extraction runs on CPU
even when the model is on GPU — MediaPipe Pose is lightweight enough
(~50 ms per image) that this is not a bottleneck for interactive demo.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# Explicit submodule import bypasses lazy mediapipe.solutions resolution
from mediapipe.python.solutions import pose as mp_pose

from augment import CLASSES, load_stats
from augment_twostream import POSE_DIM, build_full_eval_transform
from eval import CLASS_NAMES, overlay_sam
from eval_twostream import load_model_for_ckpt

# MediaPipe Pose landmark indices (mirrors extract_pose.py; duplicated here
# so this module is importable without invoking extract_pose's top-level
# mediapipe import — useful for IDE / linting on hosts without mediapipe).
NOSE = 0
L_EYE, R_EYE = 2, 5
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22
L_HIP, R_HIP = 23, 24

UNSURE_THRESHOLD = 0.40

# Landmark groups for overlay rendering — colors in BGR (cv2)
LANDMARK_GROUPS = {
    "head":     {"indices": [0, 2, 5, 7, 8],          "color": (60, 76, 231),  "radius": 5},   # red
    "shoulder": {"indices": [11, 12],                 "color": (50, 168, 82),  "radius": 6},   # green
    "elbow":    {"indices": [13, 14],                 "color": (50, 168, 82),  "radius": 6},   # green
    "wrist":    {"indices": [15, 16],                 "color": (255, 144, 30), "radius": 8},   # blue
    "finger":   {"indices": [17, 18, 19, 20, 21, 22], "color": (255, 144, 30), "radius": 4},   # blue
    "hip":      {"indices": [23, 24],                 "color": (0, 215, 255),  "radius": 6},   # yellow
}
# Skeleton connections to draw
SKELETON = [
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),   # left arm + hand
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),   # right arm + hand
    (11, 12), (23, 24), (11, 23), (12, 24),                       # torso
    (0, 2), (0, 5), (2, 7), (5, 8),                               # face
]


def pose_vector_from_landmarks(landmarks) -> np.ndarray:
    """Compute the 36-d pose feature vector from MediaPipe landmarks.

    Same feature design as extract_pose.head_pose_vector but takes
    landmarks directly to avoid running pose.process() twice.
    """
    lm = landmarks.landmark
    nose = lm[NOSE]
    l_eye, r_eye = lm[L_EYE], lm[R_EYE]
    l_ear, r_ear = lm[L_EAR], lm[R_EAR]
    l_sh, r_sh = lm[L_SHOULDER], lm[R_SHOULDER]
    l_el, r_el = lm[L_ELBOW], lm[R_ELBOW]
    l_wr, r_wr = lm[L_WRIST], lm[R_WRIST]
    l_px, r_px = lm[L_PINKY], lm[R_PINKY]
    l_ix, r_ix = lm[L_INDEX], lm[R_INDEX]
    l_th, r_th = lm[L_THUMB], lm[R_THUMB]
    l_hi, r_hi = lm[L_HIP], lm[R_HIP]

    def dist(a, b):
        return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)

    return np.array([
        r_ear.x - l_ear.x,
        nose.y - (l_eye.y + r_eye.y) / 2.0,
        r_eye.y - l_eye.y,
        r_sh.x - l_sh.x,
        r_sh.y - l_sh.y,
        float(l_ear.visibility),
        float(r_ear.visibility),
        1.0,
        l_wr.x, l_wr.y, r_wr.x, r_wr.y,
        float(l_wr.visibility), float(r_wr.visibility),
        l_wr.y - r_wr.y,
        r_wr.x - l_wr.x,
        l_el.x, l_el.y, r_el.x, r_el.y,
        l_ix.y - l_wr.y, r_ix.y - r_wr.y,
        dist(l_th, l_px), dist(r_th, r_px),
        l_hi.x, l_hi.y, r_hi.x, r_hi.y,
        l_wr.y - l_hi.y, r_wr.y - r_hi.y,
        l_wr.x - l_sh.x, r_wr.x - r_sh.x,
        float(l_el.visibility), float(r_el.visibility),
        float(l_hi.visibility), float(r_hi.visibility),
    ], dtype=np.float32)


def draw_pose_overlay(img_rgb: np.ndarray, landmarks, success: bool) -> np.ndarray:
    """Draw MediaPipe pose landmarks + skeleton on an RGB numpy image."""
    out = img_rgb.copy()
    h, w = out.shape[:2]

    if not success or landmarks is None:
        cv2.putText(out, "Pose: not detected (using zero vector)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (60, 76, 231), 2, cv2.LINE_AA)
        return out

    # Pixel coordinates per landmark
    pts = {}
    for i, lm in enumerate(landmarks.landmark):
        pts[i] = (int(lm.x * w), int(lm.y * h), float(lm.visibility))

    # Skeleton lines first (so dots overlay them)
    for a, b in SKELETON:
        if a in pts and b in pts and pts[a][2] > 0.3 and pts[b][2] > 0.3:
            cv2.line(out, pts[a][:2], pts[b][:2], (180, 180, 180), 2, cv2.LINE_AA)

    # Dots, colored by group
    for _, g in LANDMARK_GROUPS.items():
        for idx in g["indices"]:
            if idx in pts and pts[idx][2] > 0.3:
                x, y, _ = pts[idx]
                cv2.circle(out, (x, y), g["radius"], g["color"], -1, cv2.LINE_AA)
                cv2.circle(out, (x, y), g["radius"], (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(out, "Pose: detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (50, 168, 82), 2, cv2.LINE_AA)
    return out


def build_predict_fn(model: torch.nn.Module, mean, std, device: torch.device,
                     full_size: int, pose_detector):
    tx_full = build_full_eval_transform(mean, std, size=full_size)
    model.to(device).eval()

    @torch.no_grad()
    def predict(image: Image.Image):
        if image is None:
            return None, None, None, "Upload an image."

        pil = image.convert("RGB")
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # --- pose extraction (CPU, ~50 ms) — single process() call ---
        pose_result = pose_detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if pose_result.pose_landmarks is None:
            pose_vec = np.zeros(POSE_DIM, dtype=np.float32)
            success = 0
        else:
            pose_vec = pose_vector_from_landmarks(pose_result.pose_landmarks)
            success = 1
        pose_tensor = torch.from_numpy(pose_vec).unsqueeze(0).to(device)

        # --- CNN forward ---
        full = tx_full(pil).unsqueeze(0).to(device)
        logits = model(full, pose_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        top = int(probs.argmax())
        top_conf = float(probs[top])
        confidences = {f"{CLASSES[i]} ({CLASS_NAMES[CLASSES[i]]})": float(probs[i])
                       for i in range(10)}

        # --- CBAM heatmap overlay ---
        sam = model.last_spatial_attention()
        if sam is not None:
            sam_up = F.interpolate(sam, size=(full_size, full_size),
                                   mode="bilinear", align_corners=False)
            sam_map = sam_up[0, 0].cpu().numpy()
            rgb_full = np.array(pil.resize((full_size, full_size)))
            overlay_full = Image.fromarray(overlay_sam(rgb_full, sam_map, alpha=0.45))
        else:
            overlay_full = None

        # --- pose landmark overlay ---
        pose_overlay_np = draw_pose_overlay(img_rgb, pose_result.pose_landmarks, success == 1)
        pose_overlay = Image.fromarray(pose_overlay_np)

        # --- verdict text ---
        verdict = (f"Prediction: **{CLASSES[top]} — {CLASS_NAMES[CLASSES[top]]}** "
                   f"(confidence {top_conf:.2%})")
        if top_conf < UNSURE_THRESHOLD:
            verdict += (f"\n\n_Model unsure_ (confidence < {UNSURE_THRESHOLD:.0%}) — "
                        f"likely out-of-distribution input.")
        if success == 0:
            verdict += ("\n\n_MediaPipe pose detection failed_ — falling back to "
                        "CNN-only prediction. Common for heavily occluded frames.")
        else:
            # Surface a few interpretable pose features for transparency
            verdict += (f"\n\n**Pose cues:** "
                        f"yaw={pose_vec[0]:+.3f} · "
                        f"left-wrist-lap={pose_vec[28]:+.3f} · "
                        f"right-wrist-lap={pose_vec[29]:+.3f} · "
                        f"right-arm-reach={pose_vec[31]:+.3f}")

        return overlay_full, pose_overlay, confidences, verdict

    return predict


def build_ui(predict_fn, examples: list[str]):
    with gr.Blocks(title="Driver distraction classifier (Run 8 — pose fusion)") as demo:
        gr.Markdown(
            "# Distracted-driver classifier — Run 8 (pose fusion)\n"
            "Single ResNet-18 + CBAM (384 input) fused with a 36-d MediaPipe "
            "pose feature vector extracted from the uploaded image. The pose "
            "panel shows the landmarks the model conditions on; the CBAM "
            "panel shows the spatial attention of the CNN backbone."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input image")
                btn = gr.Button("Classify", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=inp, label="Samples")
            with gr.Column():
                overlay = gr.Image(label="CBAM spatial attention (CNN stream)", type="pil")
                pose_img = gr.Image(label="MediaPipe pose landmarks (pose stream)", type="pil")
                probs = gr.Label(num_top_classes=5, label="Class probabilities")
                verdict = gr.Markdown()
        btn.click(predict_fn, inputs=inp, outputs=[overlay, pose_img, probs, verdict])
        inp.change(predict_fn, inputs=inp, outputs=[overlay, pose_img, probs, verdict])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path(os.environ.get("CKPT_PATH", "checkpoints/run8_best.pt")))
    ap.add_argument("--stats", type=Path,
                    default=Path(os.environ.get("STATS_PATH", "splits/stats.json")))
    ap.add_argument("--examples-dir", type=Path,
                    default=Path(os.environ.get("EXAMPLES_DIR", "examples")))
    ap.add_argument("--full-size", type=int, default=384)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--server-port", type=int, default=7860)
    args = ap.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {args.ckpt}")
    if not args.stats.exists():
        raise SystemExit(f"Stats file not found: {args.stats}")

    mean, std = load_stats(args.stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, mode = load_model_for_ckpt(args.ckpt, use_ema=True)
    if mode != "pose":
        raise SystemExit(
            f"Checkpoint mode = {mode!r}, expected 'pose'. Use app.py for Run 6 "
            f"single-stream or app_twostream.py for Run 7 two-stream checkpoints."
        )

    # Initialize MediaPipe Pose once (heavy load), reuse per request
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.3,
    )

    predict_fn = build_predict_fn(model, mean, std, device,
                                  full_size=args.full_size,
                                  pose_detector=pose_detector)

    examples: list[str] = []
    if args.examples_dir.exists():
        examples = sorted(str(p) for p in args.examples_dir.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    demo = build_ui(predict_fn, examples)
    demo.launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
