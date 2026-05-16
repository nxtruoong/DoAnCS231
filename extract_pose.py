"""Precompute MediaPipe Pose features for all training images (Run 8).

Output: splits/pose.parquet with columns
    filename, p0..p35
where p0..p35 are 36 engineered pose features and p7 is the
detection-success flag (kept at index 7 for back-compat with the
8-feature prototype).

Run once on Kaggle:
    pip install mediapipe polars
    python extract_pose.py \\
        --img-root /kaggle/input/competitions/state-farm-distracted-driver-detection/imgs/train \\
        --out      /kaggle/working/splits/pose.parquet

Feature layout (36 dims):

  Head + torso (8):
    p0  = r_ear.x - l_ear.x              # yaw proxy (look-right < 0)
    p1  = nose.y - mean(eye.y)           # pitch proxy
    p2  = r_eye.y - l_eye.y              # roll proxy
    p3  = r_shoulder.x - l_shoulder.x    # torso twist
    p4  = r_shoulder.y - l_shoulder.y    # shoulder slope
    p5  = l_ear.visibility
    p6  = r_ear.visibility
    p7  = 1.0 if detection ok, else 0.0  # global gate

  Wrists (8):
    p8  = l_wrist.x       p9  = l_wrist.y
    p10 = r_wrist.x       p11 = r_wrist.y
    p12 = l_wrist.visibility
    p13 = r_wrist.visibility
    p14 = l_wrist.y - r_wrist.y          # vertical asymmetry (drinking, c6)
    p15 = r_wrist.x - l_wrist.x          # horizontal spread (c0 wide, c5 narrow)

  Elbows (4):
    p16 = l_elbow.x       p17 = l_elbow.y
    p18 = r_elbow.x       p19 = r_elbow.y

  Hand orientation / fingers (4):
    p20 = l_index.y - l_wrist.y          # fingers up? (texting c3)
    p21 = r_index.y - r_wrist.y          # fingers up? (texting c1)
    p22 = dist(l_thumb, l_pinky)         # left-hand spread
    p23 = dist(r_thumb, r_pinky)         # right-hand spread

  Hip anchors (4):
    p24 = l_hip.x         p25 = l_hip.y
    p26 = r_hip.x         p27 = r_hip.y

  Derived (4):
    p28 = l_wrist.y - l_hip.y            # left-hand lap proximity (c3, c4)
    p29 = r_wrist.y - r_hip.y            # right-hand lap proximity (c0)
    p30 = l_wrist.x - l_shoulder.x       # left-arm reach direction
    p31 = r_wrist.x - r_shoulder.x       # right-arm reach direction (c7 large)

  Visibility gates (4):
    p32 = l_elbow.visibility
    p33 = r_elbow.visibility
    p34 = l_hip.visibility
    p35 = r_hip.visibility
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import polars as pl
from tqdm import tqdm

# Explicit submodule import bypasses lazy `mediapipe.solutions` resolution,
# which can fail on Kaggle when TF/XLA registers CUDA factories before
# MediaPipe is initialized.
from mediapipe.python.solutions import pose as mp_pose

POSE_DIM = 36

# MediaPipe Pose landmark indices
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


def head_pose_vector(img_bgr: np.ndarray, pose) -> tuple[np.ndarray, int]:
    res = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return np.zeros(POSE_DIM, dtype=np.float32), 0

    lm = res.pose_landmarks.landmark

    # Head + torso
    nose = lm[NOSE]
    l_eye, r_eye = lm[L_EYE], lm[R_EYE]
    l_ear, r_ear = lm[L_EAR], lm[R_EAR]
    l_sh, r_sh = lm[L_SHOULDER], lm[R_SHOULDER]

    # Arms / hands
    l_el, r_el = lm[L_ELBOW], lm[R_ELBOW]
    l_wr, r_wr = lm[L_WRIST], lm[R_WRIST]
    l_px, r_px = lm[L_PINKY], lm[R_PINKY]
    l_ix, r_ix = lm[L_INDEX], lm[R_INDEX]
    l_th, r_th = lm[L_THUMB], lm[R_THUMB]
    l_hi, r_hi = lm[L_HIP], lm[R_HIP]

    def dist(a, b) -> float:
        return float(((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5)

    vec = np.array([
        # Head + torso (8)
        r_ear.x - l_ear.x,
        nose.y - (l_eye.y + r_eye.y) / 2.0,
        r_eye.y - l_eye.y,
        r_sh.x - l_sh.x,
        r_sh.y - l_sh.y,
        float(l_ear.visibility),
        float(r_ear.visibility),
        1.0,
        # Wrists (8)
        l_wr.x, l_wr.y,
        r_wr.x, r_wr.y,
        float(l_wr.visibility),
        float(r_wr.visibility),
        l_wr.y - r_wr.y,
        r_wr.x - l_wr.x,
        # Elbows (4)
        l_el.x, l_el.y,
        r_el.x, r_el.y,
        # Hand orientation / fingers (4)
        l_ix.y - l_wr.y,
        r_ix.y - r_wr.y,
        dist(l_th, l_px),
        dist(r_th, r_px),
        # Hip anchors (4)
        l_hi.x, l_hi.y,
        r_hi.x, r_hi.y,
        # Derived (4)
        l_wr.y - l_hi.y,
        r_wr.y - r_hi.y,
        l_wr.x - l_sh.x,
        r_wr.x - r_sh.x,
        # Visibility gates (4)
        float(l_el.visibility),
        float(r_el.visibility),
        float(l_hi.visibility),
        float(r_hi.visibility),
    ], dtype=np.float32)
    return vec, 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-root", type=Path, required=True,
                    help="Root containing class subdirs (c0..c9) of training images")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output parquet path")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional sample limit for smoke tests")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.img_root.rglob("*.jpg"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No .jpg images under {args.img_root}")

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.3,
    )

    rows = []
    success = 0
    t0 = time.time()
    for path in tqdm(paths, desc="pose"):
        img = cv2.imread(str(path))
        if img is None:
            vec, flag = np.zeros(POSE_DIM, dtype=np.float32), 0
        else:
            vec, flag = head_pose_vector(img, pose)
        success += flag
        rows.append({
            "filename": path.name,
            **{f"p{i}": float(vec[i]) for i in range(POSE_DIM)},
        })

    df = pl.DataFrame(rows)
    df.write_parquet(args.out)

    rate = success / len(paths)
    print(f"\nDone. {success}/{len(paths)} detections ({rate:.1%}). "
          f"Wrote {args.out} ({df.estimated_size('mb'):.1f} MB). "
          f"Elapsed: {(time.time() - t0)/60:.1f} min.")
    if rate < 0.60:
        print("WARNING: detection rate <60%. Consider YOLOv8n-face fallback.")


if __name__ == "__main__":
    main()
