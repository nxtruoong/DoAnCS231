"""Precompute MediaPipe Pose head-pose features for all training images.

Output: splits/pose.parquet with columns
    filename, p0..p7
where p0..p6 are pose scalars and p7 is the detection-success flag.

Run once on Kaggle:
    pip install mediapipe
    python extract_pose.py \
        --img-root /kaggle/input/competitions/state-farm-distracted-driver-detection/imgs/train \
        --out      /kaggle/working/splits/pose.parquet

Pose features (see RUN8_PLAN.md for rationale):
    p0 = r_ear.x - l_ear.x              # yaw proxy (look-right < 0)
    p1 = nose.y - mean(eye.y)           # pitch proxy
    p2 = r_eye.y - l_eye.y              # roll proxy
    p3 = r_shoulder.x - l_shoulder.x    # torso twist
    p4 = r_shoulder.y - l_shoulder.y    # shoulder slope
    p5 = l_ear.visibility
    p6 = r_ear.visibility
    p7 = 1.0 if detection succeeded, else 0.0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import polars as pl
from tqdm import tqdm

POSE_DIM = 8

# MediaPipe Pose landmark indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 2, 5, 7, 8, 11, 12


def head_pose_vector(img_bgr: np.ndarray, pose) -> tuple[np.ndarray, int]:
    res = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return np.zeros(POSE_DIM, dtype=np.float32), 0

    lm = res.pose_landmarks.landmark
    nose = lm[NOSE]
    l_eye, r_eye = lm[L_EYE], lm[R_EYE]
    l_ear, r_ear = lm[L_EAR], lm[R_EAR]
    l_sh, r_sh = lm[L_SHOULDER], lm[R_SHOULDER]

    vec = np.array([
        r_ear.x - l_ear.x,                       # p0 yaw proxy
        nose.y - (l_eye.y + r_eye.y) / 2.0,      # p1 pitch proxy
        r_eye.y - l_eye.y,                       # p2 roll proxy
        r_sh.x - l_sh.x,                         # p3 torso twist
        r_sh.y - l_sh.y,                         # p4 shoulder slope
        float(l_ear.visibility),                 # p5
        float(r_ear.visibility),                 # p6
        1.0,                                     # p7 gate
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

    pose = mp.solutions.pose.Pose(
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
