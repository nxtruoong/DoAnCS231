# RUN8_HOWTO — End-to-end execution guide for Run 8

Step-by-step Kaggle T4×2 recipe for the three-stream model (full image
+ hand/lap crop + MediaPipe head-pose features). Assumes Run 5/6/7
setup (splits + stats.json in `/kaggle/working/splits/`).

Design rationale: `RUN8_PLAN.md`. Run-7 retrospective: `log.md`.

---

## 0. Prerequisites

- Subject-wise splits already produced (`splits/train.csv`,
  `splits/val.csv`, `splits/stats.json`). If a previous Run already
  produced these, skip §1b.
- Code dataset attached at `/kaggle/input/driver-distraction-cbam` (or
  cloned to `/kaggle/working/code`). Must include the Run 8 additions:
  - `extract_pose.py`
  - `model_twostream.py` (with `ThreeStreamCBAM` + `build_threestream`)
  - `augment_twostream.py` (with `RegionCrop`, `ThreeStreamDataset`,
    `build_hand_*_transform`, `load_pose_lookup`)
  - `train_twostream.py` (with `--three-stream` + `--pose-parquet` flags)
  - `eval_twostream.py` (auto-dispatches on ckpt mode)

If using the GitHub mirror:
```python
!rm -rf /kaggle/working/code
!git clone https://github.com/nxtruoong/DoAnCS231-V2 /kaggle/working/code
CODE_DIR = "/kaggle/working/code"
```

---

## 1a. Paths (Cell 1a)

```python
import os, sys
COMP_DIR = "/kaggle/input/competitions/state-farm-distracted-driver-detection"
CODE_DIR = "/kaggle/input/driver-distraction-cbam"   # or /kaggle/working/code
WORK     = "/kaggle/working"
RUN      = f"{WORK}/run8"

assert os.path.exists(COMP_DIR + "/driver_imgs_list.csv"), "Competition dataset not attached"
assert os.path.exists(CODE_DIR + "/extract_pose.py"),      "Run 8 code missing"
sys.path.insert(0, CODE_DIR)
```

---

## 1b. Data prep (only if `splits/stats.json` missing)

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/data_prep.py",
    "--data-root", COMP_DIR,
    "--out-dir",   f"{WORK}/splits",
    "--batch-size", "64",
    "--num-workers", "4",
], check=True)
```

---

## 1c. Install MediaPipe + polars (Cell 1c)

MediaPipe is not preinstalled on Kaggle GPU images by default.

```python
!pip install -q mediapipe polars
```

Polars used by `extract_pose.py` for parquet write and by
`augment_twostream.load_pose_lookup` for read.

---

## 2. Precompute pose features (Cell 2, one-time, ~18 min)

Runs MediaPipe Pose over every training image, writes
`splits/pose.parquet`. Cached across re-runs — skip this cell if the
parquet already exists.

```python
import subprocess, os
if not os.path.exists(f"{WORK}/splits/pose.parquet"):
    subprocess.run([
        "python", f"{CODE_DIR}/extract_pose.py",
        "--img-root", f"{COMP_DIR}/imgs/train",
        "--out",      f"{WORK}/splits/pose.parquet",
    ], check=True)
else:
    print("pose.parquet already exists, skipping")
```

**Expected end-of-cell line:**

```
Done. XXXXX/22424 detections (XX%). Wrote .../pose.parquet (X.X MB). Elapsed: ~18 min.
```

**Pass criterion:** detection rate ≥ 0.70. If lower, MediaPipe is
failing on too many frames (cabin lighting / occlusion). Halt and
either:
- Lower `min_detection_confidence` in `extract_pose.py` (currently
  0.3), or
- Fall back to YOLOv8n-face for head-bbox features.

---

## 3. Sanity check — inspect hand/lap crop coverage (Cell 3)

**Run this before training.** If the bottom-center crop misses the
phone/lap region, training is wasted GPU.

```python
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from augment_twostream import RegionCrop

crop = RegionCrop(top_frac=0.45, bottom_frac=1.0,
                  left_frac=0.20, right_frac=0.85)

df = pd.read_csv(f"{WORK}/splits/train.csv").sample(20, random_state=0)
fig, ax = plt.subplots(4, 5, figsize=(15, 12))
for a, (_, r) in zip(ax.ravel(), df.iterrows()):
    img = Image.open(f"{COMP_DIR}/imgs/train/{r['classname']}/{r['img']}").convert("RGB")
    a.imshow(crop(img))
    a.set_title(r["classname"], fontsize=9)
    a.axis("off")
plt.tight_layout(); plt.show()
```

**Pass criterion:** each crop shows steering wheel bottom + driver's
lap. The phone/cup/radio object (when present) is fully inside the
crop.

**If fail:**
- Lap cut off → bump `--hand-top-frac` down from 0.45 to 0.40.
- Right hand on far-right console missing → widen `--hand-right-frac`
  from 0.85 to 0.90.
- Left-hand-on-wheel mostly outside → widen `--hand-left-frac` from
  0.20 down to 0.10.

---

## 4. Pose vector sanity check (Cell 4)

Confirm the precomputed pose values separate classes the way we
expect (e.g. c9 should have negative yaw on average — head turned
right toward passenger).

```python
import polars as pl, pandas as pd
df_pose = pl.read_parquet(f"{WORK}/splits/pose.parquet").to_pandas()
df_split = pd.read_csv(f"{WORK}/splits/train.csv")
m = df_split.merge(df_pose, left_on="img", right_on="filename")

# p0 = yaw proxy: r_ear.x - l_ear.x. Negative when looking right.
# p7 = detection success flag.
print("yaw mean per class (only successful detections):")
print(m[m.p7 > 0].groupby("classname")["p0"].mean().round(3))
print("\ndetection rate per class:")
print(m.groupby("classname")["p7"].mean().round(3))
```

**Expected pattern:**
- c9 (talk passenger): most-negative or most-positive yaw (head turned).
- c0 / c1 / c2 / c3 / c4 (eyes-forward classes): yaw near 0.
- Detection rate ≥ 0.70 in every class. If a class has rate < 0.50,
  pose feature won't help that class (missing data) — note in eval.

---

## 5. Smoke test (Cell 5, ~5 min)

Two-epoch run on the full pipeline before the long training cell.

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train_twostream.py",
    "--three-stream",
    "--pose-parquet", f"{WORK}/splits/pose.parquet",
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    f"{WORK}/run8_smoke",
    "--epochs", "2",
    "--batch-size", "48",
    "--num-workers", "2",
    "--no-cutmix",
    "--data-parallel",
], check=True)
```

Expect: 2 epochs complete, no OOM, val acc in 0.10–0.30 (too early to
be high). If OOM → drop `--batch-size` to 32 or remove
`--data-parallel`.

---

## 6. Full Run 8 training (Cell 6, ~3-3.5 hr)

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train_twostream.py",
    "--three-stream",
    "--pose-parquet", f"{WORK}/splits/pose.parquet",
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    RUN,
    "--epochs", "60",
    "--batch-size", "48",
    "--num-workers", "2",
    "--lr", "0.03",
    "--warmup-epochs", "2",
    "--ema-decay", "0.99",
    "--no-cutmix",
    "--full-size", "384",
    "--hand-size", "224",
    "--hand-top-frac", "0.45",
    "--hand-bottom-frac", "1.0",
    "--hand-left-frac", "0.20",
    "--hand-right-frac", "0.85",
    "--label-smoothing", "0.1",
    "--early-stop-patience", "8",
    "--early-stop-min-delta", "0.000",
    "--ckpt-every", "2",
    "--data-parallel",
], check=True)
```

**Monitoring during training:**
- Each epoch prints `train loss/acc | val loss/acc | ema val acc | elapsed`.
- Expected per-epoch time on T4×2 batch 48: **~3.2 min**. Total ~3.2 hr.
- Checkpoints land in `/kaggle/working/run8/`: `best.pt`,
  `ckpt_e02.pt`, ..., `final.pt`.

**Save-and-commit recommendation.** Use **Save Version → Save & Run
All — Commit** so the run survives browser disconnects.

**Watch milestones (Run 8 should beat Run 6 and Run 7 at each):**

| ep | target ema val acc | Run 7 actual | Run 6 actual |
|---:|---:|---:|---:|
| 10 | ≥ 0.55 | 0.50 | 0.78 |
| 20 | ≥ 0.72 | 0.64 | 0.82 |
| 30 | ≥ 0.80 | 0.68 | 0.84 |
| 40 | ≥ 0.85 | 0.75 | 0.86 |
| 50 | ≥ 0.87 | (n/a) | 0.87 |
| end | ≥ 0.88 | (n/a) | 0.87 |

Run 8 will start slower than Run 6 (two backbones to train from
scratch instead of one) but should match by ep 30 and pass by ep 50.

If ep 20 < 0.65 → architecture not helping. Kill the run, inspect
`ckpt_e10.pt` per-class metrics, decide whether to widen the hand
crop or fall back to Run 6.

---

## 7. Resume if interrupted (Cell 7)

Run 8 saves checkpoints every 2 epochs (`--ckpt-every 2`). Resume from
the latest:

```python
import subprocess, glob
ckpts = sorted(glob.glob(f"{RUN}/ckpt_e*.pt"))
last_ckpt = ckpts[-1] if ckpts else None
print("Resuming from", last_ckpt)

subprocess.run([
    "python", f"{CODE_DIR}/train_twostream.py",
    "--three-stream",
    "--pose-parquet", f"{WORK}/splits/pose.parquet",
    "--resume", last_ckpt,
    # ... same args as Cell 6 ...
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    RUN,
    "--epochs", "60",
    "--batch-size", "48",
    "--num-workers", "2",
    "--lr", "0.03",
    "--warmup-epochs", "2",
    "--ema-decay", "0.99",
    "--no-cutmix",
    "--full-size", "384",
    "--hand-size", "224",
    "--hand-top-frac", "0.45",
    "--hand-bottom-frac", "1.0",
    "--hand-left-frac", "0.20",
    "--hand-right-frac", "0.85",
    "--label-smoothing", "0.1",
    "--early-stop-patience", "8",
    "--early-stop-min-delta", "0.000",
    "--ckpt-every", "2",
    "--data-parallel",
], check=True)
```

---

## 8. Peek at history (Cell 8)

```python
import json
hist = json.loads(open(f"{RUN}/history.json").read())
print(f"Last epoch: {hist[-1]['epoch']}")
best_idx = max(range(len(hist)), key=lambda i: hist[i]['ema_val_acc'])
print(f"Best EMA:   {hist[best_idx]['ema_val_acc']:.4f} at ep {hist[best_idx]['epoch']}")
print(f"Best raw:   {max(x['val_acc'] for x in hist):.4f}")
```

---

## 9. Eval + figures (Cell 9, ~5-10 min)

`eval_twostream.py` auto-detects three-stream checkpoints from the
saved `three_stream` flag.

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/eval_twostream.py",
    "--ckpt",         f"{RUN}/best.pt",
    "--pose-parquet", f"{WORK}/splits/pose.parquet",
    "--data-root",    COMP_DIR,
    "--splits-dir",   f"{WORK}/splits",
    "--out-dir",      f"{RUN}/eval",
    "--history-json", f"{RUN}/history.json",
    "--full-size", "384",
    "--hand-size", "224",
    "--hand-top-frac", "0.45",
    "--hand-bottom-frac", "1.0",
    "--hand-left-frac", "0.20",
    "--hand-right-frac", "0.85",
], check=True)
```

Generates:
- `classification_report.txt`
- `metrics.json`
- `confusion_matrix.png`
- `per_driver_accuracy.{png,csv}`
- `training_curves.png`
- `attention_grid.png` (full-stream SAM)
- `failures.png`

Print the line `Loaded three-stream model from .../best.pt
(use_ema=True)` to confirm dispatch worked.

---

## 10. Compare to Run 6 and Run 7 (Cell 10)

```python
import json, pandas as pd

def metrics(p):
    m = json.load(open(p))
    return {"accuracy": m["accuracy"],
            "macro_f1": m["macro avg"]["f1-score"],
            "weighted_f1": m["weighted avg"]["f1-score"]}

table = pd.DataFrame({
    "Run 6 (single stream)":     metrics(f"{WORK}/run6/eval/metrics.json"),
    "Run 7 (two stream)":        metrics(f"{WORK}/run7/eval/metrics.json"),
    "Run 8 (three stream)":      metrics(f"{RUN}/eval/metrics.json"),
}).T
print(table.to_string())
table.to_csv(f"{WORK}/run8_vs_others.csv")
```

**Pass criterion for Run 8:** macro F1 ≥ 0.85 (passes Run 7's 0.748
and approaches Run 6's 0.873). Stretch goal: ≥ 0.88 macro F1.

If macro F1 < 0.80 → Run 8 underperformed. Run 6 stays as the headline.

---

## 11. Per-class deep dive (Cell 11)

Check whether starving classes (c0, c5, c9) and dumping classes (c3,
c7) recovered vs Run 7:

```python
import json
m6 = json.load(open(f"{WORK}/run6/eval/metrics.json"))
m7 = json.load(open(f"{WORK}/run7/eval/metrics.json"))
m8 = json.load(open(f"{RUN}/eval/metrics.json"))

print(f"{'class':<25} {'R6 F1':>8} {'R7 F1':>8} {'R8 F1':>8} {'Δ vs R7':>8}")
for k in m8:
    if k.startswith("c") and "(" in k:
        f6 = m6[k]["f1-score"]; f7 = m7[k]["f1-score"]; f8 = m8[k]["f1-score"]
        marker = "  <-- target" if k.startswith(("c0", "c3", "c5", "c9")) else ""
        print(f"{k:<25} {f6:>8.4f} {f7:>8.4f} {f8:>8.4f} {f8-f7:>+8.4f}{marker}")
```

Targets vs Run 7:
- c0 safe: +0.24 (0.51 → 0.75)
- c3 text-left: +0.30 (0.55 → 0.85)
- c5 radio: +0.16 (0.69 → 0.85)
- c9 talk-passenger: +0.36 (0.44 → 0.80)
- everything else: within ±0.03 (don't regress on solved classes)

---

## 12. Bundle artifacts for download (Cell 12)

```python
import zipfile
from pathlib import Path

OUT = Path(f"{WORK}/artifacts_run8.zip")
OUT.unlink(missing_ok=True)

paths = [
    Path(f"{RUN}/best.pt"),
    Path(f"{RUN}/history.json"),
    *Path(f"{RUN}/eval").iterdir(),
    Path(f"{WORK}/splits/stats.json"),
    Path(f"{WORK}/splits/pose.parquet"),
    Path(f"{WORK}/splits/train.csv"),
    Path(f"{WORK}/splits/val.csv"),
    Path(f"{WORK}/run8_vs_others.csv"),
]
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        if p.exists():
            z.write(p, p.relative_to(WORK))

print(f"{OUT.name}: {OUT.stat().st_size / 1e6:.1f} MB")
from IPython.display import FileLink, display
display(FileLink(str(OUT)))
```

Run 8 `best.pt` is ~270 MB (same backbone count as Run 7; pose MLP is
tiny). For HuggingFace Spaces (<50 MB without LFS):

```python
import torch
ck = torch.load(f"{RUN}/best.pt", map_location="cpu", weights_only=False)
torch.save({"ema": ck["ema"], "args": ck["args"],
            "three_stream": True, "two_stream": False},
           f"{RUN}/best_demo.pt")
```

`best_demo.pt` is ~90 MB. Demo needs to load pose.parquet too —
roughly 1 MB extra. Use LFS or split-zip for Spaces upload.

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `--three-stream requires --pose-parquet PATH` | flag missing | add `--pose-parquet /kaggle/working/splits/pose.parquet` |
| `mediapipe: not found` | install missing | `!pip install -q mediapipe` |
| Pose detection rate < 60% | cabin lighting / occlusion | lower `min_detection_confidence` in `extract_pose.py`, or fall back to YOLOv8n-face |
| OOM at batch 48 | host RAM, not VRAM | drop `--num-workers 2 → 0`, `--batch-size 48 → 32` |
| Per-epoch time > 5 min | DataLoader bottleneck | check `--num-workers 4`, `--batch-size 64`; verify `pose.parquet` is on local SSD (`/kaggle/working`), not `/kaggle/input` |
| Val acc stuck < 0.50 by ep 15 | pose-MLP init / fusion-head LR | inspect `ckpt_e10.pt`; check `args_dict["three_stream"]==True` |
| `Three-stream checkpoint requires --pose-parquet` at eval | eval invoked without pose | pass `--pose-parquet PATH` to `eval_twostream.py` |
| ckpt load `size mismatch` | trying to load Run 7 ckpt with Run 8 code or vice versa | use matching ckpt + script pair |
| `attention_grid.png` empty | `use_cbam=False` in ckpt args | re-train without `--no-cbam` |

---

## 14. Decision matrix after Run 8

| Run 8 macro F1 | Action |
|---|---|
| ≥ 0.88 | Headline result. Replace Run 6 in SLIDES + README. |
| 0.85 – 0.88 | Solid win over Run 7, near-match Run 6. Report Run 8 as final architecture with explanation of pose-feature ablation. |
| 0.80 – 0.85 | Partial recovery from Run 7 regression. Report all three runs in an ablation; pick Run 6 as headline unless c0/c5/c9 wins matter for the report narrative. |
| < 0.80 | Architecture-change hypothesis failed. Keep Run 6 as headline; report Run 7 + Run 8 as negative results. |
