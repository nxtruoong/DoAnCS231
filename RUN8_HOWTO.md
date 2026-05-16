# RUN8_HOWTO — End-to-end execution guide for Run 8 (pose-fusion)

Step-by-step Kaggle T4×2 recipe for the pose-fusion model: single
ResNet18+CBAM (full image at 384) + 36-d MediaPipe pose vector
fused at a 256-d MLP head.

Design rationale: `RUN8_PLAN.md`. Run-7 retrospective: `log.md`.

---

## 0. Prerequisites

- Subject-wise splits already produced (`splits/train.csv`,
  `splits/val.csv`, `splits/stats.json`). If a previous Run produced
  these, skip §1b.
- Code dataset attached at `/kaggle/input/driver-distraction-cbam` (or
  cloned to `/kaggle/working/code`). Must include the Run 8 files:
  - `extract_pose.py`
  - `model_twostream.py` (with `PoseFusionCBAM` + `build_posefusion`)
  - `augment_twostream.py` (with `PoseFusionDataset`, `load_pose_lookup`)
  - `train_twostream.py` (with `--pose-fusion` + `--pose-parquet`)
  - `eval_twostream.py` (auto-dispatch on `pose_fusion` flag)

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

Pin to an older version of MediaPipe on Kaggle Python 3.12 — newer
versions lazy-load `mp.solutions` and fail when TF/XLA registers CUDA
factories first.

```python
!pip install --no-cache-dir mediapipe==0.10.13 polars
!python -c "from mediapipe.python.solutions import pose as mp_pose; print('OK,', mp_pose.Pose)"
```

If the second line errors, try `mediapipe==0.10.14` or `0.10.15`. The
explicit submodule import in `extract_pose.py` is the safety net.

---

## 2. Precompute pose features (Cell 2, one-time, ~18 min)

Runs MediaPipe Pose over every training image, writes
`splits/pose.parquet`. 36 engineered features per image: head + wrists
+ elbows + fingers + hips + derived signals + visibility gates. See
`extract_pose.py` docstring for the full feature index.

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

**Pass criterion:** detection rate ≥ 0.70. If lower:
- Lower `min_detection_confidence` in `extract_pose.py` (currently
  0.3), or
- Fall back to YOLOv8n-face for head-bbox features.

---

## 3. Pose vector sanity check (Cell 3)

Verify the precomputed pose values separate classes the way the design
expects.

```python
import polars as pl, pandas as pd
df_pose = pl.read_parquet(f"{WORK}/splits/pose.parquet").to_pandas()
df_split = pd.read_csv(f"{WORK}/splits/train.csv")
m = df_split.merge(df_pose, left_on="img", right_on="filename")
ok = m[m.p7 > 0]

print("Detection rate per class:")
print(m.groupby("classname")["p7"].mean().round(3).to_string())

print("\nKey feature means per class (only successful detections):")
print(ok.groupby("classname")[[
    "p0",   # yaw   (negative = looking right toward passenger, expect c9 negative)
    "p28",  # l_wrist.y - l_hip.y (negative = wrist above hip; expect c3 near 0, c0 above)
    "p29",  # r_wrist.y - r_hip.y
    "p31",  # r_wrist.x - r_shoulder.x (large = right arm extended, expect c7 high)
]].mean().round(3).to_string())
```

**Expected pattern:**
- c9 (talk passenger): most-negative `p0` (head yaw to right).
- c7 (reach behind): largest `p31` (right arm extended past shoulder).
- c3 (text left): `p28` near 0 (left wrist on lap).
- c0 (safe): `p28`, `p29` both negative (both wrists above hips on wheel).
- Detection rate ≥ 0.70 in every class. If a class has rate < 0.50,
  pose features won't help it — note in eval.

If the pattern doesn't match expectations, the pose features are
weaker discriminators than predicted — consider falling back to Run 6
or adding more features.

---

## 4. Smoke test (Cell 4, ~3-5 min)

Two-epoch run on the full pipeline before the long training cell.

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train_twostream.py",
    "--pose-fusion",
    "--pose-parquet", f"{WORK}/splits/pose.parquet",
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    f"{WORK}/run8_smoke",
    "--epochs", "2",
    "--batch-size", "48",
    "--num-workers", "2",
    "--data-parallel",
], check=True)
```

Expect: 2 epochs complete, no OOM, val acc in 0.15–0.40. CutMix is
auto-disabled on the pose-fusion path even without `--no-cutmix`.

---

## 4b. Restart kernel before training (recommended)

Cell 2 loaded MediaPipe (~200 MB model). Cell 4 spun up a smoke train
process. Both leave residue in the kernel that competes with training
for the 13 GB Kaggle host RAM cap.

**Restart the kernel now** (Run → Restart & Clear Cell Outputs), then
re-run Cell 1a only. Skip Cells 1b / 1c / 2 (their outputs are already
on disk). Go straight to Cell 5.

---

## 5. Full Run 8 training (Cell 5, ~1.8-2 hr)

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train_twostream.py",
    "--pose-fusion",
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
    "--full-size", "384",
    "--label-smoothing", "0.1",
    "--early-stop-patience", "8",
    "--early-stop-min-delta", "0.000",
    "--ckpt-every", "2",
    "--data-parallel",
], check=True)
```

**Monitoring during training:**
- Each epoch prints `train loss/acc | val loss/acc | ema val acc | elapsed`.
- Expected per-epoch time on T4×2 batch 48: **~1.8 min**. Total ~1.8 hr.
- Single CNN backbone → ~50% faster than Run 7 per epoch.
- Checkpoints land in `/kaggle/working/run8/`: `best.pt`,
  `ckpt_e02.pt`, ..., `final.pt`.

**Save-and-commit recommendation.** Use **Save Version → Save & Run
All — Commit** so the run survives browser disconnects.

**Watch milestones:**

| ep | target ema val acc | Run 6 actual | Run 7 actual |
|---:|---:|---:|---:|
| 10 | ≥ 0.65 | 0.78 | 0.50 |
| 20 | ≥ 0.78 | 0.82 | 0.64 |
| 30 | ≥ 0.84 | 0.84 | 0.68 |
| 40 | ≥ 0.86 | 0.86 | 0.75 |
| 50 | ≥ 0.87 | 0.87 | (n/a) |
| end | ≥ 0.88 | 0.87 | (n/a) |

Run 8 starts slightly slower than Run 6 (pose MLP needs to learn the
projection from scratch) but should match by ep 20 and pass by ep 35
if pose features carry the signal.

If ep 20 < 0.70 → pose features not helping. Kill the run, inspect
`ckpt_e10.pt` per-class metrics, decide whether to retreat to Run 6.

---

## 6. Resume if interrupted (Cell 6)

```python
import subprocess, glob
ckpts = sorted(glob.glob(f"{RUN}/ckpt_e*.pt"))
last_ckpt = ckpts[-1] if ckpts else None
print("Resuming from", last_ckpt)

subprocess.run([
    "python", f"{CODE_DIR}/train_twostream.py",
    "--pose-fusion",
    "--pose-parquet", f"{WORK}/splits/pose.parquet",
    "--resume", last_ckpt,
    # ... same args as Cell 5 ...
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    RUN,
    "--epochs", "60",
    "--batch-size", "48",
    "--num-workers", "2",
    "--lr", "0.03",
    "--warmup-epochs", "2",
    "--ema-decay", "0.99",
    "--full-size", "384",
    "--label-smoothing", "0.1",
    "--early-stop-patience", "8",
    "--early-stop-min-delta", "0.000",
    "--ckpt-every", "2",
    "--data-parallel",
], check=True)
```

---

## 7. Peek at history (Cell 7)

```python
import json
hist = json.loads(open(f"{RUN}/history.json").read())
print(f"Last epoch: {hist[-1]['epoch']}")
best_idx = max(range(len(hist)), key=lambda i: hist[i]['ema_val_acc'])
print(f"Best EMA:   {hist[best_idx]['ema_val_acc']:.4f} at ep {hist[best_idx]['epoch']}")
print(f"Best raw:   {max(x['val_acc'] for x in hist):.4f}")
```

---

## 8. Eval + figures (Cell 8, ~5 min)

`eval_twostream.py` auto-detects pose-fusion checkpoints from the
saved `pose_fusion` flag.

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

Confirm the printed line `Loaded pose-stream model from .../best.pt
(use_ema=True)` — that means dispatch worked.

---

## 9. Compare to Run 6 and Run 7 (Cell 9)

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
    "Run 8 (pose fusion)":       metrics(f"{RUN}/eval/metrics.json"),
}).T
print(table.to_string())
table.to_csv(f"{WORK}/run8_vs_others.csv")
```

**Pass criterion for Run 8:** macro F1 ≥ 0.85 (passes Run 7's 0.748
and approaches Run 6's 0.873). Stretch goal: ≥ 0.88 macro F1.

If macro F1 < 0.80 → Run 6 stays as the headline.

---

## 10. Per-class deep dive (Cell 10)

Check whether starving classes (c0, c5, c9) and dumping classes (c3,
c7) recovered:

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
- c0 safe: +0.29 (0.51 → 0.80)
- c3 text-left: +0.33 (0.55 → 0.88)
- c5 radio: +0.21 (0.69 → 0.90)
- c9 talk-passenger: +0.38 (0.44 → 0.82)
- everything else: within ±0.03 (don't regress on solved classes)

---

## 11. Bundle artifacts for download (Cell 11)

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

Run 8 `best.pt` is ~135 MB (single ResNet18 backbone + small pose
head). For HuggingFace Spaces (<50 MB without LFS):

```python
import torch
ck = torch.load(f"{RUN}/best.pt", map_location="cpu", weights_only=False)
torch.save({"ema": ck["ema"], "args": ck["args"], "pose_fusion": True},
           f"{RUN}/best_demo.pt")
```

`best_demo.pt` is ~45 MB. Fits HF Spaces without LFS.

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `--pose-fusion requires --pose-parquet PATH` | flag missing | add `--pose-parquet /kaggle/working/splits/pose.parquet` |
| `AttributeError: 'mediapipe' has no attribute 'solutions'` | newer mediapipe lazy-load broken on Kaggle | `pip install mediapipe==0.10.13` |
| Pose detection rate < 60% | cabin lighting / occlusion | lower `min_detection_confidence` in `extract_pose.py` |
| OOM at batch 48 | host RAM, not VRAM | Tier B: drop `--data-parallel` (single T4, +60% wall). Tier C: also `--num-workers 0` |
| Host RAM creeps over 10 GB by ep 3 | DP + worker queue churn | kill, restart kernel, relaunch without `--data-parallel` |
| Per-epoch time > 4 min | DataLoader bottleneck | check `--num-workers 4`, `--batch-size 64`; verify `pose.parquet` on `/kaggle/working` (local SSD), not `/kaggle/input` |
| Val acc stuck < 0.50 by ep 15 | pose MLP init / fusion-head LR | inspect `ckpt_e10.pt`; verify `args_dict["pose_fusion"]==True` |
| `Pose-fusion checkpoint requires --pose-parquet` at eval | eval invoked without pose | pass `--pose-parquet PATH` to `eval_twostream.py` |
| ckpt load `size mismatch` | wrong arch (Run 7 ckpt with Run 8 code or vice versa) | match ckpt + script pair |
| `attention_grid.png` empty | `use_cbam=False` in ckpt args | re-train without `--no-cbam` |

---

## 13. Decision matrix after Run 8

| Run 8 macro F1 | Action |
|---|---|
| ≥ 0.88 | Headline result. Replace Run 6 in SLIDES + README. |
| 0.85 – 0.88 | Solid win over Run 7, near-match Run 6. Report Run 8 as final architecture with pose-feature ablation. |
| 0.80 – 0.85 | Partial recovery from Run 7 regression. Report all three runs in ablation; pick Run 6 as headline unless c0/c5/c9 wins matter for the report narrative. |
| < 0.80 | Pose features didn't disambiguate enough. Keep Run 6 as headline; report Run 7 + Run 8 as negative results. |
