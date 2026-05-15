# RUN7_HOWTO — End-to-end execution guide for Run 7

Step-by-step Kaggle T4×2 recipe for the two-stream model. Assumes you
already have a working Run 5/6 setup (splits + stats.json in
`/kaggle/working/splits/`).

For design rationale see `RUN7_PLAN.md`. This file is the runbook.

---

## 0. Prerequisites

- Subject-wise splits already produced (`splits/train.csv`,
  `splits/val.csv`, `splits/stats.json`). If not, run RUNPROJECT.md §4
  first.
- Code dataset attached at `/kaggle/input/driver-distraction-cbam` (or
  cloned to `/kaggle/working/code`). Must include the new files:
  - `model_twostream.py`
  - `augment_twostream.py`
  - `train_twostream.py`
  - `eval_twostream.py`
  - `model.py` (with the new `features()` method)

If using the GitHub mirror:
```python
!rm -rf /kaggle/working/code
!git clone https://github.com/nxtruoong/DoAnCS231-V2 /kaggle/working/code
CODE_DIR = "/kaggle/working/code"
```

---

## 1. Notebook setup (Cell 1)

```python
import os, sys
COMP_DIR = "/kaggle/input/competitions/state-farm-distracted-driver-detection"
CODE_DIR = "/kaggle/input/driver-distraction-cbam"   # or /kaggle/working/code
WORK     = "/kaggle/working"
RUN      = f"{WORK}/run7"

assert os.path.exists(COMP_DIR + "/driver_imgs_list.csv"), "Competition dataset not attached"
assert os.path.exists(CODE_DIR + "/train_twostream.py"),   "Run 7 code missing"
assert os.path.exists(WORK + "/splits/stats.json"),        "Run data_prep.py first (RUNPROJECT §4)"

sys.path.insert(0, CODE_DIR)
print("OK. GPU count:", __import__("torch").cuda.device_count())
```

---

## 2. Sanity check — inspect the top-crop coverage (Cell 2)

**Run this before training.** If the fixed top-45% crop misses too
many heads, training is wasted GPU.

```python
import random
from PIL import Image, ImageDraw
import pandas as pd

df = pd.read_csv(f"{WORK}/splits/train.csv")
samples = df.sample(20, random_state=42)

ok = 0
for _, row in samples.iterrows():
    p = f"{COMP_DIR}/imgs/train/{row['classname']}/{row['img']}"
    img = Image.open(p).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, int(h * 0.45)], outline="red", width=4)
    display(img.resize((320, 240)))
    print(f"  {row['classname']} / {row['img']}")
    # eyeball: head fully inside the box?

# Manual tally rule: if >= 18 of 20 heads are fully in the box,
# frac=0.45 is fine. If 10+ heads partially outside, raise frac.
```

**Pass criterion:** ≥ 18 / 20 heads fully inside the red box.
**If fail:** edit `--top-frac` in the train cell below. Common values:
0.50 (covers leaning-forward poses), 0.55 (safety margin).

---

## 3. Smoke test (Cell 3, ~5 min)

Verify the pipeline runs end-to-end before launching the full job.

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train_twostream.py",
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    f"{WORK}/run7_smoke",
    "--epochs", "2",
    "--batch-size", "64",
    "--num-workers", "4",
    "--data-parallel",
], check=True)
```

Expect: 2 epochs complete, no OOM, val acc somewhere in 0.10–0.40
(too early to be high). If OOM → drop `--batch-size` to 48 or 32.

---

## 4. Full Run 7 training (Cell 4, ~3-4 hr)

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/train_twostream.py",
    "--data-root", COMP_DIR,
    "--splits-dir", f"{WORK}/splits",
    "--out-dir",    RUN,
    "--epochs", "50",
    "--batch-size", "64",
    "--num-workers", "4",
    "--lr", "0.03",
    "--warmup-epochs", "2",
    "--ema-decay", "0.99",
    "--cutmix-alpha", "0.5",
    "--cutmix-p", "0.20",
    "--full-size", "384",
    "--face-size", "224",
    "--top-frac", "0.45",
    "--label-smoothing", "0.1",
    "--early-stop-patience", "8",
    "--early-stop-min-delta", "0.005",
    "--ckpt-every", "5",
    "--data-parallel",
], check=True)
```

**Monitoring during training:**
- Each epoch prints `train loss/acc | val loss/acc | ema val acc | elapsed`.
- Expected per-epoch time on T4×2 batch 64: **~4-5 min** (2× Run 6 due
  to ~2× params).
- Checkpoints land in `/kaggle/working/run7/`: `best.pt`, `ckpt_e05.pt`,
  ..., `final.pt`.

**Save-and-commit recommendation.** Use **Save Version → Save & Run
All — Commit** so the run survives browser disconnects. 3-4 hr is well
within the 12-hr Kaggle session limit but a closed tab kills interactive
runs.

**Watch milestones** (Run 7 should beat Run 6 at each checkpoint):

| ep | target ema val acc | Run 6 actual |
|---:|---:|---:|
| 10 | ≥ 0.75 | ~0.78 |
| 20 | ≥ 0.82 | 0.82 |
| 30 | ≥ 0.86 | 0.84 |
| 40 | ≥ 0.88 | 0.86 |
| end | ≥ 0.89 | 0.87 |

If ep 20 < 0.75 → something's wrong, kill the run and check
top-crop coverage + smoke test result.

---

## 5. Peek at history (Cell 5, instant)

```python
import json
hist = json.loads(open(f"{RUN}/history.json").read())
print(f"Last epoch: {hist[-1]['epoch']}")
best_idx = max(range(len(hist)), key=lambda i: hist[i]['ema_val_acc'])
print(f"Best EMA:   {hist[best_idx]['ema_val_acc']:.4f} at ep {hist[best_idx]['epoch']}")
print(f"Best raw:   {max(x['val_acc'] for x in hist):.4f}")
```

---

## 6. Eval + figures (Cell 6, ~5-10 min)

```python
import subprocess
subprocess.run([
    "python", f"{CODE_DIR}/eval_twostream.py",
    "--ckpt",         f"{RUN}/best.pt",
    "--data-root",    COMP_DIR,
    "--splits-dir",   f"{WORK}/splits",
    "--out-dir",      f"{RUN}/eval",
    "--history-json", f"{RUN}/history.json",
    "--full-size", "384",
    "--face-size", "224",
    "--top-frac", "0.45",
], check=True)
```

Generates the same artifact set as `eval.py`:
- `classification_report.txt`
- `metrics.json`
- `confusion_matrix.png`
- `per_driver_accuracy.{png,csv}`
- `training_curves.png`
- `attention_grid.png` (full-stream SAM)
- `failures.png`

---

## 7. Compare to Run 6 (Cell 7)

```python
import json, pandas as pd

def metrics(p):
    m = json.load(open(p))
    return {"accuracy": m["accuracy"],
            "macro_f1": m["macro avg"]["f1-score"],
            "weighted_f1": m["weighted avg"]["f1-score"]}

table = pd.DataFrame({
    "Run 6 (single stream)": metrics(f"{WORK}/run6/eval/metrics.json"),
    "Run 7 (two stream)":    metrics(f"{RUN}/eval/metrics.json"),
}).T
print(table.to_string())
table.to_csv(f"{WORK}/run7_vs_run6.csv")
```

**Pass criterion for Run 7:** macro F1 ≥ 0.886 (≥ +1pp over Run 6's
0.873). Stretch goal: 0.89-0.91.

If macro F1 < 0.86 → architecture not helping (see stop conditions in
`RUN7_PLAN.md`). Revert to Run 6 for the report.

---

## 8. Per-class deep dive (Cell 8)

Check whether the passive classes (c0, c8, c9) actually improved:

```python
import json
m6 = json.load(open(f"{WORK}/run6/eval/metrics.json"))
m7 = json.load(open(f"{RUN}/eval/metrics.json"))

print(f"{'class':<25} {'Run 6 F1':>10} {'Run 7 F1':>10} {'Δ':>8}")
for k in m7:
    if k.startswith("c") and "(" in k:
        f6 = m6[k]["f1-score"]; f7 = m7[k]["f1-score"]
        marker = "  <-- target" if k.startswith(("c0", "c8", "c9")) else ""
        print(f"{k:<25} {f6:>10.4f} {f7:>10.4f} {f7-f6:>+8.4f}{marker}")
```

Target deltas:
- c0: +9pp to +14pp (was 0.66)
- c8: +5pp to +8pp  (was 0.75)
- c9: +7pp to +12pp (was 0.68)
- everything else: within ±1pp (don't regress on solved classes)

---

## 9. Bundle artifacts for download (Cell 9)

```python
import zipfile
from pathlib import Path

OUT = Path(f"{WORK}/artifacts_run7.zip")
OUT.unlink(missing_ok=True)

paths = [
    Path(f"{RUN}/best.pt"),
    Path(f"{RUN}/history.json"),
    *Path(f"{RUN}/eval").iterdir(),
    Path(f"{WORK}/splits/stats.json"),
    Path(f"{WORK}/splits/train.csv"),
    Path(f"{WORK}/splits/val.csv"),
    Path(f"{WORK}/run7_vs_run6.csv"),
]
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        if p.exists():
            z.write(p, p.relative_to(WORK))

print(f"{OUT.name}: {OUT.stat().st_size / 1e6:.1f} MB")
from IPython.display import FileLink, display
display(FileLink(str(OUT)))
```

Note: `best.pt` for Run 7 is **~270 MB** (twice Run 6 — two backbones).
If HuggingFace Spaces upload is < 50 MB unless using LFS, you can save
EMA-only weights via:

```python
import torch
ck = torch.load(f"{RUN}/best.pt", map_location="cpu", weights_only=False)
torch.save({"ema": ck["ema"], "args": ck["args"], "two_stream": True},
           f"{RUN}/best_demo.pt")
```

`best_demo.pt` is ~90 MB. Still needs LFS for HF Spaces or split-zip.

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| OOM at batch 64 | 2× model VRAM | drop `--batch-size 48` (or 32) |
| Per-epoch time > 8 min | Wrong batch size or DataLoader bottleneck | check `--num-workers 6`, batch 64 |
| Val acc plateau ~0.84 (no gain over Run 6) | Top-crop misses heads | raise `--top-frac 0.50` |
| c8 precision crashes again | CutMix p still too low | raise `--cutmix-p 0.25` |
| `attention_grid.png` empty | `last_spatial_attention()` not populated | verify `use_cbam=True` in ckpt args |
| ckpt load error "size mismatch" | trying to load Run 6 ckpt into two-stream model | use `train_twostream.py` ckpts only with `eval_twostream.py` |
| Smoke test val acc stuck at 0.10 | Init bug regression (see log.md Run 1) | verify model.py final-Linear init is `normal_(0, 0.01)` |

---

## 11. Decision matrix after Run 7

| Run 7 macro F1 | Action |
|---|---|
| ≥ 0.89 | Headline result. Update SLIDES + README with two-stream architecture. |
| 0.886 – 0.89 | Marginal win. Report both Run 6 and Run 7 in ablation; pick the one with better c0/c8/c9. |
| 0.873 – 0.886 | No clear win. Keep Run 6 as headline; mention Run 7 as attempted ablation. |
| < 0.873 | Regression. Investigate top-crop coverage, batch size, fusion-head capacity. Don't ship Run 7. |
