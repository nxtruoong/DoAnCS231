# RUNPROJECT — End-to-end execution guide

Step-by-step instructions to run the full project on Kaggle (T4 x2 free
tier) and deploy the demo. Follow top-to-bottom. Total wall time:
~5 hours of GPU + ~1 hour of human-in-the-loop steps.

---

## 0. Prerequisites

- Kaggle account, **phone-verified** (required for GPU access).
- Joined the competition page once (one-time "I Understand and Accept"
  click) so the dataset is accessible:
  https://www.kaggle.com/c/state-farm-distracted-driver-detection
- Project code packaged as either:
  - a **GitHub repo** (public or with a Kaggle secret token), OR
  - a **zip** ready to upload as a Kaggle dataset.

Recommended: GitHub. Easier to iterate.

---

## 1. Package the project as a Kaggle dataset

Kaggle notebooks cannot import arbitrary code unless it is on disk under
`/kaggle/input/` or `/kaggle/working/`. Two ways to expose the scripts.

### Option A — Upload as a Kaggle dataset (recommended)

1. From the repo root (`D:\UIT homework\CVV`), zip these files (exclude
   the local `splits/`, `checkpoints/`, `run*` directories — they are
   gitignored):
   ```
   data_prep.py
   augment.py
   model.py
   train.py
   eval.py
   app.py
   requirements.txt
   CONTEXT.md
   README.md
   docs/
   notebooks/
   ```
2. Go to https://www.kaggle.com/datasets → **New Dataset**.
3. Title: `driver-distraction-cbam`. Visibility: Private (you only).
4. Drag the zip in, wait for upload + "Create".
5. After processing it mounts at
   `/kaggle/input/driver-distraction-cbam/`.

### Option B — Clone from GitHub inside the notebook

Skip the dataset upload. In notebook cell 1:
```python
!rm -rf /kaggle/working/code
!git clone https://github.com/<you>/driver-distraction-cbam /kaggle/working/code
CODE_DIR = "/kaggle/working/code"
```

Use this if you will iterate on code — re-running the cell pulls the
latest commit.

---

## 2. Create the Kaggle notebook

1. Go to https://www.kaggle.com/code → **New Notebook**.
2. Right-hand sidebar, **Settings**:
   - **Accelerator:** GPU T4 x2.
   - **Internet:** On (needed by `pip` if you add packages).
   - **Persistence:** Files only (so `/kaggle/working/` survives between
     sessions of the same notebook).
3. **Add Data** (`+` icon):
   - Competition dataset: search **"state-farm-distracted-driver-detection"**.
     It mounts at
     `/kaggle/input/competitions/state-farm-distracted-driver-detection/`.
   - Your code dataset (Option A only): search
     **"driver-distraction-cbam"**. Mounts at
     `/kaggle/input/driver-distraction-cbam/`.
4. Rename the notebook **"01 - stats + split"** (or whichever step
   you're running). Repeat steps 1-3 for each of the four notebooks
   below.

You can also run all four steps in one big notebook to save setup time;
the four-notebook split is so you can re-run any single stage cleanly.

---

## 3. Verify mount paths (sanity cell)

Always start each notebook with this cell:

```python
import os, sys

# === paths ===
COMP_DIR  = "/kaggle/input/competitions/state-farm-distracted-driver-detection"
CODE_DIR  = "/kaggle/input/driver-distraction-cbam"   # or /kaggle/working/code (Option B)
WORK      = "/kaggle/working"

# === verify ===
assert os.path.exists(COMP_DIR + "/driver_imgs_list.csv"), "Competition dataset not attached"
assert os.path.exists(COMP_DIR + "/imgs/train/c0"),        "imgs/train/c0 missing"
assert os.path.exists(CODE_DIR + "/data_prep.py"),         "Code dataset not attached"

sys.path.insert(0, CODE_DIR)
print("OK. CODE_DIR =", CODE_DIR)
print("OK. COMP_DIR =", COMP_DIR)
print("GPU count   =", __import__("torch").cuda.device_count())
```

If any `assert` fires, fix the attached datasets before going further.

---

## 4. Notebook 01 — Stats + Split (~5-10 min)

**Purpose:** build subject-wise train/val CSVs and compute dataset RGB
mean/std (needed by both training and demo for consistent normalization).

```python
# Cell 1: paths (copy from section 3 above)
# Cell 2: run data prep
!cd {CODE_DIR} && python data_prep.py \
    --data-root {COMP_DIR} \
    --out-dir   {WORK}/splits \
    --batch-size 64 --num-workers 4
```

Expected stdout near the end:
```
Train: ~17xxx imgs from 21 subjects
Val  : ~4xxx  imgs from 5 subjects
Class distribution (train): c0 1850 c1 1822 ... (roughly uniform)
Class distribution (val):   c0 ~430 c1 ~430 ...
Val subjects x classes:     each subject has at least 1 image per class
Dataset stats: mean=[..., ..., ...] std=[..., ..., ...]
```

**Check before proceeding:**
- Every held-out subject (`p022, p035, p047, p056, p075`) has ≥ 1 image
  in **every** class. If one is missing, swap that subject out (edit
  `HELD_OUT_SUBJECTS` in `data_prep.py`, re-run).
- `stats.json` was written to `/kaggle/working/splits/stats.json`.

```python
# Cell 3: smoke-test training pipeline (2 epochs, ~3 min)
!cd {CODE_DIR} && python train.py \
    --data-root  {COMP_DIR} \
    --splits-dir {WORK}/splits \
    --out-dir    {WORK}/sanity \
    --epochs 2 --batch-size 128 --num-workers 4 \
    --data-parallel
```

If this OOMs, drop `--batch-size` to 96 (or 64). Then commit the notebook
("Save Version → Save & Run All — Commit") so `splits/` persists for the
next notebook session.

---

## 5. Notebook 02 — Full Train (~1.5 hr, early stop)

**Purpose:** Run 5 canonical config — ResNet-18 + CBAM, TrivialAugment,
320x320 input, max 80 epochs with early stopping.

```python
# Cell 1: paths (copy section 3)
# Cell 2: Run 5 training command
!cd {CODE_DIR} && python train.py \
    --data-root  {COMP_DIR} \
    --splits-dir {WORK}/splits \
    --out-dir    {WORK}/run5 \
    --epochs 80 --batch-size 128 --num-workers 4 \
    --lr 0.03 --warmup-epochs 2 \
    --momentum 0.9 --weight-decay 5e-4 \
    --label-smoothing 0.1 --ema-decay 0.99 \
    --cutmix-alpha 0.5 --cutmix-p 0.3 \
    --img-size 320 --trivialaugment \
    --early-stop-patience 8 --early-stop-min-delta 0.005 \
    --ckpt-every 5 \
    --data-parallel
```

**Monitoring:**
- Each epoch prints `train loss | train acc | val loss | val acc | ema val acc | elapsed`.
- Per-epoch time on T4 x2 with batch 128, img 320: ~2.2 min.
- Run 5 stopped at ep 38/80 via early stop (8 epochs no gain >= 0.005
  over best `max(val_acc, ema_val_acc) = 0.8431`). Total: ~83 min.
- Checkpoints land in `/kaggle/working/run5/`:
  `best.pt`, `ckpt_e05.pt`, `ckpt_e10.pt`, ..., `final.pt`.

**Expected milestones (from Run 5 log):**

| epoch | val acc | ema val acc |
|------:|--------:|------------:|
| 7  | 0.63 | 0.69 |
| 14 | 0.76 | 0.82 |
| 20 | 0.80 | 0.82 |
| 30 | 0.79 | **0.84** ← best EMA |
| 31 | **0.83** ← best raw | 0.80 |
| 38 | 0.77 | 0.81 (early stop) |

**Tier-1 fallback (built-in):** at epoch 20, if `max(val_acc, ema_val_acc) < 0.50`,
script aborts and prints a hint. Run 5 hit 0.82 by ep 20 — fallback
did not trigger. If it does, restart cell 2 with:
```
--no-cutmix --no-grayscale --out-dir {WORK}/run5_fallback
```

**Kaggle session limits:**
- Notebook session: 12 hr — well above our budget.
- Interactive session may disconnect if browser tab closes. Two
  defenses:
  1. Use **Save Version → Save & Run All — Commit**: runs the notebook
     headless from scratch, safe across disconnects. **Recommended for
     the full training cell.**
  2. Keep the browser tab open + active.

After training, verify:
```python
# Cell 3: peek at history
import json
hist = json.loads(open(f"{WORK}/run5/history.json").read())
print(f"Final epoch: {hist[-1]}")
print(f"Best EMA val acc seen: {max(h['ema_val_acc'] for h in hist):.4f}")
# Expected: best EMA ~0.84 (Run 5 reference)
```

**Tier-2 fallback (manual):** if final EMA val acc < 0.55, also run
notebook 03 with `--no-cbam` and present both — ablation table
becomes headline result.

Commit notebook so `run5/` persists.

---

## 6. Notebook 03 — Ablation (no CBAM, ~1 hr)

**Purpose:** baseline ResNet-18 without CBAM, same Run 5 schedule
otherwise. Populates ablation table.

```python
# Cell 1: paths (copy section 3)
# Cell 2: baseline run — Run 5 config minus CBAM
!cd {CODE_DIR} && python train.py \
    --data-root  {COMP_DIR} \
    --splits-dir {WORK}/splits \
    --out-dir    {WORK}/run_baseline \
    --epochs 80 --batch-size 128 --num-workers 4 \
    --lr 0.03 --warmup-epochs 2 \
    --label-smoothing 0.1 --ema-decay 0.99 \
    --cutmix-alpha 0.5 --cutmix-p 0.3 \
    --img-size 320 --trivialaugment \
    --early-stop-patience 8 --early-stop-min-delta 0.005 \
    --no-cbam \
    --data-parallel
```

Commit when done.

---

## 7. Notebook 04 — Eval + Figures (~10-15 min)

**Purpose:** turn checkpoints into `classification_report`, confusion
matrix, per-driver breakdown, training curves, attention grid, and
failure overlays. Plus the ablation comparison table.

```python
# Cell 1: paths (copy section 3)

# Cell 2: eval main model (img-size must match training: 320)
!cd {CODE_DIR} && python eval.py \
    --ckpt        {WORK}/run5/best.pt \
    --data-root   {COMP_DIR} \
    --splits-dir  {WORK}/splits \
    --out-dir     {WORK}/run5/eval \
    --history-json {WORK}/run5/history.json \
    --img-size 320

# Cell 3: eval baseline
!cd {CODE_DIR} && python eval.py \
    --ckpt        {WORK}/run_baseline/best.pt \
    --data-root   {COMP_DIR} \
    --splits-dir  {WORK}/splits \
    --out-dir     {WORK}/run_baseline/eval \
    --history-json {WORK}/run_baseline/history.json \
    --img-size 320

# Cell 4: ablation table
import json, pandas as pd
def metrics(p):
    m = json.loads(open(p).read())
    return {"accuracy":    m["accuracy"],
            "macro_f1":    m["macro avg"]["f1-score"],
            "weighted_f1": m["weighted avg"]["f1-score"]}
table = pd.DataFrame({
    "ResNet-18 + CBAM (full)": metrics(f"{WORK}/run5/eval/metrics.json"),
    "ResNet-18 baseline":      metrics(f"{WORK}/run_baseline/eval/metrics.json"),
}).T
print(table.to_string())
table.to_csv(f"{WORK}/ablation_table.csv")

# Cell 5: bundle all artifacts for download
!cd {WORK} && zip -r artifacts.zip \
    run5/eval run_baseline/eval \
    run5/history.json run_baseline/history.json \
    run5/best.pt run_baseline/best.pt \
    splits/stats.json splits/train.csv splits/val.csv \
    ablation_table.csv
print("Size:", __import__("os").path.getsize(f"{WORK}/artifacts.zip") / 1e6, "MB")
```

After commit, the notebook's **Output** tab exposes `artifacts.zip` as a
downloadable file. Pull it to your local machine.

---

## 8. Unpack artifacts locally

On your machine (`D:\UIT homework\CVV`):

```powershell
# unzip artifacts.zip into the repo root
Expand-Archive -Path artifacts.zip -DestinationPath .
```

You should now have:
```
run5/eval/
    classification_report.txt
    metrics.json
    confusion_matrix.png
    per_driver_accuracy.png
    per_driver_accuracy.csv
    training_curves.png
    attention_grid.png
    failures.png
run5/best.pt
run_baseline/eval/...
splits/stats.json
ablation_table.csv
```

These feed directly into the report.

---

## 9. Prepare OOD demo images

1. Create `examples/` in the repo root.
2. Google Image search: `"driver texting side view"`,
   `"driver phone dashcam"`, `"distracted driving side profile"`.
3. Hand-pick ~10 images that **match the State Farm framing** (side
   view, full torso, steering wheel visible). Reject close-ups,
   front-on shots, billboards, watermarked stock photos.
4. Rename for clarity: `ood_01_texting.jpg`, `ood_02_phone.jpg`, ...
5. Drop them into `examples/`.

---

## 10. Run the demo locally

```powershell
pip install -r requirements.txt
python app.py --ckpt run5/best.pt --stats splits/stats.json --examples-dir examples
```

Caveat: `app.py` hardcodes 224x224 preprocessing — mismatches Run 5
training at 320. Demo accuracy will lag eval metrics until aligned.

Gradio prints `http://127.0.0.1:7860` — open in browser. Test:
- Upload one dataset image → high-confidence correct class.
- Upload one OOD image → see prediction + heatmap + "unsure" flag if
  confidence < 40%.

For a public link (e.g. to share with the teacher): add `--share`.

---

## 11. Deploy to HuggingFace Spaces (optional but recommended)

1. Create account at https://huggingface.co.
2. **New Space** → SDK: Gradio, hardware: CPU basic (free).
3. In the Space repo, upload:
   - `app.py`, `model.py`, `augment.py`, `eval.py`, `requirements.txt`
   - `run5/best.pt` (rename `checkpoints/best.pt`, ≤ 50 MB, OK)
   - `splits/stats.json` (rename `splits/stats.json`)
   - `examples/`
4. In the Space's **Settings → Variables**, add (optional, since the
   defaults in `app.py` already match):
   ```
   CKPT_PATH=checkpoints/best.pt
   STATS_PATH=splits/stats.json
   EXAMPLES_DIR=examples
   ```
5. Space auto-builds and serves the Gradio UI at
   `https://huggingface.co/spaces/<you>/<space-name>`. Put this URL in
   the report and slides.

---

## 12. Final deliverables checklist

- [ ] `run5/eval/classification_report.txt` (sklearn output: precision,
      recall, f1-score, support per class + accuracy + macro avg +
      weighted avg)
- [ ] `run5/eval/confusion_matrix.png`
- [ ] `run5/eval/per_driver_accuracy.{png,csv}`
- [ ] `run5/eval/training_curves.png`
- [ ] `run5/eval/attention_grid.png`
- [ ] `run5/eval/failures.png`
- [ ] `ablation_table.csv` + `run_baseline/eval/*`
- [ ] `examples/` with 10 OOD images
- [ ] HuggingFace Spaces URL (or screen recording of local demo)
- [ ] Code repo (GitHub link)
- [ ] Report PDF (sections per `CONTEXT.md` plan)
- [ ] Slides (mirror report sections, heavy on visuals)

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `imgs/train/c0` missing | Competition not joined | Open comp page, accept rules, re-attach dataset |
| `CUDA out of memory` | Batch too large w/ CBAM | `--batch-size 96` or 64 |
| Per-epoch time > 8 min | DataLoader bottleneck | Increase `--num-workers` to 6-8 |
| EMA val acc stuck near 0.10 | Bug / NaNs | Lower `--lr 0.05`, disable CutMix one run |
| Tier-1 fired at epoch 20 | Aug too aggressive | Restart `--no-cutmix --no-grayscale` |
| Per-driver acc tanks on 1 driver | That driver has unusual cabin / clothing | Expected on subject-wise; note in report |
| Notebook killed during commit | 12 hr session limit | Lower `--epochs` or shrink ablation to 15 ep |
| `gradio` not available on Kaggle | Demo runs locally, not on Kaggle | Run `app.py` only on local machine / HF Spaces |
| `best.pt` > 100 MB blocks HF upload | LFS needed | `huggingface-cli lfs-enable-largefiles .` |

---

## 14. Time budget summary (Run 5 actuals)

| Stage | Wall time | GPU time |
|---|---|---|
| 01 stats + split + sanity train | ~10-15 min | ~5 min |
| 02 full train (Run 5, early stop ep 38) | ~1.5 hr | ~1.5 hr |
| 03 ablation (no CBAM, same schedule) | ~1 hr | ~1 hr |
| 04 eval + figures | ~10-15 min | ~5 min |
| OOD image curation + local demo | ~1 hr | 0 |
| HF Space deploy | ~30 min | 0 |
| Report + slides | separate evening | 0 |
| **GPU total** | | **~3 hr** ✓ well within 5 hr budget |
