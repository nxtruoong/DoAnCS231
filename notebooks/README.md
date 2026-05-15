# Kaggle notebook templates

Each section below is the body of one Kaggle notebook. Create a new
Kaggle notebook, attach the State Farm dataset
(`/kaggle/input/competitions/state-farm-distracted-driver-detection`), set GPU = T4
x2, Internet = On, and copy the cells in.

To make the project scripts importable on Kaggle, do **one** of:

- **Option A (recommended):** upload the repo as a Kaggle dataset
  (`/kaggle/input/driver-distraction-cbam`) and prepend `sys.path` in
  cell 1.
- **Option B:** `!git clone <your-repo-url> /kaggle/working/code` and
  prepend that to `sys.path`.

Cell 1 of every notebook starts with the same setup block. Adjust
`CODE_DIR` if you used Option B.

---

## 01 — Stats + Split

```python
# Cell 1: setup
import sys
CODE_DIR = "/kaggle/input/driver-distraction-cbam"   # or /kaggle/working/code
sys.path.insert(0, CODE_DIR)
DATA_ROOT = "/kaggle/input/competitions/state-farm-distracted-driver-detection"
WORK = "/kaggle/working"
```

```python
# Cell 2: build subject-wise splits + compute dataset stats
!cd {CODE_DIR} && python data_prep.py \
    --data-root {DATA_ROOT} \
    --out-dir {WORK}/splits \
    --batch-size 64 --num-workers 4
```

```python
# Cell 3: quick sanity-train (2 epochs) — verifies pipeline works
!cd {CODE_DIR} && python train.py \
    --data-root {DATA_ROOT} \
    --splits-dir {WORK}/splits \
    --out-dir {WORK}/sanity \
    --epochs 2 --batch-size 128 --num-workers 4 \
    --data-parallel
```

Confirm train+val acc are improving and no OOM before kicking off the
full run.

---

## 02 — Full Train

```python
# Cell 1: setup (same as 01)
import sys
CODE_DIR = "/kaggle/input/driver-distraction-cbam"
sys.path.insert(0, CODE_DIR)
DATA_ROOT = "/kaggle/input/competitions/state-farm-distracted-driver-detection"
WORK = "/kaggle/working"
```

```python
# Cell 2: full training run (~2.5-3 hr on T4x2)
!cd {CODE_DIR} && python train.py \
    --data-root {DATA_ROOT} \
    --splits-dir {WORK}/splits \
    --out-dir {WORK}/run1 \
    --epochs 40 --batch-size 128 --num-workers 4 \
    --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \
    --label-smoothing 0.1 --ema-decay 0.999 \
    --cutmix-alpha 1.0 --cutmix-p 0.5 \
    --ckpt-every 5 \
    --data-parallel
```

```python
# Cell 3: peek at history
import json
hist = json.loads(open(f"{WORK}/run1/history.json").read())
for h in hist[-5:]:
    print(h)
```

**If tier-1 abort triggered**: re-run cell 2 with `--no-cutmix
--no-grayscale --out-dir {WORK}/run1_fallback`.

---

## 03 — Ablation (no CBAM)

```python
import sys
CODE_DIR = "/kaggle/input/driver-distraction-cbam"
sys.path.insert(0, CODE_DIR)
DATA_ROOT = "/kaggle/input/competitions/state-farm-distracted-driver-detection"
WORK = "/kaggle/working"

!cd {CODE_DIR} && python train.py \
    --data-root {DATA_ROOT} \
    --splits-dir {WORK}/splits \
    --out-dir {WORK}/run_baseline \
    --epochs 25 --batch-size 128 --num-workers 4 \
    --no-cbam \
    --data-parallel
```

---

## 04 — Eval + Figures

```python
import sys
CODE_DIR = "/kaggle/input/driver-distraction-cbam"
sys.path.insert(0, CODE_DIR)
DATA_ROOT = "/kaggle/input/competitions/state-farm-distracted-driver-detection"
WORK = "/kaggle/working"

# Main model (with CBAM)
!cd {CODE_DIR} && python eval.py \
    --ckpt {WORK}/run1/best.pt \
    --data-root {DATA_ROOT} \
    --splits-dir {WORK}/splits \
    --out-dir {WORK}/run1/eval \
    --history-json {WORK}/run1/history.json

# Baseline (no CBAM) — for ablation table
!cd {CODE_DIR} && python eval.py \
    --ckpt {WORK}/run_baseline/best.pt \
    --data-root {DATA_ROOT} \
    --splits-dir {WORK}/splits \
    --out-dir {WORK}/run_baseline/eval \
    --history-json {WORK}/run_baseline/history.json
```

```python
# Build ablation table for the report
import json, pandas as pd
def metrics(p):
    m = json.loads(open(p).read())
    return {"accuracy": m["accuracy"],
            "macro_f1": m["macro avg"]["f1-score"],
            "weighted_f1": m["weighted avg"]["f1-score"]}
table = pd.DataFrame({
    "ResNet-18 + CBAM (full)": metrics(f"{WORK}/run1/eval/metrics.json"),
    "ResNet-18 baseline":      metrics(f"{WORK}/run_baseline/eval/metrics.json"),
}).T
print(table.to_string())
table.to_csv(f"{WORK}/ablation_table.csv")
```

```python
# Bundle artifacts for download
!cd {WORK} && zip -r artifacts.zip run1/eval run_baseline/eval \
    run1/history.json run_baseline/history.json \
    run1/best.pt splits/stats.json ablation_table.csv
```

Download `artifacts.zip` from the Kaggle output panel; figures go into
the report, `best.pt` + `stats.json` go into the demo Space.
