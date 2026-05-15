# CONTEXT — Driver Distraction Classification Project

End-term Computer Vision project, UIT. Solve State Farm Distracted Driver
Detection (10-class) with ResNet-18 + CBAM, trained from scratch (no
ImageNet pretrain — teacher constraint). Demo via simple website on
out-of-distribution images. Training budget: < 5 hr on Kaggle T4×2.

---

## Glossary

### State Farm dataset
Kaggle competition "State Farm Distracted Driver Detection". ~22k labeled
training images, 640×480, fixed cabin-camera angle (right-side view of
driver). 10 classes c0–c9.

### Classes (10)
- `c0` safe driving
- `c1` texting — right
- `c2` talking on phone — right
- `c3` texting — left
- `c4` talking on phone — left
- `c5` operating radio
- `c6` drinking
- `c7` reaching behind
- `c8` hair and makeup
- `c9` talking to passenger

Left/right asymmetry means **horizontal flip changes class label** — see
[[adr-no-hflip]].

### Subject
A unique driver in the dataset, identified by `subject` column in
`driver_imgs_list.csv` (e.g. `p022`). 26 subjects total in train set.
Multiple images per subject across all classes.

### Subject-wise split
Train/val partition where each subject appears in *only one* of train or
val. Prevents identity leakage. Held-out subjects for this project:
`p022, p035, p047, p056, p075` (5/26 ≈ 19%). See [[adr-subject-wise-split]].

### Random split
Train/val partition done without regard for subject. Same driver appears
in both → model memorizes driver identity instead of action →
inflated val acc (typically ~99%). **Not used here** except as
reference baseline if val acc collapses (tier-3 fallback).

### OOD (out-of-distribution) images
Demo images not from State Farm dataset. Sourced via Google Images
hand-filter: search "driver texting side view", "driver phone dashcam",
"distracted driving side profile". Keep only frames matching dataset
framing (side view, full torso, steering wheel visible). ~10 images.

### Dataset stats
Per-channel RGB mean + std computed over training images (after
subject-wise split removes held-out). Used for normalization instead of
ImageNet stats — consistent with "no ImageNet pretrain" rule.

### CBAM (Convolutional Block Attention Module)
Attention block from Woo et al. 2018. Two sub-modules: Channel Attention
(CAM, MLP over channel pool with ratio=16) + Spatial Attention (SAM,
7×7 conv over spatial pool). Inserted after each of ResNet-18's 4 stages
(layer1–4). SAM map from layer4 → upsampled 224×224 → demo heatmap
overlay.

### Heavy aug pipeline
Augmentation set: RandomResizedCrop(0.7–1.0), color jitter (brightness
0.5, hue 0.2), GaussianBlur, RandomGrayscale(p=0.2), RandomErasing(p=0.3),
CutMix(p=0.5), Normalize(dataset stats). **No HorizontalFlip.**

### EMA (Exponential Moving Average)
Maintain shadow copy of model weights, updated `θ_ema ← 0.999·θ_ema +
0.001·θ` each step. Used for evaluation + demo. Typically +1–2% val acc
free.

---

## Decisions log

| Topic | Decision |
|-------|----------|
| Creative angle | CBAM attention + heavy aug + subject-wise split |
| Architecture | ResNet-18 + CBAM (CAM+SAM) after each stage |
| Init | Kaiming, pure scratch — no pretrain of any kind |
| Split | Subject-wise, 5 held-out drivers: `p022, p035, p047, p056, p075` |
| Augmentation | Heavy pipeline above, no HFlip |
| Image size | 224×224 |
| Batch size | 128 (single T4 + DataParallel on 2nd T4) |
| Optimizer | SGD momentum=0.9, weight decay 5e-4 |
| LR schedule | Cosine 0.1 → 0 over 40 epochs |
| Loss | CrossEntropy with label smoothing 0.1 |
| EMA decay | 0.999 |
| Normalization | Dataset-computed RGB stats |
| Demo stack | Gradio + HuggingFace Spaces |
| Demo features | Top-1 + probability bar + CBAM heatmap + sample buttons + "unsure" threshold (max prob < 0.4) |
| Metrics | sklearn classification_report (precision/recall/F1/support per class + accuracy + macro avg + weighted avg) + confusion matrix + per-driver breakdown + ablation table + training curves + attention viz + failure cases |
| Fallback tier 1 | val acc < 50% at epoch 20 → kill, drop CutMix + grayscale, restart |
| Fallback tier 2 | final val < 55% → retrain without CBAM as backup |
| Fallback tier 3 | demo broken → train random-split baseline for context |
| Reproducibility | seed=42 across torch/numpy/random; `cudnn.deterministic=True`; note Kaggle non-determinism caveat in report |
| Deliverables | Code repo + report PDF + live demo + slides |

---

## Execution plan (Kaggle notebooks)

1. **01_stats_split** — compute dataset RGB stats, build subject-wise
   split CSVs, verify class distribution per split, sanity-train 2 ep.
2. **02_train** — full 40-epoch run, ResNet-18 + CBAM, heavy aug, EMA.
   Save checkpoint every 5 ep + final. Target: ~2.5–3 hr.
3. **03_ablation** — baseline ResNet-18 *without* CBAM, lighter aug,
   25 ep. Target: ~1.5 hr.
4. **04_eval_figs** — generate classification_report, confusion matrix,
   per-driver table, training curves, attention overlays, failure cases.
   Export ONNX for demo.

Total budget: ~5 hr including overhead + checkpoints.

---

## Repo layout

```
driver-distraction-cbam/
├── README.md
├── data_prep.py
├── model.py            # ResNet-18 + CBAM
├── train.py
├── eval.py
├── augment.py
├── app.py              # Gradio demo
├── notebooks/01..04
├── checkpoints/        # gitignored
├── figures/
├── docs/adr/
└── requirements.txt
```

---

## Open items

- Confirm exact 5 held-out driver IDs by inspecting class distribution
  per subject (avoid holding out a driver missing some class).
- Verify batch 128 + CBAM fits on single T4 16GB; fallback batch 96 +
  grad accum if OOM.
- Search ablation slot: if budget allows, also compare CBAM-after-layer4-only
  vs CBAM-after-each-stage.
