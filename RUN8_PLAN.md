# Run 8 Plan — Pose-fusion (single CNN + 36-d MediaPipe pose), no CutMix

Status: PLANNED. Predecessor: Run 7 (two-stream full+face, eval acc
0.751, see `log.md`).

## Motivation

Run 7 (two-stream: full + top-crop face) regressed by ~9pp on macro F1
vs Run 6. Per-class eval revealed bipolar failure: starving classes c0
(R=0.38), c5 (R=0.52), c9 (R=0.29); dumping classes c3 (P=0.38) and c7
(P=0.80, R=0.98). Discriminator for these classes is **body posture**
— where the hands are, where the head is looking — not pixels in any
single crop region.

Run 7's mistake was using a second CNN backbone to *learn* posture
implicitly from pixels. ResNet18+CBAM on 22k images = limited capacity
to discover "this wrist coordinate equals lap-region" from scratch.
Worse, two CNNs doubled overfit pressure on driver-specific cues
(face, shirt, seat) that don't transfer across subjects.

**Pose features are the cheap structural insight Run 7 missed.**
MediaPipe Pose returns 33 body landmarks per image — wrist, elbow,
finger, shoulder, hip positions are all directly available. Feeding
them as engineered features lets the model use posture without
spending CNN capacity to rediscover it.

## Architecture change

| Component | Run 6 (baseline) | Run 7 | Run 8 |
|---|---|---|---|
| Stream A | ResNet18+CBAM @ 384 | Same | **Same** |
| Stream B | — | ResNet18+CBAM top-crop @ 224 | — |
| Pose features | — | — | **36-d MediaPipe vector → MLP(128)** |
| Fusion | Linear(512→10) | concat 512+512 → Linear(1024→256) → 10 | **concat 512+128 → Linear(640→256) → 10** |
| Total params | ~11.2 M | ~22.7 M | **~11.4 M** (≈ Run 6 + 0.2 M pose head) |
| CutMix | p=0.30, α=0.5 | p=0.20, α=0.5 | **disabled** |
| Loss | CE + label smoothing 0.1 | Same | Same |

Single CNN, same capacity as Run 6, plus a small pose MLP. **No hand
crop, no second CNN.**

## Why dropping the hand crop

Original Run 8 plan included a dedicated bottom-center crop CNN as
"hand stream". After analysis, that stream is **redundant** when the
pose vector includes wrist/elbow/finger landmarks:

| Original "hand stream" role | Covered by pose vector? |
|---|---|
| where is left wrist | yes — `p8, p9, p12` |
| where is right wrist | yes — `p10, p11, p13` |
| is hand on lap / wheel / face | yes — `p28, p29` (wrist–hip Δy) |
| left vs right arm reach | yes — `p30, p31` (wrist–shoulder Δx) |
| finger gesture (typing pose) | yes — `p20, p21, p22, p23` (index–wrist Δy, thumb–pinky spread) |
| **is there a phone/cup in hand** | no — only full-stream pixels see this |
| **fine object texture** | no — only full-stream pixels |

The hand stream's only unique value was higher per-pixel resolution on
the lap region for small-object discrimination. With full stream at
384×384, lap region already gets ~120×80 pixels — adequate for
phone-in-hand detection. Adding a second ResNet18+CBAM (11M params)
for the same job is a bad trade.

## The 36-d pose feature vector

Precomputed by `extract_pose.py` once, cached to
`splits/pose.parquet`. Per-image features:

| Indices | Group | What it captures |
|---|---|---|
| p0–p7 | Head + torso | yaw, pitch, roll, shoulder twist, ear visibility, detection gate |
| p8–p15 | Wrists | x/y/visibility of both wrists, vertical asymmetry, horizontal spread |
| p16–p19 | Elbows | x/y of both elbows (arm reach proxies) |
| p20–p23 | Hand orientation | index–wrist Δy (texting), thumb–pinky spread (grip vs flat) |
| p24–p27 | Hip anchors | both hips, define "lap" reference frame |
| p28–p31 | Derived | wrist–hip Δy (lap proximity), wrist–shoulder Δx (reach direction) |
| p32–p35 | Visibility gates | elbow + hip visibility (per-side) |

All features in normalised [0, 1] image coordinates from MediaPipe.
Missing detections zero-imputed (gating bit p7=0 → model learns "pose
unavailable, fall back to CNN").

## Why drop CutMix

c0 = absence of distractor by definition. CutMix pastes a patch from
another class into c0, polluting "no-distractor" signal → c0 boundary
blurs → predictions leak into nearest dense class (c3 in Run 7). Same
logic for c9. Removing CutMix cleans up the negative class.

Empirical justification: Run 7 with CutMix p=0.2 produced the c0/c9
collapse. Run 6 with CutMix p=0.15 had milder version of same problem
(c0 F1 dropped from Run 5's higher value). Trend: lower CutMix helps
passive classes. Zero CutMix = end of that gradient.

Risk: c1/c2 may lose some regularisation. Mitigation: label-smoothing
0.1 + RandomErasing already in stack, both provide regularisation
without object-mixing.

## Implementation

### Files added

- `extract_pose.py` — MediaPipe Pose extractor, 36-d feature vector
  per image, output `splits/pose.parquet`.

### Files extended

- `augment_twostream.py` — `POSE_DIM=36`, `PoseFusionDataset` (yields
  full + pose + label), `load_pose_lookup`. `RegionCrop` utility kept
  (general-purpose). Hand-crop transforms removed.
- `model_twostream.py` — `PoseFusionCBAM`: single ResNet18+CBAM + 36→128
  pose MLP → 512+128 fused → MLP(256) → 10.
- `train_twostream.py` — `--pose-fusion` flag. Loads pose parquet,
  builds `PoseFusionDataset` + `PoseFusionCBAM`, forward takes
  (full, pose). Pose is global per-image; CutMix disabled on this
  path even if not explicitly `--no-cutmix`.
- `eval_twostream.py` — auto-dispatches: `pose_fusion` flag in ckpt
  metadata → builds `PoseFusionCBAM`, requires `--pose-parquet`.

### Files unchanged

`model.py`, `augment.py`, `train.py`, `eval.py`. Run 1-6 single-stream
and Run 7 two-stream paths remain runnable.

## Expected gains

Targeted per-class deltas vs Run 7 eval (and vs Run 6 where relevant):

| class | Run 6 F1 | Run 7 F1 | Run 8 target | mechanism |
|---|---:|---:|---:|---|
| c0 safe | 0.66 | 0.51 | **≥ 0.80** | no CutMix → clean c0 boundary; pose wrist-on-wheel signal |
| c3 text-left | 0.92 | 0.55 | **≥ 0.88** | pose wrist-on-lap + finger-up signal; no CutMix |
| c5 radio | 0.94 | 0.69 | **≥ 0.90** | pose right-arm-forward distinguishes from c7 (right-arm-back) |
| c7 reach-behind | 0.98 | 0.88 | **≥ 0.94** | pose r-wrist–shoulder Δx large positive |
| c8 hair/makeup | 0.75 | 0.65 | **≥ 0.75** | pose wrist-near-head + full-stream object check |
| c9 talk-passenger | 0.68 | 0.44 | **≥ 0.82** | pose yaw + shoulder twist directly encode head turn |
| c1/c2/c4/c6 | 0.93–0.97 | 0.93–0.96 | hold ≥0.93 | no architectural reason to regress |
| **macro F1** | **0.873** | 0.748 | **≥ 0.88** | matches or exceeds Run 6 |

Conservative outcome: match Run 6 (0.873 macro F1). Optimistic:
0.88-0.90 driven by pose-disambiguated c0/c9.

## Risks

- **MediaPipe detection rate < 70%** on dashcam imagery. Mitigation:
  per-landmark visibility flags in the pose vector + global gate
  (p7); model learns "pose missing → trust CNN only".
- **Pose features over-rated by fusion MLP.** Pose is 36-d, CNN is
  512-d. Concat directly might let model lean too hard on pose,
  losing pixel discrimination for object-identity classes (c2, c6).
  Mitigation: pose MLP has Dropout(0.3) before projection to 128;
  fusion has further Dropout(0.3).
- **Cannot resume from Run 7 checkpoint.** Different architecture.
  Run 8 trains from scratch (~60 epochs).

## Stop conditions

- Best EMA val acc < 0.75 after epoch 20 → pose features not
  helping. Inspect `ckpt_e10.pt` per-class metrics. If still bipolar
  (c0/c9 starving) → pose features uninformative for this dataset.
- Pose detection rate < 60% in `extract_pose.py` summary → halt
  before training. Switch to YOLOv8n-face for head-bbox features.
- Training acc < 0.85 by epoch 15 → optimisation issue. Compare
  against Run 6 ep 15 (~0.84).

## Runtime estimate

- **Pose precompute:** one-time, ~18 min CPU on Kaggle, 22k imgs ×
  ~50 ms.
- **Training:** 60 epochs, batch 48, `num_workers=2`, single CNN
  (≈ Run 6 compute) + tiny pose MLP.
  - 2x T4 DataParallel: ~1.8 min/epoch × 60 = **~1.8 hr**.
  - Single T4 (DP-disabled, safer host RAM): ~3.0 min/epoch × 60 =
    **~3.0 hr**.
- **Eval:** ~5 min on val split.
- **Total wall (DP path):** pose 18 min + train ~1.8 h + eval 5 min ≈
  **~2.2 hours**. Well under Kaggle's 9-hour session cap.

Total wall is ~40% less than the original three-stream Run 8 plan
(~3.6 hr) because we dropped one ResNet18+CBAM backbone.
