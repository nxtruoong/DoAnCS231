# Run 8 Plan — Hand-stream + MediaPipe head-pose fusion, no CutMix

Status: PLANNED. Predecessor: Run 7 (two-stream full+face, eval acc
0.751, see `log.md`).

## Motivation

Run 7 architecture (full + top-crop face) regressed by ~9pp on macro F1
vs Run 6 single-stream. Per-class eval revealed bipolar failure:

- **Starving classes:** c0 (R=0.38), c5 (R=0.52), c9 (R=0.29).
- **Dumping classes:** c3 (P=0.38, R=0.99), c7 (P=0.80, R=0.98).

Discriminative information for the failing classes lives in regions
the face stream **does not see**:

- c3 (text-left): phone in left hand, near lap / lower steering wheel.
- c0 (safe): absence of object in lap region.
- c9 (talk-passenger): head yaw to the right (cue is in *isolated*
  face landmarks, not a coarse top-crop that includes shoulders).
- c5 (radio): right hand on center console, below the wheel.

Top-50% crop captures the head but cuts away exactly the lap/console
region where the discriminator for half the failing classes lives.

## Architecture change

| Component | Run 7 | Run 8 |
|---|---|---|
| Stream A | Full image @ 384 (ResNet18+CBAM) | Same |
| Stream B | Top-50% crop @ 224 (ResNet18+CBAM) | **Hand/lap crop @ 224** (ResNet18+CBAM) — bottom 55%, center 65% width |
| Stream C | — | **Head-pose vector** (8-dim, precomputed via MediaPipe Pose) → MLP(64) |
| Fusion | concat 512+512 → Linear(1024→256) → 10 | concat 512+512+64 → Linear(1088→256) → 10 |
| CutMix | p=0.20, α=0.5 | **Disabled** |
| Loss | CE + label smoothing 0.1 | Same |
| Other hyperparams | unchanged | unchanged |

Three streams of input but only two image streams (pose is a tiny
scalar vector). Compute ≈ Run 7.

## Why hand stream (bottom crop)

State Farm dashcam fixed mount → driver's lap/lower-wheel region is
always in the lower-center of the frame. Static crop captures the
discriminative region for phone-in-hand classes (c3, c4 cross-check)
and the "empty" region for c0/c9.

Crop box: `top=0.45, bottom=1.0, left=0.20, right=0.85`. Cuts the
right-side window glare (passenger seat) and the dashboard above the
wheel. Sanity-plot 20 samples before training.

## Why MediaPipe head-pose

c9 (talk-passenger) is defined by head yaw to the right. Run 7's face
stream had this information *somewhere* in its 224x224 crop but never
isolated it — model learned generic posture instead of gaze. Injecting
8 pose scalars provides an **explicit, low-dimensional** gaze signal
the fusion head can directly weight, bypassing the need for the CNN
to discover head pose from pixels.

**MediaPipe Pose (not Face Mesh).** Face Mesh fails on extreme yaw
(profile view = c9) and on occluded face (hand near face = c6, c8) —
exactly the cases we want to help. Pose ships nose/ear/eye landmarks
that survive at extreme yaw because it tracks the whole skeleton.

Pose features (8-dim):

```
p0 = (r_ear.x - l_ear.x)               # signed yaw proxy (look right < 0)
p1 = nose.y - mean(eye.y)              # pitch proxy
p2 = r_eye.y - l_eye.y                 # roll proxy
p3 = r_shoulder.x - l_shoulder.x       # torso twist (c7 reach-behind, c9 turn)
p4 = r_shoulder.y - l_shoulder.y       # shoulder slope
p5 = l_ear.visibility                  # left ear visible? (look-right hides it)
p6 = r_ear.visibility                  # right ear visible? (look-left hides it)
p7 = 1.0 if detection succeeded else 0 # gating bit; zero-imputed otherwise
```

Stored as parquet keyed by image filename. Loaded into a Python dict
at training start, O(1) lookup per sample.

## Why drop CutMix

c0 = absence of distractor by definition. CutMix pastes a patch from
another image into c0, polluting the "no-distractor" signal. With
shared permutation across two image streams (Run 7's setup), the
pasted patch *also* lands in the hand-crop stream → hand stream
learns "c0 sometimes has phone-shaped patches in lap region" → c0
boundary blurred → predictions leak into the dense neighbouring class
(c3). Same logic for c9.

Run 7's CutMix p=0.20 was modest, but the two-stream sharing
amplified its effect. Cleanest test: ablate CutMix entirely. If c0/c9
recall recovers and other classes hold, the diagnosis is confirmed.

If macro F1 regresses on c1/c2/c4 vs Run 7's eval, can revisit with
class-conditional CutMix (skip batches dominated by c0/c9) — defer to
Run 9.

## Implementation

### Files added

- `extract_pose.py` — runs MediaPipe Pose once over all training
  images, writes `splits/pose.parquet` (filename + 8 pose scalars).

### Files extended

- `augment_twostream.py` — adds `RegionCrop` (general bottom/top/side
  crop), `build_hand_train_transform`, `build_hand_eval_transform`,
  `ThreeStreamDataset` (yields full, hand, pose_vec, label).
- `model_twostream.py` — adds `ThreeStreamCBAM` (two CNN streams + pose
  MLP fusion).
- `train_twostream.py` — adds `--three-stream` flag. When set: loads
  pose parquet, builds `ThreeStreamDataset` and `ThreeStreamCBAM`,
  forward takes (full, hand, pose). Pose is **not** mixed by CutMix
  even if CutMix is enabled (pose is global per-image; mixing yaw
  values is meaningless).
- `eval_twostream.py` — load + eval path for three-stream ckpts.

### Files unchanged

`model.py`, `augment.py`, `train.py`, `eval.py`. Existing Run 1-6
single-stream + Run 7 two-stream paths remain runnable.

## Expected gains

Targeted per-class deltas vs Run 7 eval:

| class | Run 7 F1 | Run 8 target | mechanism |
|---|---:|---:|---|
| c0 safe | 0.51 | 0.75+ | drop CutMix (clean "no distractor" signal); hand stream sees empty lap explicitly |
| c3 text-left | 0.55 | 0.85+ | hand stream isolates phone-in-lap; P recovers as model stops dumping |
| c5 radio | 0.69 | 0.85+ | hand stream centers on console area; right-arm distinct from c7 |
| c7 reach-behind | 0.88 | 0.92+ | pose-vec shoulder twist (p3, p4) separates from c5 |
| c8 hair/makeup | 0.65 | 0.72+ | head-pose pitch (p1) separates from c2 |
| c9 talk-passenger | 0.44 | 0.80+ | explicit yaw signal (p0) + ear visibility flags |
| c1, c2, c4, c6 | 0.93-0.96 | hold ≥0.93 | no architectural reason to regress |
| **macro F1** | 0.748 | **0.85-0.88** | matches Run 6 + addresses Run 7 regressions |

Conservative outcome: matches Run 6 (0.873 macro F1). Optimistic: passes
Run 6 by 1-3pp.

## Risks

- **MediaPipe detection rate < 70%** on dashcam imagery (cabin lighting,
  partial occlusion). Mitigation: pose-vec includes `p7` gating bit;
  zero-impute lets model learn "pose missing = use CNN only". If rate
  is poor, fall back to YOLOv8n-face for head bbox (same fusion path,
  different feature source).
- **Hand crop misses object on tall drivers.** Mitigation: crop box
  `top=0.45` keeps a generous margin. Sanity-check 20 random samples
  before training, adjust if needed.
- **Resume from Run 7 checkpoint impossible.** Different architecture
  (ThreeStreamCBAM vs TwoStreamCBAM) → state dict shape mismatch. Run 8
  trains from scratch.

## Stop conditions

- Best EMA val acc still < 0.80 after epoch 30 → architecture not
  helping. Inspect per-class metrics on `ckpt_e30.pt`. If only c0/c9
  recovered but c3 still low → hand crop too high; widen.
- Pose detection rate < 60% in `extract_pose.py` summary → halt before
  training. Switch to YOLO-face fallback.
- Training acc < 0.85 by epoch 15 → optimization issue (pose-MLP init,
  fusion-head LR). Compare against Run 7 ep 15 train acc (0.90).

## Runtime estimate

- **Pose precompute:** one-time, ~22k images × ~50 ms CPU = ~18 min on
  Kaggle CPU. Cached to parquet.
- **Training:** 60 epochs, batch 48, `num_workers=2`, two CNN streams
  (same compute as Run 7) + tiny pose MLP. No CutMix → ~3% faster
  per-step than Run 7 (skips beta sample + index copy).
  - 2x T4 DataParallel: ~3.2 min/epoch × 60 = **~3.2 hours**.
  - Single T4 (DP-disabled, safer host RAM): ~5.5 min/epoch × 60 =
    **~5.5 hours**.
- **Eval:** ~5 min on val split.
- **Total wall (DP path):** pose 18 min + train ~3.2 h + eval 5 min ≈
  **~3.6 hours**. Comfortably under Kaggle's 9-hour session cap.
