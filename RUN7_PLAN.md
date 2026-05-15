# Run 7 — Two-stream CBAM with static top-crop face stream

## Motivation

Run 6 hit a clean macro-F1 ceiling of 0.873 with three residual hard
classes — c0 (safe), c8 (hair/makeup), c9 (talk passenger). All three
are "passive" classes: no distinct object-in-hand cue, model has to
read the driver's pose and gaze. Run 6's full-frame attention map at
resolution 384 produces a 12×12 SAM grid — too coarse to localize the
head + eyes region cleanly when most of the frame is cabin/torso.

A two-stream architecture gives one stream a zoomed-in view of just
the head region. The full stream keeps Run 6's responsibilities
(hand + steering + object). The face stream specialises in gaze + head
pose + hand-to-head proximity — exactly the cues that disambiguate
c0/c8/c9.

## Why no face detector

Adding MTCNN or RetinaFace would add:
- A second model + its dependencies (~20-50 MB extra weights).
- Per-batch detection latency (slower training + demo).
- A failure mode where detection misses → no face crop → fallback logic.

State Farm dashcam is mounted in a fixed position. Drivers' heads sit
in the upper portion of every frame for 95%+ of samples. A static
`top 45%` crop captures the head region without any detection at all.

## Architecture

Same `ResNet18CBAM` backbone as Run 6, instantiated **twice with
separate weights** (no parameter sharing). Each backbone outputs its
512-d pre-fc feature vector. Features concatenate → 1024-d → MLP head.

```
full image  (3, 384, 384)  ─► ResNet18CBAM ─► features ─► (512)
                                                              │
                                                              ▼ concat
                                                            (1024) ─► Linear(256) ─► ReLU ─► Dropout(0.3) ─► Linear(10)
                                                              ▲
top crop    (3, 224, 224)  ─► ResNet18CBAM ─► features ─► (512)
```

**Parameter cost vs Run 6:**
- Run 6: ~11.2 M (single ResNet-18 + CBAM).
- Run 7: ~22.5 M (two copies) + 0.26 M classifier = **~22.8 M**.
- Roughly 2× the params. FLOPs ~2× per forward.

**Why separate weights, not shared.** Shared weights would force both
streams to learn the same representation of two very different views.
The face stream operates on a low-resolution head crop; the full stream
on a high-resolution scene. Different statistics, different optimal
filters. Separate weights = one extra ResNet's worth of capacity (cheap
on Kaggle T4×2).

## TopCrop implementation

```python
class TopCrop:
    def __init__(self, frac=0.45):
        self.frac = frac
    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, w, int(h * self.frac)))
```

State Farm frames are 640×480. `frac=0.45` → top crop is 640×216. That
captures the driver's head + upper torso + the area where a phone at
the ear would appear. Just enough context for the face stream to also
catch hand-near-head poses without leaking the steering wheel.

**Before training:** inspect 20 random samples from `splits/train.csv`
overlaid with the crop box. If more than ~10% of heads fall outside
the box, raise `frac` to 0.50. Cells in `notebooks/inspect_topcrop.py`
(see RUN7_PLAN.md §"Verification").

## Augmentation policy

Both streams use the Run 6 stack (`RandomResizedCrop` + TrivialAugment
+ RandomErasing) with **independent random draws**. No HFlip in either
stream (project-wide rule, ADR 0002), so the "share flip decision
between streams" caveat from the original Run 7 sketch is automatically
satisfied — there is no flip to share.

Face stream uses:
- Smaller resolution (224 not 384). Crop is already smaller content;
  upsampling to 384 would just blur.
- Lighter RandomErasing (`p=0.15, scale=(0.02, 0.06)`). Erasing a chunk
  of a small face crop removes the only signal the stream has.

## CutMix

CutMix must be applied to **both streams with the same permutation +
box geometry** (scaled per-stream resolution). If streams receive
independent CutMix permutations, the model is asked to fuse two
different label-mix targets — impossible. See
`augment_twostream.cutmix_twostream`.

CutMix probability returned to **`p=0.20`** (was Run 6's 0.15, Run 5's
0.30). Middle ground: Run 6 showed `p=0.15` was too low — passive
classes collapsed. Restore some cross-mixing without re-introducing
the Run 5 c2↔c8 confusion.

## Training config

| Knob | Value | Why |
|---|---|---|
| epochs | 50 | Same schedule as Run 6 |
| batch size | 64 (down from 128) | 2× model + larger features → ~2× VRAM |
| LR | 0.03 | Same |
| warmup | 2 ep | Same |
| optimizer | SGD nesterov, mom 0.9, wd 5e-4 | Same |
| label smoothing | 0.1 | Same |
| EMA decay | 0.99 | Same |
| img-size full | 384 | Same as Run 6 |
| img-size face | 224 | Smaller content → smaller resolution |
| top_frac | 0.45 | Captures head + upper torso |
| CutMix p / alpha | 0.20 / 0.5 | Middle ground between Run 5 / Run 6 |
| early stop | patience 8, min-delta 0.005 | Same |

**Estimated wall time:** Run 6 was 137 min at batch 128 for ~50 ep.
Run 7 batch 64 + 2× params → expect ~3-4 hr on T4×2.

## Expected results

| Class | Run 6 F1 | Run 7 target | Reason |
|---|---:|---:|---|
| c0 safe | 0.66 | 0.75-0.80 | Face stream sees calm gaze, no hand-to-head |
| c1 text right | 0.95 | 0.95 | Already solved; protect |
| c2 phone right | 0.95 | 0.95 | Already solved; protect |
| c3 text left | 0.92 | 0.92 | Already solved; protect |
| c4 phone left | 0.97 | 0.97 | Already solved; protect |
| c5 radio | 0.94 | 0.94 | Already solved; protect |
| c6 drinking | 0.94 | 0.94 | Already solved; protect |
| c7 reach behind | 0.98 | 0.98 | Already solved; protect |
| c8 hair/makeup | 0.75 | 0.80-0.83 | Face stream resolves "hand at face" |
| c9 talk passenger | 0.68 | 0.75-0.80 | Face stream sees side-glance |
| **macro F1** | 0.873 | **0.89-0.91** | +2 to +4 pp |
| **accuracy** | 0.875 | **0.89-0.91** | +2 to +4 pp |

## Risks + caveats

1. **Fixed crop misses leaning-forward drivers.** Some frames have the
   head partially below the 45% line. Verify with 20-sample inspection
   before training. If >10% miss rate, raise `frac` to 0.50 (top
   480×240) or 0.55.
2. **Doubled compute.** 2× model means slower epochs + larger VRAM.
   Batch 128 unlikely to fit; drop to 64 (per config above).
3. **Demo + Gradio app must be updated.** `app.py` currently expects
   one input. New `app.py` (or new entrypoint) must crop on upload and
   feed both streams.
4. **Two-stream may *not* help if the bottleneck is data quantity.**
   Subject-wise val on 22k images has a hard ceiling; we may already
   be near it. Fallback plan: keep Run 6 as headline if Run 7 fails to
   improve.
5. **Face-stream regularization risk.** With only 22k training images
   and a smaller-content stream, the face stream may overfit faster
   than the full stream. Watch the EMA gap between the two streams' raw
   accuracy if logged separately. If face stream overfits hard, lower
   its `RandomErasing` further or increase the hidden classifier's
   dropout to 0.5.

## Verification before launch

Drop into a Kaggle cell:

```python
import random
from PIL import Image, ImageDraw
import pandas as pd

df = pd.read_csv("/kaggle/working/splits/train.csv")
samples = df.sample(20, random_state=42)

ROOT = "/kaggle/input/competitions/state-farm-distracted-driver-detection/imgs/train"
for _, row in samples.iterrows():
    p = f"{ROOT}/{row['classname']}/{row['img']}"
    img = Image.open(p).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, int(h * 0.45)], outline="red", width=4)
    display(img.resize((320, 240)))
    print(f"{row['classname']} / {row['img']}")
```

Eyeball: is the head fully inside the red box in 18+ of 20? If yes,
`frac=0.45` is fine. If 10+ heads partially outside, raise `frac`.

## Files in this plan

- `model_twostream.py` — `TwoStreamCBAM` wrapper around two
  `ResNet18CBAM` instances + MLP fusion head.
- `augment_twostream.py` — `TopCrop`, `TwoStreamDataset`,
  paired `cutmix_twostream`.
- `model.py` — added `features()` method to `ResNet18CBAM`
  (returns pre-fc 512-d vector). Single non-invasive change.
- This `RUN7_PLAN.md`.

## What's NOT in this PR

- `train_twostream.py` (trainer with two-stream forward + paired
  CutMix). Build it by copying `train.py` and changing:
  - dataset → `TwoStreamDataset`
  - model build → `build_twostream`
  - training step → `model(full, face)` instead of `model(images)`;
    use `cutmix_twostream` instead of `cutmix_batch`.
- `eval_twostream.py` (eval with paired forward). Same pattern.
- Updated `app.py` for two-stream demo.

These follow mechanically from the existing code; intentional to keep
the V2 repo focused on the architectural change.

## Stop conditions

- Run 7 macro F1 < 0.86 (Run 6 was 0.873) → architecture not helping;
  revert to Run 6 and ship.
- c0/c8/c9 individual F1 doesn't improve by +5pp avg → face stream
  not isolating the right cue; investigate top-crop coverage and
  retry with `frac=0.50`.
- Wall time > 5 hr on T4×2 → batch size too high; drop to 48.
