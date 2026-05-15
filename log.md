# Training Log

Reading order: each run has **Config** (what was tried), **Result**
(numbers), **What went wrong / right** (plain-language story), **Root
cause** (the technical reason), **Fix** (what changed in code).

If you're new to deep-learning training: skim the "What went wrong"
boxes first — they explain symptoms in everyday terms before the math.

---

## Run 1 — Full config, dead training

**Config:** `use_cbam=True, use_cutmix=True, no_grayscale=False, lr=0.1 (no warmup), 40 epochs, batch=128, 2x T4`

**Result (5 epochs shown):**

| ep | train loss | train acc | val loss | val acc | ema val acc |
|----|-----------|-----------|----------|---------|-------------|
| 1  | 2.3017 | 0.1090 | 2.3026 | 0.1028 | 0.0935 |
| 2  | 2.3011 | 0.1086 | 2.3023 | 0.1043 | 0.0942 |
| 3  | 2.3014 | 0.1100 | 2.3028 | 0.1019 | 0.1019 |
| 4  | 2.3012 | 0.1094 | 2.3024 | 0.1019 | 0.1019 |
| 5  | 2.3008 | 0.1068 | 2.3027 | 0.1494 | 0.1019 |

### What went wrong

Model refused to learn anything. Loss stuck at **2.3026**, accuracy stuck
at **10%** = chance level (10 classes → random guess is 1/10).

The magic number `2.3026` is the dead giveaway: it equals `ln(10)`. That
is the loss value cross-entropy gives when the model outputs the same
probability for every class (1/10 for each). Translation: model said "I
don't know, all classes equally likely" for every single image, every
single epoch. Network had collapsed before learning began.

### Root cause

Two bugs stacked on top of each other:

**Bug A — wrong init on final layer (`model.py:142`).**
Final layer maps `512 features → 10 classes`. It was initialised with
`kaiming_normal_(mode="fan_out", nonlinearity="relu")`. That formula
sets weight std = `sqrt(2 / fan_out) = sqrt(2/10) ≈ 0.45`. Way too
large for a layer that produces logits.

Consequence chain:
1. Initial logits have variance ~100 (huge numbers like `+15, -12, +8`).
2. Softmax of huge numbers → one class gets probability ≈ 1.0, others ≈ 0.
3. Cross-entropy gradient against that near-one-hot output is enormous.
4. Huge gradient flows back through ReLUs, pushing many activations
   below zero permanently. Once a ReLU outputs 0 forever, its gradient
   is 0 forever — the neuron is **dead**.
5. Same bad init also saturated the sigmoid in CBAM's channel attention
   block → CBAM started outputting near-constant values, killing its
   signal too.

Net effect: most of the network was dead before epoch 1 finished.

**Bug B — no learning rate warmup.**
Plain SGD with `lr=0.1` from a random init on a small dataset (~22k
images). First few weight updates are based on garbage gradients from
the still-random network. With `lr=0.1` those garbage updates are big
enough to push weights far from anything useful, locking in the dead
state from Bug A.

### Fix (commit `ed889e3`)

- `model.py`: replace final-layer init with `normal_(0, 0.01)`. Tiny
  weights → tiny initial logits → softmax stays near uniform → gradient
  is small and well-behaved at the start.
- `train.py`: add `--warmup-epochs` (default 5). For first N epochs LR
  ramps linearly from `0.001 × lr` up to `lr`, then cosine decay takes
  over. Slow start lets the network find a sane region before big
  updates arrive.

---

## Run 2 — Smoke test, minimal aug, no CBAM, no CutMix

**Config:** `--no-cutmix --no-cbam --minimal-aug --label-smoothing 0.0 --epochs 3 --lr 0.03 --warmup-epochs 1`

Strip everything down: no CBAM, no CutMix, no label smoothing, no fancy
aug. Just `RandomResizedCrop + Normalize`. Goal — answer "does the
basic pipeline learn at all after the Run 1 fixes?"

**Result:**

| ep | train loss | train acc | val loss | val acc | ema val acc |
|----|-----------|-----------|----------|---------|-------------|
| 1  | 2.0103 | 0.2436 | 2.5098 | 0.2204 | 0.1039 |
| 2  | 0.3513 | 0.8857 | 2.6580 | 0.4777 | 0.1008 |
| 3  | 0.0641 | 0.9825 | 1.9219 | 0.6185 | 0.1404 |

**Best val acc:** 0.6185. Total time: 4.4 min.

### What went right

Train loss collapsed from 2.0 to 0.06 in 3 epochs and train accuracy
hit **98%**. Pipeline + data + Run 1 fixes are correct. The killer in
Run 1 really was the init bug, not something deeper.

### What looked weird (but wasn't bugs)

**1. Massive train/val gap (98% train vs 62% val).** Model nearly
memorized training set in 3 epochs. Means it's overfitting fast.
Expected — with no augmentation, the model can latch onto driver-specific
features (face, shirt colour, seat). Subject-wise val uses **different
drivers**, so memorization doesn't transfer → val acc much lower.
Diagnosis: need augmentation back, but lighter than Run 1's heavy stack.

**2. EMA val acc stuck near 0.10 (random).** This *looked* like a bug
but wasn't.

EMA (Exponential Moving Average) keeps a "shadow" copy of the weights
that is a slow average of every training step's weights. Update rule
each step: `shadow = 0.999 × shadow + 0.001 × current`.

After only 3 epochs of training there were `3 × 139 = 417` steps. After
417 steps, the shadow still has `0.999^417 ≈ 66%` of the *original
random* weights mixed in. So the shadow is mostly random → predicts
randomly → 10% accuracy.

Rule of thumb: EMA with decay `d` needs roughly `1/(1−d)` steps to
"forget" the init. Decay 0.999 → 1000 steps before shadow is mostly
trained weights. For 3-epoch smoke tests, drop decay to 0.99 (≈100-step
window) so EMA reaches useful values.

**3. Val loss bumpy (2.51 → 2.66 → 1.92).** Small subject-wise val set
(~4.5k images, 5 drivers) → single hard driver can swing the number
several percentage points. Watch trend over many epochs, don't trust
single-epoch dips.

### Decision: reintroduce augmentation incrementally

Original Run 1 stack used everything at once — too aggressive for a
from-scratch model. Plan a stepwise reintroduction so any future
regression points to one component:

| Step | Add | Expect |
|------|-----|--------|
| 1 | Light ColorJitter `b=0.2, c=0.2, s=0.2, h=0.05` | val ↑, train ↓ |
| 2 | + RandomErasing `p=0.25, scale=(0.02, 0.10)` | val ↑ |
| 3 | + RandomGrayscale `p=0.1` | val ≈ |
| 4 | + GaussianBlur `sigma=(0.1, 0.8)` | val ≈ |
| 5 | + CBAM | val ↑ slightly |
| 6 | + CutMix `p=0.3, alpha=0.5` | val ↑ |
| 7 | + label_smoothing `0.1` | val ≈ |

Original Run 1 aug stack — too aggressive from scratch:
- ColorJitter `b=0.5, c=0.4, s=0.4, h=0.2` (hue ±72° is brutal — turns
  skin green/purple).
- RandomGrayscale `p=0.2`
- GaussianBlur `sigma=(0.1, 1.5)`
- RandomErasing `p=0.3, scale=(0.02, 0.20)`
- CutMix `p=0.5, alpha=1.0`
- label_smoothing `0.1`

All at once → input distribution so distorted the network never sees
clean signal during the critical early-training window.

---

## Run 3 — Full aug stack + CBAM + CutMix, EMA broken

**Config:** `--epochs 10 --lr 0.03 --warmup-epochs 2 --data-parallel`
(CBAM on, CutMix on, lighter aug from commit `b272d30`, 2x T4)

**Result:**

| ep | train loss | train acc | val loss | val acc | ema val acc |
|----|-----------|-----------|----------|---------|-------------|
| 1  | 2.3009 | 0.1092 | 2.3021 | 0.1028 | 0.1019 |
| 2  | 2.2977 | 0.1167 | 2.3072 | 0.1023 | 0.1028 |
| 3  | 1.9781 | 0.3106 | 1.9126 | 0.3079 | 0.1030 |
| 4  | 1.2594 | 0.6997 | 1.6195 | 0.5544 | 0.1030 |
| 5  | 0.9538 | 0.8409 | 1.3521 | 0.6642 | 0.1030 |
| 6  | 0.8371 | 0.8773 | 1.3474 | 0.6364 | 0.1030 |
| 7  | 0.8674 | 0.8564 | 1.2080 | 0.7431 | 0.1030 |
| 8  | 0.7704 | 0.9078 | 1.1418 | 0.7431 | 0.1030 |
| 9  | 0.7172 | 0.9178 | 1.1764 | 0.7319 | 0.1030 |
| 10 | 0.7722 | 0.8840 | 1.1675 | 0.7431 | 0.1030 |

**Best val acc (raw):** 0.7431. EMA dead at random (0.1030 = 1/10).
Total time: 25.5 min.

### What went right

Raw model is healthy. After warmup ends at ep 2, training takes off
cleanly — val acc climbs from 10% → 74% in 8 epochs. Confirms the full
augmentation + CBAM + CutMix combo works post-warmup. Run 1's collapse
was init+LR, nothing else.

### What went wrong

EMA shadow accuracy stuck at 10.3% for 10 epochs. This is *not* the
Run 2 "warmup hasn't finished yet" explanation — by ep 10 the shadow
has had 1390 steps, enough to forget random init even at decay 0.999.
Real bug.

### Root cause

The EMA update copied every floating-point entry in `state_dict()` with
the decay rule, including BatchNorm running statistics.

Background on BatchNorm: each BN layer keeps two running averages —
`running_mean` and `running_var` — that summarise the mean/variance of
activations seen during training. At inference these replace
batch-wise statistics. They are normally updated each step with
`momentum=0.1` (fast — new batch contributes 10%).

The buggy EMA was decaying them at `0.999` (slow — new batch contributes
0.1%). That is **100× slower** than BN's intended update speed. Result:
the shadow's BN stats stayed close to their initial random-network
values, while the shadow's weights drifted toward the real (trained)
weights.

Mismatched stats + trained weights = garbage. BN normalises activations
using stats from a *different network*. Outputs become noise, model
predicts whichever class has the largest logit bias, accuracy pins
near 10%.

### Fix (commit `7c5a2a8`)

Split parameters from buffers in `EMA.update`. Decay only parameters
(weights, biases). Copy buffers (BN stats, `num_batches_tracked`)
directly from the live model — no averaging.

```python
# train.py EMA.update
m_params = dict(model.named_parameters())
s_params = dict(self.shadow.named_parameters())
for k, sp in s_params.items():
    mp = m_params[k].detach()
    if sp.dtype.is_floating_point:
        sp.mul_(self.decay).add_(mp, alpha=1.0 - self.decay)
    else:
        sp.copy_(mp)
m_buf = dict(model.named_buffers())
for k, sb in self.shadow.named_buffers():
    sb.copy_(m_buf[k])
```

Why this is the right split: BN stats are not learned parameters — they
are summary statistics of the data. Averaging them across time the way
you average weights makes no sense. Weights benefit from smoothing
(reduces noise from minibatch SGD); statistics need to track the
current network exactly.

### Next (Run 4)

Same config, 25 epochs, fresh out-dir. Expect EMA val acc to catch up
to raw within 3–5 epochs after warmup, then surpass raw at the end via
smoothing.

---

## Run 4 — EMA fix verified, 25 epochs

**Config:** `--epochs 25 --lr 0.03 --warmup-epochs 2 --data-parallel`
(post commit `7c5a2a8`, batch=128, 2x T4)

**Result (key epochs):**

| ep | train loss | train acc | val loss | val acc | ema val acc |
|----|-----------|-----------|----------|---------|-------------|
| 1  | 2.3009 | 0.1092 | 2.3021 | 0.1028 | 0.1015 |
| 5  | 0.9420 | 0.8448 | 1.5974 | 0.5838 | 0.1286 |
| 10 | 0.7705 | 0.8867 | 1.2539 | 0.7031 | 0.1667 |
| 15 | 0.7014 | 0.9093 | 1.1340 | 0.7557 | 0.2266 |
| 20 | 0.7394 | 0.8922 | 1.0740 | 0.7916 | 0.3612 |
| 22 | 0.7169 | 0.8997 | 1.0576 | **0.7971** | 0.3831 |
| 25 | 0.6932 | 0.9292 | 1.0731 | 0.7876 | 0.5893 |

**Best val acc (raw):** 0.7971 (ep 22). **Best EMA val acc:** 0.5893
(ep 25, still climbing). Total time: 63.2 min.

### What went right

EMA shadow is alive — accuracy climbs monotonically from 0.10 → 0.59.
Confirms the Run 3 fix. The BN-buffers-vs-parameters hypothesis was
correct.

### What went only half-right

EMA at 0.59 still trails raw at 0.80 after 25 epochs. EMA is *supposed*
to match or beat raw eventually. The catch: decay 0.999 = "EMA averages
over the last ~1000 steps ≈ 5.8 epochs of weights". So the shadow at
epoch 25 is effectively a snapshot of weights from epoch ~19. While the
live model is still rapidly improving, the shadow stays behind by ~3
epochs of trajectory and never catches up within a short run.

Fix options: either lower the decay (shorter window, shadow tracks
faster) or run for many more epochs (live model plateaus, shadow has
time to catch up).

### Hard ceiling observed

Raw val accuracy plateaus around **0.79** from epoch 16 onward, while
train accuracy reaches 0.93. The 14-point gap is the subject-wise
penalty: with no horizontal flip and no ImageNet pretrain, a from-scratch
ResNet-18 on 22k images can memorize a lot of driver-specific cues that
don't transfer to held-out drivers. ~80% honest val acc is roughly the
realistic ceiling for this configuration.

### Decision tree

Three options for Run 5; rank by what each tells us:

- **A. Lower EMA decay to 0.99.** Cheapest. Tests "is EMA actually
  helping?" cleanly because shadow tracks within ~1 epoch.
- **B. Use ImageNet pretrained weights.** Would jump straight to ~0.92,
  but **violates teacher's "from-scratch" constraint**. Off the table.
- **C. Stronger regularization.** MixUp + harder RandomErasing +
  DropBlock. Probably marginal — won't break the 0.85 ceiling without
  better init.

Picked **A** + a few orthogonal changes (TrivialAugment for cleaner
aug story; bigger input for more pixel detail; early stopping to save
GPU when training plateaus).

---

## Run 5 — TrivialAugment + 320 input + EMA decay 0.99 + early stop

**Config:** `--epochs 80 --lr 0.03 --warmup-epochs 2 --ema-decay 0.99
--img-size 320 --trivialaugment --early-stop-min-delta 0.005
--data-parallel` (post commit `1cf73d9`, batch=128, 2x T4)

**Aug stack:**
`RandomResizedCrop(320, scale=(0.7, 1.0), ratio=(0.85, 1.15))` →
`TrivialAugmentWide` → `Normalize` → `RandomErasing(p=0.25)`.
CutMix `p=0.3, alpha=0.5` at batch level. CBAM on.

### Why each change

- **TrivialAugmentWide.** Replaces the four hand-tuned augments
  (ColorJitter + Grayscale + Blur + manual ranges) with one policy that
  picks a random op + random magnitude per image from a fixed menu. No
  hyperparameters to tune. Empirically as strong as RandAugment with
  zero tuning budget.
- **320×320 input** (up from 224). State Farm phones are small — at 224
  a phone is ~7-14 px wide; at 320 it's ~10-20 px. More pixels on the
  discriminative object → better classification.
- **EMA decay 0.99** (down from 0.999). Window shrinks from ~1000 steps
  to ~100 steps. Shadow now tracks within 1 epoch instead of 6, so the
  final EMA actually reflects the trained model.
- **Early stop, patience 8, min-delta 0.005.** Stop if no gain ≥ 0.5pp
  on `max(val_acc, ema_val_acc)` for 8 consecutive epochs. Cuts wasted
  GPU once training plateaus.

**Result (key epochs, stopped early at ep 38):**

| ep | train loss | train acc | val loss | val acc | ema val acc |
|----|-----------|-----------|----------|---------|-------------|
| 1  | 2.3012 | 0.1109 | 2.3020 | 0.1028 | 0.1028 |
| 7  | 1.2032 | 0.7180 | 1.4635 | 0.6256 | 0.6903 |
| 14 | 0.8726 | 0.8416 | 1.1591 | 0.7626 | 0.8218 |
| 20 | 0.8808 | 0.8452 | 1.0352 | 0.8022 | 0.8223 |
| 22 | 0.8381 | 0.8572 | 1.1478 | 0.7363 | 0.8296 |
| **30** | 0.8339 | 0.8626 | 1.0566 | 0.7924 | **0.8431** ← best EMA |
| **31** | 0.7644 | 0.8798 | 0.9449 | **0.8327** ← best raw | 0.7955 |
| 35 | 0.7369 | 0.8985 | 0.9993 | 0.8031 | 0.8380 |
| 37 | 0.7753 | 0.8793 | 1.0931 | 0.7584 | 0.8402 |
| 38 | 0.7312 | 0.9113 | 1.0787 | 0.7714 | 0.8068 |

**Best raw val acc:** 0.8327 (ep 31). **Best EMA val acc:** 0.8431 (ep 30).
Total time: ~83 min. Stopped at ep 38/80.

### What went right

- **+5pp over Run 4.** EMA val acc 0.7971 → 0.8431. The combination
  (TrivialAug + 320 + decay 0.99) is net positive.
- **EMA catches up to raw inside 7 epochs.** Surpasses raw at ep 14
  (0.8218 vs 0.7626) and stays competitive. Confirms decay 0.99 is the
  right window for this schedule.
- **Early stop fired correctly.** Saved ~42 epochs × 2.2 min/ep ≈ 92
  minutes of GPU.

### Things that looked weird (but aren't bugs)

**Val accuracy bounces around a lot** (0.81 ep 16 → 0.48 ep 15 → 0.80
ep 16; 0.83 ep 31 → 0.74 ep 33). Small subject-wise val set + heavy
batch-level augmentation = volatile single-epoch numbers. EMA smooths
it cleanly. Don't react to single epochs.

**Train acc plateaus around 0.91, never reaches 1.0.** Three
intentional reasons stacked:
1. **CutMix** mixes two images and their labels (e.g. 60% c1 + 40% c2).
   The training accuracy code measures predictions against the
   **original** labels only (`augment.py:104`, `train.py:116`), so when
   CutMix pastes a big chunk of class B into a class-A image, the
   prediction is often "wrong" by this measurement even though training
   is going well.
2. **Label smoothing 0.1** rewrites one-hot targets as
   `[0.91, 0.01, 0.01, ...]`. The model is literally never trained to
   output a confident one-hot — so it never gets 100% train accuracy
   by design.
3. **TrivialAugmentWide + RandomErasing** sometimes produce inputs
   genuinely hard to classify (heavy distortion + 10% area erased).

A plateau ≠ a problem. The honest measure is **val** accuracy.

### Per-class findings (after eval on Run 5 weights)

Three classes underperform. Each has a different cause.

**1. c5 (operating radio) — under-trained because of cropping.**
The radio sits on the driver's right edge (left side of image, dashboard).
`RandomResizedCrop` with `scale=(0.7, 1.0)` removes up to 30% of the
area, and it chooses the crop region uniformly — so the radio often
gets cropped away during training. Then at eval time a `CenterCrop`
also chopped ~84 pixels from each side at 320 input, removing the
radio at inference too. Double whammy.

Re-running eval with the new no-crop eval transform (commit `954218a`)
bumped c5 accuracy noticeably without retraining → confirms the eval
crop was a big chunk of the c5 problem. Train-side crop will be
tightened next run.

**2. c8 (hair/makeup) confused with c2 (phone-right), ~66% acc.**
Both classes have **right hand near the right side of the face**.
Without a clear "phone object" cue, model collapses both into one
"hand-to-head" pattern. CutMix makes this worse: at p=0.3 a c8 image
often gets a rectangle from a c2 image (with phone visible) pasted into
the face/hand region, training the model to label hand-to-head ambiguously.

**3. c3 (texting-left) ~77% acc, hard by camera geometry.**
Camera is mounted on the driver's right side. For c3 (texting with
*left* hand), the phone is on the **far side of the cabin**, partially
hidden by the body, and the phone object is foreshortened to ~10-20
pixels at 320 input. Common confusions: c4 (phone-left, same hand
region) and c0 (safe driving, when the hand drops low out of view).
c1↔c3 should be near zero given no-HFlip aug — confirm in confusion
matrix.

### Fixes applied (commit `954218a`, train crop + eval crop)

- **Train:** `RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(0.95, 1.05))`.
  Maximum ~10% area loss instead of ~30%. Keeps cabin edges (radio,
  far-side phone) in the frame most of the time.
- **Eval:** `Resize((size, size))` directly, no `CenterCrop`. Zero edge
  loss at inference — model sees full frame.

### Next (Run 6 plan)

- **Train aug:** tightened crop (above). Preserves c5 cabin context.
- **Eval aug:** no crop. Preserves periphery at inference.
- **`--img-size 320 → 384`.** Small far-side phone (c3) jumps from
  ~10-20 px to ~12-24 px (+40% pixels on the cue). SAM map at layer4
  goes from 10×10 to 12×12 (finer attention grid).
- **`--cutmix-p 0.3 → 0.15`.** Halves the rate at which c2/c8 (and c1/c3)
  pairs get cross-pasted, sharpening "phone present vs absent" signal.
- **`--epochs 50` and `--early-stop-patience 0`.** Run full schedule,
  no early termination. Cosine reaches eta_min at ep 50.
- Same other hyperparams. New `--out-dir run6`.
- Estimated wall time: ~50 × ~3 min/ep at 384 input on T4×2 ≈ 150 min.
- Expected: c5 stays high (already fixed in Run 5 re-eval); c8 0.66 →
  0.72-0.75; c3 0.77 → 0.80+; overall best EMA 0.84 → 0.86-0.87. Macro
  F1 should improve more than weighted F1 (rare classes c5/c8/c3 lift
  more than common c0).

### Confusion-matrix sanity checks for Run 6 eval

- **c2 ↔ c8 block:** should shrink most (CutMix p halved → less hand-to-head label confusion).
- **c1 ↔ c3 block:** should stay near zero (validates no-HFlip choice — if non-zero, something else flips labels).
- **c3 ↔ c4 block:** if still high → hard ceiling from camera geometry; would need higher resolution (448+) or an auxiliary phone-detection head.
- **c2 ↔ c1 block:** should stay low; if it climbs after dropping CutMix p, regularization gap is the cause and CutMix p needs a middle value (~0.2).

---

## Run 6 — 384 input + tightened crop + cutmix p halved

**Config:** `--epochs 50 --lr 0.03 --warmup-epochs 2 --ema-decay 0.99
--img-size 384 --trivialaugment --cutmix-p 0.15
--early-stop-patience 0 --data-parallel` (post commit `5b174c7`,
batch=128, 2x T4)

**Aug stack:**
`RandomResizedCrop(384, scale=(0.9, 1.0), ratio=(0.95, 1.05))` →
`TrivialAugmentWide` → `Normalize` → `RandomErasing(p=0.25)`.
CutMix `p=0.15, alpha=0.5` at batch level. CBAM on.

### Why each change vs Run 5

- **384×384 input** (up from 320). Far-side phone in c3 jumps from
  ~10-20 px to ~12-24 px. SAM map at layer4 goes 10×10 → 12×12 (finer
  attention grid).
- **Tighter `RandomResizedCrop` scale (0.9, 1.0)** (was 0.7, 1.0). Max
  10% area loss vs 30%. Stops cropping radio (c5) and far-side phone
  (c3) out of frame during training.
- **No-crop eval transform** (already applied in commit `954218a`).
- **`--cutmix-p 0.3 → 0.15`.** Halves rate of cross-pasting c2 phone
  patches onto c8 face regions. Goal: sharper "phone present vs absent"
  signal.
- **`--epochs 50` and `--early-stop-patience 0`.** Run full schedule.
  Cosine reaches eta_min at ep 50.

**Result (key epochs, cancelled ^C during ep 50, ep 49 complete):**

| ep | train loss | train acc | val loss | val acc | ema val acc |
|----|-----------|-----------|----------|---------|-------------|
| 30 | 0.6849 | 0.9217 | 1.0004 | 0.8254 | 0.8271 |
| 32 | 0.6561 | 0.9456 | 0.9622 | 0.8672 | 0.8621 |
| **35** | 0.6282 | 0.9497 | 0.9448 | 0.8446 | **0.8747** ← best EMA |
| 38 | 0.7023 | 0.9317 | 0.9889 | 0.8550 | 0.8636 |
| **39** | 0.6451 | 0.9431 | 0.9265 | **0.8683** ← best raw | 0.8141 |
| 44 | 0.6015 | 0.9610 | 0.9627 | 0.8607 | 0.8380 |
| 46 | 0.6029 | 0.9672 | 0.9687 | 0.8543 | 0.8599 |
| 49 | 0.6130 | 0.9481 | 0.9783 | 0.8541 | 0.8539 |

**Best raw val acc:** 0.8683 (ep 39). **Best EMA val acc:** 0.8747
(ep 35). Total time: ~137 min on T4×2. Run cancelled mid ep 50 — final
metrics already at peak, no loss of state. `best.pt` saved at ep 35.

### Per-class result (eval on Run 6 weights, EMA, --img-size 384)

| class | precision | recall | F1 | support | vs Run 5 (qualitative) |
|---|---:|---:|---:|---:|---|
| c0 safe driving | 0.69 | 0.64 | **0.66** | 465 | ↓ regressed |
| c1 text right | 0.90 | 1.00 | 0.95 | 462 | ≈ |
| c2 phone right | 0.99 | 0.91 | 0.95 | 462 | ↑ |
| c3 text left | 0.92 | 0.92 | **0.92** | 461 | ↑↑ (was ~0.77) |
| c4 phone left | 0.96 | 0.97 | 0.97 | 472 | ↑ |
| c5 radio | 0.99 | 0.89 | 0.94 | 466 | ↑ (eval-crop fix locked in) |
| c6 drinking | 0.99 | 0.91 | 0.94 | 468 | ≈ |
| c7 reach behind | 0.99 | 0.97 | 0.98 | 423 | ≈ |
| c8 hair/makeup | 0.61 | 0.97 | **0.75** | 398 | ↑ F1 but **precision crashed** |
| c9 talk passenger | 0.80 | 0.58 | **0.68** | 447 | ↓ regressed |
| **accuracy** | | | **0.875** | 4524 | +3pp over Run 5 EMA |
| **macro F1** | | | 0.873 | | +5pp |
| **weighted F1** | | | 0.875 | | +3pp |

### What went right

- **+3pp on EMA val acc** (0.8431 → 0.8747). Bigger lift than Run 6 plan
  predicted (0.86-0.87 target).
- **c3 jumped from ~0.77 to 0.92 F1.** Bigger input + tighter crop did
  exactly what was planned. Far-side phone now resolved enough for the
  classifier.
- **c1, c2, c4, c5, c6, c7 all ≥ 0.94 F1.** Six classes effectively
  solved on subject-wise val.
- **c1↔c3 confusion stays near zero** (both F1 > 0.92). Validates
  no-HFlip choice — model truly learned left vs right, not just
  flipped one to the other.

### What went wrong (surprises)

Three classes now problematic, in different ways:

**1. c8 precision crashed to 0.61 while recall jumped to 0.97.**
Model over-predicts c8. Recall 0.97 means it catches almost every real
hair/makeup case, but precision 0.61 means **39% of "c8 predictions"
are actually other classes**. With CutMix p halved, the model lost a
regularizer that previously kept c8 conservative. Now it labels any
"hand near head/face" image as c8 — sweeping up c0 (when driver
adjusts hair briefly while driving), c9 (when talking with hand near
face), and edge-case c2 (phone-right but model not seeing phone).

**2. c0 (safe driving) F1 dropped to 0.66.** Recall 0.64 means 36% of
real safe-driving frames mislabelled. Combined with c8's over-prediction
above — many c0 cases now stolen into c8. "Safe driving" has no
characteristic action; model needs to recognize it as **absence** of
distraction. Halving CutMix removed cross-pollination that taught the
model "no clear cue → c0", so default class collapsed.

**3. c9 (talk passenger) recall 0.58.** Same root cause. c9 is also a
"passive" class — driver looks ahead or sideways, sometimes hand-near-face.
Without enough CutMix mixing, model has no strong cue and falls back
to c8 (the dominant "hand near head" attractor).

### Root cause (single hypothesis)

**Cutmix-p 0.15 was too aggressive a cut.** Run 5 used p=0.3 — that
level was actually doing useful work for the passive/ambiguous classes
(c0, c9, c8). Halving it broke a regularization equilibrium:

- Active classes (c1-c7, all distinct hand+object actions): improved
  because they no longer get c8 phone patches glued onto them.
- Passive classes (c0, c8, c9, all "hand near head/face" or "no clear
  action"): degraded because they lost the cross-mixing that kept the
  model from collapsing them into one category.

Net effect: macro F1 still rose (5pp gain) because lifts on c2/c3/c4
were huge, but the passive-class triangle (c0 ↔ c8 ↔ c9) got worse.

### Confusion-matrix predictions for Run 6 (sanity-check on output)

- **c2 ↔ c8 block:** likely shrank as planned (both classes' F1 up).
- **c0 → c8 block:** likely big — c0 recall 0.64, c8 precision 0.61.
- **c9 → c8 block:** likely big — c9 recall 0.58.
- **c1 ↔ c3 block:** near zero (no-HFlip still holds).
- **c3 ↔ c4 block:** should be small (both ≥ 0.92 F1 → no shared
  errors).

### Decision: keep Run 6 as headline result

EMA 0.8747 is a clean +3pp over Run 5. Macro F1 0.873 is a +5pp lift.
Six of ten classes effectively solved. Three passive classes (c0/c8/c9)
are the residual problem and are realistically the hardest classes on
this dataset (no object cue, just pose).

### Next options if running a Run 7

- **A. Restore CutMix-p to 0.20** (middle value between Run 5's 0.3 and
  Run 6's 0.15). Hypothesis: lift passive classes back without losing
  c3 gains. Cheapest test.
- **B. Class-balanced sampling for c0/c8/c9.** Make sure each batch
  contains roughly equal counts of all classes; currently c8 has 398
  examples vs c4 with 472 — a 19% gap that may amplify with imbalance.
- **C. Add a "hand near head" auxiliary head.** Two-class output
  (yes/no on hand position) trained jointly with main 10-way head.
  Forces the network to keep a distinct hand-position feature instead
  of collapsing into c8. Architecturally invasive — only if A and B
  fail.
- **D. Accept current result.** Macro F1 0.873 on subject-wise val from
  scratch is a solid headline. Report c0/c8/c9 as the residual hard
  classes with cabin-camera-geometry explanation. Move on to demo +
  report.

Recommend **D** for the project deadline, **A** if time allows one more
~2.3 hr GPU run.
