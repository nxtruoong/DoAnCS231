# Training Log

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

**Diagnosis:** Loss locked at `ln(10) = 2.3026` = uniform softmax output. Network collapsed.

**Root causes:**
1. **Bad `nn.Linear` init** (`model.py:142`): `kaiming_normal_(mode="fan_out", nonlinearity="relu")` on final fc (512→10) gave std `sqrt(2/10) ≈ 0.45`. Initial logit variance ~100 → softmax saturates one-hot → huge gradient → ReLU death → output collapse. Same init also saturated CBAM `ChannelAttention` MLP sigmoid, zeroing CBAM signal.
2. **No LR warmup:** SGD nesterov `lr=0.1` from scratch on 22k images. First steps blow up.

**Fixes applied (commit `ed889e3`):**
- `model.py`: replace Linear init with `normal_(0, 0.01)`.
- `train.py`: add `--warmup-epochs` (default 5), wrap cosine in `SequentialLR` with linear warmup `1e-3*lr → lr`.

---

## Run 2 — Smoke test, minimal aug, no CBAM, no CutMix

**Config:** `--no-cutmix --no-cbam --minimal-aug --label-smoothing 0.0 --epochs 3 --lr 0.03 --warmup-epochs 1`

Aug stripped to `RandomResizedCrop + Normalize` only.

**Result:**

| ep | train loss | train acc | val loss | val acc | ema val acc |
|----|-----------|-----------|----------|---------|-------------|
| 1  | 2.0103 | 0.2436 | 2.5098 | 0.2204 | 0.1039 |
| 2  | 0.3513 | 0.8857 | 2.6580 | 0.4777 | 0.1008 |
| 3  | 0.0641 | 0.9825 | 1.9219 | 0.6185 | 0.1404 |

**Best val acc:** 0.6185. Total time: 4.4 min.

**Diagnosis:** Pipeline + data + model fine. Original Run 1 aug stack was the killer.

**Findings:**
1. **Train/val gap huge** (98% / 62%) — subject-wise split + zero aug → model memorizes drivers. Expected. Need aug, but lighter than Run 1.
2. **EMA stuck at ~0.10** = NOT a bug. With decay `0.999` and only `3 * 139 = 417` steps, shadow weight on init = `0.999^417 ≈ 0.66` → still 66% random init. Full 40 epochs = 5560 steps, EMA will converge. For short smoke runs, lower decay to `0.99`.
3. **Val loss bumpy** (2.51 → 2.66 → 1.92) — small subject-wise val set + distribution shift. Watch trend over more epochs.

**Original Run 1 aug stack (too heavy for from-scratch):**
- ColorJitter `b=0.5, c=0.4, s=0.4, h=0.2` (hue ±72° brutal)
- RandomGrayscale `p=0.2`
- GaussianBlur `sigma=(0.1, 1.5)`
- RandomErasing `p=0.3, scale=(0.02, 0.20)`
- CutMix `p=0.5, alpha=1.0`
- label_smoothing `0.1`

All simultaneously from scratch → no learnable signal.

**Next: reintroduce aug incrementally.**

| Step | Add | Expect |
|------|-----|--------|
| 1 | Light ColorJitter `b=0.2, c=0.2, s=0.2, h=0.05` | val ↑, train ↓ |
| 2 | + RandomErasing `p=0.25, scale=(0.02, 0.10)` | val ↑ |
| 3 | + RandomGrayscale `p=0.1` | val ≈ |
| 4 | + GaussianBlur `sigma=(0.1, 0.8)` | val ≈ |
| 5 | + CBAM | val ↑ slightly |
| 6 | + CutMix `p=0.3, alpha=0.5` | val ↑ |
| 7 | + label_smoothing `0.1` | val ≈ |

---

## Run 3 — Full aug stack + CBAM + CutMix, EMA broken

**Config:** `--epochs 10 --lr 0.03 --warmup-epochs 2 --data-parallel` (CBAM on, CutMix on, lighter aug from commit `b272d30`, 2x T4)

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

**Best val acc (raw):** 0.7431. EMA dead at random (0.1030 = 1/10). Total time: 25.5 min.

**Findings:**
1. **Raw model healthy.** Train loss drops, val acc climbs steadily — full aug + CBAM + CutMix work fine post-warmup. First 2 epochs spent in warmup → real learning starts ep 3.
2. **EMA frozen at ~0.103 from ep 3 onward** — not slow convergence (Run 2 hypothesis wrong here). Real bug.
3. **Val acc still rising at ep 10** — schedule too short. Cosine barely past midpoint.

**Root cause (EMA):** `EMA.update` applied decay `0.999` to ALL floating-point state_dict entries, including BN `running_mean` / `running_var`. BN running stats normally update with momentum 0.1 per step (much faster than EMA). Decaying them at 0.999 froze them near random-init values. Shadow then had near-current weights with random-net BN stats → garbage activations → predicts dominant class → ~10% acc forever.

**Fix applied (commit pending):**
```python
# train.py EMA.update — split params from buffers
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

EMA decays parameters only; buffers (BN stats, num_batches_tracked) copied verbatim from live model.

**Next (Run 4):**
- Same config, `--epochs 25` (cosine reaches eta_min, EMA settles).
- New `--out-dir run4`.
- Expect: EMA val acc tracks raw val acc within ~3-5 epochs after warmup, surpasses raw by end via smoothing.

---

## Run 4 — EMA fix verified, 25 epochs

**Config:** `--epochs 25 --lr 0.03 --warmup-epochs 2 --data-parallel` (post commit `7c5a2a8`, batch=128, 2x T4)

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

**Best val acc (raw):** 0.7971 (ep 22). **Best EMA val acc:** 0.5893 (ep 25, still climbing). Total time: 63.2 min.

**Findings:**
1. **EMA fix works** — shadow no longer frozen, monotone climb after warmup. Confirms BN-buffer hypothesis from Run 3.
2. **EMA still trails raw badly** (0.59 vs 0.79). Cause: decay `0.999` too high for short schedule. Window = `1/(1-decay) = 1000 steps` ≈ 5.8 epochs (172 steps/ep). Shadow weighted-avg of last ~6 epochs lags real model by ~3 epochs of trajectory.
3. **Raw val plateau ~0.79 from ep 16 onward.** Train acc 0.93. Subject-wise split + scratch ResNet-18 = hard ceiling. Model memorizes drivers.
4. **Val loss / acc bumpy** (ep 13 dip 0.69, ep 22 peak 0.80). Small subject-wise val set + driver shift.

**Next options (pick one):**
- **A. Lower EMA decay** → `--ema-decay 0.99` (window ~100 steps ≈ 0.6 epoch, EMA tracks raw within 1-2 epochs). Cheapest test of EMA benefit.
- **B. Pretrained backbone** → swap `build_model` to load `torchvision.models.resnet18(weights="IMAGENET1K_V1")` and freeze early stages 1-2 epochs. Expected jump to 0.92+ val.
- **C. Stronger reg** → MixUp + heavier RandomErasing + DropBlock. Marginal gain, won't break 0.85 ceiling without pretrain.

Recommend **B** for accuracy ceiling, **A** for diagnosing EMA contribution cleanly.

---

## Run 5 — TrivialAugment + 320 input + EMA decay 0.99 + early stop

**Config:** `--epochs 80 --lr 0.03 --warmup-epochs 2 --ema-decay 0.99 --img-size 320 --trivialaugment --early-stop-min-delta 0.005 --data-parallel` (post commit `1cf73d9`, batch=128, 2x T4)

**Aug stack:** `RandomResizedCrop(320, scale=(0.7, 1.0), ratio=(0.85, 1.15))` → `TrivialAugmentWide` → `Normalize` → `RandomErasing(p=0.25)`. CutMix `p=0.3 alpha=0.5` at batch level. CBAM on.

**Result (key epochs, run stopped early at ep 38 via patience=8 / min-delta=0.005):**

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

**Best raw val acc:** 0.8327 (ep 31). **Best EMA val acc:** 0.8431 (ep 30). Total time: ~83 min. Training stopped at ep 38/80 (early stop fired: 8 consecutive epochs no gain ≥ 0.5pp over best `max(val_acc, ema_val_acc) = 0.8431`).

**Findings:**
1. **+5pp over Run 4** (0.7971 → 0.8431) on best EMA. Confirms TrivialAugment + 320 input + decay 0.99 are net positive.
2. **EMA tracks raw within ~7 epochs** (decay 0.99 = window ~140 steps ≈ 1 epoch). Cleanly surpasses raw at ep 14 (0.8218 vs 0.7626) and stays competitive after.
3. **Val acc highly volatile** (0.81 ep 16 → 0.48 ep 15 → 0.81 ep 16; 0.83 ep 31 → 0.74 ep 33). Small subject-wise val (~4.5k) + heavy aug. EMA smooths nicely.
4. **Train acc plateau ~0.91, NOT a bug.** Three intentional ceilings:
   - CutMix mixes labels but `train_acc` measured against original `labels` only (`augment.py:104`, `train.py:116`) → caps acc when cut region dominates.
   - Label smoothing 0.1 → soft targets, model never trained to one-hot.
   - TrivialAugmentWide + RandomErasing → some augmented inputs genuinely hard.
5. **Early stop fired correctly at ep 38.** Saved ~42 epochs × ~2.2 min = ~92 min of GPU time vs full 80.

**Per-class findings (after eval on Run 5 weights):**

1. **c5 (operating radio) — under-trained because of crop.** RandomResizedCrop `scale=(0.7, 1.0)` removes up to 30% area; the radio sits on the driver's right edge (left side of frame), so it is the first thing cropped out. Eval transform also center-crops `366→320` at `--img-size 320`, losing ~84 px each side of longer dim → radio chopped off at inference too. Re-eval with the new no-crop eval transform (commit `954218a`) bumped c5 acc significantly without retraining — confirms eval cropping was a big chunk of the c5 problem.
2. **c8 (hair / makeup) — confused with c2 (phone-right), ~66% acc.** Both classes have **right hand near the right side of the face**. Without a strong "phone object" cue, model collapses both into one "hand-to-head" pattern. CutMix amplifies this: at p=0.3 a c8 image often gets a rectangle from a c2 image (phone visible) pasted into the face/hand region, training the model to treat hand-to-head as an ambiguous label.
3. **c3 (texting-left) — ~77% acc, hard by camera geometry.** Camera mounted on driver's right side → left hand is on the **far side of the cabin**, partially occluded by the body, and the phone object is foreshortened (~10-20 px at 320 input). Likely confusions: c4 (phone-left, same hand region) and c0 (safe driving if hand drops low). c1↔c3 should be near-zero given no-HFlip aug — verify in confusion matrix.

**Fixes applied (commits `954218a` aug, see Run 6 below for cutmix/resolution):**
- Train: `RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(0.95, 1.05))` — max ~10% area loss.
- Eval: `Resize((size, size))` directly — zero edge loss.

**Next (Run 6):**
- Train aug: tightened crop (above) — preserves c5 cabin context.
- Eval aug: no crop — preserves periphery at inference.
- `--img-size 320 → 384` — small far-side phone (c3) goes from ~10-20 px to ~12-24 px (+40% pixels on the cue); SAM map at layer4 goes from 10×10 to 12×12 (finer attention grid).
- `--cutmix-p 0.3 → 0.15` — halves the rate at which c2/c8 (and c1/c3) get cross-pasted, sharpening "phone present vs absent" signal.
- Same other hyperparams. New `--out-dir run6`.
- Expected: c5 stays high (already fixed in Run 5 re-eval); c8 0.66 → 0.72-0.75; c3 0.77 → 0.80+; overall best EMA 0.84 → 0.86-0.87. Macro F1 should improve more than weighted F1.

**Confusion-matrix sanity checks for Run 6 eval:**
- c2 ↔ c8 block: should shrink most.
- c1 ↔ c3 block: should stay near zero (validates no-HFlip choice).
- c3 ↔ c4 block: if still high → hard ceiling from camera geometry; would need higher resolution (448+) or auxiliary phone-detection head.
- c2 ↔ c1 block: should stay low; if it climbs after dropping CutMix p, regularization gap is the cause.
