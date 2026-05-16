# Driver Distraction Classification (ResNet-18 + CBAM, from scratch)

End-term project for Computer Vision (UIT). Classifies the 10 State Farm
distracted-driver classes using a ResNet-18 with CBAM attention blocks,
trained **from scratch** (no ImageNet pretrain). Evaluated with a
**subject-wise split** so the val accuracy reflects real generalization,
not driver-identity memorization. Demo via a Gradio web app on
out-of-distribution images.

See [`CONTEXT.md`](CONTEXT.md) for the full glossary and decisions, and
[`docs/adr/`](docs/adr/) for architectural decision records.

## Approach summary

| | |
|---|---|
| Architecture | ResNet-18 + CBAM (CAM + SAM) after each of layer1..4 |
| Init | Kaiming (conv), `N(0, 0.01)` (final fc), no pretrain |
| Split | Subject-wise; held-out: `p022, p035, p047, p056, p075` |
| Augmentation | RandomResizedCrop + **TrivialAugmentWide** + Normalize + RandomErasing(p=0.25). CutMix(p=0.3, alpha=0.5) at batch level. **No HFlip** (left/right class asymmetry — see ADR 0002) |
| Optimizer | SGD (Nesterov), momentum=0.9, weight_decay=5e-4 |
| LR schedule | Linear warmup (2 ep) → Cosine 0.03 → 0 |
| Loss | CrossEntropy with label smoothing 0.1 |
| EMA decay | 0.99 (eval on EMA weights; BN buffers copied verbatim) |
| Normalization | Dataset-computed RGB stats |
| Batch / size | 128 / 320x320 |
| Max epochs / early stop | 80 / patience 8 on `max(val_acc, ema_val_acc)`, min-delta 0.005 |
| Target time | < 5 hr on Kaggle T4x2 (Run 5: ~83 min, stopped ep 38) |
| Run 5 result | best raw val acc **0.8327** (ep 31), best EMA val acc **0.8431** (ep 30) |

## Repo layout

```
.
|-- CONTEXT.md             # glossary + decisions log
|-- docs/adr/              # architectural decision records
|-- data_prep.py           # subject-wise split + dataset stats
|-- augment.py             # heavy aug pipeline + CutMix + dataset class
|-- model.py               # ResNet-18 + CBAM
|-- train.py               # single-stream training loop (Run 1-6)
|-- eval.py                # classification_report + figures + attention viz
|-- model_twostream.py     # TwoStreamCBAM (Run 7) + ThreeStreamCBAM (Run 8)
|-- augment_twostream.py   # two/three-stream datasets, crops, pose lookup
|-- train_twostream.py     # multi-stream training loop, --three-stream flag for Run 8
|-- eval_twostream.py      # auto-dispatch eval for two/three-stream ckpts
|-- extract_pose.py        # Run 8 MediaPipe head-pose precompute → pose.parquet
|-- log.md                 # per-run training log + per-class diagnosis
|-- RUN7_PLAN.md / RUN7_HOWTO.md  # Run 7 design + runbook (two-stream)
|-- RUN8_PLAN.md / RUN8_HOWTO.md  # Run 8 design + runbook (three-stream + pose)
|-- demo/                  # Gradio demos (Run 6 + Run 7) — see demo/DEMO_HOWTO.md
|-- notebooks/             # Kaggle notebook templates (run on T4x2 free tier)
|-- requirements.txt
`-- README.md
```

**Architecture history:**

| Run | Architecture | Eval macro F1 |
|---|---|---:|
| Run 5 (canonical baseline) | ResNet18+CBAM, 320 input | 0.83 |
| Run 6 (current headline) | ResNet18+CBAM, 384 input, tightened crop | **0.873** |
| Run 7 | Two-stream (full + top-crop face), CutMix p=0.2 | 0.748 (regressed) |
| Run 8 (planned) | Three-stream (full + hand-crop + MediaPipe pose), no CutMix | target ≥ 0.85 |

See `log.md` for per-run diagnosis and `RUN8_PLAN.md` for the Run 8
motivation.

## Quickstart (Kaggle T4x2)

1. Open a new Kaggle notebook, attach the **State Farm Distracted Driver
   Detection** dataset, set Accelerator = GPU T4 x2, Internet = On.
2. Upload this repo as a Kaggle dataset (or `!git clone` from your own
   GitHub) so the scripts are importable.
3. Run cell recipes in [`notebooks/README.md`](notebooks/README.md) in order:
   - `01_stats_split` — verify split + compute dataset RGB stats (~5 min)
   - `02_train` — full Run 5 schedule, max 80 ep, early stop ~ep 38 (~1.5 hr)
   - `03_ablation` — baseline without CBAM, 25 epochs (~1.5 hr)
   - `04_eval_figs` — generate classification_report + all figures

Run 5 canonical training command:

```bash
python train.py \
    --data-root /kaggle/input/competitions/state-farm-distracted-driver-detection \
    --splits-dir /kaggle/working/splits \
    --out-dir    /kaggle/working/run5 \
    --epochs 80 --batch-size 128 --num-workers 4 \
    --lr 0.03 --warmup-epochs 2 --ema-decay 0.99 \
    --img-size 320 --trivialaugment \
    --early-stop-patience 8 --early-stop-min-delta 0.005 \
    --data-parallel
```

## Tier-1 fallback (built-in)

`train.py` auto-checks val accuracy at epoch 20. If `max(val_acc, ema_val_acc) < 0.50`,
run aborts. Restart with:

```bash
python train.py ... --no-cutmix --no-grayscale
```

Run 5 hit ema val acc 0.82 by ep 20 — fallback not triggered.

## Tier-2 fallback (manual)

If final EMA val acc < 0.55, retrain without CBAM as baseline:

```bash
python train.py ... --no-cbam --out-dir run_baseline
```

## Demo (local)

Gradio web demo lives in [`demo/`](demo/DEMO_HOWTO.md). Both Run 6
and Run 7 supported. CPU inference fine.

```powershell
cd demo
python app.py --ckpt checkpoints/best.pt --stats splits/stats.json            # Run 6
python app_twostream.py --ckpt checkpoints/run7_best.pt --stats splits/stats.json   # Run 7
```

See `demo/DEMO_HOWTO.md` for full setup (deps, Kaggle artifact
download, HuggingFace Spaces deployment, troubleshooting).

## Reproducibility caveat

`torch.backends.cudnn.deterministic=True` and `seed=42` everywhere, but
CUDA + cuDNN versions on Kaggle change over time, so exact bitwise
reproducibility across runs is not guaranteed.

## References

Method components are grounded in the following prior work. arXiv IDs
are given so the report bibliography can be filled in directly.

**Backbone + initialization**
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.*
  CVPR. arXiv:1512.03385. — ResNet-18 architecture (BasicBlock layout,
  global average pool, fc head).
- He, K. et al. (2015). *Delving Deep into Rectifiers: Surpassing
  Human-Level Performance on ImageNet Classification.* ICCV.
  arXiv:1502.01852. — Kaiming initialization used since we train from
  scratch (no ImageNet pretrain).

**Attention module**
- Woo, S. et al. (2018). *CBAM: Convolutional Block Attention Module.*
  ECCV. arXiv:1807.06521. — Channel + spatial attention block inserted
  after each ResNet stage. The spatial attention map from `layer4`
  produces the demo heatmap.
- Hu, J. et al. (2018). *Squeeze-and-Excitation Networks.* CVPR.
  arXiv:1709.01507. — Predecessor of CBAM (channel attention only); a
  natural alternative if ablation favors a lighter block.

**Regularization + augmentation**
- Yun, S. et al. (2019). *CutMix: Regularization Strategy to Train
  Strong Classifiers with Localizable Features.* ICCV. arXiv:1905.04899.
  — Batch-level patch mixing used during training.
- Zhong, Z. et al. (2020). *Random Erasing Data Augmentation.* AAAI.
  arXiv:1708.04896. — RandomErasing transform.
- Szegedy, C. et al. (2016). *Rethinking the Inception Architecture for
  Computer Vision.* CVPR. arXiv:1512.00567. — Label smoothing.
- DeVries, T. & Taylor, G. W. (2017). *Improved Regularization of
  Convolutional Neural Networks with Cutout.* arXiv:1708.04552. —
  Related cutout-style aug; cited as a baseline for CutMix.

**Optimization**
- Loshchilov, I. & Hutter, F. (2017). *SGDR: Stochastic Gradient Descent
  with Warm Restarts.* ICLR. arXiv:1608.03983. — Cosine LR schedule.
- Polyak, B. & Juditsky, A. (1992). *Acceleration of Stochastic
  Approximation by Averaging.* SIAM J. Control Optim. — Theoretical
  basis for EMA of weights at evaluation time.

**Interpretability + related work on State Farm**
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from
  Deep Networks via Gradient-based Localization.* ICCV.
  arXiv:1610.02391. — Alternative visualization; CBAM's SAM is used here
  instead since it is intrinsic to the forward pass.
- Eraqi, H. M. et al. (2019). *Driver Distraction Identification with an
  Ensemble of Convolutional Neural Networks.* J. Adv. Transp.
  arXiv:1901.09097. — Prior State Farm benchmark + AUC dataset; useful
  for related-work section.
- Masood, S. et al. (2018). *Detecting Distracted Driver Using
  Convolutional Neural Network.* CVPR Workshops. — VGG-16 baseline on
  the same dataset; reference for the "popular solution" framing.

**Self-supervised pretraining (referenced but not used)**
- Chen, T. et al. (2020). *A Simple Framework for Contrastive Learning
  of Visual Representations.* ICML. arXiv:2002.05709. — SimCLR; was
  considered as an alternative pretrain step and rejected to keep the
  training budget under 5 hr (see CONTEXT.md).
