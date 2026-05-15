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
| Init | Kaiming, no pretrain |
| Split | Subject-wise; held-out: `p022, p035, p047, p056, p075` |
| Augmentation | RandomResizedCrop, ColorJitter, RandomGrayscale, GaussianBlur, RandomErasing, CutMix. **No HFlip** (left/right class asymmetry — see ADR 0002) |
| Optimizer | SGD (Nesterov), momentum=0.9, weight_decay=5e-4 |
| LR schedule | Cosine 0.1 -> 0 over 40 epochs |
| Loss | CrossEntropy with label smoothing 0.1 |
| EMA decay | 0.999 (eval on EMA weights) |
| Normalization | Dataset-computed RGB stats |
| Batch / size | 128 / 224x224 |
| Target time | < 5 hr on Kaggle T4x2 |

## Repo layout

```
.
|-- CONTEXT.md             # glossary + decisions log
|-- docs/adr/              # architectural decision records
|-- data_prep.py           # subject-wise split + dataset stats
|-- augment.py             # heavy aug pipeline + CutMix + dataset class
|-- model.py               # ResNet-18 + CBAM
|-- train.py               # full training loop (EMA, cosine, CutMix, tier-1 trigger)
|-- eval.py                # classification_report + figures + attention viz
|-- app.py                 # Gradio demo
|-- notebooks/             # Kaggle notebook templates (run on T4x2 free tier)
|-- requirements.txt
`-- README.md
```

## Quickstart (Kaggle T4x2)

1. Open a new Kaggle notebook, attach the **State Farm Distracted Driver
   Detection** dataset, set Accelerator = GPU T4 x2, Internet = On.
2. Upload this repo as a Kaggle dataset (or `!git clone` from your own
   GitHub) so the scripts are importable.
3. Run the four notebooks in [`notebooks/`](notebooks/) in order:
   - `01_stats_split` — verify split + compute dataset RGB stats (~5 min)
   - `02_train` — full 40-epoch training run (~2.5-3 hr)
   - `03_ablation` — baseline without CBAM, 25 epochs (~1.5 hr)
   - `04_eval_figs` — generate classification_report + all figures

## Tier-1 fallback (built-in)

`train.py` auto-checks val accuracy at epoch 20. If EMA val acc < 0.50,
the run aborts. Restart with:

```bash
python train.py ... --no-cutmix --no-grayscale
```

## Tier-2 fallback (manual)

If final EMA val acc < 0.55, retrain without CBAM as a baseline:

```bash
python train.py ... --no-cbam --no-cutmix --out-dir run_baseline
```

## Demo (local)

```bash
python app.py --ckpt checkpoints/best.pt --stats splits/stats.json
```

For HuggingFace Spaces deployment: set `CKPT_PATH`, `STATS_PATH`,
`EXAMPLES_DIR` env vars; the Space runs `python app.py`.

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
