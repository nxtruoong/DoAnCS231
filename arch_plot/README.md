# PlotNeuralNet diagram for ResNet-18 + CBAM

Renders the project architecture as a 3D TikZ diagram using
[PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet).

## Prerequisites

- LaTeX with `pdflatex` (TeX Live, MiKTeX, or MacTeX).
- Bash (Git Bash on Windows works).
- Python 3.

## Steps

```bash
# 1. Clone PlotNeuralNet next to this repo (siblings, NOT inside)
cd ..
git clone https://github.com/HarisIqbal88/PlotNeuralNet.git
cd PlotNeuralNet

# 2. Drop our script into examples/
mkdir -p examples/cvv_arch
cp ../CVV/arch_plot/plot_arch.py examples/cvv_arch/

# 3. Build (generates .tex, runs pdflatex, opens viewer)
cd examples/cvv_arch
bash ../../tikzmake.sh plot_arch
```

Output: `plot_arch.pdf` in `examples/cvv_arch/`. Copy it back to
`arch_plot/` if you want it next to this README.

## What the diagram shows

Mirrors `model.py` exactly:
- Stem: `conv1 7x7 s=2` -> `maxpool 3x3 s=2`
- 4 stages: `layer1..layer4` (2 BasicBlocks each), channels 64 -> 128 -> 256 -> 512
- A **CAM + SAM** pair drawn after each stage = the CBAM block
- Head: `GAP` -> `FC(512 -> 10)` -> Softmax

Spatial dims annotated assume input 384x384 (Run 6 setting):
192 → 96 → 48 → 24 → 12. Swap to other resolutions by editing the
`s_filer` values in `plot_arch.py` (each ResNet stage halves the
spatial dim after stride-2 conv).

## Tweaks

- Change input image preview: edit the `to_input(...)` path in
  `plot_arch.py`. Defaults to PlotNeuralNet's bundled `cats.jpg` so
  the build does not fail when our project images are not present.
- Slim CBAM down to 1 slab (CAM only, ablation): delete the SAM block
  in the `cbam()` helper.
- Hide CBAM entirely: comment out the four `*cbam(...)` lines for the
  no-CBAM ablation diagram.

## Why PlotNeuralNet (vs Netron / torchviz)

- Static, paper-quality, vector PDF output. Looks the same in slides
  as in the report.
- Hand-controlled layout: emphasize the CBAM blocks the way the slides
  need.
- Netron renders the actual computation graph (every BN + ReLU shows
  up), which is too noisy for a presentation.
- torchviz is autograd-based; same noise problem.
