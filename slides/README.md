# slides/ — Build the CS231 presentation deck

Generates `run68_deck.pptx` (15 slides) for the final CS231 presentation.
Featuring Run 6 (headline 0.873 macro F1) + Run 8 (pose-fusion, result
pending) per the outline approved in `SLIDES_OUTLINE.md`.

Built with [pptxgenjs](https://gitbrent.github.io/PptxGenJS/) following
the rules in `.claude/skills/academic-pptx/SKILL.md` (action titles,
ghost-deck argument, one exhibit per slide, citations on slide,
sandwich title/conclusions on dark navy).

## Build

```bash
cd slides
npm install pptxgenjs
node build_slides.js
```

Output: `slides/run68_deck.pptx`. Open in PowerPoint, Keynote, or
LibreOffice Impress to inspect / edit.

Re-run anytime — script is idempotent, overwrites the .pptx.

## Asset checklist

Drop PNG files into `slides/assets/` with these exact names. Missing
files render as dashed-border placeholders so layout iteration is
unblocked.

| Filename | Slide | What it shows | Source / how to make |
|---|---|---|---|
| `classes_grid_2x5.png` | 2 | 1 image per class c0..c9, labeled | Run `SLIDES_OUTLINE.md` Phụ lục C script |
| `arch_resnet18_cbam.png` | 4 | ResNet-18 + CBAM diagram | `arch_plot/plot_arch.tex` → render PDF → export PNG |
| `cbam_visual_c1_c5_c8.png` | 5 | 3 rows × 3 cols (original, SAM, overlay) for c1/c5/c8 | `SLIDES_OUTLINE.md` Phụ lục A script |
| `run6_training_curves.png` | 7 | train/val/EMA curves for Run 6 | `run6/eval/training_curves.png` |
| `run6_confusion_matrix.png` | 8 | 10×10 confusion matrix | `run6/eval/confusion_matrix.png` |
| `run7_vs_run6_delta.png` | 9 | Per-class F1 delta bar chart Run 7 − Run 6 | matplotlib bar chart, see snippet below |
| `arch_run8_pose_fusion.png` | 10 | Run 8 architecture: full CNN + pose MLP fusion | redraw `arch_plot/plot_arch.tex` for Run 8 |
| `run8_training_curves.png` | 12 | Run 8 curves (fill after training) | `run8/eval/training_curves.png` |
| `demo_high_conf.png` | 13 | Gradio screenshot, high-confidence | screenshot from `demo/app.py` |
| `demo_ood_low_conf.png` | 13 | Gradio screenshot, OOD low-conf warning | screenshot from `demo/app.py` |

## Pending updates (after Run 8 training completes)

In `build_slides.js`:

1. **Slide 12 action title** — replace `"[chờ kết quả train, điền sau]"` with the actual macro F1.
2. **Slide 12 comparison table** — fill `[TBD]` cell with Run 8 macro F1.
3. **Slide 12 target per-class** — swap targets for actual per-class deltas.
4. **Slide 9 → Slide 12 reading flow** — if Run 8 beats Run 6, consider promoting Run 8 visually (e.g. add ⭐ glyph to slide 12 action title and remove from slide 7).

In `slides/assets/`:

1. Drop `run8_training_curves.png` in.
2. Re-run `node build_slides.js`.

## Quick snippet — Slide 9 delta bar chart

Save as `arch_plot/make_run7_run6_delta.py`, run locally:

```python
import json, matplotlib.pyplot as plt, numpy as np
m6 = json.load(open("run6/eval/metrics.json"))
m7 = json.load(open("run7/eval/metrics.json"))
classes = [k for k in m6 if k.startswith("c") and "(" in k]
deltas = [m7[k]["f1-score"] - m6[k]["f1-score"] for k in classes]
labels = [k.split(" ")[0] for k in classes]
colors = ["#C00000" if d < 0 else "#548235" for d in deltas]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(labels, deltas, color=colors)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("F1 Δ (Run 7 − Run 6)")
ax.set_title("Run 7 regress trên 8/10 lớp")
plt.tight_layout()
plt.savefig("slides/assets/run7_vs_run6_delta.png", dpi=150)
```

## Asset extraction shortcuts (copy from existing eval outputs)

```powershell
# Copy ready-made Run 6 evals into the slide assets dir
cp run6/eval/training_curves.png   slides/assets/run6_training_curves.png
cp run6/eval/confusion_matrix.png  slides/assets/run6_confusion_matrix.png
# (when Run 8 done)
cp run8/eval/training_curves.png   slides/assets/run8_training_curves.png
```

## Editing principles (from academic-pptx skill)

- Each content slide has an **action title** (complete sentence stating the takeaway), not a topic label.
- One exhibit per slide.
- Body text ≤ 40 words / slide.
- Body font ≥ 20 pt (floor, not target).
- Cite borrowed material on the slide (small muted footer).
- Conclusions slide is the last main slide (stays on screen during Q&A).

If you want to override these (the skill is opinionated), edit the
`COLORS` / `FS` tokens at the top of `build_slides.js`.
