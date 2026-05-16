# DEMO_HOWTO — Run the project locally on your laptop

Gradio web demo for Run 6 (single-stream), Run 7 (two-stream), and
Run 8 (pose-fusion). CPU-only inference is fine — ResNet-18 takes
~200 ms per image on CPU; pose-fusion adds ~50 ms MediaPipe overhead.

```
demo/
├── app.py                  # Run 6 demo (single stream)
├── app_twostream.py        # Run 7 demo (two stream)
├── app_posefusion.py       # Run 8 demo (single CNN + MediaPipe pose)
├── DEMO_HOWTO.md           # this file
├── requirements-demo.txt   # subset of repo requirements
├── checkpoints/            # drop best.pt / run7_best.pt / run8_best.pt here
├── splits/                 # drop stats.json here
└── examples/               # optional sample images
```

The scripts inject the repo root onto `sys.path`, so they import the
shared modules (`augment.py`, `eval.py`, `model.py`, etc.) without
needing the demo files to live next to them.

---

## 1. Install dependencies (one time)

From the repo root (`D:\UIT homework\CVV`):

```powershell
# CPU-only PyTorch (smaller install, no CUDA needed for inference)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Everything else
pip install -r demo/requirements-demo.txt
```

Or use the full repo `requirements.txt` if you also want to
re-run eval/training scripts locally.

Verify:
```powershell
python -c "import torch, gradio, cv2; print(torch.__version__, gradio.__version__, cv2.__version__)"
```

---

## 2. Download artifacts from Kaggle

### Run 6
- `/kaggle/working/run6/best.pt`        → `demo/checkpoints/best.pt`        (~45 MB)
- `/kaggle/working/splits/stats.json`   → `demo/splits/stats.json`          (<1 KB)

### Run 7
- `/kaggle/working/run7/best.pt`        → `demo/checkpoints/run7_best.pt`   (~270 MB)
- `/kaggle/working/splits/stats.json`   → `demo/splits/stats.json`          (same file as Run 6)

### Run 8
- `/kaggle/working/run8/best.pt`        → `demo/checkpoints/run8_best.pt`   (~135 MB)
- `/kaggle/working/splits/stats.json`   → `demo/splits/stats.json`          (same file)
- Pose features are extracted at inference time from the uploaded image
  (no need to ship `pose.parquet` to the demo machine)

In a Kaggle cell:
```python
from IPython.display import FileLink
FileLink("/kaggle/working/run6/best.pt")        # click → save
FileLink("/kaggle/working/run7/best.pt")        # click → save
FileLink("/kaggle/working/splits/stats.json")
```

Or zip + download (Cell 9 in `RUN7_HOWTO.md` does this for Run 7).

**Smaller Run 7 ckpt for sharing** (~90 MB, EMA only): see
`RUN7_HOWTO.md` §9.

---

## 3. Run the Run 6 demo

```powershell
cd demo
python app.py --ckpt checkpoints/best.pt --stats splits/stats.json
```

Console:
```
Running on local URL:  http://127.0.0.1:7860
```

Open the URL in your browser. Upload a driver image (side view, full
torso visible) or click an example. You get:
- Top-1 prediction + confidence
- Per-class probability bars
- CBAM spatial-attention heatmap overlay

Public link (Gradio share, 72 h lifetime):
```powershell
python app.py --ckpt checkpoints/best.pt --stats splits/stats.json --share
```

Change port if 7860 is busy:
```powershell
python app.py --ckpt checkpoints/best.pt --stats splits/stats.json --server-port 7861
```

---

## 4. Run the Run 7 demo

Same shape, different script + ckpt:

```powershell
cd demo
python app_twostream.py --ckpt checkpoints/run7_best.pt --stats splits/stats.json
```

UI shows two image panels:
- **Full-stream CBAM attention** — heatmap from layer4 CBAM of the
  full-frame branch (where the model is looking globally).
- **Face-stream input (TopCrop)** — the top 50% of the frame that the
  face branch actually sees.

Match the training crop fraction with `--top-frac` (default 0.50).
Override if your Run 7 ckpt used a different value:
```powershell
python app_twostream.py --ckpt checkpoints/run7_best.pt --stats splits/stats.json --top-frac 0.55
```

---

## 4b. Run the Run 8 demo (pose fusion)

Install MediaPipe (pinned — newer wheels lazy-load `mp.solutions` and
break on some Python builds):

```powershell
pip install mediapipe==0.10.13 polars
```

Run:

```powershell
cd demo
python app_posefusion.py --ckpt checkpoints/run8_best.pt --stats splits/stats.json
```

UI panels:
- **CBAM spatial attention** — heatmap from layer4 of the single
  ResNet-18 backbone (where the CNN is looking).
- **MediaPipe pose landmarks** — dots + skeleton overlaid on the
  uploaded image showing what landmarks the pose vector encodes.
  Colors: red = head, green = arms/shoulders, blue = hands/fingers,
  yellow = hips.
- **Verdict** — top-1 + confidence + interpretable pose cue values
  (yaw, lap proximity for both hands, right-arm reach).

If MediaPipe fails to detect a pose (heavy occlusion, unusual angle),
the verdict notes "pose detection failed — falling back to CNN-only".
The pose vector goes to zero, and the model's gating bit (p7 = 0)
makes it ignore pose for that prediction.

Auto-dispatch: `app_posefusion.py` calls `load_model_for_ckpt()` which
reads the `pose_fusion` flag in the ckpt metadata and refuses to load
non-pose-fusion checkpoints — friendly error pointing at the right
script.

---

## 5. Optional: preload example images

Drop 5-10 JPGs into `demo/examples/`. They'll appear as clickable
thumbnails under the input panel.

Suggested mix:
- 3-4 frames from `imgs/train/c*/` (in-distribution)
- 2-3 hand-picked OOD images (different camera angle, daytime
  highway shot, etc.) to demonstrate the "unsure" fallback.

The unsure-threshold is hardcoded at 0.40 (see `UNSURE_THRESHOLD` in
both apps). Inputs with top-1 confidence below 0.40 get a warning.

---

## 6. Verify before opening to friends

Smoke test from CLI:

```powershell
cd demo
python app.py --ckpt checkpoints/best.pt --stats splits/stats.json --server-port 7862
```

In a second terminal:
```powershell
curl http://127.0.0.1:7862
```
Should return HTML. If yes, demo is live.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Checkpoint not found: checkpoints/best.pt` | Path wrong or ckpt not copied | Verify with `ls checkpoints/`; rerun from `demo/` |
| `Stats file not found: splits/stats.json` | `stats.json` not copied | Re-download from Kaggle (`/kaggle/working/splits/stats.json`) |
| `size mismatch for ...` loading ckpt | Run 7 ckpt loaded into `app.py` (or vice versa) | Use the matching script: Run 6 → `app.py`, Run 7 → `app_twostream.py` |
| `ModuleNotFoundError: augment` | Running from wrong directory | `cd demo` first; the `sys.path` injection is relative to `demo/app*.py` |
| `ModuleNotFoundError: cv2` | `opencv-python` not installed | `pip install opencv-python` |
| First inference takes ~5 s | PyTorch JIT warmup | Normal; subsequent calls are ~200 ms |
| Port 7860 busy | Prior demo still running | `--server-port 7861` or kill the prior `python` process |
| Browser shows "no overlay" | `last_spatial_attention()` returned None | Ckpt trained with `--no-cbam`; rebuild without that flag |

---

## 8. HuggingFace Spaces (optional, for hosted demo)

Both apps support env var configuration so they work out-of-the-box
on Spaces:

```
CKPT_PATH=run7_best.pt
STATS_PATH=stats.json
EXAMPLES_DIR=examples
```

Spaces hard-caps free uploads at 50 MB without LFS. Options:
- Run 6 `best.pt` is ~45 MB → fits directly.
- Run 7 `best.pt` is ~270 MB → use `best_demo.pt` (~90 MB, EMA-only)
  via the script in `RUN7_HOWTO.md` §9, and enable Git LFS on the
  Space repo.

Pick the entrypoint by renaming the matching file to `app.py` in the
Space root, or by setting `app_file: app_twostream.py` in
`README.md` frontmatter.

---

## 9. Decision matrix

| Goal | Use |
|---|---|
| Class presentation, in-room laptop | `app.py` (Run 6), no `--share` |
| Send link to classmate or instructor | `app.py --share` or `app_twostream.py --share` |
| Two-stream ablation comparison in slides | run both, swap browser tabs |
| Long-lived public URL | HuggingFace Spaces |
