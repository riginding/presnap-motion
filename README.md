# NFL Pre-Snap Motion Spike

Baseline project to detect the snap frame in NFL All-22 video clips using frame-difference motion spikes.

## Contents

- `scripts/find_snap.py` — motion spike detector script
- `scripts/colab_notebook_cells.md` — ready-to-paste Google Colab cells
- `docs/NFL Pre-Snap Motion Spike Requirements.md` — copied requirements document
- `requirements.txt` — Python dependencies

## Quick Start (Local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/find_snap.py --video /path/to/clip.mp4 --out_dir ./out --plot
```

## Batch mode via environment variables

```bash
export PRESNAP_VIDEO_DIR='C:\Users\your_user\programming\nfl clips'
export PRESNAP_WINDOWS_RESULTS_DIR='C:\Users\your_user\programming\results'
export PRESNAP_WSL_VIDEO_CACHE_DIR='./wsl_video_cache'
export PRESNAP_OUT_DIR='./snap_output'
python scripts/find_snap.py --plot
```

CLI flags still override env vars.

## Quick Start (Google Colab)

1. Open `scripts/colab_notebook_cells.md`
2. Paste each cell into Colab
3. Upload `scripts/find_snap.py` and your `.mp4`

## Output

- Snap frame image: `*_snap_frame_<frame>.jpg`
- Motion scores: `*_motion_scores.npy`
- Optional plot: `*_motion_plot.png`

## Notes

- Uses grayscale + Gaussian blur + `cv2.absdiff`
- Detects first post-warmup threshold crossing as snap
- Supports dynamic thresholding (`median + sigma * MAD`) or fixed threshold
