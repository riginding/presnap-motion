# Google Colab - Ready to Paste Cells

## Cell 1 — Install deps
```python
!pip -q install opencv-python matplotlib numpy
```

## Cell 2 — Upload script + video
```python
from google.colab import files

print("Upload find_snap.py and your .mp4 clip")
uploaded = files.upload()
print("Uploaded:", list(uploaded.keys()))
```

## Cell 3 — Pick input + run detection
```python
import os
import glob

py_files = [f for f in glob.glob("/content/*.py") if os.path.basename(f) == "find_snap.py"]
mp4_files = glob.glob("/content/*.mp4")

assert py_files, "find_snap.py not found in /content. Upload it first."
assert mp4_files, "No .mp4 found in /content. Upload your clip first."

script = py_files[0]
video = mp4_files[0]
out_dir = "/content/out"

print("Script:", script)
print("Video:", video)

!python "$script" \
  --video "$video" \
  --out_dir "$out_dir" \
  --plot \
  --sigma 6.0 \
  --blur_kernel 5 \
  --frame_step 1 \
  --warmup_frames 20
```

## Cell 4 — Show outputs inline
```python
import glob
from IPython.display import Image, display
import os

out_dir = "/content/out"
print("Output files:")
for f in sorted(glob.glob(f"{out_dir}/*")):
    print("-", os.path.basename(f))

snap_imgs = sorted(glob.glob(f"{out_dir}/*_snap_frame_*.jpg"))
plots = sorted(glob.glob(f"{out_dir}/*_motion_plot.png"))

if snap_imgs:
    print("\nSnap frame:")
    display(Image(filename=snap_imgs[-1], width=1200))

if plots:
    print("\nMotion plot:")
    display(Image(filename=plots[-1], width=1200))
```

## Cell 5 — Download results
```python
from google.colab import files
import glob

for f in glob.glob("/content/out/*"):
    files.download(f)
```
