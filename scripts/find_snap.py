#!/usr/bin/env python3
"""
find_snap.py
Baseline motion-spike detector for NFL All-22 clips.

Usage (local or Colab):
  python find_snap.py --video /path/to/clip.mp4 --out_dir ./out

Optional:
  --threshold 2.5e6      # fixed absolute threshold (motion score)
  --sigma 6.0            # dynamic threshold = median + sigma * MAD
  --warmup_frames 20     # ignore first N frames for detection
  --plot                 # save motion score plot
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Detect NFL snap frame via motion spike.")
    p.add_argument("--video", required=True, help="Path to input .mp4 clip")
    p.add_argument("--out_dir", default="./snap_output", help="Output directory")
    p.add_argument("--blur_kernel", type=int, default=5, help="Gaussian blur kernel (odd integer)")
    p.add_argument("--frame_step", type=int, default=1, help="Process every Nth frame")
    p.add_argument("--warmup_frames", type=int, default=20, help="Ignore first N processed frames for spike detection")
    p.add_argument("--threshold", type=float, default=None, help="Fixed motion threshold (optional)")
    p.add_argument("--sigma", type=float, default=6.0, help="Dynamic threshold = median + sigma * MAD")
    p.add_argument("--plot", action="store_true", help="Save motion score plot (requires matplotlib)")
    return p.parse_args()


def ensure_odd(x: int) -> int:
    return x if x % 2 == 1 else x + 1


def compute_motion_scores(video_path: str, blur_kernel: int, frame_step: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    scores = []
    processed_frame_indices = []
    frame_cache = {}  # keep processed original BGR frames for saving snap image

    prev_gray = None
    idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        if idx % frame_step != 0:
            continue

        frame_cache[idx] = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        diff = cv2.absdiff(gray, prev_gray)
        motion_score = float(np.sum(diff))

        scores.append(motion_score)
        processed_frame_indices.append(idx)

        prev_gray = gray

    cap.release()

    return {
        "scores": np.array(scores, dtype=np.float64),
        "frame_indices": np.array(processed_frame_indices, dtype=np.int64),
        "frame_cache": frame_cache,
        "fps": fps,
        "total_frames": total_frames,
    }


def detect_spike(scores: np.ndarray, warmup_frames: int, fixed_threshold: float | None, sigma: float):
    if len(scores) == 0:
        raise RuntimeError("No motion scores computed (video may be too short or unreadable).")

    warmup = min(max(warmup_frames, 0), len(scores) - 1)
    baseline = scores[: max(warmup, 1)]

    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    robust_sigma = 1.4826 * mad

    if fixed_threshold is None:
        threshold = median + sigma * robust_sigma
    else:
        threshold = fixed_threshold

    # first crossing after warmup
    for i in range(warmup, len(scores)):
        if scores[i] >= threshold:
            return i, threshold, median, robust_sigma

    # fallback: global max
    i = int(np.argmax(scores))
    return i, threshold, median, robust_sigma


def save_plot(scores, snap_i, threshold, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(scores, label="motion_score")
    plt.axhline(threshold, color="orange", linestyle="--", label=f"threshold={threshold:.0f}")
    plt.axvline(snap_i, color="red", linestyle="-", label=f"snap_index={snap_i}")
    plt.title("Motion Score Spike Detection")
    plt.xlabel("Processed frame index")
    plt.ylabel("Sum(abs(frame_t - frame_t-1))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    blur_kernel = ensure_odd(args.blur_kernel)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = compute_motion_scores(args.video, blur_kernel, args.frame_step)
    scores = result["scores"]
    frame_indices = result["frame_indices"]
    frame_cache = result["frame_cache"]
    fps = result["fps"]

    snap_i, threshold, median, robust_sigma = detect_spike(
        scores=scores,
        warmup_frames=args.warmup_frames,
        fixed_threshold=args.threshold,
        sigma=args.sigma,
    )

    snap_frame_idx = int(frame_indices[snap_i])
    snap_time_sec = (snap_frame_idx / fps) if fps > 0 else None

    snap_frame = frame_cache.get(snap_frame_idx)
    if snap_frame is None:
        raise RuntimeError("Snap frame image missing from cache.")

    video_stem = Path(args.video).stem
    snap_img_path = out_dir / f"{video_stem}_snap_frame_{snap_frame_idx}.jpg"
    cv2.imwrite(str(snap_img_path), snap_frame)

    scores_path = out_dir / f"{video_stem}_motion_scores.npy"
    np.save(scores_path, scores)

    if args.plot:
        plot_path = out_dir / f"{video_stem}_motion_plot.png"
        save_plot(scores, snap_i, threshold, plot_path)

    print("=== SNAP DETECTION RESULT ===")
    print(f"video:              {args.video}")
    print(f"fps:                {fps:.3f}")
    print(f"total_frames:       {result['total_frames']}")
    print(f"processed_frames:   {len(scores)}")
    print(f"baseline_median:    {median:.2f}")
    print(f"baseline_robustσ:   {robust_sigma:.2f}")
    print(f"threshold:          {threshold:.2f}")
    print(f"snap_processed_i:   {snap_i}")
    print(f"snap_frame_index:   {snap_frame_idx}")
    if snap_time_sec is not None:
        print(f"snap_timestamp_sec: {snap_time_sec:.3f}")
    print(f"snap_image:         {snap_img_path}")
    print(f"scores_npy:         {scores_path}")


if __name__ == "__main__":
    main()
