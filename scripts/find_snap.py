#!/usr/bin/env python3
"""
find_snap.py
Baseline motion-spike detector for NFL All-22 clips.

Single-video usage:
  python find_snap.py --video /path/to/clip.mp4 --out_dir ./out

Batch mode (dirs configurable via env vars):
  python find_snap.py --plot
"""

import argparse
import csv
import os
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}


def parse_args():
    p = argparse.ArgumentParser(description="Detect NFL snap frame via motion spike.")

    # single-file mode
    p.add_argument("--video", default=None, help="Path to a single input video")

    # batch mode: copy videos from source dir -> WSL cache -> process all
    p.add_argument(
        "--video_dir",
        default=os.getenv("PRESNAP_VIDEO_DIR", "./videos_in"),
        help="Directory containing videos (Windows or WSL path). Env: PRESNAP_VIDEO_DIR",
    )
    p.add_argument(
        "--windows_results_dir",
        default=os.getenv("PRESNAP_WINDOWS_RESULTS_DIR", "./results_out"),
        help="Destination directory for final results (Windows or WSL path). Env: PRESNAP_WINDOWS_RESULTS_DIR",
    )
    p.add_argument(
        "--wsl_video_cache_dir",
        default=os.getenv("PRESNAP_WSL_VIDEO_CACHE_DIR", "./wsl_video_cache"),
        help="WSL-local directory where input videos are copied before processing. Env: PRESNAP_WSL_VIDEO_CACHE_DIR",
    )

    p.add_argument(
        "--out_dir",
        default=os.getenv("PRESNAP_OUT_DIR", "./snap_output"),
        help="WSL output directory. Env: PRESNAP_OUT_DIR",
    )
    p.add_argument("--blur_kernel", type=int, default=5, help="Gaussian blur kernel (odd integer)")
    p.add_argument("--frame_step", type=int, default=1, help="Process every Nth frame")
    p.add_argument("--warmup_frames", type=int, default=20, help="Ignore first N processed frames for spike detection")
    p.add_argument("--threshold", type=float, default=None, help="Fixed motion threshold (optional)")
    p.add_argument("--sigma", type=float, default=6.0, help="Dynamic threshold = median + sigma * MAD")
    p.add_argument("--plot", action="store_true", help="Save motion score plot (requires matplotlib)")
    return p.parse_args()


def ensure_odd(x: int) -> int:
    return x if x % 2 == 1 else x + 1


def normalize_windows_path(path_like: str) -> str:
    """Convert Windows path (e.g. C:\\foo\\bar) to WSL path (/mnt/c/foo/bar)."""
    m = re.match(r"^([a-zA-Z]):\\(.*)$", path_like)
    if not m:
        return path_like
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def list_video_files(dir_path: Path):
    return sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def copy_videos_to_wsl(src_dir: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    videos = list_video_files(src_dir)
    copied = []
    for src in videos:
        dst = cache_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def compute_motion_scores(video_path: str, blur_kernel: int, frame_step: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    scores = []
    processed_frame_indices = []
    frame_cache = {}

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

    for i in range(warmup, len(scores)):
        if scores[i] >= threshold:
            return i, threshold, median, robust_sigma

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


def format_time_for_filename(time_sec: float | None) -> str:
    if time_sec is None:
        return "t_na"
    # filesystem-friendly, e.g. t6p166s instead of t6.166s
    return f"t{time_sec:.3f}s".replace(".", "p")


def load_frame_at_index(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def process_one_video(video_path: Path, out_dir: Path, args, blur_kernel: int):
    result = compute_motion_scores(str(video_path), blur_kernel, args.frame_step)
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

    half_sec_offset_frames = int(round(0.5 * fps)) if fps > 0 else 0
    pre_snap_frame_idx = max(0, snap_frame_idx - half_sec_offset_frames)
    post_snap_frame_idx = min(result["total_frames"] - 1, snap_frame_idx + half_sec_offset_frames)

    pre_snap_time_sec = (pre_snap_frame_idx / fps) if fps > 0 else None
    post_snap_time_sec = (post_snap_frame_idx / fps) if fps > 0 else None

    snap_frame = frame_cache.get(snap_frame_idx)
    if snap_frame is None:
        snap_frame = load_frame_at_index(video_path, snap_frame_idx)
    if snap_frame is None:
        raise RuntimeError("Could not load snap frame image.")

    pre_snap_frame = frame_cache.get(pre_snap_frame_idx)
    if pre_snap_frame is None:
        pre_snap_frame = load_frame_at_index(video_path, pre_snap_frame_idx)
    if pre_snap_frame is None:
        raise RuntimeError("Could not load pre-snap frame image.")

    post_snap_frame = frame_cache.get(post_snap_frame_idx)
    if post_snap_frame is None:
        post_snap_frame = load_frame_at_index(video_path, post_snap_frame_idx)
    if post_snap_frame is None:
        raise RuntimeError("Could not load post-snap frame image.")

    video_stem = video_path.stem
    snap_time_tag = format_time_for_filename(snap_time_sec)
    pre_snap_time_tag = format_time_for_filename(pre_snap_time_sec)
    post_snap_time_tag = format_time_for_filename(post_snap_time_sec)

    snap_img_path = out_dir / f"{video_stem}_{snap_time_tag}_snap_frame_{snap_frame_idx}.jpg"
    pre_snap_img_path = out_dir / f"{video_stem}_{pre_snap_time_tag}_presnap_frame_{pre_snap_frame_idx}.jpg"
    post_snap_img_path = out_dir / f"{video_stem}_{post_snap_time_tag}_postsnap_frame_{post_snap_frame_idx}.jpg"

    cv2.imwrite(str(snap_img_path), snap_frame)
    cv2.imwrite(str(pre_snap_img_path), pre_snap_frame)
    cv2.imwrite(str(post_snap_img_path), post_snap_frame)

    scores_path = out_dir / f"{video_stem}_{snap_time_tag}_motion_scores.npy"
    np.save(scores_path, scores)

    plot_path = None
    if args.plot:
        plot_path = out_dir / f"{video_stem}_{snap_time_tag}_motion_plot.png"
        save_plot(scores, snap_i, threshold, plot_path)

    return {
        "video": str(video_path),
        "fps": fps,
        "total_frames": result["total_frames"],
        "processed_frames": len(scores),
        "baseline_median": median,
        "baseline_robust_sigma": robust_sigma,
        "threshold": threshold,
        "snap_processed_i": snap_i,
        "snap_frame_index": snap_frame_idx,
        "snap_timestamp_sec": snap_time_sec,
        "snap_image": str(snap_img_path),
        "presnap_frame_index": pre_snap_frame_idx,
        "presnap_timestamp_sec": pre_snap_time_sec,
        "presnap_image": str(pre_snap_img_path),
        "postsnap_frame_index": post_snap_frame_idx,
        "postsnap_timestamp_sec": post_snap_time_sec,
        "postsnap_image": str(post_snap_img_path),
        "scores_npy": str(scores_path),
        "plot": str(plot_path) if plot_path else "",
    }


def write_summary_csv(rows, out_path: Path):
    fields = [
        "video",
        "fps",
        "total_frames",
        "processed_frames",
        "baseline_median",
        "baseline_robust_sigma",
        "threshold",
        "snap_processed_i",
        "snap_frame_index",
        "snap_timestamp_sec",
        "snap_image",
        "presnap_frame_index",
        "presnap_timestamp_sec",
        "presnap_image",
        "postsnap_frame_index",
        "postsnap_timestamp_sec",
        "postsnap_image",
        "scores_npy",
        "plot",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def copy_results_to_windows(wsl_out_dir: Path, windows_results_dir: Path):
    windows_results_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in wsl_out_dir.iterdir():
        if p.is_file():
            dst = windows_results_dir / p.name
            shutil.copy2(p, dst)
            copied.append(dst)
    return copied


def main():
    args = parse_args()

    blur_kernel = ensure_odd(args.blur_kernel)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    if args.video:
        # single-video mode
        video_path = Path(normalize_windows_path(args.video))
        row = process_one_video(video_path, out_dir, args, blur_kernel)
        rows.append(row)
    else:
        # batch mode
        src_dir = Path(normalize_windows_path(args.video_dir))
        if not src_dir.exists():
            raise RuntimeError(f"Input directory does not exist: {src_dir}")

        cache_dir = Path(args.wsl_video_cache_dir)
        copied_videos = copy_videos_to_wsl(src_dir, cache_dir)
        if not copied_videos:
            raise RuntimeError(f"No video files found in: {src_dir}")

        for video_path in copied_videos:
            row = process_one_video(video_path, out_dir, args, blur_kernel)
            rows.append(row)

    summary_path = out_dir / "results_summary.csv"
    write_summary_csv(rows, summary_path)

    windows_results_dir = Path(normalize_windows_path(args.windows_results_dir))
    copied_results = copy_results_to_windows(out_dir, windows_results_dir)

    print("=== SNAP DETECTION RESULTS ===")
    print(f"processed_videos:    {len(rows)}")
    print(f"wsl_out_dir:         {out_dir}")
    print(f"summary_csv:         {summary_path}")
    print(f"windows_results_dir: {windows_results_dir}")
    print(f"copied_results:      {len(copied_results)}")
    for r in rows:
        ts = "n/a" if r["snap_timestamp_sec"] is None else f"{r['snap_timestamp_sec']:.3f}"
        print(
            f"- {Path(r['video']).name}: "
            f"snap_frame={r['snap_frame_index']} timestamp={ts}s "
            f"presnap_frame={r['presnap_frame_index']} "
            f"postsnap_frame={r['postsnap_frame_index']}"
        )


if __name__ == "__main__":
    main()
