# NFL Pre-Snap Motion Spike - Requirements Document

## 1. Project Overview
**Goal:** Automatically identify the exact frame in an NFL All-22 video clip where the ball is snapped.
**Method:** "Motion Spike" detection. Calculate frame-to-frame pixel differences to identify the moment the offensive line transitions from a static set to explosive movement.
**Role in Pipeline:** This is Step 1 of a larger ML pipeline to identify offensive personnel formations. Isolating the pre-snap frame allows downstream models to analyze a single, static image rather than processing an entire video.

## 2. Technical Approach
*   **Technique:** Absolute Difference (Frame differencing) or Dense Optical Flow.
*   **Trigger Condition:** Calculate a "Motion Score" for every frame. The snap is identified when the score crosses a pre-defined threshold (spikes from ~0 to a high number).
*   **Fallback/Alternative:** YOLO bounding box center tracking (if simple pixel math is too noisy due to camera panning/zooming).

## 3. Hardware & Compute Requirements
*   **Local Development:** Any standard laptop (Mac/Windows). OpenCV frame differencing is highly optimized and CPU-efficient.
*   **Cloud Testing/Prototyping:** Google Colab (Free Tier). Provides a free Jupyter Notebook environment in the browser.
*   **Scaling (Future):** RunPod.io or Vast.ai (approx. $0.20 - $0.40/hour) for bulk processing thousands of clips.

## 4. Software & Library Requirements
*   **Language:** Python 3.10+
*   **Core Libraries:**
    *   `opencv-python` (`cv2`): For reading the video file, converting frames to grayscale, and calculating absolute differences.
    *   `numpy`: For fast array and matrix math (summing the motion score).
    *   `matplotlib` (Optional): To plot the motion score across frames and visually verify the "spike".

## 5. Required Scripts
We need a minimal pipeline consisting of a single Python script (`find_snap.py`) that handles:
1.  **Input:** Path to an `.mp4` All-22 clip.
2.  **Process:**
    *   Load video and iterate through frames.
    *   Convert frames to grayscale (reduces compute).
    *   Apply Gaussian Blur (reduces visual noise/artifacts).
    *   Compute the absolute difference (`cv2.absdiff`) between the current frame and the previous frame.
    *   Sum the pixel differences to generate a `motion_score`.
3.  **Output:** The frame number (or timestamp) where the `motion_score` significantly breaches a threshold. Saves that specific frame as a `.jpg` for the next ML step.

## 6. Data Requirements
*   **Test Data:** 3 to 5 sample All-22 clips (short clips, ideally 5-15 seconds per play).
*   **Resolution:** 720p or 1080p is sufficient.

## 7. Next Steps & Action Items
*   [ ] Write baseline `find_snap.py` script.
*   [ ] Set up a Google Colab notebook or local Python environment.
*   [ ] Upload sample clips and test the script.
*   [ ] Tune the sensitivity (blur size, frame step gap, and threshold trigger) to handle camera panning.