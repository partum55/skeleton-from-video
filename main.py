"""
skeleton extraction from video — main entry point.
real-time exercise detection and repetition counting using mediapipe pose estimation.
"""

# Configure environment BEFORE importing any libraries
import os
import sys
import contextlib

# Fix 1: Qt/Wayland compatibility - prevent Qt from seeing Wayland session
# Qt checks XDG_SESSION_TYPE and prints a warning if it's 'wayland' but QT_QPA_PLATFORM is 'xcb'
# By removing the variable, Qt won't know about Wayland and won't print the warning
if os.environ.get("XDG_SESSION_TYPE") == "wayland":
    os.environ.pop("XDG_SESSION_TYPE", None)
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Fix 2: MediaPipe/TensorFlow C++ logging - must be set before imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Fix 3: Suppress C++ stderr output during library initialization
# MediaPipe's C++ code writes to stderr before Python can configure logging
@contextlib.contextmanager
def suppress_stderr():
    """Temporarily redirect stderr to /dev/null."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

# Fix 4: Initialize absl logging to prevent Python-level warnings
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.use_python_logging()

import argparse
import signal
import time
from collections import deque

# Import cv2 with stderr suppression to avoid Qt/Wayland warning
with suppress_stderr():
    import cv2

import numpy as np

# Import skeleton module (which imports MediaPipe) with stderr suppression
# to avoid C++ warnings during TensorFlow Lite initialization
with suppress_stderr():
    from src.skeleton import (
        PoseEstimator,
        build_adjacency_matrix,
        build_graph_laplacian,
        LandmarksTemporalFilter,
    )
from src.normalize import normalize_skeleton, NormalizedSkeletonTemporalFilter
from src.features import (
    compute_key_angles,
    compute_body_position_features,
    BodyPositionTracker,
    flatten_skeleton,
    euclidean_distance,
    cosine_similarity,
    AngleTemporalSmoother,
    get_primary_angle,
)
from src.classify import ExerciseClassifier
from src.visualize import draw_skeleton, draw_info_overlay, create_angle_plot
from src.pca import PCA
from src.repetition import count_reps_and_classify_with_confidence, fuse_exercise_labels


# global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def run_live(source: int | str = 0, show_angles: bool = True,
             show_plot: bool = True, apply_rotation: bool = False,
             frame_width: int = 1280, frame_height: int = 720,
             window_width: int = 1600, window_height: int = 900,
             mirror: bool = True):
    """run the full pipeline on a live webcam or video file.

    pipeline: video -> skeleton extraction -> normalization -> angles -> classify -> visualize
    """
    global _shutdown_requested
    _shutdown_requested = False

    # install signal handler for graceful Ctrl+C
    prev_handler = signal.signal(signal.SIGINT, _signal_handler)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"error: cannot open video source '{source}'")
        signal.signal(signal.SIGINT, prev_handler)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(max(320, frame_width)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(max(240, frame_height)))

    window_name = "Skeleton Extraction from Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window_name, max(640, window_width), max(480, window_height))
    is_fullscreen = False

    # Suppress MediaPipe C++ warnings during model initialization
    with suppress_stderr():
        estimator = PoseEstimator(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    classifier = ExerciseClassifier(default_fps=30.0)
    # Reduced EMA cascade with higher alpha for faster response:
    # - landmarks_filter: primary jitter reduction, moderate smoothing
    # - normalized_filter: secondary smoothing for stable normalization
    # - angle_smoother: final smoothing, higher alpha for responsiveness
    landmarks_filter = LandmarksTemporalFilter(min_visibility=0.4, ema_alpha=0.50)  # increased from 0.35
    normalized_filter = NormalizedSkeletonTemporalFilter(alpha=0.60)  # increased from 0.40
    angle_smoother = AngleTemporalSmoother(alpha=0.55)  # increased from 0.45
    body_tracker = BodyPositionTracker(history_frames=5)  # NEW: track body position for jump detection

    # precompute the adjacency matrix and laplacian (static graph structure)
    A = build_adjacency_matrix()
    L = build_graph_laplacian(A)
    print(f"adjacency matrix shape: {A.shape}, laplacian shape: {L.shape}")
    print(f"laplacian smallest eigenvalues: {np.sort(np.linalg.eigvalsh(L))[:3]}")

    prev_time = time.time()
    primary_angle_history: list[float] = []

    # PCA + repetition counting state
    WARMUP_SECONDS = 2.0
    REFIT_INTERVAL_SECONDS = 1.0
    BUFFER_SECONDS = 10.0
    skeleton_buffer: deque = deque()
    buffer_duration_s: float = 0.0
    refit_elapsed_s: float = 0.0
    pca_label: str | None = None
    pca_confidence: float = 0.0
    warming_up: bool = True
    pca_reference_components: np.ndarray | None = None

    # UI state
    prev_pca_reps: int = 0
    rep_flash_frames: int = 0   # counts down; >0 means flash is active

    print("press 'q' to quit, 'r' to reset rep counter, 'f' to toggle fullscreen")

    first_frame = True  # track first frame for one-time warning suppression
    try:
        while not _shutdown_requested:
            # stop immediately if user closed the window via window controls (X button)
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            ret, frame = cap.read()
            if not ret:
                # loop video files
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Suppress MediaPipe warning on first inference only
            if first_frame:
                with suppress_stderr():
                    landmarks = estimator.extract_landmarks(frame_rgb)
                first_frame = False
            else:
                landmarks = estimator.extract_landmarks(frame_rgb)

            curr_time = time.time()
            dt = max(curr_time - prev_time, 1e-3)
            fps = 1.0 / dt
            prev_time = curr_time

            exercise = classifier.current_exercise
            reps = classifier.rep_count
            angles = None

            if landmarks is not None:
                filtered_landmarks = landmarks_filter.update(landmarks)
                if filtered_landmarks is None:
                    continue

                # draw raw skeleton overlay
                frame = draw_skeleton(frame, filtered_landmarks)

                # normalize skeleton (translation + scaling + optional rotation)
                normalized = normalize_skeleton(filtered_landmarks, apply_procrustes=apply_rotation)
                normalized = normalized_filter.update(normalized)

                # buffer normalized skeleton for PCA-based rep counting
                flat = flatten_skeleton(normalized)   # (66,)
                skeleton_buffer.append((flat, dt))
                buffer_duration_s += dt
                while len(skeleton_buffer) > 1 and buffer_duration_s > BUFFER_SECONDS:
                    _, old_dt = skeleton_buffer.popleft()
                    buffer_duration_s -= old_dt

                refit_elapsed_s += dt
                warming_up = buffer_duration_s < WARMUP_SECONDS

                if (not warming_up) and refit_elapsed_s >= REFIT_INTERVAL_SECONDS:
                    try:
                        X = np.array([sample for sample, _ in skeleton_buffer], dtype=np.float64)   # (T, 66)
                        pca = PCA(variance_threshold=0.95)
                        Z = pca.fit_transform(X, reference_components=pca_reference_components)       # (T, k)
                        pca_reference_components = pca.components_.copy() if pca.components_ is not None else None
                        _, pca_label, _, pca_confidence = count_reps_and_classify_with_confidence(Z)
                        refit_elapsed_s = 0.0
                    except Exception:
                        pass

                # compute joint angles using the dot product formula
                angles_raw = compute_key_angles(normalized)
                angles = angle_smoother.update(angles_raw)
                
                # compute body position features for improved detection
                body_features_raw = compute_body_position_features(normalized)
                body_motion = body_tracker.update(body_features_raw)
                body_features = {**body_features_raw, **body_motion}

                # update classifier with angles AND body features
                exercise, reps = classifier.update(angles, dt=dt, body_features=body_features)

                # track primary angle for the plot
                avg = get_primary_angle(angles, exercise)
                primary_angle_history.append(avg)

            fused_exercise = fuse_exercise_labels(
                angle_label=exercise,
                pca_label=pca_label,
                pca_confidence=pca_confidence,
                min_pca_confidence=0.65,
            )

            # rep flash: trigger on new rep, count down for ~10 frames
            display_reps = reps
            if display_reps > prev_pca_reps:
                rep_flash_frames = 10
            prev_pca_reps = display_reps
            rep_flash = rep_flash_frames > 0
            if rep_flash_frames > 0:
                rep_flash_frames -= 1

            # warm-up progress for the progress bar
            warmup_progress = min(1.0, buffer_duration_s / max(WARMUP_SECONDS, 1e-6))

            # draw info overlay
            frame = draw_info_overlay(
                frame,
                exercise=fused_exercise,
                rep_count=display_reps,
                angles=angles if show_angles else None,
                fps=fps,
                warming_up=warming_up,
                warmup_progress=warmup_progress,
                rep_flash=rep_flash,
            )

            # draw angle plot top-right (avoids overlap with skeleton body)
            if show_plot and len(primary_angle_history) > 1:
                from src.visualize import TOP_BAR_H
                plot_title = f"{fused_exercise or 'knee'} angle"
                plot_img = create_angle_plot(primary_angle_history, title=plot_title)
                ph, pw = plot_img.shape[:2]
                fh, fw = frame.shape[:2]
                x_start = fw - pw - 10
                y_start = TOP_BAR_H + 10
                if y_start + ph < fh and x_start > 0:
                    frame[y_start:y_start + ph, x_start:x_start + pw] = plot_img

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                classifier.reset()
                landmarks_filter.reset()
                normalized_filter.reset()
                angle_smoother.reset()
                body_tracker.reset()
                primary_angle_history.clear()
                skeleton_buffer.clear()
                buffer_duration_s = 0.0
                refit_elapsed_s = 0.0
                pca_label = None
                pca_confidence = 0.0
                pca_reference_components = None
                warming_up = True
                prev_pca_reps = 0
                rep_flash_frames = 0
                print("reset rep counter")
            elif key == ord("f"):
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
                )

    finally:
        # always clean up, even on Ctrl+C or exception
        estimator.close()
        cap.release()
        cv2.destroyAllWindows()
        # pump event loop so windows actually close
        for _ in range(5):
            cv2.waitKey(1)
        signal.signal(signal.SIGINT, prev_handler)
        print("\nshutdown complete.")


def run_analysis(video_path: str, output_path: str | None = None):
    """run offline analysis on a video file — extract skeletons and save data."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"error: cannot open video '{video_path}'")
        return

    with suppress_stderr():
        estimator = PoseEstimator()
    all_skeletons = []
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = estimator.extract_xy(frame_rgb)

            if landmarks is not None:
                all_skeletons.append(landmarks)
            frame_count += 1
    finally:
        estimator.close()
        cap.release()

    if not all_skeletons:
        print("no skeletons detected")
        return

    # stack into pose matrix P in R^{T x 33 x 2}
    P = np.array(all_skeletons)
    print(f"processed {frame_count} frames, extracted {P.shape[0]} skeletons")
    print(f"pose tensor shape: {P.shape}")

    if output_path:
        np.save(output_path, P)
        print(f"saved skeleton data to {output_path}")

    # compute pairwise distances for first vs last frame
    if len(all_skeletons) >= 2:
        s_first = flatten_skeleton(all_skeletons[0])
        s_last = flatten_skeleton(all_skeletons[-1])
        dist = euclidean_distance(s_first, s_last)
        sim = cosine_similarity(s_first, s_last)
        print(f"euclidean distance (first vs last): {dist:.4f}")
        print(f"cosine similarity (first vs last): {sim:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="skeleton extraction from video — exercise detection & rep counting"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="video source: '0' for webcam, or path to video file"
    )
    parser.add_argument(
        "--mode", choices=["live", "analyze"], default="live",
        help="'live' for real-time detection, 'analyze' for offline processing"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="output path for saving skeleton data (.npy) in analyze mode"
    )
    parser.add_argument(
        "--no-angles", action="store_true",
        help="hide angle display on the overlay"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="hide the angle plot"
    )
    parser.add_argument(
        "--rotate", action="store_true",
        help="apply rotation normalization"
    )
    parser.add_argument(
        "--frame-width", type=int, default=1280,
        help="capture frame width (camera/video decode target)"
    )
    parser.add_argument(
        "--frame-height", type=int, default=720,
        help="capture frame height (camera/video decode target)"
    )
    parser.add_argument(
        "--window-width", type=int, default=1600,
        help="initial window width (resizable window mode)"
    )
    parser.add_argument(
        "--window-height", type=int, default=900,
        help="initial window height (resizable window mode)"
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="disable selfie mirror mode"
    )
    args = parser.parse_args()

    # parse source — integer for webcam, string for file
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    if args.mode == "live":
        run_live(
            source=source,
            show_angles=not args.no_angles,
            show_plot=not args.no_plot,
            apply_rotation=args.rotate,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            window_width=args.window_width,
            window_height=args.window_height,
            mirror=not args.no_mirror,
        )
    elif args.mode == "analyze":
        run_analysis(
            video_path=args.source,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
