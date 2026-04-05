"""
skeleton extraction from video — main entry point.
real-time exercise detection and repetition counting using mediapipe pose estimation.
"""

import argparse
import signal
import sys
import time

import cv2
import numpy as np

from src.skeleton import PoseEstimator, build_adjacency_matrix, build_graph_laplacian
from src.normalize import normalize_skeleton
from src.features import compute_key_angles, flatten_skeleton, euclidean_distance, cosine_similarity
from src.classify import ExerciseClassifier
from src.visualize import draw_skeleton, draw_info_overlay, create_angle_plot
from collections import deque
from src.pca import PCA
from src.repetition import count_reps_and_classify


# global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def run_live(source: int | str = 0, show_angles: bool = True,
             show_plot: bool = True, apply_rotation: bool = False):
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

    estimator = PoseEstimator()
    classifier = ExerciseClassifier()

    # precompute the adjacency matrix and laplacian (static graph structure)
    A = build_adjacency_matrix()
    L = build_graph_laplacian(A)
    print(f"adjacency matrix shape: {A.shape}, laplacian shape: {L.shape}")
    print(f"laplacian smallest eigenvalues: {np.sort(np.linalg.eigvalsh(L))[:3]}")

    prev_time = time.time()
    prev_skeleton = None
    primary_angle_history: list[float] = []

    # PCA + repetition counting state
    MIN_FRAMES = 60       # warm-up: ~2 сек при 30fps
    REFIT_EVERY = 30      # переобчислення PCA кожні ~1 сек
    skeleton_buffer: deque = deque(maxlen=300)
    frame_counter: int = 0
    pca_reps: int = 0
    pca_label: str | None = None
    warming_up: bool = True

    # UI state
    prev_pca_reps: int = 0
    rep_flash_frames: int = 0   # counts down; >0 means flash is active

    print("press 'q' to quit, 'r' to reset rep counter")

    try:
        while not _shutdown_requested:
            ret, frame = cap.read()
            if not ret:
                # loop video files
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = estimator.extract_landmarks(frame_rgb)

            exercise = classifier.current_exercise
            reps = classifier.rep_count
            angles = None

            if landmarks is not None:
                # draw raw skeleton overlay
                frame = draw_skeleton(frame, landmarks)

                # normalize skeleton (translation + scaling + optional rotation)
                normalized = normalize_skeleton(landmarks, apply_procrustes=apply_rotation)

                # buffer normalized skeleton for PCA-based rep counting
                flat = flatten_skeleton(normalized)   # (66,)
                skeleton_buffer.append(flat)
                frame_counter += 1

                if len(skeleton_buffer) >= MIN_FRAMES and frame_counter % REFIT_EVERY == 0:
                    try:
                        X = np.array(skeleton_buffer, dtype=np.float64)   # (T, 66)
                        pca = PCA(variance_threshold=0.95)
                        Z = pca.fit_transform(X)                           # (T, k)
                        pca_reps, pca_label, _ = count_reps_and_classify(Z)
                        warming_up = False
                    except Exception:
                        pass

                # compute joint angles using the dot product formula
                angles = compute_key_angles(landmarks)

                # update classifier with new angles
                exercise, reps = classifier.update(angles)

                # track primary angle for the plot
                if exercise == "squat":
                    avg = (angles["left_knee"] + angles["right_knee"]) / 2
                elif exercise == "pushup":
                    avg = (angles["left_elbow"] + angles["right_elbow"]) / 2
                elif exercise == "jumping_jack":
                    avg = (angles["left_shoulder"] + angles["right_shoulder"]) / 2
                else:
                    avg = (angles["left_knee"] + angles["right_knee"]) / 2
                primary_angle_history.append(avg)

                prev_skeleton = landmarks[:, :2].copy()

            # compute fps
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # rep flash: trigger on new rep, count down for ~10 frames
            display_reps = pca_reps if not warming_up else classifier.rep_count
            if display_reps > prev_pca_reps:
                rep_flash_frames = 10
            prev_pca_reps = display_reps
            rep_flash = rep_flash_frames > 0
            if rep_flash_frames > 0:
                rep_flash_frames -= 1

            # warm-up progress for the progress bar
            warmup_progress = len(skeleton_buffer) / MIN_FRAMES

            # draw info overlay
            frame = draw_info_overlay(
                frame,
                exercise=pca_label if not warming_up else None,
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
                plot_title = f"{exercise or 'knee'} angle"
                plot_img = create_angle_plot(primary_angle_history, title=plot_title)
                ph, pw = plot_img.shape[:2]
                fh, fw = frame.shape[:2]
                x_start = fw - pw - 10
                y_start = TOP_BAR_H + 10
                if y_start + ph < fh and x_start > 0:
                    frame[y_start:y_start + ph, x_start:x_start + pw] = plot_img

            cv2.imshow("Skeleton Extraction from Video", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                classifier.reset()
                primary_angle_history.clear()
                skeleton_buffer.clear()
                frame_counter = 0
                pca_reps = 0
                pca_label = None
                warming_up = True
                prev_pca_reps = 0
                rep_flash_frames = 0
                print("reset rep counter")

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
        )
    elif args.mode == "analyze":
        run_analysis(
            video_path=args.source,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
