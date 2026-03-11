"""
visualization utilities: drawing skeletons, overlaying info on frames, plotting angles.
"""

import cv2
import numpy as np

from src.skeleton import SKELETON_CONNECTIONS, NUM_LANDMARKS


# color scheme
JOINT_COLOR = (0, 255, 0)       # green
BONE_COLOR = (255, 255, 255)    # white
TEXT_COLOR = (0, 255, 255)      # yellow
ANGLE_TEXT_COLOR = (0, 200, 255)
BG_COLOR = (0, 0, 0)


def draw_skeleton(frame: np.ndarray, landmarks: np.ndarray,
                  joint_radius: int = 4, bone_thickness: int = 2) -> np.ndarray:
    """draw the skeleton (joints + bones) on a video frame.

    landmarks should be in R^{33x2} or R^{33x3} with x, y in [0, 1] (normalized coords).
    """
    h, w = frame.shape[:2]
    output = frame.copy()

    # convert normalized coordinates to pixel coordinates
    points = []
    for i in range(min(NUM_LANDMARKS, landmarks.shape[0])):
        x = int(landmarks[i, 0] * w)
        y = int(landmarks[i, 1] * h)
        points.append((x, y))

    # draw bones first (so joints appear on top)
    for i, j in SKELETON_CONNECTIONS:
        if i < len(points) and j < len(points):
            cv2.line(output, points[i], points[j], BONE_COLOR, bone_thickness)

    # draw joints
    for pt in points:
        cv2.circle(output, pt, joint_radius, JOINT_COLOR, -1)

    return output


def draw_info_overlay(frame: np.ndarray, exercise: str | None, rep_count: int,
                      angles: dict[str, float] | None = None,
                      fps: float | None = None) -> np.ndarray:
    """overlay exercise info, rep count, and optional angles on the frame."""
    output = frame.copy()
    h, w = frame.shape[:2]

    # semi-transparent background for text
    overlay = output.copy()
    cv2.rectangle(overlay, (10, 10), (350, 130), BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)

    # exercise name
    ex_text = exercise.upper().replace("_", " ") if exercise else "DETECTING..."
    cv2.putText(output, f"Exercise: {ex_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

    # rep count
    cv2.putText(output, f"Reps: {rep_count}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # fps
    if fps is not None:
        cv2.putText(output, f"FPS: {fps:.0f}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # draw key angles on the right side
    if angles:
        y_offset = 40
        cv2.rectangle(overlay, (w - 280, 10), (w - 10, 10 + len(angles) * 25 + 10),
                      BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        for name, value in angles.items():
            text = f"{name}: {value:.0f} deg"
            cv2.putText(output, text, (w - 270, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, ANGLE_TEXT_COLOR, 1)
            y_offset += 25

    return output


def create_angle_plot(angle_history: list[float], title: str = "Angle over time",
                      width: int = 400, height: int = 200) -> np.ndarray:
    """create a simple angle-vs-time plot as a numpy image (for embedding in opencv window)."""
    plot = np.zeros((height, width, 3), dtype=np.uint8)

    if len(angle_history) < 2:
        return plot

    # normalize angles to plot area
    angles = np.array(angle_history[-width:])  # last N frames
    min_a, max_a = 0, 180
    margin = 20

    # draw grid lines
    for val in [45, 90, 135]:
        y = height - margin - int((val - min_a) / (max_a - min_a) * (height - 2 * margin))
        cv2.line(plot, (margin, y), (width - margin, y), (40, 40, 40), 1)
        cv2.putText(plot, f"{val}", (2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    # draw the angle curve
    n = len(angles)
    x_step = (width - 2 * margin) / max(n - 1, 1)
    for i in range(n - 1):
        x1 = margin + int(i * x_step)
        x2 = margin + int((i + 1) * x_step)
        y1 = height - margin - int((angles[i] - min_a) / (max_a - min_a) * (height - 2 * margin))
        y2 = height - margin - int((angles[i + 1] - min_a) / (max_a - min_a) * (height - 2 * margin))
        cv2.line(plot, (x1, y1), (x2, y2), (0, 200, 255), 1)

    cv2.putText(plot, title, (margin, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)
    return plot
