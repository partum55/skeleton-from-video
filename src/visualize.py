"""
visualization utilities: drawing skeletons, HUD overlay, angle plot.
"""

import cv2
import numpy as np

from src.skeleton import SKELETON_CONNECTIONS, NUM_LANDMARKS


# ── Color palette ─────────────────────────────────────────────────────────────
JOINT_COLOR      = (0, 255, 120)    # mint green
BONE_COLOR       = (210, 210, 210)  # light gray
BAR_BG           = (18, 18, 18)     # near-black bars
TEXT_PRIMARY     = (255, 255, 255)  # white
TEXT_SECONDARY   = (160, 160, 160)  # gray
TEXT_ACCENT      = (0, 220, 255)    # cyan — exercise label
TEXT_REPS        = (80, 255, 80)    # bright green — reps
TEXT_REPS_FLASH  = (255, 255, 255)  # white flash on new rep
TEXT_HINT        = (100, 100, 100)  # dark gray — keyboard hints
WARMUP_BAR_BG    = (55, 55, 55)     # progress bar background
WARMUP_BAR_FG    = (0, 180, 255)    # blue progress fill
PLOT_BG          = (28, 28, 28)     # plot background
PLOT_LINE        = (0, 210, 255)    # cyan plot curve
PLOT_GRID        = (48, 48, 48)     # subtle grid lines

TOP_BAR_H  = 72   # px — top HUD bar height
BOT_BAR_H  = 44   # px — bottom HUD bar height
BAR_ALPHA  = 0.78 # bar transparency (higher = more opaque)


# ── Skeleton ──────────────────────────────────────────────────────────────────

def draw_skeleton(frame: np.ndarray, landmarks: np.ndarray,
                  joint_radius: int = 5, bone_thickness: int = 2) -> np.ndarray:
    """draw skeleton joints and bones on the frame.

    landmarks: (33, 2) or (33, 3) with x, y in [0, 1].
    """
    h, w = frame.shape[:2]
    out = frame.copy()

    points: list[tuple[int, int]] = []
    for i in range(min(NUM_LANDMARKS, landmarks.shape[0])):
        points.append((int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)))

    for i, j in SKELETON_CONNECTIONS:
        if i < len(points) and j < len(points):
            cv2.line(out, points[i], points[j], BONE_COLOR, bone_thickness, cv2.LINE_AA)

    for pt in points:
        cv2.circle(out, pt, joint_radius, JOINT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, pt, joint_radius, (0, 170, 70), 1, cv2.LINE_AA)  # dark outline

    return out


# ── Internal helpers ──────────────────────────────────────────────────────────

def _blend_rect(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                color: tuple, alpha: float) -> np.ndarray:
    """draw a filled rectangle blended over the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def _put(frame: np.ndarray, text: str, x: int, y: int,
         scale: float, color: tuple, thickness: int = 1) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _text_w(text: str, scale: float, thickness: int = 1) -> int:
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return w


# ── Main HUD ─────────────────────────────────────────────────────────────────

def draw_info_overlay(
    frame: np.ndarray,
    exercise: str | None,
    rep_count: int,
    angles: dict[str, float] | None = None,
    fps: float | None = None,
    warming_up: bool = False,
    warmup_progress: float = 0.0,
    rep_flash: bool = False,
) -> np.ndarray:
    """draw the full HUD: top bar + bottom bar.

    Parameters
    ----------
    exercise:
        Exercise label or None (shows "DETECTING...").
    rep_count:
        Current repetition count.
    angles:
        Dict of joint angles for the bottom bar.
    fps:
        Frames per second for the top-right corner.
    warming_up:
        If True, shows warm-up progress bar instead of normal content.
    warmup_progress:
        Float in [0, 1] — how far through warm-up.
    rep_flash:
        If True, rep counter renders in white (flash on new rep).
    """
    h, w = frame.shape[:2]
    out = frame.copy()

    # ── Top bar ───────────────────────────────────────────────────────────────
    out = _blend_rect(out, 0, 0, w, TOP_BAR_H, BAR_BG, BAR_ALPHA)

    if warming_up:
        # centred label
        label = "WARMING UP"
        lw = _text_w(label, 0.85, 2)
        _put(out, label, (w - lw) // 2, 30, 0.85, WARMUP_BAR_FG, 2)

        # progress bar
        bx0, bx1 = 20, w - 60
        by, bh = 46, 10
        cv2.rectangle(out, (bx0, by), (bx1, by + bh), WARMUP_BAR_BG, -1)
        fill = bx0 + int((bx1 - bx0) * min(warmup_progress, 1.0))
        if fill > bx0:
            cv2.rectangle(out, (bx0, by), (fill, by + bh), WARMUP_BAR_FG, -1)
        _put(out, f"{int(warmup_progress * 100)}%", bx1 + 6, by + 9, 0.4, TEXT_SECONDARY)

    else:
        # ── Left: exercise label ──────────────────────────────────────────────
        ex = exercise.upper().replace("_", " ") if exercise else "DETECTING..."
        _put(out, ex, 16, 46, 0.9, TEXT_ACCENT, 2)

        # ── Centre: rep counter ───────────────────────────────────────────────
        rep_str  = str(rep_count)
        rep_col  = TEXT_REPS_FLASH if rep_flash else TEXT_REPS
        rw       = _text_w(rep_str, 2.2, 4)
        lw_reps  = _text_w("REPS", 0.4, 1)
        cx       = w // 2
        _put(out, "REPS",    cx - lw_reps // 2, 18,  0.4,  TEXT_SECONDARY, 1)
        _put(out, rep_str,   cx - rw // 2,       60,  2.2,  rep_col,        4)

        # ── Right: FPS ────────────────────────────────────────────────────────
        if fps is not None:
            fps_txt = f"{fps:.0f} fps"
            _put(out, fps_txt, w - _text_w(fps_txt, 0.48) - 14, 46, 0.48, TEXT_SECONDARY)

    # ── Bottom bar ────────────────────────────────────────────────────────────
    out = _blend_rect(out, 0, h - BOT_BAR_H, w, h, BAR_BG, BAR_ALPHA)

    # keyboard hints — right side
    hints = "[Q] Quit   [R] Reset"
    _put(out, hints, w - _text_w(hints, 0.42) - 14, h - 14, 0.42, TEXT_HINT)

    # angles — left side, three horizontal groups
    if angles:
        lk  = angles.get("left_knee",      0.0)
        rk  = angles.get("right_knee",     0.0)
        le  = angles.get("left_elbow",     0.0)
        re_ = angles.get("right_elbow",    0.0)
        ls  = angles.get("left_shoulder",  0.0)
        rs_ = angles.get("right_shoulder", 0.0)

        groups = [
            f"KNEE  L{lk:.0f}° R{rk:.0f}°",
            f"ELBOW L{le:.0f}° R{re_:.0f}°",
            f"SHLD  L{ls:.0f}° R{rs_:.0f}°",
        ]
        x = 14
        for txt in groups:
            _put(out, txt, x, h - 14, 0.42, TEXT_SECONDARY)
            x += _text_w(txt, 0.42) + 24

    return out


# ── Angle plot ────────────────────────────────────────────────────────────────

def create_angle_plot(angle_history: list[float], title: str = "angle",
                      width: int = 280, height: int = 130) -> np.ndarray:
    """create a compact angle-vs-time plot for the top-right corner."""
    plot = np.full((height, width, 3), PLOT_BG, dtype=np.uint8)

    # thin border
    cv2.rectangle(plot, (0, 0), (width - 1, height - 1), (55, 55, 55), 1)

    if len(angle_history) < 2:
        return plot

    angles = np.array(angle_history[-width:])
    min_a, max_a = 0.0, 180.0
    mg = 16  # margin

    # grid lines at 45 / 90 / 135 °
    for val in [45, 90, 135]:
        y = height - mg - int((val - min_a) / (max_a - min_a) * (height - 2 * mg))
        cv2.line(plot, (mg, y), (width - 4, y), PLOT_GRID, 1)
        cv2.putText(plot, str(val), (2, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_HINT, 1)

    # curve
    n      = len(angles)
    x_step = (width - mg - 4) / max(n - 1, 1)
    pts    = []
    for i, a in enumerate(angles):
        px = mg + int(i * x_step)
        py = height - mg - int((a - min_a) / (max_a - min_a) * (height - 2 * mg))
        py = max(mg, min(height - mg, py))
        pts.append((px, py))

    for i in range(len(pts) - 1):
        cv2.line(plot, pts[i], pts[i + 1], PLOT_LINE, 1, cv2.LINE_AA)

    # title
    cv2.putText(plot, title, (mg, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_ACCENT, 1)

    return plot
