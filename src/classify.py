"""
rule-based exercise classification and repetition counting using angle thresholds.
"""

import numpy as np
from scipy.signal import find_peaks


# number of consecutive frames an exercise must be detected before we commit to it
_CONFIDENCE_FRAMES = 8

# number of consecutive idle frames before we drop back to idle
_IDLE_FRAMES = 20


class ExerciseClassifier:
    """classifies exercises and counts reps based on joint angle patterns.

    uses strict angle ranges for each exercise and requires multiple
    consecutive frames of agreement before switching exercise type.
    """

    # angle thresholds for each exercise (degrees)
    EXERCISE_RULES = {
        "squat": {
            "primary_angle": "knee",
            "up_threshold": 160.0,       # standing
            "down_threshold": 100.0,     # squat position
        },
        "pushup": {
            "primary_angle": "elbow",
            "up_threshold": 160.0,       # arms extended
            "down_threshold": 100.0,     # arms bent
        },
        "jumping_jack": {
            "primary_angle": "shoulder",
            "up_threshold": 140.0,       # arms up
            "down_threshold": 40.0,      # arms down
        },
    }

    def __init__(self):
        self.angle_history: dict[str, list[float]] = {
            "knee": [],
            "elbow": [],
            "shoulder": [],
        }
        self.current_exercise: str | None = None
        self.rep_count: int = 0
        self._state: str = "up"  # tracks if we're in "up" or "down" phase

        # hysteresis: track how many consecutive frames each candidate has
        self._candidate: str | None = None
        self._candidate_frames: int = 0
        self._idle_frames: int = 0
        self._last_active_exercise: str | None = None

    def reset(self):
        """reset all state for a new session."""
        for key in self.angle_history:
            self.angle_history[key] = []
        self.current_exercise = None
        self.rep_count = 0
        self._state = "up"
        self._candidate = None
        self._candidate_frames = 0
        self._idle_frames = 0
        self._last_active_exercise = None

    def update(self, angles: dict[str, float]) -> tuple[str | None, int]:
        """update with new frame angles and return (exercise_name, rep_count).

        angles should contain keys like 'left_knee', 'right_knee', etc.
        """
        # average left and right angles
        knee_angle = (angles.get("left_knee", 180) + angles.get("right_knee", 180)) / 2
        elbow_angle = (angles.get("left_elbow", 180) + angles.get("right_elbow", 180)) / 2
        shoulder_angle = (angles.get("left_shoulder", 180) + angles.get("right_shoulder", 180)) / 2
        hip_angle = (angles.get("left_hip", 180) + angles.get("right_hip", 180)) / 2

        self.angle_history["knee"].append(knee_angle)
        self.angle_history["elbow"].append(elbow_angle)
        self.angle_history["shoulder"].append(shoulder_angle)

        # detect exercise type with confidence gating
        raw_exercise = self._detect_exercise(knee_angle, elbow_angle, shoulder_angle, hip_angle)
        exercise = self._apply_hysteresis(raw_exercise)

        self.current_exercise = exercise

        # count reps only if we have a confirmed exercise
        if exercise is not None:
            angle_key = self.EXERCISE_RULES[exercise]["primary_angle"]
            self.rep_count = self._count_reps(angle_key, exercise)

        return self.current_exercise, self.rep_count

    @staticmethod
    def _detect_exercise(knee: float, elbow: float,
                         shoulder: float, hip: float) -> str | None:
        """determine which exercise is being performed from a single frame.

        returns None (idle) when the body is in a neutral standing position.

        the thresholds are chosen so that normal standing does NOT trigger:
        - standing knee angle: ~165-175  -> we require < 120 for squat
        - standing elbow angle: ~155-175 -> we require < 120 for pushup
        - standing shoulder angle: ~20-45 -> we require > 90 for jumping jack
        - standing hip angle: ~165-175   -> we require < 140 for squat confirmation
        """
        # squat: knees significantly bent AND hips bent (torso leaning forward)
        squat_active = knee < 120 and hip < 140

        # pushup: elbows significantly bent AND body is not upright (hips bent = plank-ish)
        pushup_active = elbow < 120 and hip < 150

        # jumping jack: shoulders raised well above resting position
        jj_active = shoulder > 90

        if squat_active and not pushup_active and not jj_active:
            return "squat"
        elif pushup_active and not squat_active and not jj_active:
            return "pushup"
        elif jj_active and not squat_active and not pushup_active:
            return "jumping_jack"
        elif squat_active and pushup_active:
            # both active — pick based on which deviates more from neutral
            return "squat" if (180 - knee) > (180 - elbow) else "pushup"

        # neutral / standing — no exercise detected
        return None

    def _apply_hysteresis(self, raw_exercise: str | None) -> str | None:
        """require multiple consecutive frames before switching exercise.

        prevents single noisy frames from changing classification.
        also returns to idle after sustained neutral frames.
        """
        if raw_exercise is None:
            # counting idle frames
            self._idle_frames += 1
            self._candidate = None
            self._candidate_frames = 0

            # drop to idle after enough neutral frames
            if self._idle_frames >= _IDLE_FRAMES:
                return None

            # keep current exercise during brief pauses (between reps)
            return self.current_exercise
        else:
            self._idle_frames = 0

            if raw_exercise == self.current_exercise:
                # already confirmed — keep it
                self._candidate = None
                self._candidate_frames = 0
                self._last_active_exercise = raw_exercise
                return self.current_exercise

            # new candidate detected
            if raw_exercise == self._candidate:
                self._candidate_frames += 1
            else:
                self._candidate = raw_exercise
                self._candidate_frames = 1

            # promote candidate to current after enough frames
            if self._candidate_frames >= _CONFIDENCE_FRAMES:
                confirmed = self._candidate
                self._candidate = None
                self._candidate_frames = 0
                # reset angle history and reps only when switching to a genuinely DIFFERENT exercise
                # (resuming the same exercise after idle should keep history for rep counting)
                if confirmed != self._last_active_exercise:
                    for key in self.angle_history:
                        self.angle_history[key] = []
                    self.rep_count = 0
                    self._state = "up"
                self._last_active_exercise = confirmed
                return confirmed

            # not enough frames yet — keep current
            return self.current_exercise

    def _count_reps(self, angle_key: str, exercise: str) -> int:
        """count repetitions by detecting peaks in the angle signal.

        uses scipy.signal.find_peaks to find local maxima (return to standing).
        each peak-to-peak cycle is one repetition.
        """
        history = self.angle_history[angle_key]
        if len(history) < 10:
            return 0

        signal = np.array(history)
        rules = self.EXERCISE_RULES[exercise]

        # find peaks (moments when angle is high = standing/extended position)
        peaks, _ = find_peaks(
            signal,
            height=rules["up_threshold"] - 20,
            distance=15,  # minimum frames between reps
            prominence=20,
        )

        return max(0, len(peaks) - 1)


def count_reps_from_signal(angle_signal: np.ndarray, up_threshold: float = 150.0,
                           min_distance: int = 15) -> int:
    """standalone rep counter for a recorded angle time series.

    finds peaks (standing positions) and counts cycles.
    """
    if len(angle_signal) < 10:
        return 0

    peaks, _ = find_peaks(
        angle_signal,
        height=up_threshold - 20,
        distance=min_distance,
        prominence=20,
    )
    return max(0, len(peaks) - 1)
