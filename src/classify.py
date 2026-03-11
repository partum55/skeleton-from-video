"""
rule-based exercise classification and repetition counting using angle thresholds.
"""

import numpy as np
from scipy.signal import find_peaks


class ExerciseClassifier:
    """classifies exercises and counts reps based on joint angle patterns."""

    # angle thresholds for each exercise (degrees)
    EXERCISE_RULES = {
        "squat": {
            "primary_angle": "knee",     # track knee angle
            "up_threshold": 160.0,       # standing
            "down_threshold": 100.0,     # squat position
        },
        "pushup": {
            "primary_angle": "elbow",    # track elbow angle
            "up_threshold": 160.0,       # arms extended
            "down_threshold": 100.0,     # arms bent
        },
        "jumping_jack": {
            "primary_angle": "shoulder", # track shoulder angle
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

    def reset(self):
        """reset all state for a new session."""
        for key in self.angle_history:
            self.angle_history[key] = []
        self.current_exercise = None
        self.rep_count = 0
        self._state = "up"

    def update(self, angles: dict[str, float]) -> tuple[str | None, int]:
        """update with new frame angles and return (exercise_name, rep_count).

        angles should contain keys like 'left_knee', 'right_knee', etc.
        """
        # average left and right angles
        knee_angle = (angles.get("left_knee", 180) + angles.get("right_knee", 180)) / 2
        elbow_angle = (angles.get("left_elbow", 180) + angles.get("right_elbow", 180)) / 2
        shoulder_angle = (angles.get("left_shoulder", 180) + angles.get("right_shoulder", 180)) / 2

        self.angle_history["knee"].append(knee_angle)
        self.angle_history["elbow"].append(elbow_angle)
        self.angle_history["shoulder"].append(shoulder_angle)

        # detect exercise type and count reps
        exercise, reps = self._classify_and_count(knee_angle, elbow_angle, shoulder_angle)
        self.current_exercise = exercise
        self.rep_count = reps
        return exercise, reps

    def _classify_and_count(self, knee: float, elbow: float,
                            shoulder: float) -> tuple[str | None, int]:
        """determine which exercise is being performed and count repetitions.

        classification uses directional thresholds:
        - squat: knee angle drops below standing (~170 -> ~80)
        - pushup: elbow angle drops below extended (~170 -> ~90)
        - jumping jack: shoulder angle rises above resting (~30 -> ~140)
        """
        knee_active = knee < 140
        elbow_active = elbow < 140
        shoulder_active = shoulder > 60

        if knee_active and not elbow_active:
            exercise = "squat"
            angle_key = "knee"
        elif elbow_active and not knee_active:
            exercise = "pushup"
            angle_key = "elbow"
        elif shoulder_active and not knee_active and not elbow_active:
            exercise = "jumping_jack"
            angle_key = "shoulder"
        elif knee_active and elbow_active:
            # both active — pick the one with more deviation
            exercise = "squat" if (180 - knee) > (180 - elbow) else "pushup"
            angle_key = "knee" if exercise == "squat" else "elbow"
        else:
            return self.current_exercise, self.rep_count

        reps = self._count_reps(angle_key, exercise)
        return exercise, reps

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
