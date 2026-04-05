"""
rule-based exercise classification and time-based FSM repetition counting.
"""

import numpy as np


class ExerciseClassifier:
    """Classify exercises and count repetitions with a finite-state machine."""

    EXERCISE_RULES = {
        "squat": {
            "primary_angle": "knee",
            "up_threshold": 160.0,
            "down_threshold": 100.0,
        },
        "pushup": {
            "primary_angle": "elbow",
            "up_threshold": 160.0,
            "down_threshold": 100.0,
        },
        "jumping_jack": {
            "primary_angle": "shoulder",
            "up_threshold": 140.0,
            "down_threshold": 40.0,
        },
    }

    def __init__(
        self,
        confidence_time_s: float = 0.30,
        idle_time_s: float = 0.70,
        dead_zone_deg: float = 5.0,
        min_phase_time_s: float = 0.20,
        default_fps: float = 30.0,
    ):
        self.angle_history: dict[str, list[float]] = {
            "knee": [],
            "elbow": [],
            "shoulder": [],
        }
        self.current_exercise: str | None = None
        self.rep_count: int = 0

        self._confidence_time_s = float(max(0.01, confidence_time_s))
        self._idle_time_s = float(max(0.05, idle_time_s))
        self._dead_zone_deg = float(max(0.0, dead_zone_deg))
        self._min_phase_time_s = float(max(0.05, min_phase_time_s))
        self._default_dt = 1.0 / float(max(1.0, default_fps))

        self._candidate: str | None = None
        self._candidate_time_s: float = 0.0
        self._idle_time_acc_s: float = 0.0
        self._last_active_exercise: str | None = None

        self._state: str = "up"
        self._state_time_s: float = 0.0

    def reset(self):
        """Reset all state for a new session."""
        for key in self.angle_history:
            self.angle_history[key] = []
        self.current_exercise = None
        self.rep_count = 0

        self._candidate = None
        self._candidate_time_s = 0.0
        self._idle_time_acc_s = 0.0
        self._last_active_exercise = None

        self._state = "up"
        self._state_time_s = 0.0

    def update(self, angles: dict[str, float], dt: float | None = None) -> tuple[str | None, int]:
        """Update classifier with new frame angles and return (exercise, reps)."""
        dt_s = self._default_dt if dt is None else float(max(1e-3, dt))

        knee_angle = (angles.get("left_knee", 180) + angles.get("right_knee", 180)) / 2.0
        elbow_angle = (angles.get("left_elbow", 180) + angles.get("right_elbow", 180)) / 2.0
        shoulder_angle = (angles.get("left_shoulder", 180) + angles.get("right_shoulder", 180)) / 2.0
        hip_angle = (angles.get("left_hip", 180) + angles.get("right_hip", 180)) / 2.0

        self.angle_history["knee"].append(knee_angle)
        self.angle_history["elbow"].append(elbow_angle)
        self.angle_history["shoulder"].append(shoulder_angle)

        raw_exercise = self._detect_exercise(knee_angle, elbow_angle, shoulder_angle, hip_angle)
        exercise = self._apply_hysteresis(raw_exercise, dt_s)
        self.current_exercise = exercise

        if exercise is not None:
            angle_key = self.EXERCISE_RULES[exercise]["primary_angle"]
            primary_angle = {
                "knee": knee_angle,
                "elbow": elbow_angle,
                "shoulder": shoulder_angle,
            }[angle_key]
            self._update_rep_fsm(exercise, primary_angle, dt_s)

        return self.current_exercise, self.rep_count

    @staticmethod
    def _detect_exercise(knee: float, elbow: float, shoulder: float, hip: float) -> str | None:
        squat_active = knee < 120 and hip < 140
        pushup_active = elbow < 120 and hip < 150
        jj_active = shoulder > 90

        if squat_active and not pushup_active and not jj_active:
            return "squat"
        if pushup_active and not squat_active and not jj_active:
            return "pushup"
        if jj_active and not squat_active and not pushup_active:
            return "jumping_jack"
        if squat_active and pushup_active:
            return "squat" if (180 - knee) > (180 - elbow) else "pushup"
        return None

    def _apply_hysteresis(self, raw_exercise: str | None, dt_s: float) -> str | None:
        if raw_exercise is None:
            self._idle_time_acc_s += dt_s
            self._candidate = None
            self._candidate_time_s = 0.0
            if self._idle_time_acc_s >= self._idle_time_s:
                return None
            return self.current_exercise

        self._idle_time_acc_s = 0.0
        if raw_exercise == self.current_exercise:
            self._candidate = None
            self._candidate_time_s = 0.0
            self._last_active_exercise = raw_exercise
            return self.current_exercise

        if raw_exercise == self._candidate:
            self._candidate_time_s += dt_s
        else:
            self._candidate = raw_exercise
            self._candidate_time_s = dt_s

        if self._candidate_time_s >= self._confidence_time_s:
            confirmed = self._candidate
            self._candidate = None
            self._candidate_time_s = 0.0
            if confirmed != self._last_active_exercise:
                for key in self.angle_history:
                    self.angle_history[key] = []
                self.rep_count = 0
            self._last_active_exercise = confirmed
            self._initialize_rep_state(confirmed)
            return confirmed

        return self.current_exercise

    def _initialize_rep_state(self, exercise: str) -> None:
        rules = self.EXERCISE_RULES[exercise]
        angle_key = rules["primary_angle"]
        history = self.angle_history[angle_key]
        if not history:
            self._state = "up"
            self._state_time_s = 0.0
            return

        current_angle = history[-1]
        down_thr = float(rules["down_threshold"]) + self._dead_zone_deg
        self._state = "down" if current_angle <= down_thr else "up"
        self._state_time_s = 0.0

    def _update_rep_fsm(self, exercise: str, angle_value: float, dt_s: float) -> None:
        rules = self.EXERCISE_RULES[exercise]
        up_thr = float(rules["up_threshold"]) - self._dead_zone_deg
        down_thr = float(rules["down_threshold"]) + self._dead_zone_deg

        self._state_time_s += dt_s

        if self._state == "up":
            if angle_value <= down_thr and self._state_time_s >= self._min_phase_time_s:
                self._state = "down"
                self._state_time_s = 0.0
            return

        if angle_value >= up_thr and self._state_time_s >= self._min_phase_time_s:
            self.rep_count += 1
            self._state = "up"
            self._state_time_s = 0.0


def count_reps_from_signal(
    angle_signal: np.ndarray,
    up_threshold: float = 150.0,
    down_threshold: float | None = None,
    sample_rate_hz: float = 30.0,
    dead_zone_deg: float = 5.0,
    min_phase_time_s: float = 0.20,
    min_distance: int | None = None,
) -> int:
    """Count reps from a 1-D angle signal using the same up/down FSM logic."""
    signal = np.asarray(angle_signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) < 10:
        return 0

    down_thr = float(up_threshold - 60.0) if down_threshold is None else float(down_threshold)
    dz = float(max(0.0, dead_zone_deg))
    min_phase = float(max(0.05, min_phase_time_s))
    if min_distance is not None:
        min_phase = max(min_phase, float(min_distance) / float(max(1.0, sample_rate_hz)))
    dt_s = 1.0 / float(max(1.0, sample_rate_hz))

    state = "down" if signal[0] <= down_thr + dz else "up"
    state_time = 0.0
    reps = 0

    for angle_value in signal:
        state_time += dt_s
        if state == "up":
            if angle_value <= down_thr + dz and state_time >= min_phase:
                state = "down"
                state_time = 0.0
        else:
            if angle_value >= up_threshold - dz and state_time >= min_phase:
                reps += 1
                state = "up"
                state_time = 0.0

    return max(0, int(reps))
