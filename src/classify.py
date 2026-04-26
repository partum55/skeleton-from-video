"""Rule-based exercise classification and FSM repetition counting."""

import numpy as np


class ExerciseClassifier:
    """Two-state FSM (up/down) for exercise detection and rep counting."""

    EXERCISE_RULES = {
        "squat": {
            "primary_angle": "knee",
            "up_threshold": 160.0,
            "down_threshold": 105.0,
            "min_amplitude": 40.0,
        },
        "pushup": {
            "primary_angle": "elbow",
            "up_threshold": 155.0,
            "down_threshold": 90.0,
            "min_amplitude": 35.0,
        },
        "jumping_jack": {
            "primary_angle": "shoulder",
            "up_threshold": 120.0,
            "down_threshold": 50.0,
            "min_amplitude": 50.0,
        },
    }

    def __init__(
        self,
        confidence_time_s: float = 0.30,
        idle_time_s: float = 0.70,
        dead_zone_deg: float = 10.0,
        min_phase_time_s: float = 0.20,
        min_hold_time_s: float = 0.05,
        min_velocity_frames: int = 2,
        default_fps: float = 30.0,
    ):
        self.angle_history: dict[str, list[float]] = {
            "knee": [],
            "elbow": [],
            "shoulder": [],
        }
        self.current_exercise: str | None = None
        self.rep_count: int = 0

        # Summary bookkeeping kept separate from the live FSM counter so it
        # cannot interfere with the detection/counting logic.
        self.rep_totals: dict[str, int] = {name: 0 for name in self.EXERCISE_RULES}
        self.session_history: list[tuple[str, int]] = []
        self._session_exercise: str | None = None
        self._session_reps: int = 0

        self._confidence_time_s = float(max(0.01, confidence_time_s))
        self._idle_time_s = float(max(0.05, idle_time_s))
        self._dead_zone_deg = float(max(0.0, dead_zone_deg))
        self._min_phase_time_s = float(max(0.05, min_phase_time_s))
        self._min_hold_time_s = float(max(0.02, min_hold_time_s))
        self._min_velocity_frames = max(1, min_velocity_frames)
        self._default_dt = 1.0 / float(max(1.0, default_fps))

        self._candidate: str | None = None
        self._candidate_time_s: float = 0.0
        self._idle_time_acc_s: float = 0.0
        self._last_active_exercise: str | None = None

        self._state: str = "up"
        self._state_time_s: float = 0.0

        self._pending_transition: str | None = None
        self._pending_time_s: float = 0.0

        self._recent_angles: list[float] = []
        self._max_recent_angles = 5

        self._phase_min_angle: float = 180.0
        self._phase_max_angle: float = 0.0

        self._recent_transition_time_s: float = 1.0
        self._hysteresis_window_s: float = 0.3
        self._skip_next_up_count: bool = False

    def reset(self):
        """Reset all state for a new session."""
        for key in self.angle_history:
            self.angle_history[key] = []
        self.current_exercise = None
        self.rep_count = 0
        self.rep_totals = {name: 0 for name in self.EXERCISE_RULES}
        self.session_history = []
        self._session_exercise = None
        self._session_reps = 0

        self._candidate = None
        self._candidate_time_s = 0.0
        self._idle_time_acc_s = 0.0
        self._last_active_exercise = None

        self._state = "up"
        self._state_time_s = 0.0

        self._pending_transition = None
        self._pending_time_s = 0.0
        self._recent_angles.clear()
        self._phase_min_angle = 180.0
        self._phase_max_angle = 0.0
        self._recent_transition_time_s = 1.0
        self._skip_next_up_count = False

    def update(
        self,
        angles: dict[str, float],
        dt: float | None = None,
        body_features: dict[str, float] | None = None,
    ) -> tuple[str | None, int]:
        """Feed new frame angles, return (exercise_label, rep_count)."""
        dt_s = self._default_dt if dt is None else float(max(1e-3, dt))

        knee_angle = (angles.get("left_knee", 180) + angles.get("right_knee", 180)) / 2.0
        elbow_angle = (angles.get("left_elbow", 180) + angles.get("right_elbow", 180)) / 2.0
        shoulder_angle = (angles.get("left_shoulder", 180) + angles.get("right_shoulder", 180)) / 2.0
        hip_angle = (angles.get("left_hip", 180) + angles.get("right_hip", 180)) / 2.0

        self.angle_history["knee"].append(knee_angle)
        self.angle_history["elbow"].append(elbow_angle)
        self.angle_history["shoulder"].append(shoulder_angle)

        raw_exercise = self._detect_exercise(
            knee_angle, elbow_angle, shoulder_angle, hip_angle,
            body_features=body_features
        )
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
    def _detect_exercise(
        knee: float,
        elbow: float,
        shoulder: float,
        hip: float,
        body_features: dict[str, float] | None = None,
    ) -> str | None:
        """Detect exercise type from joint angles and body position."""
        torso_vert = 1.0
        leg_spread = 0.3
        if body_features:
            torso_vert = body_features.get("torso_verticality", 1.0)
            leg_spread = body_features.get("leg_spread", 0.3)

        squat_active = knee < 130 and hip < 145 and torso_vert > 0.4

        is_horizontal = torso_vert < 0.5
        pushup_active = is_horizontal and elbow < 140 and knee > 120

        jj_active = (
            shoulder > 100
            and knee > 150
            and hip > 140
            and torso_vert > 0.7
            and (leg_spread > 1.0 or shoulder > 140)
        )

        if squat_active and not jj_active:
            if pushup_active:
                return "squat" if (180 - knee) > (180 - elbow) else "pushup"
            return "squat"

        if pushup_active and not squat_active and not jj_active:
            return "pushup"

        if jj_active and not squat_active and not pushup_active:
            return "jumping_jack"

        return None

    def _apply_hysteresis(self, raw_exercise: str | None, dt_s: float) -> str | None:
        if raw_exercise is None:
            self._idle_time_acc_s += dt_s
            self._candidate = None
            self._candidate_time_s = 0.0
            if self._idle_time_acc_s >= self._idle_time_s:
                self._end_session()
                self._last_active_exercise = None
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
            switched = confirmed != self._last_active_exercise
            if switched:
                self._end_session()
                self._start_session(confirmed)
            prev_active = self._last_active_exercise
            self._last_active_exercise = confirmed
            self._initialize_rep_state(confirmed)
            if prev_active is None and confirmed is not None:
                self.rep_count += 1
                self._record_rep(confirmed)
                if self._state == "down":
                    self._skip_next_up_count = True
            return confirmed

        return self.current_exercise

    def _start_session(self, exercise: str | None) -> None:
        if exercise is None:
            return
        if self._session_exercise == exercise:
            return
        self._session_exercise = exercise
        self._session_reps = 0

    def _end_session(self) -> None:
        if self._session_exercise is None:
            return
        if self._session_reps > 0:
            self.session_history.append((self._session_exercise, self._session_reps))
        self._session_exercise = None
        self._session_reps = 0

    def _record_rep(self, exercise: str) -> None:
        self.rep_totals[exercise] = self.rep_totals.get(exercise, 0) + 1
        if self._session_exercise != exercise:
            self._start_session(exercise)
        self._session_reps += 1

    def finalize(self) -> dict:
        """Close any open session and return a printable summary."""
        self._end_session()
        return {
            "totals": dict(self.rep_totals),
            "history": list(self.session_history),
            "grand_total": int(sum(self.rep_totals.values())),
        }

    def _initialize_rep_state(self, exercise: str) -> None:
        """Reset FSM when switching to a different exercise."""
        rules = self.EXERCISE_RULES[exercise]
        angle_key = rules["primary_angle"]
        history = self.angle_history[angle_key]

        self._pending_transition = None
        self._pending_time_s = 0.0
        self._recent_angles.clear()
        self._phase_min_angle = 180.0
        self._phase_max_angle = 0.0
        self._recent_transition_time_s = 1.0

        if not history:
            self._state = "up"
            self._state_time_s = 0.0
            return

        current_angle = float(history[-1])
        down_thr = float(rules["down_threshold"]) + self._dead_zone_deg
        self._state = "down" if current_angle <= down_thr else "up"

        recent = history[-max(2, int(self._confidence_time_s / self._default_dt) + 1):]
        if recent:
            self._phase_min_angle = float(min(recent))
            self._phase_max_angle = float(max(recent))
        else:
            self._phase_min_angle = current_angle
            self._phase_max_angle = current_angle

        self._state_time_s = self._confidence_time_s

    def _compute_angle_velocity(self) -> float:
        """Angle change over the recent window (positive = increasing)."""
        if len(self._recent_angles) < 2:
            return 0.0
        return self._recent_angles[-1] - self._recent_angles[0]

    def _is_velocity_valid_for_transition(self, target_state: str) -> bool:
        """True if the angle is moving in the right direction for the transition."""
        if len(self._recent_angles) < self._min_velocity_frames:
            return True

        velocity = self._compute_angle_velocity()

        if target_state == "down":
            return velocity < 5.0
        return velocity > -5.0

    def _get_effective_thresholds(self, rules: dict) -> tuple[float, float]:
        """Up/down thresholds, widened briefly after a recent transition."""
        up_thr = float(rules["up_threshold"]) - self._dead_zone_deg
        down_thr = float(rules["down_threshold"]) + self._dead_zone_deg

        if self._recent_transition_time_s < self._hysteresis_window_s:
            hysteresis_factor = 1.0 - (self._recent_transition_time_s / self._hysteresis_window_s)
            hysteresis_deg = 8.0 * hysteresis_factor

            if self._state == "up":
                down_thr -= hysteresis_deg
            else:
                up_thr += hysteresis_deg

        return up_thr, down_thr

    def _update_rep_fsm(self, exercise: str, angle_value: float, dt_s: float) -> None:
        """Advance the up/down state machine, count a rep on confirmed up transition."""
        rules = self.EXERCISE_RULES[exercise]
        min_amplitude = float(rules.get("min_amplitude", 30.0))
        up_thr, down_thr = self._get_effective_thresholds(rules)

        self._state_time_s += dt_s
        self._recent_transition_time_s += dt_s

        self._recent_angles.append(angle_value)
        if len(self._recent_angles) > self._max_recent_angles:
            self._recent_angles.pop(0)

        self._phase_min_angle = min(self._phase_min_angle, angle_value)
        self._phase_max_angle = max(self._phase_max_angle, angle_value)

        if self._state == "up":
            if angle_value <= down_thr:
                if self._pending_transition == "down":
                    self._pending_time_s += dt_s
                else:
                    self._pending_transition = "down"
                    self._pending_time_s = dt_s

                phase_time_ok = self._state_time_s >= self._min_phase_time_s
                hold_time_ok = self._pending_time_s >= self._min_hold_time_s
                velocity_ok = self._is_velocity_valid_for_transition("down")

                if phase_time_ok and hold_time_ok and velocity_ok:
                    self._state = "down"
                    self._state_time_s = 0.0
                    self._pending_transition = None
                    self._pending_time_s = 0.0
                    self._recent_transition_time_s = 0.0
                    self._phase_min_angle = angle_value
                    self._phase_max_angle = angle_value
            else:
                self._pending_transition = None
                self._pending_time_s = 0.0
            return

        if angle_value >= up_thr:
            if self._pending_transition == "up":
                self._pending_time_s += dt_s
            else:
                self._pending_transition = "up"
                self._pending_time_s = dt_s

            phase_time_ok = self._state_time_s >= self._min_phase_time_s
            hold_time_ok = self._pending_time_s >= self._min_hold_time_s
            velocity_ok = self._is_velocity_valid_for_transition("up")

            amplitude = self._phase_max_angle - self._phase_min_angle
            amplitude_ok = amplitude >= min_amplitude

            if phase_time_ok and hold_time_ok and velocity_ok and amplitude_ok:
                if self._skip_next_up_count:
                    self._skip_next_up_count = False
                else:
                    self.rep_count += 1
                    self._record_rep(exercise)
                self._state = "up"
                self._state_time_s = 0.0
                self._pending_transition = None
                self._pending_time_s = 0.0
                self._recent_transition_time_s = 0.0
                self._phase_min_angle = angle_value
                self._phase_max_angle = angle_value
        else:
            self._pending_transition = None
            self._pending_time_s = 0.0


def count_reps_from_signal(
    angle_signal: np.ndarray,
    up_threshold: float = 150.0,
    down_threshold: float | None = None,
    sample_rate_hz: float = 30.0,
    dead_zone_deg: float = 10.0,
    min_phase_time_s: float = 0.25,
    min_distance: int | None = None,
    min_amplitude_deg: float = 30.0,
    min_hold_time_s: float = 0.08,
) -> int:
    """Count reps from a 1-D angle signal using the same FSM logic."""
    signal = np.asarray(angle_signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) < 10:
        return 0

    down_thr = float(up_threshold - 60.0) if down_threshold is None else float(down_threshold)
    dz = float(max(0.0, dead_zone_deg))
    min_phase = float(max(0.05, min_phase_time_s))
    min_hold = float(max(0.02, min_hold_time_s))
    min_amp = float(max(0.0, min_amplitude_deg))

    if min_distance is not None:
        min_phase = max(min_phase, float(min_distance) / float(max(1.0, sample_rate_hz)))
    dt_s = 1.0 / float(max(1.0, sample_rate_hz))

    state = "down" if signal[0] <= down_thr + dz else "up"
    state_time = 0.0
    hold_time = 0.0
    pending_state: str | None = None
    phase_min = signal[0]
    phase_max = signal[0]
    reps = 0

    for angle_value in signal:
        state_time += dt_s
        phase_min = min(phase_min, angle_value)
        phase_max = max(phase_max, angle_value)

        if state == "up":
            if angle_value <= down_thr + dz:
                if pending_state == "down":
                    hold_time += dt_s
                else:
                    pending_state = "down"
                    hold_time = dt_s

                if state_time >= min_phase and hold_time >= min_hold:
                    state = "down"
                    state_time = 0.0
                    pending_state = None
                    hold_time = 0.0
                    phase_min = angle_value
                    phase_max = angle_value
            else:
                pending_state = None
                hold_time = 0.0
        else:
            if angle_value >= up_threshold - dz:
                if pending_state == "up":
                    hold_time += dt_s
                else:
                    pending_state = "up"
                    hold_time = dt_s

                amplitude = phase_max - phase_min
                if state_time >= min_phase and hold_time >= min_hold and amplitude >= min_amp:
                    reps += 1
                    state = "up"
                    state_time = 0.0
                    pending_state = None
                    hold_time = 0.0
                    phase_min = angle_value
                    phase_max = angle_value
            else:
                pending_state = None
                hold_time = 0.0

    return max(0, int(reps))
