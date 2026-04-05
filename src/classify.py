"""
rule-based exercise classification and time-based FSM repetition counting.

Key accuracy features:
- Hold-time validation: angle must stay in threshold zone for min duration
- Velocity direction check: motion must be in correct direction for transition
- Amplitude validation: meaningful range of motion required before counting rep
- Temporal hysteresis: wider threshold gap after recent transitions
"""

import numpy as np


class ExerciseClassifier:
    """Classify exercises and count repetitions with a finite-state machine.
    
    The FSM uses multiple validation layers to prevent false counts:
    1. Hold-time: angle must remain past threshold for min_hold_time_s
    2. Velocity: angle derivative must have correct sign for transition
    3. Amplitude: rep only counts if angle range exceeds min_amplitude_deg
    4. Phase time: minimum time must pass in each phase
    """

    EXERCISE_RULES = {
        "squat": {
            "primary_angle": "knee",
            "up_threshold": 160.0,
            "down_threshold": 90.0,  # lowered from 100 for deeper squat detection
            "min_amplitude": 40.0,   # minimum angle change for valid rep
        },
        "pushup": {
            "primary_angle": "elbow",
            "up_threshold": 155.0,   # slightly lowered for partial extension
            "down_threshold": 90.0,  # lowered from 100 for shallow pushups
            "min_amplitude": 35.0,
        },
        "jumping_jack": {
            "primary_angle": "shoulder",
            "up_threshold": 120.0,   # lowered from 140 for normal arm raises
            "down_threshold": 50.0,  # raised from 40 for faster detection
            "min_amplitude": 50.0,
        },
    }

    def __init__(
        self,
        confidence_time_s: float = 0.30,
        idle_time_s: float = 0.70,
        dead_zone_deg: float = 10.0,       # increased from 5 for more hysteresis
        min_phase_time_s: float = 0.25,    # increased from 0.20
        min_hold_time_s: float = 0.08,     # NEW: hold time before confirming transition
        min_velocity_frames: int = 2,       # NEW: frames to check velocity direction
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
        self._min_hold_time_s = float(max(0.02, min_hold_time_s))
        self._min_velocity_frames = max(1, min_velocity_frames)
        self._default_dt = 1.0 / float(max(1.0, default_fps))

        self._candidate: str | None = None
        self._candidate_time_s: float = 0.0
        self._idle_time_acc_s: float = 0.0
        self._last_active_exercise: str | None = None

        # FSM state
        self._state: str = "up"
        self._state_time_s: float = 0.0
        
        # Hold-time validation: pending transition tracking
        self._pending_transition: str | None = None  # "down" or "up"
        self._pending_time_s: float = 0.0
        
        # Velocity tracking for direction validation
        self._recent_angles: list[float] = []
        self._max_recent_angles = 5
        
        # Amplitude tracking for rep validation
        self._phase_min_angle: float = 180.0
        self._phase_max_angle: float = 0.0
        
        # Hysteresis: tighten thresholds after recent transition
        self._recent_transition_time_s: float = 1.0  # time since last transition
        self._hysteresis_window_s: float = 0.3       # apply hysteresis for this long

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
        
        # Reset new validation state
        self._pending_transition = None
        self._pending_time_s = 0.0
        self._recent_angles.clear()
        self._phase_min_angle = 180.0
        self._phase_max_angle = 0.0
        self._recent_transition_time_s = 1.0

    def update(
        self, 
        angles: dict[str, float], 
        dt: float | None = None,
        body_features: dict[str, float] | None = None,
    ) -> tuple[str | None, int]:
        """Update classifier with new frame angles and return (exercise, reps).
        
        Parameters
        ----------
        angles: Joint angles from compute_key_angles()
        dt: Time delta since last frame (seconds)
        body_features: Optional body position features from compute_body_position_features()
                      Used for improved exercise detection (torso orientation, leg spread)
        """
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
        """Detect exercise type from joint angles and body position.
        
        Improved detection with strict body position checks:
        - Squat: requires both bent knee AND bent hip (seated position)
        - Push-up: REQUIRES horizontal body position (torso_verticality < 0.5)
        - Jumping jack: requires raised arms + upright stance + leg spread
        
        Walking/standing will NOT trigger push-up because torso is vertical.
        """
        # Extract body features with defaults for upright standing
        torso_vert = 1.0  # default = upright (prevents false push-up detection)
        leg_spread = 0.3
        if body_features:
            torso_vert = body_features.get("torso_verticality", 1.0)
            leg_spread = body_features.get("leg_spread", 0.3)
        
        # Squat: knee bent, hip flexed, semi-upright torso
        squat_active = knee < 130 and hip < 145 and torso_vert > 0.4
        
        # Push-up: STRICT horizontal body requirement
        # torso_verticality < 0.5 means body is more horizontal than 45°
        # This prevents standing/walking from triggering push-up
        is_horizontal = torso_vert < 0.5
        pushup_active = (
            is_horizontal            # REQUIRED: body must be horizontal
            and elbow < 140          # arms bent (relaxed threshold)
            and knee > 120           # legs mostly extended
        )
        
        # Jumping jack: arms raised significantly, upright stance
        jj_active = shoulder > 100 and knee > 150 and hip > 150 and torso_vert > 0.7
        
        # Priority-based classification with conflict resolution
        if squat_active and not jj_active:
            # Squat takes priority when legs are bent
            if pushup_active:
                # Both active: choose based on which angle is more extreme
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
        """Initialize FSM state when switching to a new exercise."""
        rules = self.EXERCISE_RULES[exercise]
        angle_key = rules["primary_angle"]
        history = self.angle_history[angle_key]
        
        # Reset all validation state
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

        current_angle = history[-1]
        down_thr = float(rules["down_threshold"]) + self._dead_zone_deg
        self._state = "down" if current_angle <= down_thr else "up"
        self._state_time_s = 0.0

    def _compute_angle_velocity(self) -> float:
        """Compute smoothed angle velocity from recent history.
        
        Returns positive value if angle is increasing, negative if decreasing.
        """
        if len(self._recent_angles) < 2:
            return 0.0
        # Use simple difference of first and last for robustness
        return self._recent_angles[-1] - self._recent_angles[0]
    
    def _is_velocity_valid_for_transition(self, target_state: str) -> bool:
        """Check if velocity direction supports the requested transition.
        
        For transition to "down": angle should be decreasing (velocity < 0)
        For transition to "up": angle should be increasing (velocity > 0)
        """
        if len(self._recent_angles) < self._min_velocity_frames:
            return True  # Not enough data, allow transition
        
        velocity = self._compute_angle_velocity()
        
        if target_state == "down":
            return velocity < 5.0  # Allow small positive velocity (tolerance for noise)
        else:  # target_state == "up"
            return velocity > -5.0  # Allow small negative velocity
    
    def _get_effective_thresholds(self, rules: dict) -> tuple[float, float]:
        """Get thresholds with hysteresis adjustment after recent transitions."""
        up_thr = float(rules["up_threshold"]) - self._dead_zone_deg
        down_thr = float(rules["down_threshold"]) + self._dead_zone_deg
        
        # Apply hysteresis: require more extreme angles after recent transition
        if self._recent_transition_time_s < self._hysteresis_window_s:
            hysteresis_factor = 1.0 - (self._recent_transition_time_s / self._hysteresis_window_s)
            hysteresis_deg = 8.0 * hysteresis_factor  # Up to 8° extra
            
            if self._state == "up":
                # Just transitioned to up, require deeper down to go back
                down_thr -= hysteresis_deg
            else:
                # Just transitioned to down, require higher up to go back
                up_thr += hysteresis_deg
        
        return up_thr, down_thr

    def _update_rep_fsm(self, exercise: str, angle_value: float, dt_s: float) -> None:
        """Update FSM with multi-layer validation for accurate rep counting.
        
        Validation layers:
        1. Phase time: minimum time must pass in current phase
        2. Hold time: angle must stay past threshold for hold duration
        3. Velocity: motion direction must be consistent with transition
        4. Amplitude: sufficient range of motion before counting rep
        """
        rules = self.EXERCISE_RULES[exercise]
        min_amplitude = float(rules.get("min_amplitude", 30.0))
        up_thr, down_thr = self._get_effective_thresholds(rules)

        # Update timing
        self._state_time_s += dt_s
        self._recent_transition_time_s += dt_s
        
        # Track angle history for velocity calculation
        self._recent_angles.append(angle_value)
        if len(self._recent_angles) > self._max_recent_angles:
            self._recent_angles.pop(0)
        
        # Track min/max angles in current phase for amplitude validation
        self._phase_min_angle = min(self._phase_min_angle, angle_value)
        self._phase_max_angle = max(self._phase_max_angle, angle_value)

        if self._state == "up":
            # Check for transition to "down"
            if angle_value <= down_thr:
                # Start or continue pending down transition
                if self._pending_transition == "down":
                    self._pending_time_s += dt_s
                else:
                    self._pending_transition = "down"
                    self._pending_time_s = dt_s
                
                # Validate transition conditions
                phase_time_ok = self._state_time_s >= self._min_phase_time_s
                hold_time_ok = self._pending_time_s >= self._min_hold_time_s
                velocity_ok = self._is_velocity_valid_for_transition("down")
                
                if phase_time_ok and hold_time_ok and velocity_ok:
                    # Confirmed transition to down
                    self._state = "down"
                    self._state_time_s = 0.0
                    self._pending_transition = None
                    self._pending_time_s = 0.0
                    self._recent_transition_time_s = 0.0
                    # Reset amplitude tracking for down phase
                    self._phase_min_angle = angle_value
                    self._phase_max_angle = angle_value
            else:
                # Angle left threshold zone, reset pending transition
                self._pending_transition = None
                self._pending_time_s = 0.0
            return

        # State is "down" - check for transition to "up"
        if angle_value >= up_thr:
            # Start or continue pending up transition
            if self._pending_transition == "up":
                self._pending_time_s += dt_s
            else:
                self._pending_transition = "up"
                self._pending_time_s = dt_s
            
            # Validate transition conditions
            phase_time_ok = self._state_time_s >= self._min_phase_time_s
            hold_time_ok = self._pending_time_s >= self._min_hold_time_s
            velocity_ok = self._is_velocity_valid_for_transition("up")
            
            # Amplitude validation: did we actually move through significant range?
            amplitude = self._phase_max_angle - self._phase_min_angle
            amplitude_ok = amplitude >= min_amplitude
            
            if phase_time_ok and hold_time_ok and velocity_ok and amplitude_ok:
                # Confirmed rep completion
                self.rep_count += 1
                self._state = "up"
                self._state_time_s = 0.0
                self._pending_transition = None
                self._pending_time_s = 0.0
                self._recent_transition_time_s = 0.0
                # Reset amplitude tracking for up phase
                self._phase_min_angle = angle_value
                self._phase_max_angle = angle_value
        else:
            # Angle left threshold zone, reset pending transition
            self._pending_transition = None
            self._pending_time_s = 0.0


def count_reps_from_signal(
    angle_signal: np.ndarray,
    up_threshold: float = 150.0,
    down_threshold: float | None = None,
    sample_rate_hz: float = 30.0,
    dead_zone_deg: float = 10.0,       # increased from 5
    min_phase_time_s: float = 0.25,    # increased from 0.20
    min_distance: int | None = None,
    min_amplitude_deg: float = 30.0,   # NEW: minimum angle range for valid rep
    min_hold_time_s: float = 0.08,     # NEW: hold time before confirming transition
) -> int:
    """Count reps from a 1-D angle signal using FSM with validation layers.
    
    Validation layers (same as ExerciseClassifier):
    1. Phase time: minimum time in each phase before transition
    2. Hold time: angle must stay past threshold for min duration
    3. Amplitude: minimum range of motion required for valid rep
    """
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
