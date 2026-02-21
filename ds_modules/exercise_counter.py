from ds_modules.angle_utils import cal_angle


class ExerciseCounter:
    """공통 카운터 상태 관리."""

    def __init__(self):
        self.count = 0
        self.state = "idle"
        self.ready_frames = 0
        self.is_active = False
        self.inactive_frames = 0

        self.active_threshold = 10
        self.inactive_threshold = 6
        self.rep_cooldown_frames = 4
        self._rep_cooldown = 0

    def update(self, npts, current_phase=None):
        raise NotImplementedError

    def reset(self):
        self.count = 0
        self.state = "idle"
        self.ready_frames = 0
        self.is_active = False
        self.inactive_frames = 0
        self._rep_cooldown = 0

    def _deactivate(self, state_after):
        self.is_active = False
        self.ready_frames = 0
        self.inactive_frames = 0
        self.state = state_after
        self._rep_cooldown = 0


class PushUpCounter(ExerciseCounter):
    """푸시업 카운터 (기본/Phase 모드 겸용)."""

    def __init__(self):
        super().__init__()
        self.state = "up"
        self.required_sequence = {"top", "bottom"}
        self.min_required = 2
        self.visited_phases = set()
        self._count_gate_phase = None
        self.active_threshold = 6
        self.inactive_threshold = 12
        self.angle_down_threshold = 105
        self.angle_up_threshold = 155

    def _compute_metrics(self, npts):
        arm_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
        arm_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
        avg_arm = (arm_l + arm_r) / 2
        wrist_y = (npts["Left Wrist"][1] + npts["Right Wrist"][1]) / 2
        knee_y = (npts["Left Knee"][1] + npts["Right Knee"][1]) / 2
        shoulder_y = (npts["Left Shoulder"][1] + npts["Right Shoulder"][1]) / 2
        return avg_arm, wrist_y, knee_y, shoulder_y

    def update(self, npts, current_phase=None):
        if self._rep_cooldown > 0:
            self._rep_cooldown -= 1

        if npts is None:
            if self.is_active:
                self.inactive_frames += 1
                if self.inactive_frames > self.inactive_threshold:
                    self._deactivate("up")
                    self.visited_phases.clear()
                    self._count_gate_phase = None
            return self.count

        avg_arm, wrist_y, knee_y, shoulder_y = self._compute_metrics(npts)

        if not self.is_active:
            # Lenient warm-up pose check to avoid missing early reps.
            is_ready_pose = (avg_arm > self.angle_up_threshold) and (
                wrist_y >= (knee_y - 0.12) or wrist_y >= (shoulder_y - 0.03)
            )
            if is_ready_pose:
                self.ready_frames += 1
            else:
                self.ready_frames = max(0, self.ready_frames - 1)

            if self.ready_frames >= self.active_threshold:
                self.is_active = True
                self.state = "up"
                self.inactive_frames = 0
                self.visited_phases.clear()
                self._count_gate_phase = None
            return self.count

        # Keep set active even when pose keypoints jitter, as long as movement exists.
        is_exercise_pose = (wrist_y >= (knee_y - 0.18)) or (avg_arm < (self.angle_up_threshold - 5))
        if is_exercise_pose:
            self.inactive_frames = 0
        else:
            self.inactive_frames += 1
            if self.inactive_frames > self.inactive_threshold:
                self._deactivate("up")
                self.visited_phases.clear()
                self._count_gate_phase = None
                return self.count

        counted_this_frame = False
        if current_phase is not None:
            if current_phase in self.required_sequence:
                self.visited_phases.add(current_phase)

            if current_phase == "top":
                matched = len(self.visited_phases & self.required_sequence)
                if (
                    matched >= self.min_required
                    and self._count_gate_phase != "top"
                    and self._rep_cooldown == 0
                ):
                    self.count += 1
                    counted_this_frame = True
                    self._rep_cooldown = self.rep_cooldown_frames
                    self.visited_phases.clear()
                    self._count_gate_phase = "top"
            else:
                self._count_gate_phase = None

        # Angle-based fallback for noisy phase labels.
        if self.state == "up":
            if avg_arm < self.angle_down_threshold:
                self.state = "down"
        elif self.state == "down":
            if avg_arm > self.angle_up_threshold:
                if (not counted_this_frame) and self._rep_cooldown == 0:
                    self.count += 1
                    self._rep_cooldown = self.rep_cooldown_frames
                self.state = "up"

        return self.count


class PullUpCounter(ExerciseCounter):
    """풀업 카운터 (기본/Phase 모드 겸용)."""

    def __init__(self):
        super().__init__()
        self.state = "down"
        self.required_sequence = {"top", "bottom"}
        self.min_required = 2
        self.visited_phases = set()
        self._count_gate_phase = None
        self.active_threshold = 5
        self.inactive_threshold = 12
        self.angle_top_threshold = 100
        self.angle_bottom_threshold = 150

    def _compute_metrics(self, npts):
        wrist_y = (npts["Left Wrist"][1] + npts["Right Wrist"][1]) / 2
        shoulder_y = (npts["Left Shoulder"][1] + npts["Right Shoulder"][1]) / 2
        nose_y = npts["Nose"][1]
        elbow_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
        elbow_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
        avg_elbow = (elbow_l + elbow_r) / 2
        return wrist_y, shoulder_y, nose_y, avg_elbow

    def update(self, npts, current_phase=None):
        if self._rep_cooldown > 0:
            self._rep_cooldown -= 1

        if npts is None:
            if self.is_active:
                self.inactive_frames += 1
                if self.inactive_frames > self.inactive_threshold:
                    self._deactivate("down")
                    self.visited_phases.clear()
                    self._count_gate_phase = None
            return self.count

        wrist_y, shoulder_y, nose_y, avg_elbow = self._compute_metrics(npts)

        if not self.is_active:
            is_ready_pose = (wrist_y < (shoulder_y + 0.04)) or (avg_elbow < (self.angle_bottom_threshold - 5))
            if is_ready_pose:
                self.ready_frames += 1
            else:
                self.ready_frames = max(0, self.ready_frames - 1)

            if self.ready_frames >= self.active_threshold:
                self.is_active = True
                self.state = "down"
                self.inactive_frames = 0
                self.visited_phases.clear()
                self._count_gate_phase = None
            return self.count

        is_exercise_pose = (wrist_y < (shoulder_y + 0.15)) or (avg_elbow < (self.angle_bottom_threshold - 5))
        if is_exercise_pose:
            self.inactive_frames = 0
        else:
            self.inactive_frames += 1
            if self.inactive_frames > self.inactive_threshold:
                self._deactivate("down")
                self.visited_phases.clear()
                self._count_gate_phase = None
                return self.count

        counted_this_frame = False
        if current_phase is not None:
            if current_phase in self.required_sequence:
                self.visited_phases.add(current_phase)

            if current_phase == "bottom":
                matched = len(self.visited_phases & self.required_sequence)
                if (
                    matched >= self.min_required
                    and self._count_gate_phase != "bottom"
                    and self._rep_cooldown == 0
                ):
                    self.count += 1
                    counted_this_frame = True
                    self._rep_cooldown = self.rep_cooldown_frames
                    self.visited_phases.clear()
                    self._count_gate_phase = "bottom"
            else:
                self._count_gate_phase = None

        # Angle-based fallback for noisy phase labels.
        if self.state == "down":
            if avg_elbow < self.angle_top_threshold:
                self.state = "up"
        elif self.state == "up":
            if avg_elbow > self.angle_bottom_threshold:
                if (not counted_this_frame) and self._rep_cooldown == 0:
                    self.count += 1
                    self._rep_cooldown = self.rep_cooldown_frames
                self.state = "down"

        return self.count
