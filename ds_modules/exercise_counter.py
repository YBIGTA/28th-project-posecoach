from ds_modules.angle_utils import cal_angle


class ExerciseCounter:
    """공통 카운터 상태 관리."""

    _BASE_FPS = 10.0
    _BASE_ACTIVE_THRESH = 4    # 기준 10 FPS에서 0.4초
    _BASE_INACTIVE_THRESH = 10 # 기준 10 FPS에서 1.0초

    def __init__(self, fps: float = 10.0):
        self.count = 0
        self.state = "idle"
        self.ready_frames = 0
        self.is_active = False
        self.inactive_frames = 0

        ratio = max(fps, 1.0) / self._BASE_FPS
        self.active_threshold = max(1, round(self._BASE_ACTIVE_THRESH * ratio))
        self.inactive_threshold = max(1, round(self._BASE_INACTIVE_THRESH * ratio))

    def update(self, npts, current_phase=None):
        raise NotImplementedError

    def reset(self):
        self.count = 0
        self.state = "idle"
        self.ready_frames = 0
        self.is_active = False
        self.inactive_frames = 0

    def _deactivate(self, state_after):
        self.is_active = False
        self.ready_frames = 0
        self.inactive_frames = 0
        self.state = state_after


class PushUpCounter(ExerciseCounter):
    """푸시업 카운터 (Phase 기반 — top+bottom 방문 시 카운트)."""

    def __init__(self, fps: float = 10.0):
        super().__init__(fps)
        self.required_sequence = {"top", "bottom"}
        self.min_required = 2
        self.visited_phases = set()
        self._count_gate_phase = None

    def _compute_metrics(self, npts):
        arm_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
        arm_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
        avg_arm = (arm_l + arm_r) / 2
        wrist_y = (npts["Left Wrist"][1] + npts["Right Wrist"][1]) / 2
        knee_y = (npts["Left Knee"][1] + npts["Right Knee"][1]) / 2
        return avg_arm, wrist_y, knee_y

    def update(self, npts, current_phase=None):
        if npts is None:
            if self.is_active:
                self.inactive_frames += 1
                if self.inactive_frames > self.inactive_threshold:
                    self._deactivate("idle")
                    self.visited_phases.clear()
                    self._count_gate_phase = None
            return self.count

        avg_arm, wrist_y, knee_y = self._compute_metrics(npts)

        if not self.is_active:
            if wrist_y > knee_y and avg_arm > 140:
                self.ready_frames += 1
            else:
                self.ready_frames = max(0, self.ready_frames - 1)

            if self.ready_frames > self.active_threshold:
                self.is_active = True
                self.inactive_frames = 0
                self.visited_phases.clear()
                self._count_gate_phase = None
            return self.count

        is_exercise_pose = wrist_y > (knee_y - 0.08)
        if is_exercise_pose:
            self.inactive_frames = 0
        else:
            self.inactive_frames += 1
            if self.inactive_frames > self.inactive_threshold:
                self._deactivate("idle")
                self.visited_phases.clear()
                self._count_gate_phase = None
                return self.count

        if current_phase in self.required_sequence:
            self.visited_phases.add(current_phase)

        if current_phase == "top":
            matched = len(self.visited_phases & self.required_sequence)
            if matched >= self.min_required and self._count_gate_phase != "top":
                self.count += 1
                self.visited_phases.clear()
                self._count_gate_phase = "top"
        else:
            self._count_gate_phase = None

        return self.count


class PullUpCounter(ExerciseCounter):
    """풀업 카운터 (Phase 기반 — top+bottom 방문 시 카운트)."""

    def __init__(self, fps: float = 10.0):
        super().__init__(fps)
        self.required_sequence = {"top", "bottom"}
        self.min_required = 2
        self.visited_phases = set()
        self._count_gate_phase = None

    def _compute_metrics(self, npts):
        wrist_y = (npts["Left Wrist"][1] + npts["Right Wrist"][1]) / 2
        shoulder_y = (npts["Left Shoulder"][1] + npts["Right Shoulder"][1]) / 2
        return wrist_y, shoulder_y

    def update(self, npts, current_phase=None):
        if npts is None:
            if self.is_active:
                self.inactive_frames += 1
                if self.inactive_frames > self.inactive_threshold:
                    self._deactivate("idle")
                    self.visited_phases.clear()
                    self._count_gate_phase = None
            return self.count

        wrist_y, shoulder_y = self._compute_metrics(npts)

        if not self.is_active:
            if wrist_y < shoulder_y + 0.05:
                self.ready_frames += 1
            else:
                self.ready_frames = max(0, self.ready_frames - 1)

            if self.ready_frames > self.active_threshold:
                self.is_active = True
                self.inactive_frames = 0
                self.visited_phases.clear()
                self._count_gate_phase = None
            return self.count

        is_exercise_pose = wrist_y < (shoulder_y + 0.12)
        if is_exercise_pose:
            self.inactive_frames = 0
        else:
            self.inactive_frames += 1
            if self.inactive_frames > self.inactive_threshold:
                self._deactivate("idle")
                self.visited_phases.clear()
                self._count_gate_phase = None
                return self.count

        if current_phase in self.required_sequence:
            self.visited_phases.add(current_phase)

        if current_phase == "bottom":
            matched = len(self.visited_phases & self.required_sequence)
            if matched >= self.min_required and self._count_gate_phase != "bottom":
                self.count += 1
                self.visited_phases.clear()
                self._count_gate_phase = "bottom"
        else:
            self._count_gate_phase = None

        return self.count