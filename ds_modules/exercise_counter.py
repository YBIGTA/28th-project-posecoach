from ds_modules.angle_utils import cal_angle

class ExerciseCounter:
    """엄격한 시퀀스 기반 카운터 부모 클래스"""
    def __init__(self):
        self.count = 0
        self.is_active = False
        self.ready_frames = 0
        self.active_threshold = 10
        self.required_sequence = set()  # 필수 Phase
        self.min_required = 2           # 최소 필요 개수
        self.visited_phases = set()

    def reset(self):
        self.count = 0
        self.is_active = False
        self.ready_frames = 0
        self.visited_phases = set()


class PushUpCounter(ExerciseCounter):
    """푸시업 시퀀스 기반 카운터"""
    def __init__(self):
        super().__init__()
        self.required_sequence = {'top', 'bottom'}  # ✅ bottom (typo 수정)
        self.min_required = 2
        
    def update(self, npts, current_phase):
        if npts is None: 
            return self.count

        # 1. 준비 감지 (Active 전환)
        if not self.is_active:
            try:
                arm_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
                arm_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
                avg_arm = (arm_l + arm_r) / 2
                wrist_y = (npts["Left Wrist"][1] + npts["Right Wrist"][1]) / 2
                knee_y = (npts["Left Knee"][1] + npts["Right Knee"][1]) / 2

                if wrist_y > knee_y and avg_arm > 160:
                    self.ready_frames += 1
                else:
                    self.ready_frames = 0
                
                if self.ready_frames > self.active_threshold:
                    self.is_active = True
                    print(">>> 푸시업 시작")
            except (KeyError, TypeError, ValueError):
                pass
            
            return self.count

        # 2. Phase 기록
        if current_phase in self.required_sequence:
            self.visited_phases.add(current_phase)

        # 3. 카운팅 (top에서 체크)
        if current_phase == 'top':
            matched = len(self.visited_phases & self.required_sequence)
            
            # ✅ min_required 활용
            if matched >= self.min_required:
                self.count += 1
                print(f"★ 푸시업 {self.count}회 성공 ({matched}/2 phases)")
                self.visited_phases = set()
        
        return self.count


class PullUpCounter(ExerciseCounter):
    """풀업 시퀀스 기반 카운터"""
    def __init__(self):
        super().__init__()
        self.required_sequence = {'top', 'bottom'}  # ✅ bottom (typo 수정)
        self.min_required = 2

    def update(self, npts, current_phase):
        if npts is None: 
            return self.count

        # 1. 준비 감지
        if not self.is_active:
            try:
                wrist_y = (npts["Left Wrist"][1] + npts["Right Wrist"][1]) / 2
                shoulder_y = (npts["Left Shoulder"][1] + npts["Right Shoulder"][1]) / 2
                
                if wrist_y < shoulder_y:  # 손이 어깨보다 위
                    self.ready_frames += 1
                else:
                    self.ready_frames = 0
                
                if self.ready_frames > self.active_threshold:
                    self.is_active = True
                    print(">>> 풀업 시작")
            except (KeyError, TypeError, ValueError):
                pass
            
            return self.count

        # 2. Phase 기록
        if current_phase in self.required_sequence:
            self.visited_phases.add(current_phase)

        # 3. 카운팅 (bottom에서 체크)
        if current_phase == 'bottom':
            matched = len(self.visited_phases & self.required_sequence)
            
            # ✅ min_required 활용
            if matched >= self.min_required:
                self.count += 1
                print(f"★ 풀업 {self.count}회 성공 ({matched}/2 phases)")
                self.visited_phases = set()
        
        return self.count