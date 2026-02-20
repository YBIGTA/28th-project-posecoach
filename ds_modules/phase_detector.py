"""
운동 Phase 감지기 (바이오메카닉스 문헌 기반)

팔꿈치 각도를 사용하여 운동의 phase를 감지합니다.
"""
from collections import deque
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PhaseDetector:
    """운동 Phase 감지 부모 클래스"""
    
    def __init__(self):
        self.phase = 'ready'
        self.angle_history = deque(maxlen=5)
        self.velocity_history = deque(maxlen=3)
        self.prev_angle = None
        self.frames_in_phase = 0
        
    def get_stable_velocity(self) -> float:
        """최근 속도들의 평균을 반환하여 노이즈 제거"""
        if len(self.velocity_history) < 2:
            return 0.0
        return sum(self.velocity_history) / len(self.velocity_history)
    
    def reset(self):
        """Phase 감지기 초기화"""
        self.phase = 'ready'
        self.angle_history.clear()
        self.velocity_history.clear()
        self.prev_angle = None
        self.frames_in_phase = 0


class PushUpPhaseDetector(PhaseDetector):
    """푸시업 Phase 감지기 (팔꿈치 각도 기반)"""
    
    # 히스테리시스 임계값
    TOP_ENTER = 150    # top 진입 (팔 펴짐)
    TOP_EXIT = 140     # top 탈출
    BOTTOM_ENTER = 110 # bottom 진입 (팔 굽힘)
    BOTTOM_EXIT = 120  # bottom 탈출
    VEL_THRESHOLD = 0.8   # 속도 임계값 (도/프레임)
    MIN_FRAMES = 1        # 최소 체류 프레임
    
    def __init__(self):
        super().__init__()
        logger.info("PushUpPhaseDetector 초기화 (팔꿈치 각도 기반)")
    
    def update(self, avg_arm_angle: float) -> str:
        """팔꿈치 각도로 phase 판별"""
        self.frames_in_phase += 1
        
        # 속도 계산
        if self.prev_angle is not None:
            velocity = avg_arm_angle - self.prev_angle
            self.velocity_history.append(velocity)
        else:
            velocity = 0.0  # 첫 프레임
        
        self.prev_angle = avg_arm_angle
        avg_velocity = self.get_stable_velocity()
        prev_phase = self.phase
        
        if self.phase == 'ready':
            if avg_arm_angle > self.TOP_ENTER:
                self.phase = 'top'
        
        elif self.phase == 'top':
            if (avg_arm_angle < self.TOP_EXIT 
                and avg_velocity < -self.VEL_THRESHOLD 
                and self.frames_in_phase >= self.MIN_FRAMES):
                self.phase = 'descending'
        
        elif self.phase == 'descending':
            if avg_arm_angle < self.BOTTOM_ENTER:
                self.phase = 'bottom'
            elif avg_velocity > self.VEL_THRESHOLD:
                self.phase = 'ascending'

        elif self.phase == 'bottom':
            if (avg_arm_angle > self.BOTTOM_EXIT
                and avg_velocity > self.VEL_THRESHOLD
                and self.frames_in_phase >= self.MIN_FRAMES):
                self.phase = 'ascending'

        elif self.phase == 'ascending':
            if avg_arm_angle > self.TOP_ENTER:
                self.phase = 'top'
            elif avg_velocity < -self.VEL_THRESHOLD:
                self.phase = 'descending'
        
        if prev_phase != self.phase:
            logger.debug(f"PushUp Phase: {prev_phase} → {self.phase} (angle={avg_arm_angle:.1f}°, vel={avg_velocity:.1f}°/f)")
            self.frames_in_phase = 0
        
        return self.phase


class PullUpPhaseDetector(PhaseDetector):
    """풀업 Phase 감지기 (팔꿈치 각도 기반 - 문헌 표준)"""
    
    BOTTOM_ENTER = 150
    BOTTOM_EXIT = 140
    TOP_ENTER = 100
    TOP_EXIT = 110
    VEL_THRESHOLD = 1.0
    MIN_FRAMES = 1
    
    def __init__(self):
        super().__init__()
        logger.info("PullUpPhaseDetector 초기화 (팔꿈치 각도 기반)")
    
    def update(self, avg_elbow_angle: float) -> str:
        """팔꿈치 각도로 phase 판별"""
        self.frames_in_phase += 1
        
        # 속도 계산
        if self.prev_angle is not None:
            velocity = avg_elbow_angle - self.prev_angle
            self.velocity_history.append(velocity)
        else:
            velocity = 0.0  # 첫 프레임
        
        self.prev_angle = avg_elbow_angle
        avg_velocity = self.get_stable_velocity()
        prev_phase = self.phase
        
        if self.phase == 'ready':
            if avg_elbow_angle > self.BOTTOM_ENTER:
                self.phase = 'bottom'
        
        elif self.phase == 'bottom':
            if (avg_elbow_angle < self.BOTTOM_EXIT 
                and avg_velocity < -self.VEL_THRESHOLD 
                and self.frames_in_phase >= self.MIN_FRAMES):
                self.phase = 'ascending'
        
        elif self.phase == 'ascending':
            if avg_elbow_angle < self.TOP_ENTER:
                self.phase = 'top'
            elif avg_velocity > self.VEL_THRESHOLD:
                self.phase = 'descending'

        elif self.phase == 'top':
            if (avg_elbow_angle > self.TOP_EXIT
                and avg_velocity > self.VEL_THRESHOLD
                and self.frames_in_phase >= self.MIN_FRAMES):
                self.phase = 'descending'

        elif self.phase == 'descending':
            if avg_elbow_angle > self.BOTTOM_ENTER:
                self.phase = 'bottom'
            elif avg_velocity < -self.VEL_THRESHOLD:
                self.phase = 'ascending'
        
        if prev_phase != self.phase:
            logger.debug(f"PullUp Phase: {prev_phase} → {self.phase} (angle={avg_elbow_angle:.1f}°, vel={avg_velocity:.1f}°/f)")
            self.frames_in_phase = 0
        
        return self.phase


def create_phase_detector(exercise_type: str) -> PhaseDetector:
    if exercise_type == '푸시업':
        return PushUpPhaseDetector()
    elif exercise_type == '풀업':
        return PullUpPhaseDetector()
    else:
        logger.warning(f"알 수 없는 운동 종류: {exercise_type}")
        return PushUpPhaseDetector()


def extract_phase_metric(npts: Optional[Dict], exercise_type: str) -> Optional[float]:
    """팔꿈치 각도 추출 (푸시업/풀업 공통)"""
    if npts is None:
        return None
    
    try:
        from ds_modules.angle_utils import cal_angle
        
        elbow_l = cal_angle(
            npts["Left Shoulder"],
            npts["Left Elbow"],
            npts["Left Wrist"]
        )
        elbow_r = cal_angle(
            npts["Right Shoulder"],
            npts["Right Elbow"],
            npts["Right Wrist"]
        )
        avg_elbow = (elbow_l + elbow_r) / 2
        return avg_elbow
            
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"Phase 지표 추출 실패: {e}")
        return None
