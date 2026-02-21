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

    _BASE_FPS = 10.0  # 임계값이 보정된 기준 FPS

    def __init__(self, fps: float = 10.0):
        self.phase = 'ready'
        self.velocity_history = deque(maxlen=3)
        self.prev_angle = None
        self.frames_in_phase = 0
        self.fps = max(fps, 1.0)
        
    def get_stable_velocity(self) -> float:
        """최근 속도들의 평균을 반환하여 노이즈 제거"""
        if len(self.velocity_history) < 2:
            return 0.0
        return sum(self.velocity_history) / len(self.velocity_history)
    
    def _fps_scaled_vel(self, base_threshold: float) -> float:
        """기준 FPS 대비 현재 FPS로 속도 임계값을 보정한다."""
        return base_threshold * (self._BASE_FPS / self.fps)

    def reset(self):
        """Phase 감지기 초기화"""
        self.phase = 'ready'
        self.velocity_history.clear()
        self.prev_angle = None
        self.frames_in_phase = 0


class PushUpPhaseDetector(PhaseDetector):
    """푸시업 Phase 감지기 (팔꿈치 각도 기반, FPS 보정)"""

    # 히스테리시스 임계값
    TOP_ENTER = 150    # top 진입 (팔 펴짐)
    TOP_EXIT = 140     # top 탈출
    BOTTOM_ENTER = 110 # bottom 진입 (팔 굽힘)
    BOTTOM_EXIT = 120  # bottom 탈출
    VEL_THRESHOLD = 0.8   # 속도 임계값 (도/프레임, 기준 10 FPS)
    MIN_FRAMES = 1        # 최소 체류 프레임

    def __init__(self, fps: float = 10.0):
        super().__init__(fps)
        self.vel_threshold = self._fps_scaled_vel(self.VEL_THRESHOLD)
        self.min_frames = max(1, round(self.MIN_FRAMES * fps / self._BASE_FPS))
        logger.info(f"PushUpPhaseDetector 초기화 (fps={fps}, vel_threshold={self.vel_threshold:.2f}, min_frames={self.min_frames})")

    def update(self, raw_angle: float) -> str:
        """팔꿈치 각도로 phase 판별"""
        self.frames_in_phase += 1

        if self.prev_angle is not None:
            velocity = raw_angle - self.prev_angle
            self.velocity_history.append(velocity)
        else:
            velocity = 0.0

        self.prev_angle = raw_angle
        avg_velocity = self.get_stable_velocity()
        prev_phase = self.phase

        if self.phase == 'ready':
            if raw_angle > self.TOP_ENTER:
                self.phase = 'top'

        elif self.phase == 'top':
            if (raw_angle < self.TOP_EXIT
                and avg_velocity < -self.vel_threshold
                and self.frames_in_phase >= self.min_frames):
                self.phase = 'descending'

        elif self.phase == 'descending':
            if raw_angle < self.BOTTOM_ENTER:
                self.phase = 'bottom'
            elif avg_velocity > self.vel_threshold:
                self.phase = 'ascending'

        elif self.phase == 'bottom':
            if (raw_angle > self.BOTTOM_EXIT
                and avg_velocity > self.vel_threshold
                and self.frames_in_phase >= self.min_frames):
                self.phase = 'ascending'

        elif self.phase == 'ascending':
            if raw_angle > self.TOP_ENTER:
                self.phase = 'top'
            elif avg_velocity < -self.vel_threshold:
                self.phase = 'descending'

        if prev_phase != self.phase:
            logger.debug(f"PushUp Phase: {prev_phase} → {self.phase} (angle={raw_angle:.1f}°, vel={avg_velocity:.2f}°/f)")
            self.frames_in_phase = 0

        return self.phase


class PullUpPhaseDetector(PhaseDetector):
    """풀업 Phase 감지기 (팔꿈치 각도 기반, FPS 보정)"""

    BOTTOM_ENTER = 150
    BOTTOM_EXIT = 140
    TOP_ENTER = 100
    TOP_EXIT = 110
    VEL_THRESHOLD = 1.0
    MIN_FRAMES = 1

    def __init__(self, fps: float = 10.0):
        super().__init__(fps)
        self.vel_threshold = self._fps_scaled_vel(self.VEL_THRESHOLD)
        self.min_frames = max(1, round(self.MIN_FRAMES * fps / self._BASE_FPS))
        logger.info(f"PullUpPhaseDetector 초기화 (fps={fps}, vel_threshold={self.vel_threshold:.2f}, min_frames={self.min_frames})")

    def update(self, raw_angle: float) -> str:
        """팔꿈치 각도로 phase 판별"""
        self.frames_in_phase += 1

        if self.prev_angle is not None:
            velocity = raw_angle - self.prev_angle
            self.velocity_history.append(velocity)
        else:
            velocity = 0.0

        self.prev_angle = raw_angle
        avg_velocity = self.get_stable_velocity()
        prev_phase = self.phase

        if self.phase == 'ready':
            if raw_angle > self.BOTTOM_ENTER:
                self.phase = 'bottom'

        elif self.phase == 'bottom':
            if (raw_angle < self.BOTTOM_EXIT
                and avg_velocity < -self.vel_threshold
                and self.frames_in_phase >= self.min_frames):
                self.phase = 'ascending'

        elif self.phase == 'ascending':
            if raw_angle < self.TOP_ENTER:
                self.phase = 'top'
            elif avg_velocity > self.vel_threshold:
                self.phase = 'descending'

        elif self.phase == 'top':
            if (raw_angle > self.TOP_EXIT
                and avg_velocity > self.vel_threshold
                and self.frames_in_phase >= self.min_frames):
                self.phase = 'descending'

        elif self.phase == 'descending':
            if raw_angle > self.BOTTOM_ENTER:
                self.phase = 'bottom'
            elif avg_velocity < -self.vel_threshold:
                self.phase = 'ascending'

        if prev_phase != self.phase:
            logger.debug(f"PullUp Phase: {prev_phase} → {self.phase} (angle={raw_angle:.1f}°, vel={avg_velocity:.2f}°/f)")
            self.frames_in_phase = 0

        return self.phase


def create_phase_detector(exercise_type: str, fps: float = 10.0) -> PhaseDetector:
    if exercise_type == '푸시업':
        return PushUpPhaseDetector(fps=fps)
    elif exercise_type == '풀업':
        return PullUpPhaseDetector(fps=fps)
    else:
        logger.warning(f"알 수 없는 운동 종류: {exercise_type}")
        return PushUpPhaseDetector(fps=fps)


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
