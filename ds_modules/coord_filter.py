"""
키포인트 좌표 스무딩 필터

이동 평균(window=3)으로 프레임 간 키포인트 떨림을 제거한다.
"""
from collections import deque


class KeypointSmoother:
    """이동 평균 기반 키포인트 스무더 (이상치 감쇠 포함)."""

    def __init__(self, window=3, jump_threshold=0.15):
        self.window = window
        self.jump_threshold = jump_threshold
        self._history = {}  # {keypoint_name: deque([(x, y), ...])}

    def reset(self):
        self._history.clear()

    def smooth(self, flat_pts):
        """
        flat_pts를 스무딩하여 반환한다.
        이전 평균 대비 jump_threshold 이상 점프하는 좌표는 70:30 비율로 블렌딩한다.

        Args:
            flat_pts: {"Nose": [x, y], ...}  (compute_virtual_keypoints 반환값)

        Returns:
            동일 구조의 스무딩된 dict
        """
        if flat_pts is None:
            return None

        smoothed = {}
        for name, coord in flat_pts.items():
            if name not in self._history:
                self._history[name] = deque(maxlen=self.window)

            buf = self._history[name]

            # 이상치 감쇠: 기존 평균 대비 큰 점프 시 블렌딩
            if len(buf) >= 1:
                prev_avg_x = sum(c[0] for c in buf) / len(buf)
                prev_avg_y = sum(c[1] for c in buf) / len(buf)
                dx = abs(coord[0] - prev_avg_x)
                dy = abs(coord[1] - prev_avg_y)
                if dx > self.jump_threshold or dy > self.jump_threshold:
                    coord = [
                        prev_avg_x * 0.7 + coord[0] * 0.3,
                        prev_avg_y * 0.7 + coord[1] * 0.3,
                    ]

            buf.append(coord)

            avg_x = sum(c[0] for c in buf) / len(buf)
            avg_y = sum(c[1] for c in buf) / len(buf)
            smoothed[name] = [avg_x, avg_y]

        return smoothed
