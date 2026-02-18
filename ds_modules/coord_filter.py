"""
키포인트 좌표 스무딩 필터

이동 평균(window=3)으로 프레임 간 키포인트 떨림을 제거한다.
"""
from collections import deque


class KeypointSmoother:
    """이동 평균 기반 키포인트 스무더."""

    def __init__(self, window=3):
        self.window = window
        self._history = {}  # {keypoint_name: deque([(x, y), ...])}

    def reset(self):
        self._history.clear()

    def smooth(self, flat_pts):
        """
        flat_pts를 스무딩하여 반환한다.

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
            self._history[name].append(coord)

            buf = self._history[name]
            avg_x = sum(c[0] for c in buf) / len(buf)
            avg_y = sum(c[1] for c in buf) / len(buf)
            smoothed[name] = [avg_x, avg_y]

        return smoothed
