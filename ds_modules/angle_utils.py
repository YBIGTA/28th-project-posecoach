"""
각도/거리 계산 및 COCO 17 → 가상 키포인트 변환 유틸리티

"""
import numpy as np
from numpy import degrees, arccos, dot
from numpy.linalg import norm


def cal_angle(A, B, C):
    """코사인 법칙으로 ∠ABC를 도(°) 단위로 반환한다."""
    A, B, C = map(np.array, (A, B, C))
    ba = A - B
    bc = C - B
    norm_ba = norm(ba)
    norm_bc = norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 180.0
    cos_val = np.clip(dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(degrees(arccos(cos_val)))


def cal_distance(A, B):
    """두 점 사이의 유클리드 거리를 반환한다."""
    A, B = map(np.array, (A, B))
    return float(norm(A - B))


def _mid(p1, p2):
    """두 점의 중점을 반환한다."""
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]


def compute_virtual_keypoints(pts):
    """
    COCO 17 키포인트 dict → 가상 키포인트를 추가한 확장 dict를 반환한다.

    추가되는 가상 키포인트:
      - Neck:    midpoint(Left Shoulder, Right Shoulder)
      - Waist:   midpoint(Left Hip, Right Hip)
      - Ankle_C: midpoint(Left Ankle, Right Ankle)

    Args:
        pts: {"Nose": {"x":..,"y":..,"vis":..}, ...}  (COCO 17)

    Returns:
        {"Nose": [x, y], "Left Shoulder": [x, y], ..., "Neck": [x, y], ...}
        좌표만 [x, y] 리스트로 통일 (규칙 계산용)
    """
    if pts is None:
        return None

    flat = {}
    for name, pt in pts.items():
        flat[name] = [pt["x"], pt["y"]]

    # 가상 키포인트 생성
    flat["Neck"] = _mid(flat["Left Shoulder"], flat["Right Shoulder"])
    flat["Waist"] = _mid(flat["Left Hip"], flat["Right Hip"])
    flat["Ankle_C"] = _mid(flat["Left Ankle"], flat["Right Ankle"])

    return flat


def normalize_pts(pts, w, h):
    """
    픽셀 좌표 dict를 [0, 1] 정규화 좌표로 변환한다.

    Args:
        pts: compute_virtual_keypoints() 반환 dict  {"Nose": [x, y], ...}
        w, h: 프레임 해상도

    Returns:
        동일 구조의 정규화된 dict
    """
    if pts is None:
        return None
    normed = {}
    for name, coord in pts.items():
        normed[name] = [coord[0] / w, coord[1] / h]
    return normed


def is_keypoint_visible(pt_dict, threshold=0.5):
    """원본 키포인트 dict의 vis 값이 임계값 이상인지 확인한다."""
    if pt_dict is None:
        return False
    return pt_dict.get("vis", 0) >= threshold
