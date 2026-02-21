"""
스켈레톤 시각화 유틸리티
프레임 이미지 위에 YOLO26n-pose 키포인트를 오버레이한다.
"""
import cv2
import numpy as np

from utils.keypoints import COCO_SKELETON, CONFIDENCE_THRESHOLD

# 스켈레톤 연결 정의 (COCO 표준 16개)
POSE_CONNECTIONS = COCO_SKELETON

# 관절 색상 (BGR)
JOINT_COLOR = (0, 255, 0)       # 초록
CONNECTION_COLOR = (255, 255, 0) # 시안
JOINT_RADIUS = 5
CONNECTION_THICKNESS = 2
VIS_THRESHOLD = CONFIDENCE_THRESHOLD


def draw_skeleton_on_frame(img_path, keypoints):
    """
    프레임 이미지 위에 관절점과 연결선을 그린다.

    Args:
        img_path: 프레임 이미지 경로 (str 또는 Path)
        keypoints: yolo_result_to_dict() 반환 dict
                   {"Nose": {"x": 960, "y": 200, "z": 0.0, "vis": 0.99}, ...}

    Returns:
        RGB numpy array (스켈레톤 오버레이 된 이미지), 실패 시 None
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    if keypoints is None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 연결선 먼저 그리기 (관절점 아래에 깔림)
    for joint_a, joint_b in POSE_CONNECTIONS:
        if joint_a not in keypoints or joint_b not in keypoints:
            continue
        pa, pb = keypoints[joint_a], keypoints[joint_b]
        if pa["vis"] < VIS_THRESHOLD or pb["vis"] < VIS_THRESHOLD:
            continue
        pt_a = (int(pa["x"]), int(pa["y"]))
        pt_b = (int(pb["x"]), int(pb["y"]))
        cv2.line(img, pt_a, pt_b, CONNECTION_COLOR, CONNECTION_THICKNESS)

    # 관절점 그리기
    for name, pt in keypoints.items():
        if pt["vis"] < VIS_THRESHOLD:
            continue
        center = (int(pt["x"]), int(pt["y"]))
        cv2.circle(img, center, JOINT_RADIUS, JOINT_COLOR, -1)

    # BGR → RGB 변환
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
