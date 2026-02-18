"""
YOLO26n-pose COCO 17 키포인트 정의 및 변환 유틸리티

모든 키포인트 추출 스크립트와 app.py가 이 모듈을 참조한다.
PDF 카운팅 함수 패턴(pts[5] = Left Shoulder)과 호환되는 설계.
"""
from ultralytics import YOLO

# ===== COCO 17 키포인트 매핑 =====
COCO_KEYPOINT_MAP = {
    "Nose": 0,
    "Left Eye": 1,
    "Right Eye": 2,
    "Left Ear": 3,
    "Right Ear": 4,
    "Left Shoulder": 5,
    "Right Shoulder": 6,
    "Left Elbow": 7,
    "Right Elbow": 8,
    "Left Wrist": 9,
    "Right Wrist": 10,
    "Left Hip": 11,
    "Right Hip": 12,
    "Left Knee": 13,
    "Right Knee": 14,
    "Left Ankle": 15,
    "Right Ankle": 16,
}

COCO_INDEX_TO_NAME = {v: k for k, v in COCO_KEYPOINT_MAP.items()}

# ===== 스켈레톤 연결 (COCO 표준 16개) =====
COCO_SKELETON = [
    # 얼굴
    ("Nose", "Left Eye"), ("Nose", "Right Eye"),
    ("Left Eye", "Left Ear"), ("Right Eye", "Right Ear"),
    # 상체
    ("Left Shoulder", "Right Shoulder"),
    ("Left Shoulder", "Left Elbow"), ("Left Elbow", "Left Wrist"),
    ("Right Shoulder", "Right Elbow"), ("Right Elbow", "Right Wrist"),
    # 몸통
    ("Left Shoulder", "Left Hip"), ("Right Shoulder", "Right Hip"),
    ("Left Hip", "Right Hip"),
    # 하체
    ("Left Hip", "Left Knee"), ("Left Knee", "Left Ankle"),
    ("Right Hip", "Right Knee"), ("Right Knee", "Right Ankle"),
]

# ===== 신뢰도 임계값 =====
CONFIDENCE_THRESHOLD = 0.5

# ===== 기본 모델 =====
DEFAULT_MODEL = "yolo26n-pose.pt"


def load_pose_model(model_name=None):
    """YOLO26n-pose 모델을 로드한다. 첫 호출 시 자동 다운로드."""
    return YOLO(model_name or DEFAULT_MODEL)


def select_best_person(result):
    """
    다중 인물 검출 시 바운딩 박스 면적이 가장 큰 사람을 선택한다.
    (운동 영상에서 주 피사체가 가장 크게 찍힘)

    Returns:
        int: 선택된 사람 인덱스, 사람 미검출 시 -1
    """
    if result.boxes is None or len(result.boxes) == 0:
        return -1
    if len(result.boxes) == 1:
        return 0
    boxes = result.boxes.xyxy.cpu()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(areas.argmax())


def yolo_result_to_dict(result):
    """
    YOLO 추론 결과를 기존 MediaPipe 호환 딕셔너리로 변환한다.

    반환 형식: {"Left Shoulder": {"x": 960, "y": 540, "z": 0.0, "vis": 0.95}, ...}

    Args:
        result: YOLO 추론 결과 (단일 이미지)

    Returns:
        dict 또는 None (사람 미검출 시)
    """
    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    person_idx = select_best_person(result)
    if person_idx < 0:
        return None

    xy = result.keypoints.xy[person_idx].cpu().numpy()       # (17, 2)
    conf = result.keypoints.conf[person_idx].cpu().numpy()   # (17,)

    pts = {}
    for name, idx in COCO_KEYPOINT_MAP.items():
        pts[name] = {
            "x": int(round(float(xy[idx][0]))),
            "y": int(round(float(xy[idx][1]))),
            "z": 0.0,
            "vis": round(float(conf[idx]), 4),
        }

    return pts
