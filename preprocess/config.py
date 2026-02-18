from pathlib import Path

# 프로젝트 루트 (preprocess/)
PROJECT_ROOT = Path(__file__).resolve().parent

# ===== 데이터 루트 (data/) =====
DATA_DIR = PROJECT_ROOT / 'data'

# ===== 원본 데이터 (rawdata/) =====
RAWDATA_DIR     = DATA_DIR / 'rawdata'
RAW_LABELS_DIR  = RAWDATA_DIR / 'labels'       # 라벨 ZIP/폴더 모음
RAW_IMAGES_DIR  = RAWDATA_DIR / 'images'       # 원천 이미지 TAR 모음

# ===== 추출 결과물 (extracted/) =====
EXTRACTED_DIR     = DATA_DIR / 'extracted'
OUT_LABELS_DIR    = EXTRACTED_DIR / 'labels'       # 추출된 2D 라벨 JSON
OUT_LABELS_3D_DIR = EXTRACTED_DIR / 'labels_3d'    # 추출된 3D 라벨 JSON
OUT_IMAGES_DIR    = EXTRACTED_DIR / 'images'       # 추출된 이미지 JPG
OUT_MEDIAPIPE_DIR = EXTRACTED_DIR / 'mediapipe'    # 레거시 (MediaPipe 결과)
OUT_YOLO_DIR      = EXTRACTED_DIR / 'yolo_pose'    # YOLO26n-pose 추출 JSON

# 대상 운동
TARGET_EXERCISES = {
    '27': '푸시업',
    '35': '풀업',
}

# ===== 영상 전처리 설정 =====
UPLOAD_VIDEO_DIR  = DATA_DIR / 'uploads'          # 사용자 업로드 영상
OUT_FRAMES_DIR    = EXTRACTED_DIR / 'frames'       # 추출된 프레임 이미지
OUT_FRAMES_MP_DIR = EXTRACTED_DIR / 'frames_mediapipe'  # 레거시 (프레임 MediaPipe 좌표)
OUT_FRAMES_YP_DIR = EXTRACTED_DIR / 'frames_yolo_pose'  # 프레임 YOLO26n-pose 좌표

# 프레임 추출 설정
FRAME_EXTRACT_FPS = 2        # 초당 추출 프레임 수 (1~3, 학습 이미지 밀도 매칭)
TARGET_RESOLUTION = (1920, 1080)  # FHD 해상도 (width, height)

# ===== YOLO 모델 설정 =====
YOLO_POSE_MODEL = "yolo26n-pose.pt"
YOLO_CONFIDENCE = 0.5
