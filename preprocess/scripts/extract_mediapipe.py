import cv2
import json
import sys
import numpy as np
from pathlib import Path
import time

import mediapipe as mp

# config.py import (ppss/ 루트)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUT_IMAGES_DIR, OUT_MEDIAPIPE_DIR

# 매핑 테이블
SHARED_MAP = {
    "Nose": 0, "Left Eye": 2, "Right Eye": 5, "Left Ear": 7, "Right Ear": 8,
    "Left Shoulder": 11, "Right Shoulder": 12, "Left Elbow": 13, "Right Elbow": 14,
    "Left Wrist": 15, "Right Wrist": 16, "Left Hip": 23, "Right Hip": 24,
    "Left Knee": 25, "Right Knee": 26, "Left Ankle": 27, "Right Ankle": 28
}

MP_ONLY_MAP = {
    "MP_Left_Pinky": 17, "MP_Right_Pinky": 18, "MP_Left_Index": 19, "MP_Right_Index": 20,
    "MP_Left_Thumb": 21, "MP_Right_Thumb": 22, "MP_Left_Heel": 29, "MP_Right_Heel": 30,
    "MP_Left_Foot_Index": 31, "MP_Right_Foot_Index": 32
}

def get_view_key(filename):
    if '_A' in filename: return 'view1'
    if '_B' in filename: return 'view2'
    if '_C' in filename: return 'view3'
    if '_D' in filename: return 'view4'
    if '_E' in filename: return 'view5'
    return 'view1'

#  모델(pose 객체) 받아오기.. // 매번 생성하지 않음
def extract_and_save(pose_model, img_path, save_folder):
    try:
        stream = open(str(img_path), "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        stream.close()
    except Exception:
        return

    if img is None: return

    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
    else:
        return

    results = pose_model.process(img_rgb)

    if not results.pose_landmarks:
        return

    target_view = get_view_key(img_path.name)
    output_data = {
        "frames": [
            {
                target_view: {
                    "pts": {},
                    "img_key": img_path.name
                }
            }
        ]
    }

    pts_container = output_data['frames'][0][target_view]['pts']

    for key, idx in SHARED_MAP.items():
        lm = results.pose_landmarks.landmark[idx]
        pts_container[key] = {
            'x': int(lm.x * w),
            'y': int(lm.y * h),
            'z': lm.z,
            'vis': lm.visibility,
            'type': 'shared'
        }

    for key, idx in MP_ONLY_MAP.items():
        lm = results.pose_landmarks.landmark[idx]
        pts_container[key] = {
            'x': int(lm.x * w),
            'y': int(lm.y * h),
            'z': lm.z,
            'vis': lm.visibility,
            'type': 'mp_only'
        }

    new_name = f"media_{img_path.stem}.json"
    save_path = save_folder / new_name

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

def main():
    print(f"이미지 경로: {OUT_IMAGES_DIR}")
    print(f"저장 경로:   {OUT_MEDIAPIPE_DIR}")

    OUT_MEDIAPIPE_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 이미지 목록 스캔 중...")
    image_files = list(OUT_IMAGES_DIR.rglob('*.jpg')) + list(OUT_IMAGES_DIR.rglob('*.png')) + list(OUT_IMAGES_DIR.rglob('*.jpeg'))

    total_files = len(image_files)
    if total_files == 0:
        print(f"이미지를 못 찾았습니다.")
        return

    print(f"총 {total_files}장의 이미지를 처리합니다.")

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as pose_model:

        start_time = time.time()
        success_cnt = 0

        for i, img_path in enumerate(image_files):
            try:
                extract_and_save(pose_model, img_path, OUT_MEDIAPIPE_DIR)
                success_cnt += 1

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    speed = (i + 1) / elapsed
                    remaining = (total_files - (i + 1)) / speed / 60
                    print(f"   [{i + 1}/{total_files}] 처리 중... (속도: {speed:.1f}장/초, 예상 잔여: {remaining:.1f}분)")

            except KeyboardInterrupt:
                print("\n사용자에 의해 중단됨!")
                break
            except Exception as e:
                print(f"에러 ({img_path.name}): {e}")

    print(f"\n작업 끝! 총 {success_cnt}개 완료.")
    print(f"확인 경로: {OUT_MEDIAPIPE_DIR}")

if __name__ == "__main__":
    main()
