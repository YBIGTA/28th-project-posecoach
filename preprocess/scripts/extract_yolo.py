"""
AI Hub 이미지 YOLO26n-pose 키포인트 추출

AI Hub 피트니스 자세 이미지에서 YOLO26n-pose를 사용하여
관절 좌표를 추출하고 JSON으로 저장한다.
"""
import cv2
import json
import sys
import numpy as np
from pathlib import Path
import time

# 경로 설정
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import OUT_IMAGES_DIR, OUT_MEDIAPIPE_DIR, TARGET_EXERCISES
from utils.keypoints import load_pose_model, yolo_result_to_dict, COCO_KEYPOINT_MAP


def get_view_key(filename):
    if '_A' in filename: return 'view1'
    if '_B' in filename: return 'view2'
    if '_C' in filename: return 'view3'
    if '_D' in filename: return 'view4'
    if '_E' in filename: return 'view5'
    return 'view1'


TARGET_IDS = list(TARGET_EXERCISES.keys())  # ['27', '35']


def get_exercise_info(img_path):
    """이미지 경로에서 Training/Validation, 운동 종류를 추출"""
    parts = img_path.parts
    source_type = "Validation" if "Validation" in parts else "Training"
    for part in parts:
        for tid in TARGET_IDS:
            if part.startswith(f"{tid}_"):
                return source_type, part
    return source_type, None


def extract_and_save(model, img_path, save_root):
    try:
        stream = open(str(img_path), "rb")
        bytes_data = bytearray(stream.read())
        numpyarray = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
        stream.close()
    except Exception:
        return

    if img is None:
        return

    if len(img.shape) != 3:
        return

    results = model(img, verbose=False)
    if not results or len(results) == 0:
        return

    pts = yolo_result_to_dict(results[0])
    if pts is None:
        return

    target_view = get_view_key(img_path.name)
    output_data = {
        "frames": [
            {
                target_view: {
                    "pts": pts,
                    "img_key": img_path.name
                }
            }
        ]
    }

    source_type, exercise_folder = get_exercise_info(img_path)
    if exercise_folder:
        save_folder = save_root / source_type / exercise_folder
    else:
        save_folder = save_root / source_type
    save_folder.mkdir(parents=True, exist_ok=True)

    new_name = f"media_{img_path.stem}.json"
    save_path = save_folder / new_name

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


def main():
    print(f"이미지 경로: {OUT_IMAGES_DIR}")
    print(f"저장 경로:   {OUT_MEDIAPIPE_DIR}")

    OUT_MEDIAPIPE_DIR.mkdir(parents=True, exist_ok=True)

    print("이미지 목록 스캔 중...")
    image_files = (
        list(OUT_IMAGES_DIR.rglob('*.jpg'))
        + list(OUT_IMAGES_DIR.rglob('*.png'))
        + list(OUT_IMAGES_DIR.rglob('*.jpeg'))
    )

    total_files = len(image_files)
    if total_files == 0:
        print(f"이미지를 못 찾았습니다.")
        return

    print(f"총 {total_files}장의 이미지를 처리합니다.")

    model = load_pose_model()

    start_time = time.time()
    success_cnt = 0

    for i, img_path in enumerate(image_files):
        try:
            extract_and_save(model, img_path, OUT_MEDIAPIPE_DIR)
            success_cnt += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                remaining = (total_files - (i + 1)) / speed / 60
                print(f"   [{i + 1}/{total_files}] 처리 중... "
                      f"(속도: {speed:.1f}장/초, 예상 잔여: {remaining:.1f}분)")

        except KeyboardInterrupt:
            print("\n사용자에 의해 중단됨!")
            break
        except Exception as e:
            print(f"에러 ({img_path.name}): {e}")

    print(f"\n작업 끝! 총 {success_cnt}개 완료.")
    print(f"확인 경로: {OUT_MEDIAPIPE_DIR}")


if __name__ == "__main__":
    main()
