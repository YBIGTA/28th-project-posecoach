"""
영상 프레임 YOLO26n-pose 키포인트 추출

video_preprocess.py로 추출된 FHD 프레임에 YOLO26n-pose를 적용하여
영상별 시계열 관절 좌표 JSON을 생성한다.
"""
import cv2
import json
import sys
from pathlib import Path
import time

# 경로 설정
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import OUT_FRAMES_DIR, OUT_FRAMES_YP_DIR, FRAME_EXTRACT_FPS
from utils.keypoints import load_pose_model, yolo_result_to_dict

def process_single_frame(model, img_path):
    """단일 프레임에서 YOLO26n-pose 키포인트를 추출하여 dict로 반환한다."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    results = model(img, verbose=False)
    if not results or len(results) == 0:
        return None

    return yolo_result_to_dict(results[0])


def process_frame_batch(model, img_paths, batch_size=8):
    """
    여러 프레임을 배치로 묶어 YOLO pose 추론을 수행한다.

    Args:
        model: YOLO pose 모델
        img_paths: 이미지 경로 리스트
        batch_size: 배치 크기 (기본 8)

    Returns:
        list[dict | None]: 각 프레임의 키포인트 dict 또는 None
    """
    all_results = [None] * len(img_paths)

    for start in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[start : start + batch_size]
        imgs = []
        valid_indices = []

        for i, fpath in enumerate(batch_paths):
            img = cv2.imread(str(fpath))
            if img is not None:
                imgs.append(img)
                valid_indices.append(start + i)

        if not imgs:
            continue

        results = model(imgs, verbose=False)
        for res, global_idx in zip(results, valid_indices):
            pts = yolo_result_to_dict(res)
            all_results[global_idx] = pts

    return all_results


def process_video_frames(video_dir, model, save_root):
    """
    하나의 영상 디렉토리(프레임 모음)를 처리하여
    시계열 키포인트 JSON 1개를 생성한다.
    """
    frame_files = sorted(
        [f for f in video_dir.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"}]
    )

    if not frame_files:
        return 0

    video_name = video_dir.name
    frames_data = []
    success = 0

    for i, fpath in enumerate(frame_files):
        pts = process_single_frame(model, fpath)
        if pts is not None:
            frames_data.append({
                "frame_idx": i,
                "img_key": fpath.name,
                "pts": pts,
            })
            success += 1

    if not frames_data:
        print(f"  [WARN] 키포인트 추출 실패: {video_name}")
        return 0

    # 첫 프레임으로 해상도 읽기
    sample_img = cv2.imread(str(frame_files[0]))
    h, w = sample_img.shape[:2] if sample_img is not None else (1080, 1920)

    output = {
        "video": video_name,
        "resolution": [w, h],
        "fps": FRAME_EXTRACT_FPS,
        "total_frames": len(frame_files),
        "extracted_keypoints": success,
        "frames": frames_data,
    }

    save_root.mkdir(parents=True, exist_ok=True)
    save_path = save_root / f"{video_name}_keypoints.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  -> {success}/{len(frame_files)}프레임 추출 -> {save_path.name}")
    return success


def main():
    print("--- 프레임 YOLO26n-pose 키포인트 추출 ---")
    print(f"프레임 경로: {OUT_FRAMES_DIR}")
    # [수정] 출력 경로 로그 수정
    print(f"저장 경로:   {OUT_FRAMES_YP_DIR}")

    if not OUT_FRAMES_DIR.exists():
        print(f"\n프레임이 없습니다. 먼저 video_preprocess.py를 실행하세요.")
        return

    # 영상별 하위 디렉토리 탐색
    video_dirs = sorted([
        d for d in OUT_FRAMES_DIR.iterdir() if d.is_dir()
    ])

    if not video_dirs:
        print(f"\n프레임 디렉토리가 비어있습니다: {OUT_FRAMES_DIR}")
        return

    print(f"발견된 영상: {len(video_dirs)}개\n")

    # [수정] 저장 디렉토리 생성 (YP_DIR)
    OUT_FRAMES_YP_DIR.mkdir(parents=True, exist_ok=True)

    model = load_pose_model()

    start = time.time()
    total_success = 0

    for i, vdir in enumerate(video_dirs, 1):
        print(f"[{i}/{len(video_dirs)}] {vdir.name}")
        # [수정] 함수 호출 시 YP_DIR 전달
        count = process_video_frames(vdir, model, OUT_FRAMES_YP_DIR)
        total_success += count

    elapsed = time.time() - start

    print(f"\n--- 완료 ---")
    print(f"총 {total_success}개 키포인트 추출 ({elapsed:.1f}초)")
    # [수정] 완료 로그 수정
    print(f"결과: {OUT_FRAMES_YP_DIR}")


if __name__ == "__main__":
    main()