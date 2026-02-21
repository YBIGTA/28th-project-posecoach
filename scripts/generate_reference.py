"""
모범 영상 → DTW 레퍼런스 JSON 변환 스크립트

기존 파이프라인(프레임 추출 → YOLO → 전처리 → 페이즈 감지)을 그대로 사용하여
페이즈별 피처 시퀀스를 JSON으로 저장한다.

[변경사항]
- 레퍼런스 프레임 이미지를 ds_modules/ref_frames/{exercise}/ 에 보관
  → app_phase.py에서 사용자 프레임과 나란히 비교할 수 있음
- JSON에 phase별 프레임 경로 목록(phase_frame_paths) 추가

사용법:
    python scripts/generate_reference.py --video model.mp4 --exercise 푸시업
    python scripts/generate_reference.py --video model.mp4 --exercise 풀업
"""
import sys
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

import cv2


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "preprocess"))
sys.path.insert(0, str(ROOT / "preprocess" / "scripts"))
sys.path.insert(0, str(ROOT))

from config import FRAME_EXTRACT_FPS, TARGET_RESOLUTION
from video_preprocess import extract_frames
from extract_yolo_frames import process_single_frame
from utils.keypoints import load_pose_model

from ds_modules import compute_virtual_keypoints, normalize_pts, KeypointSmoother
from ds_modules.phase_detector import create_phase_detector, extract_phase_metric
from ds_modules.exercise_counter import PushUpCounter, PullUpCounter
from ds_modules.dtw_scorer import extract_feature_vector

# 레퍼런스 프레임 보관 디렉토리
REF_FRAMES_BASE = ROOT / "ds_modules" / "ref_frames"


def generate_reference(
    video_path: str,
    exercise_type: str,
    output_path: str,
    extract_fps: int = FRAME_EXTRACT_FPS,
    pose_model=None,
):
    """
    모범 영상에서 페이즈별 피처 시퀀스를 추출하여 JSON으로 저장한다.
    레퍼런스 프레임 이미지도 ds_modules/ref_frames/{exercise}/ 에 보관한다.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] 영상 파일 없음: {video_path}")
        return False

    # 레퍼런스 프레임 저장 디렉토리 (기존 삭제 후 새로 생성)
    ref_frames_dir = REF_FRAMES_BASE / exercise_type
    if ref_frames_dir.exists():
        shutil.rmtree(ref_frames_dir)
    ref_frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[1/4] 프레임 추출 중... (FPS={extract_fps})")
        extracted_count = extract_frames(video_path, ref_frames_dir, extract_fps, TARGET_RESOLUTION)
        if extracted_count == 0:
            print("[ERROR] 프레임 추출 실패")
            return False
        print(f"  -> {extracted_count}개 프레임 추출 (보관 위치: {ref_frames_dir})")

        print("[2/4] YOLO 키포인트 추출 중...")
        if pose_model is None:
            pose_model = load_pose_model()
        frame_files = sorted(
            f for f in ref_frames_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".png", ".jpeg"}
        )
        

        import gc
        all_keypoints = []
        for i, fpath in enumerate(frame_files):
            pts = process_single_frame(pose_model, fpath)
            all_keypoints.append((fpath, pts))
            if i % 10 == 0:
                gc.collect()

        success = sum(1 for _, p in all_keypoints if p is not None)
        print(f"  -> {success}/{len(frame_files)}개 키포인트 추출")

        sample_img = cv2.imread(str(frame_files[0]))
        img_h, img_w = sample_img.shape[:2] if sample_img is not None else (TARGET_RESOLUTION[1], TARGET_RESOLUTION[0])

        print("[3/4] 페이즈 감지 및 피처 추출 중...")
        smoother = KeypointSmoother(window=3)
        phase_detector = create_phase_detector(exercise_type)
        counter = PushUpCounter() if exercise_type == "푸시업" else PullUpCounter()

        phase_features: dict    = defaultdict(list)
        phase_frame_paths: dict = defaultdict(list)   # ← 추가: phase별 프레임 경로
        phase_frame_counts: dict = defaultdict(int)

        for fpath, pts in all_keypoints:
            flat     = compute_virtual_keypoints(pts)
            smoothed = smoother.smooth(flat)
            npts     = normalize_pts(smoothed, img_w, img_h) if smoothed else None

            phase_metric  = extract_phase_metric(npts, exercise_type)
            current_phase = phase_detector.update(phase_metric) if phase_metric is not None else 'ready'

            counter.update(npts, current_phase)

            if counter.is_active and npts is not None:
                vec = extract_feature_vector(npts, exercise_type)
                if vec is not None:
                    phase_features[current_phase].append(vec.tolist())
                    phase_frame_paths[current_phase].append(str(fpath))   # ← 추가
                    phase_frame_counts[current_phase] += 1

        print("[4/4] 레퍼런스 JSON 저장 중...")
        output_data = {
            "video":             video_path.name,
            "exercise_type":     exercise_type,
            "fps":               extract_fps,
            "resolution":        [img_w, img_h],
            "exercise_count":    counter.count,
            "ref_frames_dir":    str(ref_frames_dir),   # ← 추가
            "phases":            dict(phase_features),
            "phase_frame_paths": dict(phase_frame_paths),  # ← 추가
            "phase_frame_counts": dict(phase_frame_counts),
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        total_vecs = sum(len(v) for v in phase_features.values())
        print(f"\n=== 레퍼런스 생성 완료 ===")
        print(f"  파일: {output_file}")
        print(f"  운동: {exercise_type}")
        print(f"  횟수: {counter.count}회")
        print(f"  총 피처 벡터: {total_vecs}개")
        for phase, vecs in phase_features.items():
            print(f"    {phase}: {len(vecs)}개 (프레임 {len(phase_frame_paths[phase])}개 보관)")

        return True

    except Exception as e:
        print(f"[ERROR] 레퍼런스 생성 중 예외: {e}")
        # 실패 시 프레임 디렉토리 정리
        shutil.rmtree(ref_frames_dir, ignore_errors=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="모범 영상 → DTW 레퍼런스 JSON 변환")
    parser.add_argument("--video",    "-v", required=True)
    parser.add_argument("--exercise", "-e", required=True, choices=["푸시업", "풀업"])
    parser.add_argument("--output",   "-o",
                        help="출력 JSON 경로 (기본: ds_modules/reference_{exercise}.json)")
    parser.add_argument("--fps", type=int, default=FRAME_EXTRACT_FPS)
    args = parser.parse_args()

    if args.output is None:
        name = "pushup" if args.exercise == "푸시업" else "pullup"
        args.output = str(ROOT / "ds_modules" / f"reference_{name}.json")

    generate_reference(args.video, args.exercise, args.output, args.fps)


if __name__ == "__main__":
    main()