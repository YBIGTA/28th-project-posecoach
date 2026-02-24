"""
모범 영상 → DTW 레퍼런스 JSON 변환 스크립트

기존 파이프라인(프레임 추출 → YOLO → 전처리 → 페이즈 감지)을 그대로 사용하여
페이즈별 피처 시퀀스를 JSON으로 저장한다.

사용법:
    python scripts/generate_reference.py --video model.mp4 --exercise 푸시업 --output ds_modules/reference_pushup.json
    python scripts/generate_reference.py --video model.mp4 --exercise 풀업 --output ds_modules/reference_pullup.json
"""
import sys
import json
import argparse
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# 경로 설정
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "preprocess"))
sys.path.insert(0, str(ROOT / "preprocess" / "scripts"))
sys.path.insert(0, str(ROOT))

from config import FRAME_EXTRACT_FPS, TARGET_RESOLUTION
from video_preprocess import extract_frames
from extract_yolo_frames import process_single_frame, process_frame_batch
from utils.keypoints import load_pose_model

from ds_modules import compute_virtual_keypoints, normalize_pts, KeypointSmoother
from ds_modules.phase_detector import create_phase_detector, extract_phase_metric
from ds_modules.exercise_counter import PushUpCounter, PullUpCounter
from ds_modules.dtw_scorer import extract_feature_vector


def generate_reference(video_path: str, exercise_type: str, output_path: str,
                       extract_fps: int = FRAME_EXTRACT_FPS, model=None):
    """모범 영상에서 페이즈별 피처 시퀀스를 추출하여 JSON으로 저장한다."""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] 영상 파일 없음: {video_path}")
        return False

    # 임시 디렉토리에 프레임 추출
    tmp_dir = Path(tempfile.mkdtemp(prefix="ref_frames_"))
    try:
        print(f"[1/4] 프레임 추출 중... (FPS={extract_fps})")
        extracted_count = extract_frames(video_path, tmp_dir, extract_fps, TARGET_RESOLUTION)
        if extracted_count == 0:
            print("[ERROR] 프레임 추출 실패")
            return False
        print(f"  -> {extracted_count}개 프레임 추출")

        # YOLO 키포인트 추출
        print("[2/4] YOLO 키포인트 추출 중...")
        import torch
        pose_model = model if model is not None else load_pose_model()
        use_half = torch.cuda.is_available()

        frame_files = sorted(
            f for f in tmp_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".png", ".jpeg"}
        )

        # 프레임을 메모리에 한 번만 로드
        preloaded_bgr = [cv2.imread(str(f)) for f in frame_files]

        # 해상도 (첫 유효 프레임에서)
        first_valid = next((img for img in preloaded_bgr if img is not None), None)
        if first_valid is not None:
            img_h, img_w = first_valid.shape[:2]
        else:
            img_w, img_h = TARGET_RESOLUTION

        all_keypoints = process_frame_batch(pose_model, preloaded_bgr, batch_size=32, use_half=use_half)

        success = sum(1 for p in all_keypoints if p is not None)
        print(f"  -> {success}/{len(frame_files)}개 키포인트 추출")

        # 전처리 + 페이즈 감지 + 피처 추출
        print("[3/4] 페이즈 감지 및 피처 추출 중...")
        smoother = KeypointSmoother(window=3)
        phase_detector = create_phase_detector(exercise_type, fps=extract_fps)

        if exercise_type == "푸시업":
            counter = PushUpCounter(fps=extract_fps)
        else:
            counter = PullUpCounter(fps=extract_fps)

        phase_features: dict = defaultdict(list)
        phase_frame_counts: dict = defaultdict(int)

        for pts in all_keypoints:
            flat = compute_virtual_keypoints(pts)
            smoothed = smoother.smooth(flat)
            npts = normalize_pts(smoothed, img_w, img_h) if smoothed else None

            phase_metric = extract_phase_metric(npts, exercise_type)
            if phase_metric is not None:
                current_phase = phase_detector.update(phase_metric)
            else:
                current_phase = 'ready'

            counter.update(npts, current_phase)

            if counter.is_active and npts is not None:
                vec = extract_feature_vector(npts, exercise_type)
                if vec is not None:
                    phase_features[current_phase].append(vec.tolist())
                    phase_frame_counts[current_phase] += 1

        # JSON 저장
        print("[4/4] 레퍼런스 JSON 저장 중...")
        output_data = {
            "video": video_path.name,
            "exercise_type": exercise_type,
            "fps": extract_fps,
            "resolution": [img_w, img_h],
            "exercise_count": counter.count,
            "phases": dict(phase_features),
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
            print(f"    {phase}: {len(vecs)}개")

        return True

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="모범 영상 → DTW 레퍼런스 JSON 변환"
    )
    parser.add_argument("--video", "-v", required=True,
                        help="모범 영상 경로 (MP4, AVI, MOV, MKV, WEBM)")
    parser.add_argument("--exercise", "-e", required=True,
                        choices=["푸시업", "풀업"],
                        help="운동 종류")
    parser.add_argument("--output", "-o", required=True,
                        help="출력 레퍼런스 JSON 경로")
    parser.add_argument("--fps", type=int, default=FRAME_EXTRACT_FPS,
                        help=f"프레임 추출 FPS (기본: {FRAME_EXTRACT_FPS})")
    args = parser.parse_args()

    generate_reference(args.video, args.exercise, args.output, args.fps)


if __name__ == "__main__":
    main()
