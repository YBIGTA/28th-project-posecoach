"""
generate_reference_json.py
==========================
모범 영상에서 reference_pushup.json / reference_pullup.json 을 생성하는 스크립트.

사용법:
    python generate_reference_json.py \\
        --video /path/to/good_pushup.mp4 \\
        --exercise pushup \\
        --fps 10

생성 위치: ds_modules/reference_pushup.json (또는 reference_pullup.json)

JSON 구조:
    {
        "exercise": "pushup",
        "phases": {
            "top":        [[v0_dim0, v0_dim1, ...], [v1_dim0, ...], ...],
            "bottom":     [...],
            "ascending":  [...],
            "descending": [...]
        },
        "frame_counts": {"top": 42, "bottom": 38, ...},
        "generated_at": "2026-02-23T00:00:00"
    }
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "preprocess" / "scripts"))

from video_preprocess import extract_frames  # type: ignore
from extract_yolo_frames import process_single_frame  # type: ignore
from utils.keypoints import load_pose_model
from ds_modules import (
    KeypointSmoother,
    compute_virtual_keypoints,
    create_phase_detector,
    extract_feature_vector,
    extract_phase_metric,
    normalize_pts,
)

TARGET_RESOLUTION = (1920, 1080)
OUT_FRAMES_DIR    = ROOT / "data" / "frames"
DS_MODULES_DIR    = ROOT / "ds_modules"


def build_reference_json(
    video_path: Path,
    exercise_type: str,   # "pushup" or "pullup"
    extract_fps: int = 10,
    min_frames_per_phase: int = 2,
) -> dict:
    """
    모범 영상에서 phase별 feature vector 시퀀스를 추출해서 dict로 반환.
    """
    exercise_ko = {"pushup": "푸시업", "pullup": "풀업"}[exercise_type]
    img_w, img_h = TARGET_RESOLUTION

    # ── 프레임 추출 ──
    frames_dir = OUT_FRAMES_DIR / (video_path.stem + "_ref_gen")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 프레임 추출 중: {video_path.name} → {extract_fps}fps")
    extract_frames(video_path, frames_dir, extract_fps, TARGET_RESOLUTION)

    frame_files = sorted(
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not frame_files:
        raise RuntimeError("프레임 추출 실패: 프레임이 없습니다.")
    print(f"    → {len(frame_files)}개 프레임 추출 완료")

    # ── 키포인트 추출 ──
    print("[2/3] 키포인트 추출 및 feature vector 생성 중...")
    pose_model     = load_pose_model()
    smoother       = KeypointSmoother(window=3)
    phase_detector = create_phase_detector(exercise_ko, fps=extract_fps)

    phase_sequences: dict[str, list[list[float]]] = {}
    success = 0

    for i, fpath in enumerate(frame_files):
        pts     = process_single_frame(pose_model, fpath)
        flat    = compute_virtual_keypoints(pts)
        smoothed = smoother.smooth(flat)
        npts    = normalize_pts(smoothed, img_w, img_h) if smoothed else None

        phase_metric  = extract_phase_metric(npts, exercise_ko)
        current_phase = (
            phase_detector.update(phase_metric)
            if phase_metric is not None
            else phase_detector.phase
        )

        feat_vec = extract_feature_vector(npts, exercise_ko)
        if feat_vec is not None:
            phase_sequences.setdefault(current_phase, []).append(feat_vec.tolist())
            success += 1

        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(frame_files)} 완료...")

    print(f"    → 성공: {success}/{len(frame_files)} 프레임")

    # ── phase별 통계 출력 ──
    print("\n[3/3] Phase별 벡터 수:")
    for phase, vecs in sorted(phase_sequences.items()):
        status = "✅" if len(vecs) >= min_frames_per_phase else "⚠️  (부족)"
        print(f"    {phase:12s}: {len(vecs):4d}개  {status}")

    # 최소 프레임 미달 phase 경고
    thin_phases = [p for p, v in phase_sequences.items() if len(v) < min_frames_per_phase]
    if thin_phases:
        print(f"\n⚠️  다음 phase는 벡터가 부족합니다 (< {min_frames_per_phase}개): {thin_phases}")
        print("   더 긴 영상을 사용하거나 FPS를 높여 재시도하세요.")

    # 최소 기준 미달 phase 제거
    valid = {p: v for p, v in phase_sequences.items() if len(v) >= min_frames_per_phase}
    if not valid:
        raise RuntimeError("모든 phase의 벡터 수가 기준 미달입니다. 더 긴 영상을 사용하세요.")

    return {
        "exercise":     exercise_type,
        "phases":       valid,
        "frame_counts": {p: len(v) for p, v in valid.items()},
        "generated_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="DTW 레퍼런스 JSON 생성기")
    parser.add_argument("--video",    required=True, help="모범 영상 경로")
    parser.add_argument("--exercise", required=True, choices=["pushup", "pullup"],
                        help="운동 종류")
    parser.add_argument("--fps",      type=int, default=10, help="추출 FPS (기본 10)")
    parser.add_argument("--output",   default=None,
                        help="출력 JSON 경로 (기본: ds_modules/reference_{exercise}.json)")
    parser.add_argument("--min-frames", type=int, default=2,
                        help="phase당 최소 프레임 수 (기본 2)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 영상 파일을 찾을 수 없습니다: {video_path}")
        sys.exit(1)

    out_path = (
        Path(args.output)
        if args.output
        else DS_MODULES_DIR / f"reference_{args.exercise}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  DTW 레퍼런스 JSON 생성")
    print(f"  영상: {video_path.name}")
    print(f"  운동: {args.exercise}  |  FPS: {args.fps}")
    print(f"  출력: {out_path}")
    print(f"{'='*50}\n")

    try:
        ref_data = build_reference_json(
            video_path=video_path,
            exercise_type=args.exercise,
            extract_fps=args.fps,
            min_frames_per_phase=args.min_frames,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ref_data, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 저장 완료: {out_path}")
        print(f"   phases: {list(ref_data['frame_counts'].keys())}")
        print(f"   총 벡터: {sum(ref_data['frame_counts'].values())}개\n")

    except Exception as e:
        print(f"\n❌ 생성 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
