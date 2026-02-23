"""
영상 전처리 파이프라인
- 업로드된 영상을 FHD(1920x1080)로 리사이징
- 초당 1~3프레임 추출 (학습 이미지 밀도 매칭)
"""
import cv2
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    UPLOAD_VIDEO_DIR, OUT_FRAMES_DIR,
    FRAME_EXTRACT_FPS, TARGET_RESOLUTION
)

SUPPORTED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def extract_frames(video_path, output_dir, extract_fps=FRAME_EXTRACT_FPS,
                   target_resolution=TARGET_RESOLUTION):
    """
    단일 영상에서 프레임을 추출한다.

    Args:
        video_path: 입력 영상 경로
        output_dir: 프레임 저장 디렉토리
        extract_fps: 초당 추출할 프레임 수 (1~3)
        target_resolution: (width, height) 리사이징 해상도

    Returns:
        추출된 프레임 수
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] 영상을 열 수 없습니다: {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / src_fps if src_fps > 0 else 0

    print(f"  원본: {src_w}x{src_h}, {src_fps:.1f}fps, "
          f"{total_frames}프레임, {duration:.1f}초")
    print(f"  설정: {target_resolution[0]}x{target_resolution[1]}, "
          f"추출 {extract_fps}fps")

    # 프레임 간격 계산: 원본 FPS / 추출 FPS
    if src_fps <= 0:
        print(f"  [ERROR] FPS를 읽을 수 없습니다.")
        cap.release()
        return 0

    frame_interval = src_fps / extract_fps

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    extracted = 0
    frame_idx = 0
    next_extract_at = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= next_extract_at:
            # FHD 리사이징 (비율 유지하며 letterbox/pillarbox)
            resized = resize_to_fhd(frame, target_resolution)

            filename = f"{stem}_frame{extracted:06d}.jpg"
            save_path = output_dir / filename
            cv2.imwrite(str(save_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 85])

            extracted += 1
            next_extract_at += frame_interval

        frame_idx += 1

    cap.release()
    print(f"  -> {extracted}개 프레임 추출 완료")
    return extracted


def resize_to_fhd(frame, target_resolution):
    """
    프레임을 FHD 해상도로 리사이징한다.
    종횡비를 유지하면서 letterbox(검정 패딩)를 적용한다.
    """
    target_w, target_h = target_resolution
    h, w = frame.shape[:2]

    # 스케일 계산 (종횡비 유지)
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # letterbox 패딩 (검정)
    canvas = cv2.copyMakeBorder(
        resized,
        top=(target_h - new_h) // 2,
        bottom=(target_h - new_h + 1) // 2,
        left=(target_w - new_w) // 2,
        right=(target_w - new_w + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )
    # 정확한 크기 보장
    return canvas[:target_h, :target_w]


def process_all_videos(input_dir=UPLOAD_VIDEO_DIR, output_dir=OUT_FRAMES_DIR,
                       extract_fps=FRAME_EXTRACT_FPS,
                       target_resolution=TARGET_RESOLUTION):
    """
    입력 디렉토리의 모든 영상에 대해 프레임 추출을 수행한다.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
        print(f"업로드 디렉토리가 생성되었습니다: {input_dir}")
        print("영상 파일을 넣고 다시 실행하세요.")
        return

    video_files = [
        f for f in sorted(input_dir.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not video_files:
        print(f"영상 파일이 없습니다: {input_dir}")
        print(f"지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    print(f"--- 영상 전처리 시작 ---")
    print(f"입력 경로: {input_dir}")
    print(f"저장 경로: {output_dir}")
    print(f"추출 FPS: {extract_fps}, 해상도: {target_resolution[0]}x{target_resolution[1]}")
    print(f"발견된 영상: {len(video_files)}개\n")

    total_extracted = 0
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] {video_path.name}")
        # 영상별 하위 디렉토리
        video_out_dir = output_dir / video_path.stem
        count = extract_frames(video_path, video_out_dir,
                               extract_fps, target_resolution)
        total_extracted += count

    print(f"\n--- 전처리 완료 ---")
    print(f"총 {total_extracted}개 프레임 추출 ({len(video_files)}개 영상)")
    print(f"저장 경로: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="영상 전처리: FHD 리사이징 + 프레임 추출")
    parser.add_argument("--input", "-i", type=str, default=str(UPLOAD_VIDEO_DIR),
                        help="입력 영상 경로 (파일 또는 디렉토리)")
    parser.add_argument("--output", "-o", type=str, default=str(OUT_FRAMES_DIR),
                        help="프레임 저장 디렉토리")
    parser.add_argument("--fps", type=int, default=FRAME_EXTRACT_FPS,
                        choices=[1, 2, 3],
                        help="초당 추출 프레임 수 (1~3)")
    parser.add_argument("--width", type=int, default=TARGET_RESOLUTION[0],
                        help="목표 해상도 너비")
    parser.add_argument("--height", type=int, default=TARGET_RESOLUTION[1],
                        help="목표 해상도 높이")
    args = parser.parse_args()

    target_res = (args.width, args.height)
    input_path = Path(args.input)

    if input_path.is_file():
        # 단일 파일 처리
        output_dir = Path(args.output) / input_path.stem
        print(f"--- 단일 영상 전처리 ---")
        print(f"입력: {input_path}")
        extract_frames(input_path, output_dir, args.fps, target_res)
    else:
        # 디렉토리 내 전체 처리
        process_all_videos(input_path, Path(args.output), args.fps, target_res)


if __name__ == "__main__":
    main()
