"""
활동 프레임 라벨링용 샘플 CSV를 생성한다.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUT_FRAMES_DIR


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _to_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path.resolve())


def _sample_indices(total_frames: int, samples_per_video: int):
    if total_frames <= 0:
        return []
    sample_count = min(total_frames, max(1, samples_per_video))
    if sample_count == total_frames:
        return list(range(total_frames))

    # 영상 전체를 고르게 커버하도록 등간격 샘플링한다.
    raw = np.linspace(0, total_frames - 1, num=sample_count)
    idxs = sorted({int(round(x)) for x in raw.tolist()})
    return idxs


def build_label_template(frames_root: Path, samples_per_video: int, copy_dir: Path | None = None):
    rows = []
    video_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])

    for vdir in video_dirs:
        frame_files = sorted([f for f in vdir.iterdir() if f.suffix.lower() in IMAGE_EXTS])
        sample_idxs = _sample_indices(len(frame_files), samples_per_video)

        for idx in sample_idxs:
            fpath = frame_files[idx]
            rows.append(
                {
                    "frame_path": _to_rel(fpath),
                    "label": -1,  # -1은 미라벨, 0/1로 수정 후 학습한다.
                    "video_dir": _to_rel(vdir),
                    "frame_idx": idx,
                    "img_key": fpath.name,
                }
            )

            if copy_dir is not None:
                out_dir = copy_dir / vdir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fpath, out_dir / fpath.name)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="활동 프레임 라벨링 템플릿 생성")
    parser.add_argument(
        "--frames-root",
        type=str,
        default=str(OUT_FRAMES_DIR),
        help="프레임 디렉토리 루트",
    )
    parser.add_argument(
        "--samples-per-video",
        type=int,
        default=80,
        help="영상당 샘플 프레임 수",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="labels.csv",
        help="생성할 CSV 경로",
    )
    parser.add_argument(
        "--copy-dir",
        type=str,
        default="",
        help="라벨링 편의용 샘플 이미지 복사 경로(선택)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="기존 CSV가 있으면 이어붙인 뒤 중복(frame_path) 제거",
    )
    args = parser.parse_args()

    frames_root = Path(args.frames_root).expanduser()
    if not frames_root.is_absolute():
        frames_root = Path.cwd() / frames_root
    frames_root = frames_root.resolve()

    if not frames_root.exists():
        raise FileNotFoundError(f"프레임 경로가 없습니다: {frames_root}")

    copy_dir = None
    if args.copy_dir:
        copy_dir = Path(args.copy_dir).expanduser()
        if not copy_dir.is_absolute():
            copy_dir = Path.cwd() / copy_dir
        copy_dir = copy_dir.resolve()
        copy_dir.mkdir(parents=True, exist_ok=True)

    df_new = build_label_template(
        frames_root=frames_root,
        samples_per_video=args.samples_per_video,
        copy_dir=copy_dir,
    )

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.append and output_path.exists():
        df_old = pd.read_csv(output_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["frame_path"], keep="first")
    else:
        df = df_new

    df = df.sort_values(by=["video_dir", "frame_idx"]).reset_index(drop=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"샘플 행 수: {len(df)}")
    print(f"저장 완료: {output_path}")
    if copy_dir is not None:
        print(f"샘플 이미지 복사 경로: {copy_dir}")


if __name__ == "__main__":
    main()