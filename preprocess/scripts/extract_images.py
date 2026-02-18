import os
import sys
import tarfile
import shutil
from pathlib import Path

# config.py import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_IMAGES_DIR, OUT_IMAGES_DIR, TARGET_EXERCISES

TARGET_IDS = list(TARGET_EXERCISES.keys())  # ['27', '35']

def is_target_filename(name):
    """파일명에 27(푸시업) 또는 35(턱걸이)가 포함되어 있는지 확인"""
    parts = name.replace('_', '-').replace('.', '-').split('-')
    for tid in TARGET_IDS:
        if tid in parts:
            return True, tid
    return False, None

def process_tar(tar_path):
    print(f"분석 중 (Image): {tar_path.name}")
    count = 0

    # Training/Validation 감지
    source_type = "Validation" if "Validation" in tar_path.parts else "Training"

    try:
        with tarfile.open(tar_path, 'r') as t:
            for member in t.getmembers():
                if member.isfile():
                    is_target, tid = is_target_filename(member.name)

                    if is_target:
                        ex_name = TARGET_EXERCISES[tid]
                        save_dir = OUT_IMAGES_DIR / source_type / f"{tid}_{ex_name}" / tar_path.stem
                        save_dir.mkdir(parents=True, exist_ok=True)

                        target_path = save_dir / os.path.basename(member.name)
                        if not target_path.exists():
                            source = t.extractfile(member)
                            with open(target_path, "wb") as dest:
                                shutil.copyfileobj(source, dest)
                            count += 1
    except Exception as e:
        print(f"에러: {e}")

    if count > 0:
        print(f"  → {source_type} > {count}개 이미지 추출 완료")

def main():
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("--- 원천(이미지/영상) 데이터 추출 시작 ---")
    print(f"원본 경로: {RAW_IMAGES_DIR}")
    print(f"저장 경로: {OUT_IMAGES_DIR}")

    for d in [RAW_IMAGES_DIR / 'Training', RAW_IMAGES_DIR / 'Validation']:
        if d.exists():
            for tar_file in sorted(d.rglob('*.tar')):
                process_tar(tar_file)

    print(f"\n이미지 추출 완료: {OUT_IMAGES_DIR}")

if __name__ == "__main__":
    main()
