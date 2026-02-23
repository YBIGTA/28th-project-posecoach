import os
import sys
import zipfile
import json
from pathlib import Path

# config.py import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_LABELS_DIR, OUT_LABELS_DIR, TARGET_EXERCISES

# TARGET_EXERCISES 
TARGET_MAP = {name: eid for eid, name in TARGET_EXERCISES.items()}

def check_json_content(json_bytes):
    """JSON 내용을 읽어 타겟 운동인지 확인"""
    try:
        data = json.loads(json_bytes.decode('utf-8'))

        # 1. type_info -> exercise 이름 확인
        type_info = data.get('type_info', {})
        exercise_name = type_info.get('exercise')

        if exercise_name in TARGET_MAP:
            return True, TARGET_MAP[exercise_name], exercise_name

        # 2. img_key 주소로 확인
        frames = data.get('frames', [])
        if frames:
            img_key = frames[0].get('view1', {}).get('img_key', '')
            if '-27-' in img_key: return True, '27', '푸시업'
            if '-35-' in img_key: return True, '35', '풀업'

    except Exception:
        pass
    return False, None, None

def process_zip(zip_path):
    print(f"분석 중 (Label): {zip_path.name}")
    count = 0

    # Training/Validation
    source_type = "Validation" if "Validation" in zip_path.parts else "Training"

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for member in z.namelist():
                if member.lower().endswith('.json'):
                    with z.open(member) as f:
                        content = f.read()
                        is_target, ex_id, ex_name = check_json_content(content)

                        if is_target:
                            save_dir = OUT_LABELS_DIR / source_type / f"{ex_id}_{ex_name}" / zip_path.stem
                            save_dir.mkdir(parents=True, exist_ok=True)

                            target_path = save_dir / os.path.basename(member)

                            with open(target_path, 'wb') as out:
                                out.write(content)
                            count += 1
    except Exception as e:
        print(f"에러: {e}")

    if count > 0:
        print(f"{source_type} > {count}개 JSON 추출 완료")

def main():
    OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    dirs = [RAW_LABELS_DIR / 'Training', RAW_LABELS_DIR / 'Validation']

    print("라벨링(JSON) 데이터 추출 시작 ---")
    print(f"원본 경로: {RAW_LABELS_DIR}")
    print(f"저장 경로: {OUT_LABELS_DIR}")
    for d in dirs:
        if d.exists():
            for zip_file in d.rglob('*.zip'):
                process_zip(zip_file)

    print(f"\n라벨링 추출 완료. 결과 확인: {OUT_LABELS_DIR}")

if __name__ == "__main__":
    main()
