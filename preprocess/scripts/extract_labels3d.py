import os
import sys
import zipfile
import json
import shutil
from pathlib import Path

# config.py import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_LABELS_DIR, OUT_LABELS_3D_DIR, TARGET_EXERCISES

TARGET_MAP = {name: eid for eid, name in TARGET_EXERCISES.items()}


def check_2d_json_content(json_bytes):
    """2D JSON 내용을 읽어 타겟 운동인지 확인"""
    try:
        data = json.loads(json_bytes.decode('utf-8'))

        # 1. type_info → exercise 이름 확인
        type_info = data.get('type_info', {})
        exercise_name = type_info.get('exercise')
        if exercise_name in TARGET_MAP:
            return True, TARGET_MAP[exercise_name], exercise_name

        # 2. img_key에서 운동 ID 확인
        frames = data.get('frames', [])
        if frames:
            img_key = frames[0].get('view1', {}).get('img_key', '')
            if '-27-' in img_key:
                return True, '27', '푸시업'
            if '-35-' in img_key:
                return True, '35', '풀업'

    except Exception:
        pass
    return False, None, None


def get_2d_counterpart(name_3d):
    """3D 파일명에서 대응하는 2D 파일명을 생성
    예: D34-1-477-3d.json → D34-1-477.json
    """
    if name_3d.endswith('-3d.json'):
        return name_3d[:-len('-3d.json')] + '.json'
    return None


# ─── ZIP 기반 처리 (Colab 환경) ───

def process_zip(zip_path):
    """ZIP 내부에서 -3d.json 파일을 찾아 대응 2D로 운동 판별 후 추출"""
    print(f"분석 중 (3D Label ZIP): {zip_path.name}")
    count = 0

    source_type = "Validation" if "Validation" in zip_path.parts else "Training"

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            all_names = set(z.namelist())

            for member in z.namelist():
                if not member.lower().endswith('-3d.json'):
                    continue

                # 대응하는 2D 파일 찾기
                counterpart_2d = get_2d_counterpart(os.path.basename(member))
                if counterpart_2d is None:
                    continue

                # ZIP 내 2D 파일 경로 구성 (같은 디렉터리)
                member_dir = os.path.dirname(member)
                counterpart_path = os.path.join(member_dir, counterpart_2d) if member_dir else counterpart_2d

                if counterpart_path not in all_names:
                    continue

                # 2D JSON 내용으로 운동 판별
                with z.open(counterpart_path) as f2d:
                    content_2d = f2d.read()
                    is_target, ex_id, ex_name = check_2d_json_content(content_2d)

                if is_target:
                    save_dir = OUT_LABELS_3D_DIR / source_type / f"{ex_id}_{ex_name}" / zip_path.stem
                    save_dir.mkdir(parents=True, exist_ok=True)

                    target_path = save_dir / os.path.basename(member)
                    with z.open(member) as f3d:
                        content_3d = f3d.read()
                        with open(target_path, 'wb') as out:
                            out.write(content_3d)
                    count += 1

    except Exception as e:
        print(f"에러: {e}")

    if count > 0:
        print(f"  → {source_type} > {count}개 3D JSON 추출 완료")


# ─── 디렉터리 기반 처리 (이미 압축 해제된 경우) ───

def process_directory(dir_path):
    """디렉터리 내 -3d.json 파일을 찾아 대응 2D로 운동 판별 후 복사"""
    print(f"분석 중 (3D Label 폴더): {dir_path.name}")
    count = 0

    source_type = "Validation" if "Validation" in dir_path.parts else "Training"

    for json3d_path in dir_path.rglob('*-3d.json'):
        counterpart_2d = get_2d_counterpart(json3d_path.name)
        if counterpart_2d is None:
            continue

        json2d_path = json3d_path.parent / counterpart_2d
        if not json2d_path.exists():
            continue

        # 2D JSON 내용으로 운동 판별
        try:
            with open(json2d_path, 'rb') as f2d:
                content_2d = f2d.read()
                is_target, ex_id, ex_name = check_2d_json_content(content_2d)
        except Exception:
            continue

        if is_target:
            # 소스 폴더 이름 결정 (상위 라벨링 폴더명)
            rel = json3d_path.relative_to(dir_path)
            # 파일이 dir_path 바로 아래에 있으면 dir_path 이름을 소스 폴더로 사용
            source_folder = rel.parts[0] if len(rel.parts) > 1 else dir_path.name

            save_dir = OUT_LABELS_3D_DIR / source_type / f"{ex_id}_{ex_name}" / source_folder
            save_dir.mkdir(parents=True, exist_ok=True)

            target_path = save_dir / json3d_path.name
            if not target_path.exists():
                shutil.copy2(json3d_path, target_path)
                count += 1

    if count > 0:
        print(f"  → {source_type} > {count}개 3D JSON 추출 완료")


def main():
    OUT_LABELS_3D_DIR.mkdir(parents=True, exist_ok=True)

    print("--- 3D 라벨링(JSON) 데이터 추출 시작 ---")
    print(f"원본 경로: {RAW_LABELS_DIR}")
    print(f"저장 경로: {OUT_LABELS_3D_DIR}")

    for d in [RAW_LABELS_DIR / 'Training', RAW_LABELS_DIR / 'Validation']:
        if not d.exists():
            continue

        # 1. ZIP 파일 처리
        zip_files = sorted(d.rglob('*.zip'))
        for zip_file in zip_files:
            process_zip(zip_file)

        # 2. 이미 추출된 디렉터리 처리 (ZIP 외의 하위 폴더)
        for sub in sorted(d.iterdir()):
            if sub.is_dir():
                process_directory(sub)

    print(f"\n3D 라벨 추출 완료. 결과 확인: {OUT_LABELS_3D_DIR}")


if __name__ == "__main__":
    main()
