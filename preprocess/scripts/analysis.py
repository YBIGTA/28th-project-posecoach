import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# config.py 경로 설정
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OUT_LABELS_DIR, OUT_MEDIAPIPE_DIR

# 비교할 관절 목록 (MediaPipe Key 기준)
TARGET_JOINTS = [
    "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow",
    "Left Hip", "Right Hip",
    "Left Knee", "Right Knee",
    "Left Ankle", "Right Ankle"
]

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_gt_map(label_dir):
    """
    모든 정답(GT) JSON을 읽어서 { '파일명.jpg': {좌표데이터} } 형태의 맵을 생성
    핵심: 파일 경로가 있든 없든 무조건 '파일명(basename)'을 키로 사용함.
    """
    print("📂 정답 데이터(GT) 로딩 및 매핑 중...")
    gt_map = {}
    
    json_files = list(label_dir.rglob('*.json'))
    
    for path in tqdm(json_files, desc="GT Parsing"):
        try:
            data = load_json(path)
            frames = data.get('frames', [])
            
            for frame in frames:
                # view1 ~ view5 순회
                for view_key, view_data in frame.items():
                    if 'img_key' in view_data and 'pts' in view_data:
                        # [수정 1] 안전장치: 경로가 포함되어 있어도 파일명만 추출
                        full_path = view_data['img_key']
                        fname = Path(full_path).name 
                        
                        gt_map[fname] = view_data['pts']
                                
        except Exception as e:
            print(f"⚠️ GT 로드 에러 ({path.name}): {e}")
            
    print(f"✅ 총 {len(gt_map)}개의 정답 프레임 로드 완료!")
    return gt_map

def main():
    print("--- 오차 분석 시작 (GT Map vs MediaPipe) ---")

    # 0. MediaPipe 추출 결과 존재 여부 확인
    if not OUT_MEDIAPIPE_DIR.exists() or not list(OUT_MEDIAPIPE_DIR.rglob('media_*.json')):
        print(f"MediaPipe 추출 결과가 없습니다: {OUT_MEDIAPIPE_DIR}")
        print("먼저 extract_mediapipe.py를 실행하세요:")
        print("  conda activate fitness && python preprocess/scripts/extract_mediapipe.py")
        return

    # 1. 정답 족보(Map) 생성
    gt_map = build_gt_map(OUT_LABELS_DIR)
    
    if not gt_map:
        print("❌ 로드된 정답 데이터가 없습니다. 경로를 확인해주세요.")
        return

    # 2. MediaPipe 예측 파일 목록 가져오기
    pred_files = list(OUT_MEDIAPIPE_DIR.rglob('media_*.json'))
    print(f"📄 예측 파일(MediaPipe) {len(pred_files)}개 발견")

    results = []

    # 3. 비교 시작
    for pred_path in tqdm(pred_files, desc="Calculating"):
        pred_data = load_json(pred_path)
        
        try:
            frames = pred_data.get('frames', [])
            if not frames: continue
            
            # [수정 2] View 키를 안전하게 탐색
            # frames[0] 안에 있는 모든 키(view1, view2 등)를 확인
            for view_key, view_data in frames[0].items():
                if 'img_key' not in view_data:
                    continue
                
                # [수정 3] 핵심: 예측값의 img_key도 무조건 파일명(basename)으로 변환 후 비교
                raw_img_key = view_data['img_key']
                img_name = Path(raw_img_key).name  # 경로 제거 (Normalize)
                
                # 이제 양쪽 다 껍데기를 뗐으니 안전하게 비교 가능
                if img_name in gt_map:
                    gt_pts = gt_map[img_name]
                    pred_pts = view_data.get('pts', {})
                    
                    for joint in TARGET_JOINTS:
                        if joint in gt_pts and joint in pred_pts:
                            gx, gy = gt_pts[joint]['x'], gt_pts[joint]['y']
                            px, py = pred_pts[joint]['x'], pred_pts[joint]['y']
                            
                            dist = np.sqrt((gx - px)**2 + (gy - py)**2)
                            
                            results.append({
                                "filename": img_name,
                                "joint": joint,
                                "gt_x": gx, "gt_y": gy,
                                "pred_x": px, "pred_y": py,
                                "error": round(dist, 2)
                            })
                            
        except Exception as e:
            print(f"Error ({pred_path.name}): {e}")
            continue

    # 4. 결과 저장
    if results:
        df = pd.DataFrame(results)
        save_path = "final_error_report.csv"
        df.to_csv(save_path, index=False)
        print(f"\n✅ 분석 완료! 결과 저장됨: {save_path}")
        print("\n[관절별 평균 오차 (Pixel)]")
        print(df.groupby('joint')['error'].mean().sort_values())
    else:
        print("\n⚠️ 매칭되는 데이터가 없습니다.")
        print("팁: extract_mediapipe.py가 생성한 JSON의 'img_key'와")
        print("    AI Hub 원본 JSON의 'img_key' 파일명이 일치하는지 확인해보세요.")

if __name__ == "__main__":
    main()