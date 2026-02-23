"""
Cohen's d 기반 관절 메트릭 가중치 산출 스크립트

AI Hub 라벨 데이터에서 정자세 vs 오답 그룹의
각 관절 메트릭별 Cohen's d를 산출하고 가중치로 변환한다.

실행:
  python3 ds_modules/compute_cohens_d.py            # 푸시업 + 풀업 모두
  python3 ds_modules/compute_cohens_d.py pushup     # 푸시업만
  python3 ds_modules/compute_cohens_d.py pullup     # 풀업만
"""

import json
import math
import os
import sys

import numpy as np

# angle_utils 재활용
sys.path.insert(0, os.path.dirname(__file__))
from angle_utils import cal_angle, cal_distance, _mid

# ── 공통 설정 ─────────────────────────────────────────
_BASE = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "preprocess", "data", "extracted", "labels", "Validation",
)
IMG_W = 1920
IMG_H = 1080


# ── 공통 함수 ─────────────────────────────────────────
def _pt(pts, name):
    """키포인트를 [x, y] 리스트로 반환한다."""
    p = pts[name]
    return [p["x"], p["y"]]


def load_data(data_dir):
    """JSON 파일들을 로드하고 correct/incorrect 그룹으로 분리한다."""
    correct, incorrect = [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
            doc = json.load(f)
        desc = doc["type_info"]["description"]
        # 풀업 정자세는 "정자세 (" 로 시작
        is_correct = desc == "정자세" or desc.startswith("정자세 ")
        target = correct if is_correct else incorrect
        target.append(doc)
    return correct, incorrect


def _extract_frames(doc, view="view1"):
    """문서에서 view별 프레임 키포인트를 순회하며 yield한다."""
    for frame in doc["frames"]:
        v = frame.get(view)
        if v is not None:
            yield v["pts"]


def cohens_d(arr_correct, arr_incorrect):
    """두 배열 간 Cohen's d를 반환한다."""
    n1, n2 = len(arr_correct), len(arr_incorrect)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(arr_correct), np.mean(arr_incorrect)
    s1, s2 = np.std(arr_correct, ddof=1), np.std(arr_incorrect, ddof=1)
    s_pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled < 1e-12:
        return 0.0
    return (m1 - m2) / s_pooled


def collect_and_compute(docs, metric_fn, metric_names):
    """문서 리스트에서 메트릭별 값 배열을 수집한다."""
    arrays = {k: [] for k in metric_names}
    for doc in docs:
        for pts in _extract_frames(doc):
            try:
                m = metric_fn(pts)
            except (KeyError, ZeroDivisionError, ValueError):
                continue
            for k, v in m.items():
                arrays[k].append(v)
    return {k: np.array(v) for k, v in arrays.items()}


def run_analysis(data_dir, metric_fn, metric_names, output_path, label):
    """Cohen's d 산출 → 가중치 변환 → JSON 저장"""
    data_dir = os.path.normpath(data_dir)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"데이터 경로: {data_dir}")

    correct_docs, incorrect_docs = load_data(data_dir)
    print(f"정자세: {len(correct_docs)}개, 오답: {len(incorrect_docs)}개")

    correct_vals = collect_and_compute(correct_docs, metric_fn, metric_names)
    incorrect_vals = collect_and_compute(incorrect_docs, metric_fn, metric_names)

    print(f"정자세 샘플 수: {len(next(iter(correct_vals.values())))}개 프레임")
    print(f"오답 샘플 수: {len(next(iter(incorrect_vals.values())))}개 프레임")
    print()

    results = {}
    for metric in metric_names:
        d = cohens_d(correct_vals[metric], incorrect_vals[metric])
        results[metric] = {"d": round(d, 4), "abs_d": round(abs(d), 4)}

    total_abs_d = sum(r["abs_d"] for r in results.values())
    for r in results.values():
        r["weight"] = round(r["abs_d"] / total_abs_d, 4) if total_abs_d > 0 else 0.0

    print(f"{'메트릭':<25} {'M_correct':>10} {'M_incorrect':>12} "
          f"{'Cohen d':>9} {'|d|':>7} {'weight':>8}")
    print("-" * 75)
    for metric in metric_names:
        mc = np.mean(correct_vals[metric])
        mi = np.mean(incorrect_vals[metric])
        r = results[metric]
        print(f"{metric:<25} {mc:10.4f} {mi:12.4f} "
              f"{r['d']:9.4f} {r['abs_d']:7.4f} {r['weight']:8.4f}")

    output = {
        metric: {"d": r["d"], "weight": r["weight"]}
        for metric, r in results.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n가중치 저장: {output_path}")


# ══════════════════════════════════════════════════════
#  푸시업 메트릭
# ══════════════════════════════════════════════════════
PUSHUP_DIR = os.path.join(_BASE, "27_푸시업", "body_01")
PUSHUP_OUTPUT = os.path.join(os.path.dirname(__file__), "weights_pushup.json")
PUSHUP_METRICS = [
    "elbow_angle", "back_angle", "hand_offset",
    "head_tilt", "shoulder_abduction",
]


def pushup_metrics(pts):
    """푸시업 프레임에서 5개 메트릭을 계산한다."""
    # elbow_angle: cal_angle(Shoulder, Elbow, Wrist) 좌우 평균
    elbow_angle = (
        cal_angle(_pt(pts, "Left Shoulder"), _pt(pts, "Left Elbow"), _pt(pts, "Left Wrist"))
        + cal_angle(_pt(pts, "Right Shoulder"), _pt(pts, "Right Elbow"), _pt(pts, "Right Wrist"))
    ) / 2

    # back_angle: cal_angle(Neck, Waist, Ankle_C)
    ankle_c = _mid(_pt(pts, "Left Ankle"), _pt(pts, "Right Ankle"))
    back_angle = cal_angle(_pt(pts, "Neck"), _pt(pts, "Waist"), ankle_c)

    # hand_offset: abs(Waist.x - mid(Wrist L/R .x)) / img_width
    waist = _pt(pts, "Waist")
    wrist_mid_x = (_pt(pts, "Left Wrist")[0] + _pt(pts, "Right Wrist")[0]) / 2
    hand_offset = abs(waist[0] - wrist_mid_x) / IMG_W

    # head_tilt: (mid(눈코 y) - mid(귀 y)) / img_height
    eye_nose_y = (
        _pt(pts, "Left Eye")[1] + _pt(pts, "Right Eye")[1] + _pt(pts, "Nose")[1]
    ) / 3
    ear_y = (_pt(pts, "Left Ear")[1] + _pt(pts, "Right Ear")[1]) / 2
    head_tilt = (eye_nose_y - ear_y) / IMG_H

    # shoulder_abduction: cal_angle(Elbow, Shoulder, Hip) 좌우 평균
    shoulder_abduction = (
        cal_angle(_pt(pts, "Left Elbow"), _pt(pts, "Left Shoulder"), _pt(pts, "Left Hip"))
        + cal_angle(_pt(pts, "Right Elbow"), _pt(pts, "Right Shoulder"), _pt(pts, "Right Hip"))
    ) / 2

    return {
        "elbow_angle": elbow_angle,
        "back_angle": back_angle,
        "hand_offset": hand_offset,
        "head_tilt": head_tilt,
        "shoulder_abduction": shoulder_abduction,
    }


# ══════════════════════════════════════════════════════
#  풀업 메트릭
# ══════════════════════════════════════════════════════
PULLUP_DIR = os.path.join(_BASE, "35_풀업", "furniture_01")
PULLUP_OUTPUT = os.path.join(os.path.dirname(__file__), "weights_pullup.json")
PULLUP_METRICS = [
    "head_tilt", "shoulder_packing", "elbow_flare", "body_sway",
]


def pullup_metrics(pts):
    """
    풀업 프레임에서 4개 메트릭을 계산한다.

    | 메트릭           | 계산                                                      | 의미           |
    |------------------|-----------------------------------------------------------|----------------|
    | head_tilt        | (mid(눈코 y) - mid(귀 y)) / img_height                    | 시선/고개 방향  |
    | shoulder_packing | (mid(어깨 y) - Neck.y) / img_height                       | 숄더패킹 정도   |
    | elbow_flare      | dist(L Elbow, R Elbow) / dist(L Shoulder, R Shoulder)     | 팔꿈치 벌림 비율 |
    | body_sway        | abs(Waist.x - mid(Shoulder L/R .x)) / img_width           | 몸통 흔들림     |
    """
    # head_tilt: 시선 위쪽 유지 여부
    eye_nose_y = (
        _pt(pts, "Left Eye")[1] + _pt(pts, "Right Eye")[1] + _pt(pts, "Nose")[1]
    ) / 3
    ear_y = (_pt(pts, "Left Ear")[1] + _pt(pts, "Right Ear")[1]) / 2
    head_tilt = (eye_nose_y - ear_y) / IMG_H

    # shoulder_packing: 어깨 중점 y - Neck y (정규화)
    shoulder_mid_y = (_pt(pts, "Left Shoulder")[1] + _pt(pts, "Right Shoulder")[1]) / 2
    neck_y = _pt(pts, "Neck")[1]
    shoulder_packing = (shoulder_mid_y - neck_y) / IMG_H

    # elbow_flare: 팔꿈치 간 거리 / 어깨 간 거리
    elbow_dist = cal_distance(_pt(pts, "Left Elbow"), _pt(pts, "Right Elbow"))
    shoulder_dist = cal_distance(_pt(pts, "Left Shoulder"), _pt(pts, "Right Shoulder"))
    if shoulder_dist < 1e-6:
        elbow_flare = 1.0
    else:
        elbow_flare = elbow_dist / shoulder_dist

    # body_sway: 몸통 중심선 대비 허리 편차 (정규화)
    waist_x = _pt(pts, "Waist")[0]
    shoulder_mid_x = (_pt(pts, "Left Shoulder")[0] + _pt(pts, "Right Shoulder")[0]) / 2
    body_sway = abs(waist_x - shoulder_mid_x) / IMG_W

    return {
        "head_tilt": head_tilt,
        "shoulder_packing": shoulder_packing,
        "elbow_flare": elbow_flare,
        "body_sway": body_sway,
    }


# ══════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════
def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target in ("all", "pushup"):
        run_analysis(PUSHUP_DIR, pushup_metrics, PUSHUP_METRICS,
                     PUSHUP_OUTPUT, "푸시업 Cohen's d 분석")

    if target in ("all", "pullup"):
        run_analysis(PULLUP_DIR, pullup_metrics, PULLUP_METRICS,
                     PULLUP_OUTPUT, "풀업 Cohen's d 분석")


if __name__ == "__main__":
    main()
