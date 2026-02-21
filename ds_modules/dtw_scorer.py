"""
DTW 기반 유사도 점수 모듈

모범 영상(레퍼런스)과 사용자 영상의 페이즈별 DTW 거리를 계산하여
가우시안 커널로 유사도 점수(0~1)를 산출한다.

피처: 관절 각도(정규화) + 정규화 좌표 혼합 (~47차원)
DTW: 순수 numpy 구현 (Sakoe-Chiba 밴드, 벡터화 쌍 거리 계산)
"""
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ds_modules.angle_utils import cal_angle, cal_distance

logger = logging.getLogger(__name__)


# ── 피처 추출 함수 ──────────────────────────────────────────

def extract_pushup_angles(npts: Dict[str, List[float]]) -> Optional[np.ndarray]:
    """
    푸시업 관절 각도 피처 7개를 추출한다. (0~1 정규화)

    - elbow_l, elbow_r: 좌/우 팔꿈치 각도
    - back: 등(Neck-Waist-Ankle) 각도
    - abd_l, abd_r: 좌/우 어깨 외전각
    - head_tilt: 고개 숙임 (eye_nose_y - ear_y)
    - hand_offset: |waist_x - hand_center_x|
    """
    try:
        elbow_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
        elbow_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
        back = cal_angle(npts["Neck"], npts["Waist"], npts["Ankle_C"])
        abd_l = cal_angle(npts["Left Elbow"], npts["Left Shoulder"], npts["Left Hip"])
        abd_r = cal_angle(npts["Right Elbow"], npts["Right Shoulder"], npts["Right Hip"])

        eye_nose_y = ((npts["Left Eye"][1] + npts["Right Eye"][1]) / 2 + npts["Nose"][1]) / 2
        ear_y = (npts["Left Ear"][1] + npts["Right Ear"][1]) / 2
        head_tilt = eye_nose_y - ear_y  # 이미 정규화 좌표이므로 소수값

        waist_x = npts["Waist"][0]
        hand_center_x = (npts["Left Wrist"][0] + npts["Right Wrist"][0]) / 2
        hand_offset = abs(waist_x - hand_center_x)

        return np.array([
            elbow_l / 180.0,
            elbow_r / 180.0,
            back / 180.0,
            abd_l / 180.0,
            abd_r / 180.0,
            head_tilt,
            hand_offset,
        ], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"푸시업 각도 추출 실패: {e}")
        return None


def extract_pullup_angles(npts: Dict[str, List[float]]) -> Optional[np.ndarray]:
    """
    풀업 관절 각도 피처 7개를 추출한다. (0~1 정규화)

    - elbow_l, elbow_r: 좌/우 팔꿈치 각도 (풀업 위치의 핵심 지표)
    - back: 등(Neck-Waist-Ankle) 각도
    - head_tilt: 고개 숙임
    - shoulder_packing: 어깨 패킹 (shoulder_mid_y - neck_y)
    - elbow_flare: 팔꿈치 벌림 비율 (elbow_dist / shoulder_dist)
    - body_sway: waist_x (흔들림 추적용 단일값)
    """
    try:
        elbow_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
        elbow_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
        back = cal_angle(npts["Neck"], npts["Waist"], npts["Ankle_C"])

        eye_nose_y = ((npts["Left Eye"][1] + npts["Right Eye"][1]) / 2 + npts["Nose"][1]) / 2
        ear_y = (npts["Left Ear"][1] + npts["Right Ear"][1]) / 2
        head_tilt = eye_nose_y - ear_y

        shoulder_mid_y = (npts["Left Shoulder"][1] + npts["Right Shoulder"][1]) / 2
        neck_y = npts["Neck"][1]
        shoulder_packing = shoulder_mid_y - neck_y

        elbow_dist = cal_distance(npts["Left Elbow"], npts["Right Elbow"])
        shoulder_dist = cal_distance(npts["Left Shoulder"], npts["Right Shoulder"])
        elbow_flare = elbow_dist / shoulder_dist if shoulder_dist > 1e-6 else 0.0
        elbow_flare = min(elbow_flare / 3.0, 1.0)

        body_sway = npts["Waist"][0]

        return np.array([
            elbow_l / 180.0,
            elbow_r / 180.0,
            back / 180.0,
            head_tilt,
            shoulder_packing,
            elbow_flare,
            body_sway,
        ], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"풀업 각도 추출 실패: {e}")
        return None


# 좌표 추출에 사용할 키포인트 (COCO 17 + 가상 3개 = 20개)
_COORDINATE_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
    "Neck", "Waist", "Ankle_C",
]


def extract_coordinates(npts: Dict[str, List[float]]) -> Optional[np.ndarray]:
    """
    20개 키포인트의 x,y 좌표 = 40개 값을 추출한다.
    이미 정규화된 좌표이므로 추가 정규화 불필요.
    """
    try:
        coords = []
        for kp in _COORDINATE_KEYPOINTS:
            coords.extend(npts[kp])
        return np.array(coords, dtype=np.float32)
    except (KeyError, TypeError) as e:
        logger.debug(f"좌표 추출 실패: {e}")
        return None


def extract_feature_vector(npts: Optional[Dict[str, List[float]]], exercise_type: str) -> Optional[np.ndarray]:
    """
    각도 + 좌표를 합친 피처 벡터를 반환한다.
    각도가 벡터 앞쪽에 위치하므로 DTW 비교 시 각도만 슬라이싱 가능.
    - 푸시업: 7(각도) + 40(좌표) = 47차원
    - 풀업: 7(각도) + 40(좌표) = 47차원
    """
    if npts is None:
        return None

    if exercise_type == "푸시업":
        angles = extract_pushup_angles(npts)
    elif exercise_type == "풀업":
        angles = extract_pullup_angles(npts)
    else:
        return None

    coords = extract_coordinates(npts)

    if angles is None or coords is None:
        return None

    return np.concatenate([angles, coords])


# ── DTW 핵심 함수 ────────────────────────────────────────────

def _dtw_distance(seq1: np.ndarray, seq2: np.ndarray, window: int = 0) -> float:
    """
    순수 numpy DTW 거리 계산 (Sakoe-Chiba 밴드 지원).

    Args:
        seq1: (N, D) float32 배열
        seq2: (M, D) float32 배열
        window: Sakoe-Chiba 밴드 폭 (0 = 제약 없음 = 전체 DTW)

    Returns:
        DTW 거리 (float)
    """
    n, m = len(seq1), len(seq2)

    # 전체 쌍별 L2 거리를 브로드캐스팅으로 한번에 계산 (N, M, D) → (N, M)
    diff = seq1[:, np.newaxis, :] - seq2[np.newaxis, :, :]
    cost = np.sqrt((diff * diff).sum(axis=-1))  # (N, M)

    # 누적 비용 행렬 초기화
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    acc[0, 0] = 0.0

    # 밴드 폭 결정: 0이면 전체 탐색, 최소한 |n-m|은 보장
    w = window if window > 0 else max(n, m)
    w = max(w, abs(n - m))

    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w) + 1
        for j in range(j_lo, j_hi):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j],      # 위 (ref 프레임 건너뜀)
                acc[i, j - 1],      # 왼쪽 (user 프레임 건너뜀)
                acc[i - 1, j - 1],  # 대각선 (둘 다 진행)
            )

    return float(acc[n, m])


# ── DTW Scorer 클래스 ───────────────────────────────────────

class DTWScorer:
    """
    페이즈별 DTW 유사도 점수를 계산하는 클래스.

    사용법:
        scorer = DTWScorer("reference_pushup.json", "푸시업")
        for frame in frames:
            vec = extract_feature_vector(npts, "푸시업")
            scorer.accumulate(vec, current_phase)
        result = scorer.finalize()
    """

    # 각도 피처 차원 수 (각도만으로 DTW 비교)
    _ANGLE_DIMS = {"푸시업": 7, "풀업": 7}

    def __init__(self, reference_path: str, exercise_type: str,
                 sigma: float = 0.25, window: int = 0):
        """
        Args:
            reference_path: 레퍼런스 JSON 경로
            exercise_type: "푸시업" 또는 "풀업"
            sigma: 가우시안 커널 sigma (클수록 점수가 너그러워짐)
            window: Sakoe-Chiba 밴드 폭 (0 = 전체 DTW)
        """
        self.exercise_type = exercise_type
        self.sigma = sigma
        self.window = window
        self.active = False
        self._n_angles = self._ANGLE_DIMS.get(exercise_type, 7)

        # 레퍼런스 로드 — 각도 부분만 float32 2D 배열로 사전 변환
        try:
            with open(reference_path, "r", encoding="utf-8") as f:
                ref_data = json.load(f)
            self._ref_angles: Dict[str, np.ndarray] = {}
            for phase, vectors in ref_data.get("phases", {}).items():
                arr = np.array(vectors, dtype=np.float32)        # (T, 47)
                self._ref_angles[phase] = arr[:, :self._n_angles]  # (T, 7)
            if self._ref_angles:
                self.active = True
                logger.info(f"DTW 레퍼런스 로드 완료: {reference_path} "
                            f"(phases: {list(self._ref_angles.keys())})")
            else:
                logger.warning(f"레퍼런스에 phase 데이터 없음: {reference_path}")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"DTW 레퍼런스 로드 실패: {e} — DTW 비활성화")

        # 세그먼트 누적 버퍼
        self._current_phase: Optional[str] = None
        self._current_segment: List[np.ndarray] = []
        self._phase_scores: Dict[str, List[float]] = defaultdict(list)

    def accumulate(self, feature_vec: Optional[np.ndarray], phase: str):
        """
        프레임별 호출. 페이즈가 전환되면 이전 세그먼트를 DTW로 평가한다.
        """
        if not self.active:
            return

        # 페이즈 전환 감지
        if phase != self._current_phase:
            if self._current_phase is not None and len(self._current_segment) >= 2:
                self._score_segment(self._current_phase)
            self._current_phase = phase
            self._current_segment = []

        if feature_vec is not None:
            self._current_segment.append(feature_vec)

    def _score_segment(self, phase: str):
        """
        numpy DTW로 세그먼트 거리 계산 → 가우시안 유사도 변환.

        각도 피처만 사용하여 폼 품질을 비교한다.
        좌표는 카메라 위치에 의존하므로 DTW 비교에서 제외.
        """
        if phase not in self._ref_angles:
            return
        ref_arr = self._ref_angles[phase]   # (T_r, 7) — 로드 시 사전 변환
        if len(ref_arr) == 0 or len(self._current_segment) < 2:
            return

        try:
            # 사용자 시퀀스: 각도 부분만 float32 2D 배열로 변환
            user_arr = np.array(
                [v[:self._n_angles] for v in self._current_segment],
                dtype=np.float32
            )  # (T_u, 7)

            n, m = len(user_arr), len(ref_arr)
            w = self.window if self.window > 0 else max(abs(n - m), int(max(n, m) * 0.2))
            distance = _dtw_distance(user_arr, ref_arr, window=w)

            # 평균 거리 = 총 거리 / max(두 시퀀스 길이)
            avg_distance = distance / max(len(user_arr), len(ref_arr))

            # 가우시안 커널: similarity = exp(-(d/σ)²)
            similarity = float(np.exp(-(avg_distance / self.sigma) ** 2))

            self._phase_scores[phase].append(similarity)
            logger.debug(f"DTW [{phase}] dist={distance:.2f}, avg={avg_distance:.4f}, "
                         f"sim={similarity:.4f} (user={len(user_arr)}, ref={len(ref_arr)})")

        except Exception as e:
            logger.warning(f"DTW 세그먼트 평가 실패 [{phase}]: {e}")

    def finalize(self) -> Dict:
        """
        마지막 세그먼트 평가 후 종합 결과를 반환한다.

        Returns:
            {
                "overall_dtw_score": float,           # 전체 평균 DTW 유사도
                "phase_dtw_scores": {phase: float},   # 페이즈별 평균
                "phase_segment_counts": {phase: int}, # 페이즈별 세그먼트 수
            }
        """
        if not self.active:
            return {
                "overall_dtw_score": None,
                "phase_dtw_scores": {},
                "phase_segment_counts": {},
            }

        # 마지막 세그먼트 처리
        if self._current_phase is not None and len(self._current_segment) >= 2:
            self._score_segment(self._current_phase)

        # 페이즈별 평균 점수
        phase_avg: Dict[str, float] = {}
        phase_counts: Dict[str, int] = {}
        all_scores: List[float] = []

        for phase, scores in self._phase_scores.items():
            if scores:
                phase_avg[phase] = round(sum(scores) / len(scores), 4)
                phase_counts[phase] = len(scores)
                all_scores.extend(scores)

        overall = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

        return {
            "overall_dtw_score": overall,
            "phase_dtw_scores": phase_avg,
            "phase_segment_counts": phase_counts,
        }
