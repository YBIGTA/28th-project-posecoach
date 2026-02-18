"""
DTW 기반 유사도 점수 모듈

모범 영상(레퍼런스)과 사용자 영상의 페이즈별 DTW 거리를 계산하여
가우시안 커널로 유사도 점수(0~1)를 산출한다.

피처: 관절 각도(정규화) + 정규화 좌표 혼합 (~47차원)
라이브러리: fastdtw (O(N) 근사, radius=1)
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
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"푸시업 각도 추출 실패: {e}")
        return None


def extract_pullup_angles(npts: Dict[str, List[float]]) -> Optional[np.ndarray]:
    """
    풀업 관절 각도 피처 4개를 추출한다. (0~1 정규화)

    - head_tilt: 고개 숙임
    - shoulder_packing: 어깨 패킹 (shoulder_mid_y - neck_y)
    - elbow_flare: 팔꿈치 벌림 비율 (elbow_dist / shoulder_dist)
    - body_sway: waist_x (흔들림 추적용 단일값)
    """
    try:
        eye_nose_y = ((npts["Left Eye"][1] + npts["Right Eye"][1]) / 2 + npts["Nose"][1]) / 2
        ear_y = (npts["Left Ear"][1] + npts["Right Ear"][1]) / 2
        head_tilt = eye_nose_y - ear_y

        shoulder_mid_y = (npts["Left Shoulder"][1] + npts["Right Shoulder"][1]) / 2
        neck_y = npts["Neck"][1]
        shoulder_packing = shoulder_mid_y - neck_y

        elbow_dist = cal_distance(npts["Left Elbow"], npts["Right Elbow"])
        shoulder_dist = cal_distance(npts["Left Shoulder"], npts["Right Shoulder"])
        elbow_flare = elbow_dist / shoulder_dist if shoulder_dist > 1e-6 else 0.0
        # 비율값을 0~1로 클리핑 (일반적으로 0.5~2.0 범위 → /3.0으로 정규화)
        elbow_flare = min(elbow_flare / 3.0, 1.0)

        body_sway = npts["Waist"][0]

        return np.array([
            head_tilt,
            shoulder_packing,
            elbow_flare,
            body_sway,
        ], dtype=np.float64)
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
        return np.array(coords, dtype=np.float64)
    except (KeyError, TypeError) as e:
        logger.debug(f"좌표 추출 실패: {e}")
        return None


def extract_feature_vector(npts: Optional[Dict[str, List[float]]], exercise_type: str) -> Optional[np.ndarray]:
    """
    각도 + 좌표를 합친 피처 벡터를 반환한다.
    - 푸시업: 7(각도) + 40(좌표) = 47차원
    - 풀업: 4(각도) + 40(좌표) = 44차원
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

    def __init__(self, reference_path: str, exercise_type: str, sigma: float = 0.5):
        self.exercise_type = exercise_type
        self.sigma = sigma
        self.active = False

        # 레퍼런스 로드
        try:
            with open(reference_path, "r", encoding="utf-8") as f:
                ref_data = json.load(f)
            self.reference: Dict[str, List[List[float]]] = {}
            for phase, vectors in ref_data.get("phases", {}).items():
                self.reference[phase] = [np.array(v, dtype=np.float64) for v in vectors]
            if self.reference:
                self.active = True
                logger.info(f"DTW 레퍼런스 로드 완료: {reference_path} "
                            f"(phases: {list(self.reference.keys())})")
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
            # 이전 세그먼트 평가
            if self._current_phase is not None and len(self._current_segment) >= 2:
                self._score_segment(self._current_phase)
            # 새 세그먼트 시작
            self._current_phase = phase
            self._current_segment = []

        # 피처 축적
        if feature_vec is not None:
            self._current_segment.append(feature_vec)

    def _score_segment(self, phase: str):
        """fastdtw로 세그먼트 거리 계산 → 가우시안 유사도 변환"""
        if phase not in self.reference or not self.reference[phase]:
            return
        if len(self._current_segment) < 2:
            return

        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean

            user_seq = self._current_segment
            ref_seq = self.reference[phase]

            distance, _ = fastdtw(user_seq, ref_seq, radius=1, dist=euclidean)

            # 평균 거리 = 총 거리 / max(두 시퀀스 길이)
            avg_distance = distance / max(len(user_seq), len(ref_seq))

            # 차원별 평균 거리로 정규화 (다차원 유클리드 보정)
            dim = len(user_seq[0]) if user_seq else 1
            normalized_distance = avg_distance / np.sqrt(dim)

            # 가우시안 커널: similarity = exp(-(d/σ)²)
            similarity = np.exp(-(normalized_distance / self.sigma) ** 2)

            self._phase_scores[phase].append(float(similarity))
            logger.debug(f"DTW [{phase}] dist={distance:.2f}, avg={avg_distance:.4f}, "
                         f"sim={similarity:.4f} (user={len(user_seq)}, ref={len(ref_seq)})")

        except ImportError:
            logger.error("fastdtw 미설치 — pip install fastdtw")
            self.active = False
        except Exception as e:
            logger.warning(f"DTW 세그먼트 평가 실패 [{phase}]: {e}")

    def finalize(self) -> Dict:
        """
        마지막 세그먼트 평가 후 종합 결과를 반환한다.

        Returns:
            {
                "overall_dtw_score": float,      # 전체 평균 DTW 유사도
                "phase_dtw_scores": {phase: float},  # 페이즈별 평균
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
