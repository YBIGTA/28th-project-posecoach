"""
DTW 기반 유사도 점수 모듈 (버그 수정 버전)

수정 사항:
- self.reference = {} 초기화를 try 블록 밖으로 이동 (FileNotFoundError 시 속성 누락 버그 수정)
- _score_segment 방어 체크 강화
- FileNotFoundError 로그 메시지 개선 (레퍼런스 영상 업로드 시 활성화 안내)
"""
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ds_modules.angle_utils import cal_angle, cal_distance

logger = logging.getLogger(__name__)


def extract_pushup_angles(npts: Dict[str, List[float]]) -> Optional[np.ndarray]:
    try:
        elbow_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
        elbow_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
        back = cal_angle(npts["Neck"], npts["Waist"], npts["Ankle_C"])
        abd_l = cal_angle(npts["Left Elbow"], npts["Left Shoulder"], npts["Left Hip"])
        abd_r = cal_angle(npts["Right Elbow"], npts["Right Shoulder"], npts["Right Hip"])
        eye_nose_y = ((npts["Left Eye"][1] + npts["Right Eye"][1]) / 2 + npts["Nose"][1]) / 2
        ear_y = (npts["Left Ear"][1] + npts["Right Ear"][1]) / 2
        head_tilt = eye_nose_y - ear_y
        waist_x = npts["Waist"][0]
        hand_center_x = (npts["Left Wrist"][0] + npts["Right Wrist"][0]) / 2
        hand_offset = abs(waist_x - hand_center_x)
        return np.array([
            elbow_l / 180.0, elbow_r / 180.0, back / 180.0,
            abd_l / 180.0, abd_r / 180.0, head_tilt, hand_offset,
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"푸시업 각도 추출 실패: {e}")
        return None


def extract_pullup_angles(npts: Dict[str, List[float]]) -> Optional[np.ndarray]:
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
            elbow_l / 180.0, elbow_r / 180.0, back / 180.0,
            head_tilt, shoulder_packing, elbow_flare, body_sway,
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"풀업 각도 추출 실패: {e}")
        return None


_COORDINATE_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
    "Neck", "Waist", "Ankle_C",
]


def extract_coordinates(npts: Dict[str, List[float]]) -> Optional[np.ndarray]:
    try:
        coords = []
        for kp in _COORDINATE_KEYPOINTS:
            coords.extend(npts[kp])
        return np.array(coords, dtype=np.float64)
    except (KeyError, TypeError) as e:
        logger.debug(f"좌표 추출 실패: {e}")
        return None


def extract_feature_vector(npts: Optional[Dict[str, List[float]]], exercise_type: str) -> Optional[np.ndarray]:
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


class DTWScorer:
    _ANGLE_DIMS = {"푸시업": 7, "풀업": 7}

    def __init__(self, reference_path: str, exercise_type: str, sigma: float = 0.25):
        self.exercise_type = exercise_type
        self.sigma = sigma
        self.active = False

        # ── 핵심 수정: try 블록 밖에서 먼저 초기화 ──
        # 기존 코드는 try 안에서만 self.reference를 만들어서
        # FileNotFoundError 발생 시 self.reference 속성 자체가 없었음
        self.reference: Dict[str, List[np.ndarray]] = {}
        self._current_phase: Optional[str] = None
        self._current_segment: List[np.ndarray] = []
        self._phase_scores: Dict[str, List[float]] = defaultdict(list)

        try:
            #print("🔥 [DTW INIT] reference_path =", reference_path)
            with open(reference_path, "r", encoding="utf-8") as f:
                ref_data = json.load(f)

            loaded: Dict[str, List[np.ndarray]] = {}
            for phase, vectors in ref_data.get("phases", {}).items():
                if vectors:
                    loaded[phase] = [np.array(v, dtype=np.float64) for v in vectors]
            
            #print("🔥 [DTW INIT] loaded phases =", list(loaded.keys()))

            if loaded:
                self.reference = loaded
                self.active = True
                logger.info(
                    f"DTW 레퍼런스 로드 완료: {reference_path} "
                    f"(phases: {list(self.reference.keys())})"
                )
            else:
                logger.warning(f"레퍼런스에 phase 데이터 없음: {reference_path}")

        except FileNotFoundError:
            # JSON 없어도 레퍼런스 영상 업로드 시 analysis.py에서 직접 주입하므로 warning만
            logger.warning(
                f"DTW 레퍼런스 JSON 없음: {reference_path} "
                f"— 레퍼런스 영상 업로드 시 자동 활성화됩니다."
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"DTW 레퍼런스 로드 실패: {e} — DTW 비활성화")

    def accumulate(self, feature_vec: Optional[np.ndarray], phase: str):
        if not self.active:
            return

        #print("🔥 [ACC] phase =", phase)    

        if phase != self._current_phase:
            if self._current_phase is not None and len(self._current_segment) >= 2:
                self._score_segment(self._current_phase)
            self._current_phase = phase
            self._current_segment = []

        if feature_vec is not None:
            self._current_segment.append(feature_vec)

    def _score_segment(self, phase: str):
        # 방어 체크 강화
        if not self.reference:
            logger.debug(f"DTW [{phase}] reference 비어있음 — 스킵")
            return
        if phase not in self.reference or not self.reference[phase]:
            logger.debug(f"DTW [{phase}] 레퍼런스에 해당 phase 없음 — 스킵")
            return
        if len(self._current_segment) < 2:
            return

        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean

            # n_angles = self._ANGLE_DIMS.get(self.exercise_type, 7)
            # user_seq = [v[:n_angles] for v in self._current_segment]
            # ref_seq  = [v[:n_angles] for v in self.reference[phase]]
            user_seq = self._current_segment
            ref_seq  = self.reference[phase]

            distance, _ = fastdtw(user_seq, ref_seq, radius=1, dist=euclidean)
            avg_distance = distance / max(len(user_seq), len(ref_seq))
            similarity   = float(np.exp(-(avg_distance / self.sigma) ** 2))

            self._phase_scores[phase].append(similarity)
            logger.debug(
                f"DTW [{phase}] dist={distance:.2f}, avg={avg_distance:.4f}, "
                f"sim={similarity:.4f} (user={len(user_seq)}, ref={len(ref_seq)})"
            )

        except ImportError:
            logger.error("fastdtw 미설치 — pip install fastdtw")
            self.active = False
        except Exception as e:
            logger.warning(f"DTW 세그먼트 평가 실패 [{phase}]: {e}")

    def finalize(self) -> Dict:
        if not self.active:
            return {"overall_dtw_score": None, "phase_dtw_scores": {}, "phase_segment_counts": {}}

        if self._current_phase is not None and len(self._current_segment) >= 2:
            self._score_segment(self._current_phase)

        phase_avg: Dict[str, float] = {}
        phase_counts: Dict[str, int] = {}
        all_scores: List[float] = []

        for phase, scores in self._phase_scores.items():
            if scores:
                phase_avg[phase]    = round(sum(scores) / len(scores), 4)
                phase_counts[phase] = len(scores)
                all_scores.extend(scores)

        overall = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
        return {
            "overall_dtw_score": overall,
            "phase_dtw_scores": phase_avg,
            "phase_segment_counts": phase_counts,
        }
