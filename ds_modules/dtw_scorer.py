"""
DTW 기반 유사도 점수 모듈

모범 영상(레퍼런스)과 사용자 영상의 페이즈별 DTW 거리를 계산하여
가우시안 커널로 유사도 점수(0~1)를 산출한다.

피처: 관절 각도(정규화) + Waist 기준 상대좌표 + 토르소 길이 정규화 (~47차원)
라이브러리: fastdtw (O(N) 근사, radius=1)

[변경사항 v2]
- extract_coordinates(): 절대좌표 → Waist 기준 상대좌표 + 토르소 길이 정규화
- _score_segment(): path 활용 → 구간별 비용 / 워핑 비율 / 차원별 기여도 분석
- finalize(): LLM 입력용 llm_context dict 추가 반환
"""
import json
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ds_modules.angle_utils import cal_angle, cal_distance

logger = logging.getLogger(__name__)

# ── 피처 이름 (차원별 기여도 레이블용) ─────────────────────
PUSHUP_FEATURE_NAMES = [
    "elbow_L", "elbow_R", "back", "abd_L", "abd_R", "head_tilt", "hand_offset",
]
PULLUP_FEATURE_NAMES = [
    "head_tilt", "shoulder_packing", "elbow_flare", "body_sway",
]

_COORDINATE_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
    "Neck", "Waist", "Ankle_C",
]
_COORD_FEATURE_NAMES = [f"{kp}_{ax}" for kp in _COORDINATE_KEYPOINTS for ax in ("x", "y")]

_MIN_TORSO_LENGTH    = 1e-6
SPEED_FAST_THRESHOLD = 0.4    # 1:1 매핑 비율 < 0.4 → 너무 빠름
SPEED_SLOW_THRESHOLD = 0.75   # 1:1 매핑 비율 > 0.75 → 너무 느림
BAD_FRAME_PERCENTILE = 80     # 상위 몇 % 를 문제 프레임으로 볼지


# ── 프레임 비교 쌍 빌더 ──────────────────────────────────────

def _build_frame_pairs(
    path: List[Tuple[int, int]],
    user_img_paths: List[str],
    ref_img_paths:  List[str],
    n: int = 5,
) -> List[Dict]:
    """
    DTW path에서 고비용 구간을 중심으로 (사용자 프레임, 레퍼런스 프레임) 쌍을 n개 선택한다.

    Returns:
        [{user_img: str, ref_img: str, user_idx: int, ref_idx: int}, ...]
    """
    if not path or not user_img_paths or not ref_img_paths:
        return []

    # 유효 인덱스 범위 클리핑
    valid_pairs = [
        (ui, ri) for ui, ri in path
        if ui < len(user_img_paths) and ri < len(ref_img_paths)
    ]
    if not valid_pairs:
        return []

    # 균등 샘플링 (전체 구간에서 n개)
    step = max(1, len(valid_pairs) // n)
    sampled = valid_pairs[::step][:n]

    return [
        {
            "user_img": user_img_paths[ui],
            "ref_img":  ref_img_paths[ri],
            "user_idx": ui,
            "ref_idx":  ri,
        }
        for ui, ri in sampled
    ]


# ── 피처 추출 함수 ──────────────────────────────────────────

def extract_pushup_angles(npts: Dict[str, List[float]]) -> Optional[np.ndarray]:
    try:
        elbow_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
        elbow_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
        back    = cal_angle(npts["Neck"], npts["Waist"], npts["Ankle_C"])
        abd_l   = cal_angle(npts["Left Elbow"],  npts["Left Shoulder"],  npts["Left Hip"])
        abd_r   = cal_angle(npts["Right Elbow"], npts["Right Shoulder"], npts["Right Hip"])

        eye_nose_y = ((npts["Left Eye"][1] + npts["Right Eye"][1]) / 2 + npts["Nose"][1]) / 2
        ear_y      = (npts["Left Ear"][1]  + npts["Right Ear"][1])  / 2
        head_tilt  = eye_nose_y - ear_y

        waist_x       = npts["Waist"][0]
        hand_center_x = (npts["Left Wrist"][0] + npts["Right Wrist"][0]) / 2
        hand_offset   = abs(waist_x - hand_center_x)

        return np.array([
            elbow_l / 180.0, elbow_r / 180.0, back / 180.0,
            abd_l / 180.0, abd_r / 180.0, head_tilt, hand_offset,
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"푸시업 각도 추출 실패: {e}")
        return None


def extract_pullup_angles(npts: Dict[str, List[float]]) -> Optional[np.ndarray]:
    try:
        eye_nose_y = ((npts["Left Eye"][1] + npts["Right Eye"][1]) / 2 + npts["Nose"][1]) / 2
        ear_y      = (npts["Left Ear"][1]  + npts["Right Ear"][1])  / 2
        head_tilt  = eye_nose_y - ear_y

        shoulder_mid_y   = (npts["Left Shoulder"][1] + npts["Right Shoulder"][1]) / 2
        neck_y           = npts["Neck"][1]
        shoulder_packing = shoulder_mid_y - neck_y

        elbow_dist    = cal_distance(npts["Left Elbow"], npts["Right Elbow"])
        shoulder_dist = cal_distance(npts["Left Shoulder"], npts["Right Shoulder"])
        elbow_flare   = elbow_dist / shoulder_dist if shoulder_dist > 1e-6 else 0.0
        elbow_flare   = min(elbow_flare / 3.0, 1.0)

        body_sway = npts["Waist"][0]

        return np.array([head_tilt, shoulder_packing, elbow_flare, body_sway], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as e:
        logger.debug(f"풀업 각도 추출 실패: {e}")
        return None


def extract_coordinates(npts: Dict[str, List[float]]) -> Optional[np.ndarray]:
    """Waist 기준 상대좌표 + 토르소 길이 정규화 (40차원)"""
    try:
        waist = np.array(npts["Waist"], dtype=np.float64)
        neck  = np.array(npts["Neck"],  dtype=np.float64)
        torso_length = float(np.linalg.norm(neck - waist))
        if torso_length < _MIN_TORSO_LENGTH:
            return None
        coords = []
        for kp_name in _COORDINATE_KEYPOINTS:
            kp       = np.array(npts[kp_name], dtype=np.float64)
            relative = (kp - waist) / torso_length
            coords.extend(relative.tolist())
        return np.array(coords, dtype=np.float64)
    except (KeyError, TypeError) as e:
        logger.debug(f"좌표 추출 실패: {e}")
        return None


def extract_feature_vector(npts: Optional[Dict], exercise_type: str) -> Optional[np.ndarray]:
    """각도 + 상대좌표 피처 벡터 (푸시업 47차원 / 풀업 44차원)"""
    if npts is None:
        return None
    angles = extract_pushup_angles(npts) if exercise_type == "푸시업" else extract_pullup_angles(npts)
    coords = extract_coordinates(npts)
    if angles is None or coords is None:
        return None
    return np.concatenate([angles, coords])


def _get_angle_feature_names(exercise_type: str) -> List[str]:
    return PUSHUP_FEATURE_NAMES if exercise_type == "푸시업" else PULLUP_FEATURE_NAMES


# ── DTW path 분석 헬퍼 ─────────────────────────────────────

def _analyze_path(
    path: List[Tuple[int, int]],
    user_seq: List[np.ndarray],
    ref_seq:  List[np.ndarray],
    exercise_type: str,
) -> Dict:
    """
    DTW path를 분석해 3가지 추가 정보를 반환한다.

    Returns:
        {
            "speed":             "fast" | "normal" | "slow",
            "warping_ratio":     float,   # 1:1 매핑 비율 (높을수록 속도 유사)
            "worst_user_frames": [int],   # 고비용 사용자 프레임 인덱스 (최대 10개)
            "worst_features":    [        # 각도 피처 중 기여도 상위 3개
                {"name": str, "avg_diff": float}
            ],
        }
    """
    angle_names   = _get_angle_feature_names(exercise_type)
    n_angle_feats = len(angle_names)

    # ── 1. 포인트별 비용 계산 ──────────────────────────────
    pointwise: List[Tuple[int, float]] = []
    one_to_one = 0
    for i, (ui, ri) in enumerate(path):
        cost = float(np.linalg.norm(user_seq[ui] - ref_seq[ri]))
        pointwise.append((ui, cost))
        if i > 0 and path[i][0] != path[i-1][0] and path[i][1] != path[i-1][1]:
            one_to_one += 1

    # ── 2. 워핑 비율 → 속도 판정 ──────────────────────────
    warping_ratio = one_to_one / len(path) if path else 0.0
    if warping_ratio < SPEED_FAST_THRESHOLD:
        speed = "fast"
    elif warping_ratio > SPEED_SLOW_THRESHOLD:
        speed = "slow"
    else:
        speed = "normal"

    # ── 3. 고비용 프레임 (상위 BAD_FRAME_PERCENTILE %) ────
    costs_arr  = np.array([c for _, c in pointwise])
    threshold  = np.percentile(costs_arr, BAD_FRAME_PERCENTILE)
    worst_user_frames = sorted(set(
        ui for ui, c in pointwise if c >= threshold
    ))[:10]

    # ── 4. 차원별 기여도 (각도 피처만) ────────────────────
    dim_costs = np.zeros(n_angle_feats)
    for ui, ri in path:
        diff       = np.abs(user_seq[ui][:n_angle_feats] - ref_seq[ri][:n_angle_feats])
        dim_costs += diff
    dim_costs /= (len(path) + 1e-12)

    top_indices   = np.argsort(dim_costs)[::-1][:3]
    worst_features = [
        {"name": angle_names[i], "avg_diff": round(float(dim_costs[i]), 4)}
        for i in top_indices
        if dim_costs[i] > 1e-4
    ]

    return {
        "speed":             speed,
        "warping_ratio":     round(warping_ratio, 3),
        "worst_user_frames": worst_user_frames,
        "worst_features":    worst_features,
    }


# ── DTW Scorer 클래스 ───────────────────────────────────────

class DTWScorer:
    """
    페이즈별 DTW 유사도 점수 + LLM 입력용 분석 정보를 산출하는 클래스.

    finalize() 반환값:
        {
            "overall_dtw_score":    float,
            "phase_dtw_scores":     {phase: float},
            "phase_segment_counts": {phase: int},
            "llm_context":          {           ← LLM 프롬프트용 구조화 정보
                "exercise":               str,
                "phase_details":          {phase: {...}},
                "overall_worst_features": [...],
            }
        }
    """

    def __init__(self, reference_path: str, exercise_type: str, sigma: float = 0.5):
        self.exercise_type = exercise_type
        self.sigma         = sigma
        self.active        = False

        try:
            with open(reference_path, "r", encoding="utf-8") as f:
                ref_data = json.load(f)
            self.reference: Dict[str, List[np.ndarray]] = {}
            for phase, vectors in ref_data.get("phases", {}).items():
                self.reference[phase] = [np.array(v, dtype=np.float64) for v in vectors]
            if self.reference:
                self.active = True
                logger.info(f"DTW 레퍼런스 로드 완료: {reference_path} "
                            f"(phases: {list(self.reference.keys())})")
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"DTW 레퍼런스 로드 실패: {e} — DTW 비활성화")

        self._current_phase:      Optional[str]          = None
        self._current_segment:    List[np.ndarray]       = []
        self._current_img_paths:  List[str]              = []
        self._phase_scores:       Dict[str, List[float]] = defaultdict(list)
        self._phase_analysis:     Dict[str, List[Dict]]  = defaultdict(list)
        # user img_path → (ref_img_path, phase) 전체 매핑 (프레임 슬라이더용)
        self._frame_mapping:      Dict[str, Dict]        = {}

        # 레퍼런스 JSON에서 phase별 프레임 경로 로드
        self._ref_frame_paths: Dict[str, List[str]] = {}
        try:
            with open(reference_path, "r", encoding="utf-8") as _f:
                _ref = json.load(_f)
            self._ref_frame_paths = _ref.get("phase_frame_paths", {})
        except Exception:
            pass

    def accumulate(
        self,
        feature_vec: Optional[np.ndarray],
        phase: str,
        img_path: Optional[str] = None,
    ):
        """
        프레임별 호출. 페이즈가 전환되면 이전 세그먼트를 DTW로 평가한다.
        img_path: 현재 프레임 이미지 경로 (비교 뷰 용)
        """
        if not self.active:
            return
        if phase != self._current_phase:
            if self._current_phase is not None and len(self._current_segment) >= 2:
                self._score_segment(self._current_phase)
            self._current_phase     = phase
            self._current_segment   = []
            self._current_img_paths = []
        if feature_vec is not None:
            self._current_segment.append(feature_vec)
            if img_path:
                self._current_img_paths.append(img_path)

    def _score_segment(self, phase: str):
        if phase not in self.reference or not self.reference[phase]:
            return
        if len(self._current_segment) < 2:
            return

        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean

            user_seq = self._current_segment
            ref_seq  = self.reference[phase]

            distance, path = fastdtw(user_seq, ref_seq, radius=1, dist=euclidean)

            avg_distance        = distance / max(len(user_seq), len(ref_seq))
            dim                 = len(user_seq[0]) if user_seq else 1
            normalized_distance = avg_distance / np.sqrt(dim)
            similarity          = float(np.exp(-(normalized_distance / self.sigma) ** 2))

            self._phase_scores[phase].append(similarity)

            # path 분석
            analysis = _analyze_path(path, user_seq, ref_seq, self.exercise_type)
            analysis["score"]       = round(similarity, 4)
            analysis["segment_len"] = len(user_seq)

            # 프레임 비교 쌍 (고비용 구간 위주 5쌍)
            ref_paths  = self._ref_frame_paths.get(phase, [])
            user_paths = self._current_img_paths
            if ref_paths and user_paths:
                analysis["frame_pairs"] = _build_frame_pairs(
                    path, user_paths, ref_paths, n=5
                )
            else:
                analysis["frame_pairs"] = []

            self._phase_analysis[phase].append(analysis)

            # 전체 path → user_img → ref_img 매핑 저장 (슬라이더용)
            ref_paths  = self._ref_frame_paths.get(phase, [])
            user_paths = self._current_img_paths
            if ref_paths and user_paths:
                for ui, ri in path:
                    if ui < len(user_paths) and ri < len(ref_paths):
                        user_img = user_paths[ui]
                        if user_img not in self._frame_mapping:
                            self._frame_mapping[user_img] = {
                                "ref_img": ref_paths[ri],
                                "phase":   phase,
                                "user_idx": ui,
                                "ref_idx":  ri,
                            }

            logger.debug(
                f"DTW [{phase}] sim={similarity:.4f} speed={analysis['speed']} "
                f"worst={[f['name'] for f in analysis['worst_features']]}"
            )

        except ImportError:
            logger.error("fastdtw 미설치 — pip install fastdtw")
            self.active = False
        except Exception as e:
            logger.warning(f"DTW 세그먼트 평가 실패 [{phase}]: {e}")

    def finalize(self) -> Dict:
        if not self.active:
            return {
                "overall_dtw_score":    None,
                "phase_dtw_scores":     {},
                "phase_segment_counts": {},
                "llm_context":          {},
            }

        if self._current_phase is not None and len(self._current_segment) >= 2:
            self._score_segment(self._current_phase)

        phase_avg:    Dict[str, float] = {}
        phase_counts: Dict[str, int]   = {}
        all_scores:   List[float]      = []

        for phase, scores in self._phase_scores.items():
            if scores:
                phase_avg[phase]    = round(sum(scores) / len(scores), 4)
                phase_counts[phase] = len(scores)
                all_scores.extend(scores)

        overall     = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
        llm_context = self._build_llm_context(phase_avg)

        return {
            "overall_dtw_score":     overall,
            "phase_dtw_scores":      phase_avg,
            "phase_segment_counts":  phase_counts,
            "llm_context":           llm_context,
            "_phase_analysis_raw":   dict(self._phase_analysis),
            "frame_mapping":         self._frame_mapping,  # user_img → ref_img 전체 매핑
        }

    def _build_llm_context(self, phase_avg: Dict[str, float]) -> Dict:
        """
        LLM 프롬프트에 넣을 구조화된 분석 결과.

        {
            "exercise": "푸시업",
            "phase_details": {
                "bottom": {
                    "dtw_score":      0.61,
                    "speed":          "fast",       # 지배적 속도 패턴
                    "worst_features": [             # 가장 문제된 관절 top3
                        {"name": "elbow_L", "avg_diff": 0.12},
                        ...
                    ],
                    "bad_frame_ratio": 0.23,        # 문제 프레임 비율
                },
                ...
            },
            "overall_worst_features": [             # 전체 운동에서 top3
                {"name": "back", "avg_diff": 0.09},
                ...
            ],
        }
        """
        phase_details: Dict = {}
        global_feature_costs:  Dict[str, float] = defaultdict(float)
        global_feature_counts: Dict[str, int]   = defaultdict(int)

        for phase, analyses in self._phase_analysis.items():
            if not analyses:
                continue

            # 속도 빈도 집계
            speed_counts: Dict[str, int] = defaultdict(int)
            for a in analyses:
                speed_counts[a["speed"]] += 1
            dominant_speed = max(speed_counts, key=speed_counts.get)

            # worst_features 합산
            feature_costs: Dict[str, float] = defaultdict(float)
            feature_cnt:   Dict[str, int]   = defaultdict(int)
            total_frames = 0
            bad_frames   = 0

            for a in analyses:
                for wf in a["worst_features"]:
                    feature_costs[wf["name"]] += wf["avg_diff"]
                    feature_cnt[wf["name"]]   += 1
                    global_feature_costs[wf["name"]]  += wf["avg_diff"]
                    global_feature_counts[wf["name"]] += 1
                total_frames += a["segment_len"]
                bad_frames   += len(a["worst_user_frames"])

            sorted_feats = sorted(feature_costs.items(), key=lambda x: x[1], reverse=True)[:3]
            worst_features = [
                {"name": name, "avg_diff": round(cost / max(feature_cnt[name], 1), 4)}
                for name, cost in sorted_feats
            ]

            phase_details[phase] = {
                "dtw_score":      phase_avg.get(phase, 0.0),
                "speed":          dominant_speed,
                "worst_features": worst_features,
                "bad_frame_ratio": round(bad_frames / max(total_frames, 1), 3),
            }

        # 전체 worst top3
        sorted_global = sorted(global_feature_costs.items(), key=lambda x: x[1], reverse=True)[:3]
        overall_worst = [
            {"name": name, "avg_diff": round(cost / max(global_feature_counts[name], 1), 4)}
            for name, cost in sorted_global
        ]

        return {
            "exercise":               self.exercise_type,
            "phase_details":          phase_details,
            "overall_worst_features": overall_worst,
        }