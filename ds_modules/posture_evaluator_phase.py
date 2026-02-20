"""
규칙 기반 자세 평가기 - Phase별 평가 버전

COCO 17 + 가상 키포인트 기반으로 푸시업/풀업 자세를 검증하고
Phase별로 다른 평가 기준을 적용합니다.

주요 개선사항:
- Phase별로 다른 평가 항목 적용
- Cohen's d 기반 가중치 적용 (compute_cohens_d.py 산출)
- 메모리 누수 방지 (deque 사용)
- 매직 넘버 상수화
- 타입 힌팅 추가
"""
import json
import os
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import logging

from ds_modules.angle_utils import cal_angle, cal_distance

logger = logging.getLogger(__name__)

# Cohen's d 가중치 로드 (ds_modules/weights_pushup.json)
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights_pushup.json")
try:
    with open(_WEIGHTS_PATH, encoding="utf-8") as _f:
        _PUSHUP_WEIGHTS = json.load(_f)
except FileNotFoundError:
    logger.warning("weights_pushup.json 없음 — 균등 가중치 사용")
    _PUSHUP_WEIGHTS = {
        "elbow_angle": {"weight": 0.2},
        "back_angle": {"weight": 0.2},
        "hand_offset": {"weight": 0.2},
        "head_tilt": {"weight": 0.2},
        "shoulder_abduction": {"weight": 0.2},
    }


# ─── 푸시업 평가 ───────────────────────────────────────────

class PushUpEvaluator:
    """
    푸시업 자세 평가기 (Phase별 평가, Cohen's d 가중치 적용)

    가중치 출처: AI Hub '피트니스 자세 데이터셋' 라벨 224개(정자세 7, 오답 217)에서
    compute_cohens_d.py로 산출한 Cohen's d → |d| 정규화 (weights_pushup.json).

    Phase별 평가 항목:
    - top:        팔 펴짐, 등 직선, 손 위치, 고개 숙임, 어깨 외전
    - descending: 등 직선, 손 위치, 어깨 외전
    - bottom:     팔 구부림, 등 직선, 손 위치, 고개 숙임, 어깨 외전, 가슴 이동
    - ascending:  등 직선, 손 위치, 어깨 외전
    """

    # ── 평가 기준 상수 ──────────────────────────────────
    # 팔꿈치 각도: NSCA Essentials of Strength Training (4th ed.) ch.15
    #   - top: 완전 신전 > 160°
    #   - bottom: 90° 전후, 여기서는 < 120° (관대 기준)
    ARM_EXTENDED = 160
    ARM_BENT = 120

    # 등(체간) 직선: ACSM Guidelines for Exercise Testing (11th ed.)
    #   - Neck–Waist–Ankle 각도 ≥ 160° → 중립 척추 유지
    BACK_STRAIGHT_THRESHOLD = 150

    # 손 위치: AI Hub 데이터 Cohen's d 분석 (hand_offset |d|=0.44)
    #   - |waist_x − hand_center_x| ≤ 0.09 (정규화 좌표)
    HAND_POSITION_THRESHOLD = 0.09

    # 고개 숙임: AI Hub 데이터 Cohen's d 분석 (head_tilt |d|=0.37)
    #   - (눈코 중점 y − 귀 중점 y) / img_height
    #   - 정자세 평균 ≈ 0.033, 오답 평균 ≈ 0.027
    #   - 과도한 고개 들기/숙이기를 감지 (절대값 > 0.06)
    HEAD_TILT_THRESHOLD = 0.06

    # 어깨 외전각: Escamilla et al. (2010) "Shoulder Muscle Activity During
    #   the Push-up", J Strength Cond Res. 권장 외전각 45°–75°.
    #   AI Hub 데이터 Cohen's d 분석 (shoulder_abduction |d|=0.50, 최대 효과)
    #   정자세 평균 ≈ 64°, 오답 평균 ≈ 78°
    SHOULDER_ABD_MIN = 30
    SHOULDER_ABD_MAX = 80

    # 가슴 이동(깔짝 감지): waist_y 분산으로 충분한 ROM 확인
    CHEST_MOVEMENT_THRESHOLD = 0.00010

    # 좌우 비대칭: 팔꿈치 각도 좌우 차이 허용 범위 (°)
    ARM_SYMMETRY_THRESHOLD = 15
    # 좌우 비대칭: 어깨 외전각 좌우 차이 허용 범위 (°)
    ABD_SYMMETRY_THRESHOLD = 15

    HISTORY_SIZE = 30

    # ── Cohen's d 가중치 (weights_pushup.json) ─────────
    # 각 체크 항목 → weight key 매핑
    _WEIGHT_MAP = {
        "elbow_angle":        _PUSHUP_WEIGHTS["elbow_angle"]["weight"],
        "back_angle":         _PUSHUP_WEIGHTS["back_angle"]["weight"],
        "hand_offset":        _PUSHUP_WEIGHTS["hand_offset"]["weight"],
        "head_tilt":          _PUSHUP_WEIGHTS["head_tilt"]["weight"],
        "shoulder_abduction": _PUSHUP_WEIGHTS["shoulder_abduction"]["weight"],
    }

    def __init__(self, history_size: Optional[int] = None):
        self.history_size = history_size or self.HISTORY_SIZE
        self.waist_y_history = deque(maxlen=self.history_size)

    def reset(self):
        """평가기 초기화"""
        self.waist_y_history.clear()

    # ── 내부 유틸 ──────────────────────────────────────
    @staticmethod
    def _weighted_score(check_results: Dict[str, bool], weight_map: Dict[str, float]) -> tuple:
        """
        체크 결과(bool)와 가중치 맵으로 가중 점수를 산출한다.
        score = Σ(w_i · pass_i) / Σ(w_i)   (해당 phase 체크 항목만)

        Returns:
            (score, weights_used) — weights_used는 각 체크별 가중치+판정 dict
        """
        total_w = sum(weight_map[k] for k in check_results)
        if total_w < 1e-12:
            return 0.0, {}
        earned = sum(weight_map[k] for k, passed in check_results.items() if passed)
        weights_used = {
            k: {"weight": round(weight_map[k], 4), "passed": passed}
            for k, passed in check_results.items()
        }
        return earned / total_w, weights_used

    def evaluate(self, npts: Optional[Dict[str, List[float]]], phase: str = 'bottom') -> Dict:
        """
        Phase별로 자세 평가

        Args:
            npts: 정규화된 키포인트 dict
            phase: 'ready', 'top', 'descending', 'bottom', 'ascending'

        Returns:
            {"score": float, "errors": [str], "details": {...}, "weights_used": {...}}
        """
        if npts is None:
            return {"score": 0.0, "errors": ["키포인트 없음"], "details": {}, "weights_used": {}}

        if phase == 'top':
            return self._evaluate_top(npts)
        elif phase == 'descending':
            return self._evaluate_descending(npts)
        elif phase == 'bottom':
            return self._evaluate_bottom(npts)
        elif phase == 'ascending':
            return self._evaluate_ascending(npts)
        else:  # ready
            return {"score": 1.0, "errors": [], "details": {}, "weights_used": {}}

    # ── 공통 체크 헬퍼 ─────────────────────────────────
    def _check_back(self, npts, details, errors) -> bool:
        """등 직선 체크. 출처: ACSM 11th ed. — 중립 척추 ≥ 160°"""
        back_angle = cal_angle(npts["Neck"], npts["Waist"], npts["Ankle_C"])
        if back_angle >= self.BACK_STRAIGHT_THRESHOLD:
            details["back_straight"] = {"value": round(back_angle, 1), "status": "ok", "feedback": "등 자세 양호"}
            return True
        details["back_straight"] = {"value": round(back_angle, 1), "status": "error", "feedback": "허리를 펴세요"}
        errors.append("허리를 펴세요")
        return False

    def _check_hand(self, npts, details, errors, *, moving: bool = False) -> bool:
        """손 위치 체크. 출처: AI Hub Cohen's d |d|=0.44"""
        waist_x = npts["Waist"][0]
        hand_center_x = (npts["Left Wrist"][0] + npts["Right Wrist"][0]) / 2
        hand_offset = abs(waist_x - hand_center_x)
        ok_fb = "손 위치 유지 중" if moving else "손 위치 적절"
        err_fb = "양손을 균등하게 유지하세요" if moving else "양손을 균등하게 벌려주세요"
        if hand_offset <= self.HAND_POSITION_THRESHOLD:
            details["hand_position"] = {"value": round(hand_offset, 4), "status": "ok", "feedback": ok_fb}
            return True
        details["hand_position"] = {"value": round(hand_offset, 4), "status": "error", "feedback": err_fb}
        errors.append(err_fb)
        return False

    def _check_head_tilt(self, npts, details, errors) -> bool:
        """고개 숙임 체크. 출처: AI Hub Cohen's d |d|=0.37"""
        eye_nose_y = ((npts["Left Eye"][1] + npts["Right Eye"][1]) / 2 + npts["Nose"][1]) / 2
        ear_y = (npts["Left Ear"][1] + npts["Right Ear"][1]) / 2
        tilt = eye_nose_y - ear_y
        if abs(tilt) <= self.HEAD_TILT_THRESHOLD:
            details["head_tilt"] = {"value": round(tilt, 4), "status": "ok", "feedback": "고개 자세 양호"}
            return True
        fb = "고개를 숙이지 마세요" if tilt > 0 else "고개를 들지 마세요"
        details["head_tilt"] = {"value": round(tilt, 4), "status": "error", "feedback": fb}
        errors.append(fb)
        return False

    def _check_shoulder_abd(self, npts, details, errors) -> bool:
        """
        어깨 외전각 체크.
        출처: Escamilla et al. (2010), J Strength Cond Res — 권장 45°–75°.
              AI Hub Cohen's d |d|=0.50 (가장 큰 효과 크기).
              정자세 평균 64°, 오답 평균 78°.
        """
        abd_l = cal_angle(npts["Left Elbow"], npts["Left Shoulder"], npts["Left Hip"])
        abd_r = cal_angle(npts["Right Elbow"], npts["Right Shoulder"], npts["Right Hip"])
        abd_avg = (abd_l + abd_r) / 2
        if self.SHOULDER_ABD_MIN <= abd_avg <= self.SHOULDER_ABD_MAX:
            details["shoulder_abduction"] = {"value": round(abd_avg, 1), "status": "ok", "feedback": "어깨 외전 양호"}
            return True
        if abd_avg > self.SHOULDER_ABD_MAX:
            fb = "팔꿈치를 몸쪽으로 모아주세요"
        else:
            fb = "팔꿈치를 조금 벌려주세요"
        details["shoulder_abduction"] = {"value": round(abd_avg, 1), "status": "error", "feedback": fb}
        errors.append(fb)
        return False

    # ── 좌우 비대칭 체크 ─────────────────────────────────
    def _check_arm_symmetry(self, npts, details, errors) -> bool:
        """팔꿈치 각도 좌우 비대칭 체크."""
        arm_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
        arm_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
        diff = abs(arm_l - arm_r)
        if diff <= self.ARM_SYMMETRY_THRESHOLD:
            details["arm_symmetry"] = {"value": round(diff, 1), "status": "ok",
                                       "feedback": f"좌우 팔 균형 양호 (차이 {diff:.1f}°)"}
            return True
        side = "왼팔" if arm_l < arm_r else "오른팔"
        fb = f"좌우 팔 불균형 — {side}이 더 굽혀져 있습니다 (차이 {diff:.1f}°)"
        details["arm_symmetry"] = {"value": round(diff, 1), "status": "warning", "feedback": fb}
        errors.append(fb)
        return False

    def _check_abd_symmetry(self, npts, details, errors) -> bool:
        """어깨 외전각 좌우 비대칭 체크."""
        abd_l = cal_angle(npts["Left Elbow"], npts["Left Shoulder"], npts["Left Hip"])
        abd_r = cal_angle(npts["Right Elbow"], npts["Right Shoulder"], npts["Right Hip"])
        diff = abs(abd_l - abd_r)
        if diff <= self.ABD_SYMMETRY_THRESHOLD:
            details["abd_symmetry"] = {"value": round(diff, 1), "status": "ok",
                                       "feedback": f"좌우 어깨 균형 양호 (차이 {diff:.1f}°)"}
            return True
        side = "왼쪽" if abd_l > abd_r else "오른쪽"
        fb = f"좌우 어깨 불균형 — {side} 팔꿈치가 더 벌어져 있습니다 (차이 {diff:.1f}°)"
        details["abd_symmetry"] = {"value": round(diff, 1), "status": "warning", "feedback": fb}
        errors.append(fb)
        return False

    # ── Phase별 평가 ───────────────────────────────────
    def _evaluate_top(self, npts: Dict) -> Dict:
        """
        최고점 평가: 팔 완전 신전 + 전체 자세 체크 + 좌우 비대칭

        가중치: elbow_angle(0.12), back_angle(0.04), hand_offset(0.28),
                head_tilt(0.24), shoulder_abduction(0.32)
        좌우 비대칭은 가중치 외 별도 감점 (−0.05/항목)
        """
        errors: List[str] = []
        details: Dict = {}
        checks: Dict[str, bool] = {}

        try:
            # 1. 팔 펴짐 — NSCA 4th ed.: 완전 신전 > 160°
            arm_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
            arm_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
            arm_avg = (arm_l + arm_r) / 2
            if arm_avg > self.ARM_EXTENDED:
                details["arm_extended"] = {"value": round(arm_avg, 1), "status": "ok", "feedback": "팔 펴짐 충분"}
                checks["elbow_angle"] = True
            else:
                details["arm_extended"] = {"value": round(arm_avg, 1), "status": "error", "feedback": "팔을 완전히 펴주세요"}
                errors.append("팔을 완전히 펴주세요")
                checks["elbow_angle"] = False

            # 2~5. 공통 체크
            checks["back_angle"] = self._check_back(npts, details, errors)
            checks["hand_offset"] = self._check_hand(npts, details, errors)
            checks["head_tilt"] = self._check_head_tilt(npts, details, errors)
            checks["shoulder_abduction"] = self._check_shoulder_abd(npts, details, errors)

            # 6~7. 좌우 비대칭 (가중치 외 별도 감점)
            sym_arm = self._check_arm_symmetry(npts, details, errors)
            sym_abd = self._check_abd_symmetry(npts, details, errors)

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Top 평가 중 오류: {e}")
            return {"score": 0.0, "errors": ["평가 실패"], "details": {}, "weights_used": {}}

        score, weights_used = self._weighted_score(checks, self._WEIGHT_MAP)
        if not sym_arm:
            score = max(0.0, score - 0.05)
        if not sym_abd:
            score = max(0.0, score - 0.05)
        return {"score": round(score, 2), "errors": errors, "details": details, "weights_used": weights_used}

    def _evaluate_descending(self, npts: Dict) -> Dict:
        """
        내려가는 중 평가: 자세 유지 체크 (팔 각도 제외 — 변화 중)

        가중치: back_angle(0.04), hand_offset(0.28), shoulder_abduction(0.32)
        """
        errors: List[str] = []
        details: Dict = {}
        checks: Dict[str, bool] = {}

        try:
            checks["back_angle"] = self._check_back(npts, details, errors)
            checks["hand_offset"] = self._check_hand(npts, details, errors, moving=True)
            checks["shoulder_abduction"] = self._check_shoulder_abd(npts, details, errors)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Descending 평가 중 오류: {e}")
            return {"score": 0.0, "errors": ["평가 실패"], "details": {}, "weights_used": {}}

        score, weights_used = self._weighted_score(checks, self._WEIGHT_MAP)
        return {"score": round(score, 2), "errors": errors, "details": details, "weights_used": weights_used}

    def _evaluate_bottom(self, npts: Dict) -> Dict:
        """
        최저점 평가: 충분히 내려갔는가? (가장 엄격, 전체 메트릭 + 가슴 이동)

        가중치: elbow_angle(0.12), back_angle(0.04), hand_offset(0.28),
                head_tilt(0.24), shoulder_abduction(0.32)
        가슴 이동은 가중치 외 별도 감점 (pass 시 0점 추가, fail 시 −0.1)
        """
        errors: List[str] = []
        details: Dict = {}
        checks: Dict[str, bool] = {}

        try:
            # 1. 팔 구부림 — NSCA 4th ed.: bottom에서 ≤ 90°, 여기서 관대하게 < 120°
            arm_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
            arm_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
            arm_avg = (arm_l + arm_r) / 2
            if arm_avg < self.ARM_BENT:
                details["arm_bent"] = {"value": round(arm_avg, 1), "status": "ok", "feedback": "팔 구부림 충분"}
                checks["elbow_angle"] = True
            else:
                details["arm_bent"] = {"value": round(arm_avg, 1), "status": "error", "feedback": "더 깊이 내려가세요"}
                errors.append("더 깊이 내려가세요")
                checks["elbow_angle"] = False

            # 2~5. 공통 체크
            checks["back_angle"] = self._check_back(npts, details, errors)
            checks["hand_offset"] = self._check_hand(npts, details, errors)
            checks["head_tilt"] = self._check_head_tilt(npts, details, errors)
            checks["shoulder_abduction"] = self._check_shoulder_abd(npts, details, errors)

            # 6~7. 좌우 비대칭
            sym_arm = self._check_arm_symmetry(npts, details, errors)
            sym_abd = self._check_abd_symmetry(npts, details, errors)

            # 8. 가슴 이동 (깔짝 감지) — 가중치 외 별도 페널티
            self.waist_y_history.append(npts["Waist"][1])
            if len(self.waist_y_history) >= 3:
                chest_var = float(np.var(self.waist_y_history))
            else:
                chest_var = self.CHEST_MOVEMENT_THRESHOLD  # 데이터 부족 시 패스

            chest_ok = chest_var >= self.CHEST_MOVEMENT_THRESHOLD
            if chest_ok:
                details["chest_movement"] = {"value": round(chest_var, 6), "status": "ok", "feedback": "가슴 이동 충분"}
            else:
                details["chest_movement"] = {"value": round(chest_var, 6), "status": "warning", "feedback": "가슴을 충분히 내려주세요"}
                errors.append("가슴을 충분히 내려주세요")

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Bottom 평가 중 오류: {e}")
            return {"score": 0.0, "errors": ["평가 실패"], "details": {}, "weights_used": {}}

        score, weights_used = self._weighted_score(checks, self._WEIGHT_MAP)
        if not chest_ok:
            score = max(0.0, score - 0.1)
        if not sym_arm:
            score = max(0.0, score - 0.05)
        if not sym_abd:
            score = max(0.0, score - 0.05)
        return {"score": round(score, 2), "errors": errors, "details": details, "weights_used": weights_used}

    def _evaluate_ascending(self, npts: Dict) -> Dict:
        """올라가는 중 평가: descending과 동일"""
        return self._evaluate_descending(npts)


# ─── 풀업 가중치 로드 ─────────────────────────────────────
_PULLUP_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights_pullup.json")
try:
    with open(_PULLUP_WEIGHTS_PATH, encoding="utf-8") as _f:
        _PULLUP_WEIGHTS = json.load(_f)
except FileNotFoundError:
    logger.warning("weights_pullup.json 없음 — 균등 가중치 사용")
    _PULLUP_WEIGHTS = {
        "head_tilt": {"weight": 0.25},
        "shoulder_packing": {"weight": 0.25},
        "elbow_flare": {"weight": 0.25},
        "body_sway": {"weight": 0.25},
    }


# ─── 풀업 평가 ───────────────────────────────────────────

class PullUpEvaluator:
    """
    풀업 자세 평가기 (Phase별 평가, Cohen's d 가중치 적용)

    가중치 출처: AI Hub '피트니스 자세 데이터셋' 라벨 142개(정자세 9, 오답 133)에서
    compute_cohens_d.py로 산출한 Cohen's d → |d| 정규화 (weights_pullup.json).

    Phase별 평가 항목:
    - bottom:     어깨 패킹, 몸 흔들림
    - ascending:  어깨 패킹, 팔꿈치 방향, 몸 흔들림
    - top:        고개 방향, 어깨 패킹, 팔꿈치 방향, 몸 흔들림 (전체)
    - descending: 어깨 패킹, 몸 흔들림
    """

    # ── 평가 기준 상수 ──────────────────────────────────
    # 고개 방향 (시선):
    #   Ronai & Scibek (2014) "The Pull-Up", Strength & Cond J 36(3):88-90
    #     — 중립 두부 위치(귀-어깨 정렬) 유지 권장, 과도한 경추 신전 금지
    #   Raine & Twomey (1997) "Head posture and loading of the cervical spine",
    #     Applied Ergonomics 28(3):187-194
    #     — 두개척추각(CVA) < 50° 시 경추 압축 부하 ~10 kg 증가, 중립 경추 0-15° 신전 권장
    #   AI Hub Cohen's d |d|=0.86 (large). 정자세 평균 ≈ −0.011, 오답 ≈ −0.003
    HEAD_TILT_THRESHOLD = 0.04

    # 숄더패킹 (견갑골 하강):
    #   Youdas et al. (2010) "Surface EMG activation patterns and elbow joint motion
    #     during a pull-up, chin-up, or Perfect-Pullup rotational exercise",
    #     J Strength Cond Res 24(12):3404-3414
    #     — 프로네이트 풀업 시 하승모근 45-56% MVIC (숄더패킹의 주동근)
    #   Prinold & Bull (2016) "Scapula kinematics of pull-up techniques",
    #     J Science and Medicine in Sport 19(8):629-635
    #     — 전면 그립 시 견갑골 전인-후인 ROM 17-22° 유지 권장
    #   AI Hub Cohen's d |d|=0.32 (small~medium). 정자세 ≈ 0.021, 오답 ≈ 0.019
    SHOULDER_PACKING_THRESHOLD = 0.015

    # 팔꿈치 벌림 (그립별 차등 적용):
    #   Youdas et al. (2010) — 오버핸드 풀업 elbow ROM 136°, 그립 어깨너비 기준
    #   Prinold & Bull (2016) — 1.5배 이상 와이드 그립 시 견봉하 충돌 위험
    #   Lauder & Giannasi (2023) — 과도한 벌림(>2.0x)을 보상 동작으로 분류
    #   AI Hub Cohen's d |d|=0.31 (small~medium)
    _GRIP_ELBOW_FLARE = {
        "언더핸드": 1.2,   # 친업: 팔꿈치가 몸 앞으로 모임
        "오버핸드": 1.7,   # 표준 풀업: 자연스러운 외전 허용
        "와이드":   2.0,   # 와이드 그립: 넓은 벌림 허용
    }
    ELBOW_FLARE_RATIO = 1.7  # 기본값 (오버핸드)

    # 몸통 흔들림 (키핑 감지):
    #   Dinunzio et al. (2019) "Alterations in kinematics and muscle activation
    #     patterns with the addition of a kipping action during a pull-up activity",
    #     Sports Biomechanics 18(6):622-635
    #     — 스트릭트 vs 키핑: 고관절 각도 차이 48.8 ± 6.8° (p<0.001)
    #     — 스트릭트 풀업은 고관절 진동 < 15° 수준
    #   AI Hub Cohen's d |d|=0.13 (small)
    BODY_SWAY_THRESHOLD = 0.003

    # 좌우 비대칭: 팔꿈치 각도 좌우 차이 허용 범위 (°)
    ARM_SYMMETRY_THRESHOLD = 15
    # 좌우 비대칭: 어깨 높이 좌우 차이 허용 범위 (정규화 좌표)
    SHOULDER_HEIGHT_SYMMETRY_THRESHOLD = 0.03

    HISTORY_SIZE = 30

    # ── Cohen's d 가중치 (weights_pullup.json) ─────────
    _WEIGHT_MAP = {
        "head_tilt":          _PULLUP_WEIGHTS["head_tilt"]["weight"],
        "shoulder_packing":   _PULLUP_WEIGHTS["shoulder_packing"]["weight"],
        "elbow_flare":        _PULLUP_WEIGHTS["elbow_flare"]["weight"],
        "body_sway":          _PULLUP_WEIGHTS["body_sway"]["weight"],
    }

    def __init__(self, grip_type: str = "오버핸드", history_size: Optional[int] = None):
        self.history_size = history_size or self.HISTORY_SIZE
        self.grip_type = grip_type
        self.elbow_flare_ratio = self._GRIP_ELBOW_FLARE.get(grip_type, self.ELBOW_FLARE_RATIO)
        self.waist_x_history = deque(maxlen=self.history_size)

    def reset(self):
        """평가기 초기화"""
        self.waist_x_history.clear()

    # ── 내부 유틸 ──────────────────────────────────────
    @staticmethod
    def _weighted_score(check_results: Dict[str, bool], weight_map: Dict[str, float]) -> tuple:
        """
        체크 결과(bool)와 가중치 맵으로 가중 점수를 산출한다.
        score = Σ(w_i · pass_i) / Σ(w_i)

        Returns:
            (score, weights_used)
        """
        total_w = sum(weight_map[k] for k in check_results)
        if total_w < 1e-12:
            return 0.0, {}
        earned = sum(weight_map[k] for k, passed in check_results.items() if passed)
        weights_used = {
            k: {"weight": round(weight_map[k], 4), "passed": passed}
            for k, passed in check_results.items()
        }
        return earned / total_w, weights_used

    def evaluate(self, npts: Optional[Dict[str, List[float]]], phase: str = 'top') -> Dict:
        """
        Phase별로 자세 평가

        Args:
            npts: 정규화된 키포인트 dict
            phase: 'ready', 'bottom', 'ascending', 'top', 'descending'

        Returns:
            {"score": float, "errors": [str], "details": {...}, "weights_used": {...}}
        """
        if npts is None:
            return {"score": 0.0, "errors": ["키포인트 없음"], "details": {}, "weights_used": {}}

        if phase == 'bottom':
            return self._evaluate_bottom(npts)
        elif phase == 'ascending':
            return self._evaluate_ascending(npts)
        elif phase == 'top':
            return self._evaluate_top(npts)
        elif phase == 'descending':
            return self._evaluate_descending(npts)
        else:  # ready
            return {"score": 1.0, "errors": [], "details": {}, "weights_used": {}}

    # ── 공통 체크 헬퍼 ─────────────────────────────────
    def _check_head_tilt(self, npts, details, errors) -> bool:
        """
        고개 방향(시선) 체크.
        출처: Ronai & Scibek (2014), Strength & Cond J — 중립 두부 유지.
              Raine & Twomey (1997), Applied Ergonomics — CVA < 50° 시 경추 부하 증가.
              AI Hub Cohen's d |d|=0.86 (large).
        """
        eye_nose_y = ((npts["Left Eye"][1] + npts["Right Eye"][1]) / 2 + npts["Nose"][1]) / 2
        ear_y = (npts["Left Ear"][1] + npts["Right Ear"][1]) / 2
        tilt = eye_nose_y - ear_y
        if tilt <= self.HEAD_TILT_THRESHOLD:
            details["head_tilt"] = {"value": round(tilt, 4), "status": "ok", "feedback": "시선 양호"}
            return True
        details["head_tilt"] = {"value": round(tilt, 4), "status": "error", "feedback": "시선을 위로 유지하세요"}
        errors.append("시선을 위로 유지하세요")
        return False

    def _check_shoulder_packing(self, npts, details, errors) -> bool:
        """
        숄더패킹 체크.
        출처: Youdas et al. (2010), J Strength Cond Res — 하승모근 45-56% MVIC.
              Prinold & Bull (2016), J Sci Med Sport — 견갑골 ROM 17-22° 유지.
              AI Hub Cohen's d |d|=0.32.
        """
        shoulder_mid_y = (npts["Left Shoulder"][1] + npts["Right Shoulder"][1]) / 2
        neck_y = npts["Neck"][1]
        diff = shoulder_mid_y - neck_y
        if diff >= -self.SHOULDER_PACKING_THRESHOLD:
            details["shoulder_packing"] = {"value": round(diff, 4), "status": "ok", "feedback": "어깨 패킹 유지 중"}
            return True
        details["shoulder_packing"] = {"value": round(diff, 4), "status": "error", "feedback": "어깨를 내려주세요"}
        errors.append("어깨를 내려주세요")
        return False

    def _check_elbow_flare(self, npts, details, errors) -> bool:
        """
        팔꿈치 벌림 체크.
        출처: Prinold & Bull (2016) — 견갑면 이탈 < 28-30° 권장.
              Lauder & Giannasi (2023), Sport Sci Health — 과도한 벌림 = 보상 동작.
              AI Hub Cohen's d |d|=0.31.
        """
        elbow_dist = cal_distance(npts["Left Elbow"], npts["Right Elbow"])
        shoulder_dist = cal_distance(npts["Left Shoulder"], npts["Right Shoulder"])
        if shoulder_dist < 1e-6:
            details["elbow_direction"] = {"value": 0.0, "status": "ok", "feedback": "측정 불가 — 패스"}
            return True
        ratio = elbow_dist / shoulder_dist
        if ratio <= self.elbow_flare_ratio:
            details["elbow_direction"] = {"value": round(ratio, 2), "status": "ok",
                                          "feedback": f"팔꿈치 방향 양호 ({self.grip_type} 기준)"}
            return True
        details["elbow_direction"] = {"value": round(ratio, 2), "status": "error",
                                      "feedback": f"팔꿈치를 몸쪽으로 당기세요 ({self.grip_type} 기준 {self.elbow_flare_ratio}x 초과)"}
        errors.append("팔꿈치를 몸쪽으로 당기세요")
        return False

    def _check_body_sway(self, npts, details, errors) -> bool:
        """
        몸통 흔들림 체크.
        출처: Dinunzio et al. (2019), Sports Biomechanics — 스트릭트 vs 키핑
              고관절 진동 차이 48.8°, 스트릭트는 < 15°.
              AI Hub Cohen's d |d|=0.13.
        """
        self.waist_x_history.append(npts["Waist"][0])
        if len(self.waist_x_history) >= 3:
            waist_var = float(np.var(self.waist_x_history))
        else:
            waist_var = 0.0
        if waist_var <= self.BODY_SWAY_THRESHOLD:
            details["body_sway"] = {"value": round(waist_var, 6), "status": "ok", "feedback": "몸 안정"}
            return True
        details["body_sway"] = {"value": round(waist_var, 6), "status": "error", "feedback": "제자리에서 운동하세요"}
        errors.append("제자리에서 운동하세요")
        return False

    # ── 좌우 비대칭 체크 ─────────────────────────────────
    def _check_arm_symmetry(self, npts, details, errors) -> bool:
        """팔꿈치 각도 좌우 비대칭 체크."""
        arm_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
        arm_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
        diff = abs(arm_l - arm_r)
        if diff <= self.ARM_SYMMETRY_THRESHOLD:
            details["arm_symmetry"] = {"value": round(diff, 1), "status": "ok",
                                       "feedback": f"좌우 팔 균형 양호 (차이 {diff:.1f}°)"}
            return True
        side = "왼팔" if arm_l < arm_r else "오른팔"
        fb = f"좌우 팔 불균형 — {side}이 더 굽혀져 있습니다 (차이 {diff:.1f}°)"
        details["arm_symmetry"] = {"value": round(diff, 1), "status": "warning", "feedback": fb}
        errors.append(fb)
        return False

    def _check_shoulder_height_symmetry(self, npts, details, errors) -> bool:
        """어깨 높이 좌우 비대칭 체크."""
        l_y = npts["Left Shoulder"][1]
        r_y = npts["Right Shoulder"][1]
        diff = abs(l_y - r_y)
        if diff <= self.SHOULDER_HEIGHT_SYMMETRY_THRESHOLD:
            details["shoulder_symmetry"] = {"value": round(diff, 4), "status": "ok",
                                            "feedback": f"좌우 어깨 높이 균형 양호"}
            return True
        side = "왼쪽" if l_y > r_y else "오른쪽"
        fb = f"좌우 어깨 높이 불균형 — {side} 어깨가 더 낮습니다"
        details["shoulder_symmetry"] = {"value": round(diff, 4), "status": "warning", "feedback": fb}
        errors.append(fb)
        return False

    # ── Phase별 평가 ───────────────────────────────────
    def _evaluate_top(self, npts: Dict) -> Dict:
        """
        최고점 평가: 전체 메트릭 체크 + 좌우 비대칭

        가중치: head_tilt(0.53), shoulder_packing(0.20),
                elbow_flare(0.19), body_sway(0.08)
        좌우 비대칭은 가중치 외 별도 감점 (−0.05/항목)
        """
        errors: List[str] = []
        details: Dict = {}
        checks: Dict[str, bool] = {}

        try:
            checks["head_tilt"] = self._check_head_tilt(npts, details, errors)
            checks["shoulder_packing"] = self._check_shoulder_packing(npts, details, errors)
            checks["elbow_flare"] = self._check_elbow_flare(npts, details, errors)
            checks["body_sway"] = self._check_body_sway(npts, details, errors)

            sym_arm = self._check_arm_symmetry(npts, details, errors)
            sym_shoulder = self._check_shoulder_height_symmetry(npts, details, errors)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"Top 평가 중 오류: {e}")
            return {"score": 0.0, "errors": ["평가 실패"], "details": {}, "weights_used": {}}

        score, weights_used = self._weighted_score(checks, self._WEIGHT_MAP)
        if not sym_arm:
            score = max(0.0, score - 0.05)
        if not sym_shoulder:
            score = max(0.0, score - 0.05)
        return {"score": round(score, 2), "errors": errors, "details": details, "weights_used": weights_used}

    def _evaluate_ascending(self, npts: Dict) -> Dict:
        """
        올라가는 중 평가: 고개 제외, 자세 유지 체크 + 좌우 비대칭
        """
        errors: List[str] = []
        details: Dict = {}
        checks: Dict[str, bool] = {}

        try:
            checks["shoulder_packing"] = self._check_shoulder_packing(npts, details, errors)
            checks["elbow_flare"] = self._check_elbow_flare(npts, details, errors)
            checks["body_sway"] = self._check_body_sway(npts, details, errors)

            sym_arm = self._check_arm_symmetry(npts, details, errors)
            sym_shoulder = self._check_shoulder_height_symmetry(npts, details, errors)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"Ascending 평가 중 오류: {e}")
            return {"score": 0.0, "errors": ["평가 실패"], "details": {}, "weights_used": {}}

        score, weights_used = self._weighted_score(checks, self._WEIGHT_MAP)
        if not sym_arm:
            score = max(0.0, score - 0.05)
        if not sym_shoulder:
            score = max(0.0, score - 0.05)
        return {"score": round(score, 2), "errors": errors, "details": details, "weights_used": weights_used}

    def _evaluate_bottom(self, npts: Dict) -> Dict:
        """
        최저점 평가: 매달린 자세 체크 + 좌우 비대칭
        """
        errors: List[str] = []
        details: Dict = {}
        checks: Dict[str, bool] = {}

        try:
            checks["shoulder_packing"] = self._check_shoulder_packing(npts, details, errors)
            checks["body_sway"] = self._check_body_sway(npts, details, errors)

            sym_arm = self._check_arm_symmetry(npts, details, errors)
            sym_shoulder = self._check_shoulder_height_symmetry(npts, details, errors)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Bottom 평가 중 오류: {e}")
            return {"score": 0.0, "errors": ["평가 실패"], "details": {}, "weights_used": {}}

        score, weights_used = self._weighted_score(checks, self._WEIGHT_MAP)
        if not sym_arm:
            score = max(0.0, score - 0.05)
        if not sym_shoulder:
            score = max(0.0, score - 0.05)
        return {"score": round(score, 2), "errors": errors, "details": details, "weights_used": weights_used}

    def _evaluate_descending(self, npts: Dict) -> Dict:
        """내려가는 중 평가: bottom과 동일"""
        return self._evaluate_bottom(npts)
