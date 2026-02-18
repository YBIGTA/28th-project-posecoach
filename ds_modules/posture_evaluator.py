"""
규칙 기반 자세 평가기

COCO 17 + 가상 키포인트 기반으로 푸시업/풀업 자세를 검증하고
오류 피드백과 점수를 반환한다.

입력: normalize_pts() 결과 (정규화된 좌표 dict)
"""
import numpy as np

from ds_modules.angle_utils import cal_angle, cal_distance


# ─── 푸시업 평가 ───────────────────────────────────────────

class PushUpEvaluator:
    """
    프레임 시퀀스를 축적하면서 푸시업 자세를 평가한다.

    체크 항목:
      1. 팔 구부림: 어깨-팔꿈치-손목 각도 < 120° = 충분
      2. 등 직선:  Neck-Waist-Ankle_C 각도 < 160° = 불량
      3. 손 위치:  등 x와 손 중심 x 차이 > 0.09 = 불량
      4. 가슴 이동: waist_y 분산 < 0.00024 = 부족
    """

    def __init__(self):
        self.waist_y_history = []

    def reset(self):
        self.waist_y_history = []

    def evaluate(self, npts):
        """
        단일 프레임을 평가한다.

        Args:
            npts: 정규화된 키포인트 dict  {"Nose": [x, y], ...}

        Returns:
            dict: {
                "score": float (0~1),
                "errors": [str, ...],
                "details": {
                    "arm_bend": {"value": float, "status": str, "feedback": str},
                    ...
                }
            }
        """
        if npts is None:
            return {"score": 0.0, "errors": ["키포인트 없음"], "details": {}}

        errors = []
        details = {}
        checks_passed = 0
        total_checks = 4

        # ── 1. 팔 구부림 ──
        arm_l = cal_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
        arm_r = cal_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
        arm_avg = (arm_l + arm_r) / 2

        if arm_avg < 120:
            details["arm_bend"] = {"value": round(arm_avg, 1), "status": "ok",
                                   "feedback": "팔 구부림 충분"}
            checks_passed += 1
        else:
            details["arm_bend"] = {"value": round(arm_avg, 1), "status": "error",
                                   "feedback": "팔을 더 구부려주세요"}
            errors.append("팔을 더 구부려주세요")

        # ── 2. 등 직선 (Neck - Waist - Ankle_C) ──
        back_angle = cal_angle(npts["Neck"], npts["Waist"], npts["Ankle_C"])

        if back_angle >= 160:
            details["back_straight"] = {"value": round(back_angle, 1), "status": "ok",
                                        "feedback": "등 자세 양호"}
            checks_passed += 1
        else:
            details["back_straight"] = {"value": round(back_angle, 1), "status": "error",
                                        "feedback": "허리를 펴세요"}
            errors.append("허리를 펴세요")

        # ── 3. 손 위치 (등 x vs 손 중심 x) ──
        waist_x = npts["Waist"][0]
        hand_center_x = (npts["Left Wrist"][0] + npts["Right Wrist"][0]) / 2
        hand_offset = abs(waist_x - hand_center_x)

        if hand_offset <= 0.09:
            details["hand_position"] = {"value": round(hand_offset, 4), "status": "ok",
                                        "feedback": "손 위치 적절"}
            checks_passed += 1
        else:
            details["hand_position"] = {"value": round(hand_offset, 4), "status": "error",
                                        "feedback": "양손을 균등하게 벌려주세요"}
            errors.append("양손을 균등하게 벌려주세요")

        # ── 4. 가슴(허리) 이동 ──
        self.waist_y_history.append(npts["Waist"][1])
        if len(self.waist_y_history) >= 3:
            chest_var = float(np.var(self.waist_y_history))
        else:
            chest_var = 0.001  # 데이터 부족 시 패스

        if chest_var >= 0.00024:
            details["chest_movement"] = {"value": round(chest_var, 6), "status": "ok",
                                         "feedback": "가슴 이동 충분"}
            checks_passed += 1
        else:
            details["chest_movement"] = {"value": round(chest_var, 6), "status": "warning",
                                         "feedback": "가슴을 충분히 내려주세요"}
            errors.append("가슴을 충분히 내려주세요")

        score = checks_passed / total_checks
        return {"score": round(score, 2), "errors": errors, "details": details}


# ─── 풀업 평가 ───────────────────────────────────────────

class PullUpEvaluator:
    """
    프레임 시퀀스를 축적하면서 풀업 자세를 평가한다.

    체크 항목:
      1. 고개 숙임:  눈/코 중점 vs 귀 높이 차이 > 0.01 = 숙임
      2. 어깨 패킹: 어깨 중점 y < Neck y = 어깨 올라감
      3. 팔꿈치 방향: 수축 시 팔꿈치 거리 > 어깨 거리 × 1.3 = 벌어짐
      4. 몸 흔들림:  waist_x 분산 > 0.0006 = 과다
    """

    def __init__(self):
        self.waist_x_history = []
        self.max_elbow_dist = 0.0

    def reset(self):
        self.waist_x_history = []
        self.max_elbow_dist = 0.0

    def evaluate(self, npts):
        """
        단일 프레임을 평가한다.

        Args:
            npts: 정규화된 키포인트 dict

        Returns:
            dict: {"score": float, "errors": [str], "details": {...}}
        """
        if npts is None:
            return {"score": 0.0, "errors": ["키포인트 없음"], "details": {}}

        errors = []
        details = {}
        checks_passed = 0
        total_checks = 4

        # ── 1. 고개 숙임 ──
        eye_nose_mid_y = ((npts["Left Eye"][1] + npts["Right Eye"][1]) / 2 + npts["Nose"][1]) / 2
        ear_mid_y = (npts["Left Ear"][1] + npts["Right Ear"][1]) / 2
        head_diff = eye_nose_mid_y - ear_mid_y

        if head_diff <= 0.01:
            details["head_tilt"] = {"value": round(head_diff, 4), "status": "ok",
                                    "feedback": "고개 자세 양호"}
            checks_passed += 1
        else:
            details["head_tilt"] = {"value": round(head_diff, 4), "status": "error",
                                    "feedback": "고개를 숙이지 마세요"}
            errors.append("고개를 숙이지 마세요")

        # ── 2. 어깨 패킹 ──
        shoulder_mid_y = (npts["Left Shoulder"][1] + npts["Right Shoulder"][1]) / 2
        neck_y = npts["Neck"][1]

        if shoulder_mid_y >= neck_y - 0.005:
            details["shoulder_packing"] = {"value": round(shoulder_mid_y - neck_y, 4),
                                           "status": "ok", "feedback": "어깨 패킹 양호"}
            checks_passed += 1
        else:
            details["shoulder_packing"] = {"value": round(shoulder_mid_y - neck_y, 4),
                                           "status": "error", "feedback": "어깨를 내려주세요"}
            errors.append("어깨를 내려주세요")

        # ── 3. 팔꿈치 방향 ──
        elbow_dist = cal_distance(npts["Left Elbow"], npts["Right Elbow"])
        shoulder_dist = cal_distance(npts["Left Shoulder"], npts["Right Shoulder"])

        # 수축 상태: 어깨 y > 팔꿈치 y (올라간 상태)
        contracted = (npts["Left Shoulder"][1] > npts["Left Elbow"][1]
                      or npts["Right Shoulder"][1] > npts["Right Elbow"][1])

        if contracted:
            if shoulder_dist > 1e-6 and elbow_dist > shoulder_dist * 1.3:
                details["elbow_direction"] = {"value": round(elbow_dist / max(shoulder_dist, 1e-6), 2),
                                              "status": "error",
                                              "feedback": "팔꿈치를 몸쪽으로 당기세요"}
                errors.append("팔꿈치를 몸쪽으로 당기세요")
            else:
                details["elbow_direction"] = {"value": round(elbow_dist / max(shoulder_dist, 1e-6), 2),
                                              "status": "ok",
                                              "feedback": "팔꿈치 방향 양호"}
                checks_passed += 1
        else:
            # 이완 상태에서는 기준 팔꿈치 거리 저장
            self.max_elbow_dist = max(self.max_elbow_dist, elbow_dist)
            details["elbow_direction"] = {"value": round(elbow_dist, 4), "status": "ok",
                                          "feedback": "이완 상태"}
            checks_passed += 1

        # ── 4. 몸 흔들림 ──
        self.waist_x_history.append(npts["Waist"][0])
        if len(self.waist_x_history) >= 3:
            waist_var = float(np.var(self.waist_x_history))
        else:
            waist_var = 0.0  # 데이터 부족 시 패스

        if waist_var <= 0.0006:
            details["body_sway"] = {"value": round(waist_var, 6), "status": "ok",
                                    "feedback": "몸 안정"}
            checks_passed += 1
        else:
            details["body_sway"] = {"value": round(waist_var, 6), "status": "error",
                                    "feedback": "제자리에서 운동하세요"}
            errors.append("제자리에서 운동하세요")

        score = checks_passed / total_checks
        return {"score": round(score, 2), "errors": errors, "details": details}
