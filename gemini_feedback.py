"""
AIFit - Gemini API 기반 운동 자세 피드백 생성기

사용법:
    - GEMINI_API_KEY 환경변수 설정 또는 직접 입력
    - SYSTEM_PROMPT / USER_PROMPT_TEMPLATE 수정으로 프롬프트 편집 가능
    - generate_feedback(analysis_results) 호출로 피드백 생성
"""

import os
import json
import logging
import traceback
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

# .env 파일을 항상 로드 (호출 시점마다 최신값 반영)
load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  🔑 API 설정
# ══════════════════════════════════════════════════════════════

# ※ 모듈 로드 시점이 아닌 generate_feedback() 호출 시점에 읽으므로
#   여기서는 기본값만 선언합니다.
GEMINI_MODEL: str = "gemini-2.5-flash"  # 모델 변경 가능: gemini-1.5-pro, gemini-2.0-flash 등
GEMINI_API_URL: str = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

# ══════════════════════════════════════════════════════════════
#  ✏️ 프롬프트 편집 구역 (자유롭게 수정하세요)
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT: str = """
당신은 전문 퍼스널 트레이너이자 운동 생체역학 전문가입니다.
AI 영상 분석 시스템이 추출한 운동 자세 데이터를 바탕으로
사용자에게 구체적이고 실용적인 피드백을 한국어로 제공합니다.

피드백 작성 원칙:
1. 잘한 점을 먼저 짧게 칭찬하고, 개선점을 구체적으로 설명하세요.
2. 수치 데이터(각도, 점수 등)를 직접 언급하여 신뢰감을 높이세요.
3. "더 깊이 내려가세요" 같은 단순 지적보다 "팔꿈치 각도가 {값}°인데, 90° 아래로 내려가면 가슴근육 자극이 증가합니다" 처럼 이유를 설명하세요.
4. Phase별 특이사항이 있으면 단계별로 정리하세요.
5. 전체 총평은 3~5문장으로 마무리하세요.
6. 마크다운 형식(##, **볼드**, - 목록)을 활용하세요.
7. DTW 유사도 점수가 있으면 모범 영상 대비 유사도를 언급하세요.
"""

# 아래 {중괄호} 변수들은 코드에서 자동으로 치환됩니다. 수정 시 유지하세요.
USER_PROMPT_TEMPLATE: str = """
## 분석 대상 운동 정보

- **운동 종류**: {exercise_type}
- **운동 횟수**: {exercise_count}회
- **평균 자세 점수**: {avg_posture_score:.1%}
- **등급**: {grade}
- **영상 FPS**: {fps}fps, **총 프레임**: {total_frames}개

## 점수 요약

{score_summary}

## Phase별 평균 점수

{phase_avg_summary}

## 주요 오류 목록 (빈도 상위)

{top_errors_summary}

## 상세 수치 샘플 (오류 프레임 기준)

{detail_samples}

## DTW 유사도 (모범 영상 대비)

{dtw_summary}

---

위 데이터를 바탕으로 이 사용자의 {exercise_type} 자세에 대한 종합 피드백을 작성해주세요.
Phase별 문제점, 개선 방법, 긍정적인 부분을 균형 있게 포함하세요.
"""

# ══════════════════════════════════════════════════════════════
#  📊 데이터 전처리 함수
# ══════════════════════════════════════════════════════════════

def _calc_avg_score(frame_scores: list) -> float:
    """frame_scores 리스트에서 평균 점수를 계산한다."""
    scores = [fs["score"] for fs in frame_scores if "score" in fs]
    return sum(scores) / len(scores) if scores else 0.0


def _get_grade(avg_score: float) -> str:
    if avg_score >= 0.9:
        return "S"
    elif avg_score >= 0.7:
        return "A"
    elif avg_score >= 0.5:
        return "B"
    return "C"


def _score_summary(avg_score: float, frame_scores: list) -> str:
    if not frame_scores:
        return "프레임 데이터 없음"
    scores = [fs["score"] for fs in frame_scores]
    return (
        f"- 평균: {avg_score:.1%}\n"
        f"- 최고: {max(scores):.1%}\n"
        f"- 최저: {min(scores):.1%}\n"
        f"- 분석 프레임 수: {len(scores)}개"
    )


def _phase_avg_summary(frame_scores: list) -> str:
    """Phase별 평균 점수를 문자열로 반환한다."""
    phase_data: Dict[str, list] = {}
    for fs in frame_scores:
        p = fs.get("phase", "unknown")
        phase_data.setdefault(p, []).append(fs["score"])
    if not phase_data:
        return "Phase 데이터 없음"
    lines = []
    for phase, scores in sorted(phase_data.items()):
        avg = sum(scores) / len(scores)
        lines.append(f"- **{phase}**: {avg:.1%} ({len(scores)}프레임)")
    return "\n".join(lines)


def _top_errors_summary(frame_scores: list, top_n: int = 8) -> str:
    """오류 메시지 빈도 상위 N개를 반환한다."""
    error_counts: Dict[str, int] = {}
    for fs in frame_scores:
        for err in fs.get("errors", []):
            error_counts[err] = error_counts.get(err, 0) + 1
    if not error_counts:
        return "감지된 오류 없음 ✅"
    sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])[:top_n]
    return "\n".join(f"- {err} ({cnt}회)" for err, cnt in sorted_errors)


def _detail_samples(error_frames: list, max_samples: int = 3) -> str:
    """오류 프레임에서 상세 수치 샘플을 추출한다."""
    if not error_frames:
        return "오류 프레임 없음"
    samples = error_frames[:max_samples]
    lines = []
    for ef in samples:
        lines.append(f"### 프레임 {ef['frame_idx']} [{ef.get('phase','?')}] — 점수 {ef['score']:.1%}")
        details = ef.get("details", {})
        for k, v in details.items():
            status_icon = "✅" if v["status"] == "ok" else "⚠️" if v["status"] == "warning" else "❌"
            lines.append(f"  {status_icon} {k}: {v['value']} — {v['feedback']}")
    return "\n".join(lines)


def _dtw_summary(dtw_result: Optional[dict]) -> str:
    """DTW 결과를 문자열로 요약한다."""
    if not dtw_result or dtw_result.get("overall_dtw_score") is None:
        return "DTW 레퍼런스 미적용 (모범 영상 없음)"
    overall = dtw_result["overall_dtw_score"]
    phase_scores = dtw_result.get("phase_dtw_scores", {})
    lines = [f"- 전체 유사도: {overall:.1%}"]
    for phase, score in sorted(phase_scores.items()):
        lines.append(f"  - {phase}: {score:.1%}")
    return "\n".join(lines)


def build_prompt(analysis_results: dict) -> str:
    """
    analysis_results(session_state['analysis_results'])를 받아
    Gemini에 전달할 최종 유저 프롬프트를 생성한다.
    """
    frame_scores = analysis_results.get("frame_scores", [])
    error_frames = analysis_results.get("error_frames", [])
    dtw_result = analysis_results.get("dtw_result")

    avg_score = _calc_avg_score(frame_scores)
    grade = _get_grade(avg_score)

    return USER_PROMPT_TEMPLATE.format(
        exercise_type=analysis_results.get("exercise_type", "알 수 없음"),
        exercise_count=analysis_results.get("exercise_count", 0),
        avg_posture_score=avg_score,
        grade=grade,
        fps=analysis_results.get("fps", "-"),
        total_frames=analysis_results.get("total_frames", 0),
        score_summary=_score_summary(avg_score, frame_scores),
        phase_avg_summary=_phase_avg_summary(frame_scores),
        top_errors_summary=_top_errors_summary(frame_scores),
        detail_samples=_detail_samples(error_frames),
        dtw_summary=_dtw_summary(dtw_result),
    )


# ══════════════════════════════════════════════════════════════
#  🚀 Gemini API 호출
# ══════════════════════════════════════════════════════════════

def generate_feedback(
    analysis_results: Dict[str, Any],
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_output_tokens: int = 1200,
) -> str:
    """
    Gemini API를 호출하여 운동 자세 피드백을 생성한다.

    Args:
        analysis_results: st.session_state['analysis_results'] dict
        api_key: Gemini API 키 (None이면 .env / 환경변수에서 읽음)
        temperature: 생성 다양성 (0.0~1.0)
        max_output_tokens: 최대 출력 토큰 수

    Returns:
        피드백 문자열 (마크다운 형식)
    """
    # 호출 시점마다 .env를 다시 읽어 최신 키를 반영
    load_dotenv(override=True)
    key = api_key or os.environ.get("GEMINI_API_KEY", "")

    if not key or not key.strip():
        raise ValueError(
            "Gemini API 키가 없습니다.\n"
            "① UI의 'Gemini API 설정'에서 직접 입력하거나\n"
            "② 프로젝트 루트의 .env 파일에 GEMINI_API_KEY=AIza... 형식으로 저장하세요."
        )

    key = key.strip()
    user_prompt = build_prompt(analysis_results)

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT.strip()}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }

    try:
        response = requests.post(
            GEMINI_API_URL,
            params={"key": key},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )

        # HTTP 오류 시 응답 본문을 먼저 확인
        if not response.ok:
            status = response.status_code
            try:
                err_body = response.json()
                err_msg = err_body.get("error", {}).get("message", response.text[:300])
            except Exception:
                err_msg = response.text[:300]

            if status == 400:
                raise RuntimeError(f"요청 오류 (400): {err_msg}")
            elif status == 403:
                raise RuntimeError(f"API 키 인증 실패 (403): 키를 확인하세요.\n{err_msg}")
            elif status == 429:
                raise RuntimeError(f"요청 한도 초과 (429): 잠시 후 재시도하세요.\n{err_msg}")
            else:
                raise RuntimeError(f"Gemini API 오류 ({status}): {err_msg}")

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"응답에 candidates 없음: {json.dumps(data, ensure_ascii=False)[:300]}")

        parts = candidates[0].get("content", {}).get("parts", [])
        feedback_text = "".join(p.get("text", "") for p in parts)

        if not feedback_text.strip():
            raise RuntimeError(f"빈 응답 반환. 전체 응답: {json.dumps(data, ensure_ascii=False)[:300]}")

        return feedback_text

    except requests.exceptions.Timeout:
        raise RuntimeError("요청 시간 초과 (60초). 네트워크 상태를 확인하세요.")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"네트워크 연결 실패: {e}\nVPN이나 방화벽을 확인하세요.")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"예상치 못한 오류:\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
#  🧪 테스트 실행 (python gemini_feedback.py 로 단독 실행 가능)
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # 더미 데이터로 프롬프트 미리보기
    dummy_results = {
        "exercise_type": "푸시업",
        "exercise_count": 5,
        "fps": 10,
        "total_frames": 120,
        "frame_scores": [
            {"frame_idx": i, "phase": ["top", "descending", "bottom", "ascending"][i % 4],
             "score": 0.6 + (i % 5) * 0.08,
             "errors": ["허리를 펴세요"] if i % 3 == 0 else [],
             "details": {"back_angle": {"value": 145.2, "status": "error", "feedback": "허리를 펴세요"}}}
            for i in range(20)
        ],
        "error_frames": [
            {"frame_idx": 3, "phase": "bottom", "score": 0.5,
             "errors": ["허리를 펴세요", "팔꿈치를 몸쪽으로 모아주세요"],
             "details": {
                 "back_angle": {"value": 142.1, "status": "error", "feedback": "허리를 펴세요"},
                 "shoulder_abduction": {"value": 85.3, "status": "error", "feedback": "팔꿈치를 몸쪽으로 모아주세요"}
             }}
        ],
        "dtw_result": {
            "overall_dtw_score": 0.72,
            "phase_dtw_scores": {"top": 0.80, "bottom": 0.65, "descending": 0.70, "ascending": 0.73},
        },
    }

    print("=" * 60)
    print("📋 생성될 프롬프트 미리보기")
    print("=" * 60)
    print(build_prompt(dummy_results))

    if "--run" in sys.argv:
        print("\n" + "=" * 60)
        print("🤖 Gemini 피드백 생성 중...")
        print("=" * 60)
        try:
            feedback = generate_feedback(dummy_results)
            print(feedback)
        except Exception as e:
            print(f"❌ 오류: {e}")
