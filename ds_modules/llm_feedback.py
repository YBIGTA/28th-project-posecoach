"""
LLM 피드백 생성 모듈 (OpenAI GPT)

DTW 분석 결과 + 자세 평가 결과를 구조화된 프롬프트로 변환하고
GPT API로 자연스러운 문단 형식의 피드백을 생성한다.

환경변수 설정 (.env):
    OPENAI_API_KEY=your_api_key_here
"""
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def _load_env():
    cur = Path(__file__).resolve()
    for _ in range(5):
        env_file = cur / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return
        cur = cur.parent

_load_env()

_FEATURE_KO: Dict[str, str] = {
    "elbow_L": "왼쪽 팔꿈치 각도", "elbow_R": "오른쪽 팔꿈치 각도",
    "back": "등(척추) 직선", "abd_L": "왼쪽 어깨 외전", "abd_R": "오른쪽 어깨 외전",
    "head_tilt": "고개 기울기", "hand_offset": "손 위치",
    "shoulder_packing": "어깨 패킹(견갑 하강)",
    "elbow_flare": "팔꿈치 벌림", "body_sway": "몸통 흔들림",
}
_SPEED_KO  = {"fast": "너무 빠름", "normal": "적절", "slow": "너무 느림"}
_PHASE_KO  = {
    "top": "최고점(팔 펴기)", "bottom": "최저점(내려가기)",
    "descending": "내려가는 구간", "ascending": "올라오는 구간", "ready": "준비 자세",
}


def aggregate_posture_errors(error_frames: List[Dict]) -> List[str]:
    all_errors: List[str] = []
    for ef in error_frames:
        all_errors.extend(ef.get("errors", []))
    return all_errors


def _analyze_rep_trend(frame_scores: List[Dict], exercise_count: int) -> str:
    """횟수별 평균 점수 추이 분석 → 텍스트 요약"""
    if not frame_scores or exercise_count < 2:
        return ""

    # 전체 프레임을 횟수 수만큼 균등 분할
    n = len(frame_scores)
    chunk = max(1, n // exercise_count)
    rep_scores = []
    for i in range(exercise_count):
        seg = frame_scores[i*chunk:(i+1)*chunk]
        if seg:
            rep_scores.append(sum(s["score"] for s in seg) / len(seg))

    if len(rep_scores) < 2:
        return ""

    first_avg = sum(rep_scores[:max(1, len(rep_scores)//3)]) / max(1, len(rep_scores)//3)
    last_avg  = sum(rep_scores[-max(1, len(rep_scores)//3):]) / max(1, len(rep_scores)//3)
    diff      = last_avg - first_avg

    lines = ["## 횟수별 자세 점수 추이"]
    for i, sc in enumerate(rep_scores, 1):
        lines.append(f"- {i}번째: {sc:.0%}")
    if diff < -0.08:
        lines.append("→ 후반부로 갈수록 자세가 무너지는 경향 (체력 저하 의심)")
    elif diff > 0.05:
        lines.append("→ 후반부로 갈수록 자세가 안정되는 경향 (워밍업 효과)")
    else:
        lines.append("→ 전반적으로 일관된 자세 유지")
    return "\n".join(lines)


def _analyze_phase_errors(error_frames: List[Dict]) -> str:
    """횟수 × Phase 조합별 오류 분석 — 몇 번째 횟수의 어느 구간에서 문제인지"""
    if not error_frames:
        return ""

    # rep_idx × phase → 오류 목록 (rep_idx=0은 준비 구간이므로 제외)
    rep_phase_err: Dict[tuple, List[str]] = defaultdict(list)
    for ef in error_frames:
        rep = ef.get("rep_idx", 0)
        if rep == 0:
            continue
        key = (rep, ef.get("phase", "unknown"))
        rep_phase_err[key].extend(ef.get("errors", []))

    lines = ["## 횟수별 × 구간별 주요 오류"]
    for (rep, phase), errs in sorted(rep_phase_err.items()):
        if not errs:
            continue
        top = Counter(errs).most_common(2)
        top_str = " / ".join(f"{e}({c}회)" for e, c in top)
        lines.append(f"- {rep}번째 횟수 [{_PHASE_KO.get(phase, phase)}]: {top_str}")
    return "\n".join(lines)


def _analyze_worst_best_frames(frame_scores: List[Dict]) -> str:
    """최고/최저 프레임 점수 및 구간 분석"""
    if not frame_scores:
        return ""

    sorted_scores = sorted(frame_scores, key=lambda x: x["score"])
    worst5 = sorted_scores[:5]
    best5  = sorted_scores[-5:]

    worst_phases = Counter(f["phase"] for f in worst5)
    best_phases  = Counter(f["phase"] for f in best5)

    lines = ["## 프레임별 극단값 분석"]
    lines.append(f"- 최저 점수 구간: {', '.join(_PHASE_KO.get(p,p) for p in worst_phases)}")
    lines.append(f"- 최고 점수 구간: {', '.join(_PHASE_KO.get(p,p) for p in best_phases)}")
    lines.append(f"- 최저 점수: {sorted_scores[0]['score']:.0%} / 최고 점수: {sorted_scores[-1]['score']:.0%}")
    return "\n".join(lines)


def build_prompt(
    llm_context: Dict,
    posture_errors: List[str],
    avg_score: float,
    exercise_count: int,
    frame_scores: Optional[List[Dict]] = None,
    error_frames: Optional[List[Dict]] = None,
    extra_notes: Optional[str] = None,
) -> tuple:
    """(system_prompt, user_message) 튜플 반환"""
    exercise      = llm_context.get("exercise", "운동")
    phase_details = llm_context.get("phase_details", {})
    overall_worst = llm_context.get("overall_worst_features", [])

    system = (
        "당신은 10년 경력의 전문 피트니스 트레이너입니다.\n"
        "아래 운동 분석 데이터를 보고 피드백을 작성하되, 반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 포함하지 마세요.\n"
        "\n"
        "{\n"
        '  \"feedback\": \"(피드백 전문 — 자연스러운 문단, 번호 목록 금지)\",\n'
        '  \"highlighted_frames\": [\n'
        '    {\"rep_idx\": N, \"phase\": \"phase_name\", \"type\": \"good\"또는\"bad\", \"reason\": \"한 줄 설명\"}\n'
        "  ]\n"
        "}\n"
        "\n"
        "feedback 작성 규칙:\n"
        "- 첫 문단 (4~5문장): 전체 평가 + 잘한 횟수/구간 구체적으로. 어떤 동작이 좋았는지, 왜 좋은지까지 설명\n"
        "- 둘째 문단 (4~5문장): 가장 시급한 개선점 — 몇 번째 횟수, 어느 구간, 어떤 동작인지 + 왜 문제인지 + 어떻게 고쳐야 하는지 구체적인 동작 큐잉까지\n"
        "- 셋째 문단 (3~4문장): 두 번째 개선점 — 마찬가지로 횟수/구간 명시 + 개선 방법\n"
        "- 마지막 문장: 짧은 응원\n"
        "- 수치(%)는 동작 언어로 표현, 전문적이되 친근한 말투\n"
        "- 각 문단은 충분히 풍부하게 작성. 너무 짧으면 안 됨\n"
        "\n"
        "highlighted_frames 작성 규칙:\n"
        "- feedback에서 언급한 횟수/구간을 정확히 매핑\n"
        "- phase는 반드시 다음 중 하나: top, bottom, ascending, descending\n"
        "- type: 잘한 구간은 good, 문제 구간은 bad\n"
        "- 총 3~5개 선택\n"
        "- rep_idx는 데이터에 있는 실제 횟수 번호 사용"
    )

    lines: List[str] = []
    lines.append("아래 분석 데이터를 바탕으로 피드백을 작성해주세요.")
    lines.append("")

    # 기본 운동 정보
    lines.append("## 운동 정보")
    lines.append(f"- 종목: {exercise}")
    lines.append(f"- 총 횟수: {exercise_count}회")
    lines.append(f"- 평균 자세 점수: {avg_score:.0%}")
    lines.append("")

    # 자세 오류 빈도
    if posture_errors:
        error_counter = Counter(posture_errors)
        lines.append("## 자세 평가 오류 (빈도순)")
        for err, cnt in error_counter.most_common(6):
            lines.append(f"- {err} → {cnt}회 감지")
        lines.append("")

    # Phase별 오류 분포
    if error_frames:
        phase_err_text = _analyze_phase_errors(error_frames)
        if phase_err_text:
            lines.append(phase_err_text)
            lines.append("")

    # 횟수별 추이
    if frame_scores and exercise_count >= 2:
        trend_text = _analyze_rep_trend(frame_scores, exercise_count)
        if trend_text:
            lines.append(trend_text)
            lines.append("")

    # 최고/최저 프레임
    if frame_scores:
        extremes_text = _analyze_worst_best_frames(frame_scores)
        if extremes_text:
            lines.append(extremes_text)
            lines.append("")

    # DTW Phase별 분석
    if phase_details:
        lines.append("## Phase별 모범 동작 대비 분석 (DTW)")
        for phase, detail in phase_details.items():
            worst = detail.get("worst_features", [])
            speed = _SPEED_KO.get(detail.get("speed", "normal"), "적절")
            bad   = detail.get("bad_frame_ratio", 0)
            lines.append(f"### {_PHASE_KO.get(phase, phase)}")
            lines.append(f"- 모범 동작과의 유사도: {detail.get('dtw_score', 0):.0%}")
            lines.append(f"- 속도: {speed}")
            lines.append(f"- 자세 불안정 구간 비율: {bad:.0%}")
            if worst:
                for w in worst:
                    lines.append(f"- {_FEATURE_KO.get(w['name'], w['name'])}: 모범 대비 차이 큼")
            lines.append("")

    # 전체 worst 관절
    if overall_worst:
        lines.append("## 운동 전체에서 가장 차이가 큰 관절")
        for wf in overall_worst:
            lines.append(f"- {_FEATURE_KO.get(wf['name'], wf['name'])}")
        lines.append("")

    if extra_notes:
        lines.append("## 사용자 추가 요청")
        lines.append(extra_notes)
        lines.append("")

    return system, "\n".join(lines)


def generate_feedback(
    prompt,
    model: str = "gpt-4o",
    max_tokens: int = 2048,
    api_key: Optional[str] = None,
) -> Optional[str]:
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        logger.error("OPENAI_API_KEY 없음 — .env 파일 또는 인자로 전달하세요.")
        return None

    try:
        from openai import OpenAI

        if isinstance(prompt, tuple):
            system_instruction, user_message = prompt
        else:
            system_instruction = "당신은 전문 피트니스 트레이너입니다."
            user_message = prompt

        import json as _json

        client = OpenAI(api_key=resolved_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        try:
            parsed = _json.loads(raw)
            return parsed  # dict: {"feedback": str, "highlighted_frames": [...]}
        except _json.JSONDecodeError:
            return {"feedback": raw, "highlighted_frames": []}

    except ImportError:
        logger.error("openai 미설치 — pip install openai")
        return None
    except Exception as e:
        logger.error(f"GPT 피드백 생성 실패: {e}")
        return None


def run_feedback_pipeline(
    dtw_result: Dict,
    error_frames: List[Dict],
    avg_score: float,
    exercise_count: int,
    frame_scores: Optional[List[Dict]] = None,
    api_key: Optional[str] = None,
    extra_notes: Optional[str] = None,
) -> Dict:
    llm_context    = dtw_result.get("llm_context", {})
    posture_errors = aggregate_posture_errors(error_frames)

    prompt = build_prompt(
        llm_context    = llm_context,
        posture_errors = posture_errors,
        avg_score      = avg_score,
        exercise_count = exercise_count,
        frame_scores   = frame_scores,
        error_frames   = error_frames,
        extra_notes    = extra_notes,
    )
    result = generate_feedback(prompt, api_key=api_key)
    if isinstance(result, dict):
        feedback_text       = result.get("feedback", "")
        highlighted_frames  = result.get("highlighted_frames", [])
    else:
        feedback_text       = result or ""
        highlighted_frames  = []
    return {
        "prompt":             prompt[1] if isinstance(prompt, tuple) else prompt,
        "feedback":           feedback_text,
        "highlighted_frames": highlighted_frames,
    }
