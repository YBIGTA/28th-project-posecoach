"""
AI 상세 피드백 페이지 (pages/feedback.py)

app_phase.py에서 분석 완료 후 st.switch_page()로 진입.
session_state["results"]에서 분석 결과를 읽어 LLM 피드백을 생성한다.
"""
import sys
import json
from pathlib import Path
from collections import Counter

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent  # pages/ 기준 상위 = 프로젝트 루트
sys.path.insert(0, str(ROOT / "preprocess"))
sys.path.insert(0, str(ROOT / "preprocess" / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ds_modules"))

from ds_modules.llm_feedback import (
    build_prompt,
    generate_feedback,
    aggregate_posture_errors,
    run_feedback_pipeline,
)

_FEATURE_KO = {
    "elbow_L": "왼쪽 팔꿈치 각도", "elbow_R": "오른쪽 팔꿈치 각도",
    "back": "등(척추) 직선", "abd_L": "왼쪽 어깨 외전", "abd_R": "오른쪽 어깨 외전",
    "head_tilt": "고개 기울기", "hand_offset": "손 위치",
    "shoulder_packing": "어깨 패킹", "elbow_flare": "팔꿈치 벌림", "body_sway": "몸통 흔들림",
}
_PHASE_KO = {
    "top": "최고점(팔 펴기)", "bottom": "최저점(내려가기)",
    "descending": "내려가는 구간", "ascending": "올라오는 구간", "ready": "준비 자세",
}

# ── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(page_title="AI 피드백", layout="wide")
st.title("🤖 AI 상세 피드백")

if st.button("← 분석 결과로 돌아가기"):
    st.switch_page("app_phase.py")

# ── 분석 결과 확인 ────────────────────────────────────────────
if "results" not in st.session_state:
    st.warning("분석 결과가 없습니다. 먼저 영상을 분석해주세요.")
    st.stop()

res          = st.session_state["results"]
frame_scores = res.get("frame_scores", [])
error_frames = res.get("error_frames", [])
dtw_result   = res.get("dtw_result") or {}
exercise_type = res.get("exercise_type", "운동")
exercise_count = res.get("exercise_count", 0)

if not frame_scores:
    st.warning("분석된 프레임이 없습니다.")
    st.stop()

avg_score = sum(fs["score"] for fs in frame_scores) / len(frame_scores)
dtw_sc    = dtw_result.get("overall_dtw_score")
combined  = avg_score * 0.7 + dtw_sc * 0.3 if dtw_sc is not None else avg_score
llm_ctx   = dtw_result.get("llm_context", {})

# ── 요약 메트릭 ───────────────────────────────────────────────
st.subheader("분석 요약")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{exercise_type} 횟수", f"{exercise_count}회")
c2.metric("평균 자세 점수",        f"{avg_score:.0%}")
c3.metric("DTW 유사도",            f"{dtw_sc:.0%}" if dtw_sc is not None else "N/A")
c4.metric("종합 점수",             f"{combined:.0%}")

# ── 주요 오류 빈도 ────────────────────────────────────────────
posture_errors = aggregate_posture_errors(error_frames)
if posture_errors:
    st.divider()
    st.subheader("주요 자세 오류 (빈도순)")
    error_counter = Counter(posture_errors)
    import pandas as pd
    err_df = pd.DataFrame([
        {"오류": err, "감지 횟수": cnt, "비율": f"{cnt/len(frame_scores):.0%}"}
        for err, cnt in error_counter.most_common(8)
    ])
    st.dataframe(err_df, width='stretch', hide_index=True)

# ── DTW 관절 분석 ─────────────────────────────────────────────
overall_worst = llm_ctx.get("overall_worst_features", [])
phase_details = llm_ctx.get("phase_details", {})

if overall_worst or phase_details:
    st.divider()
    st.subheader("DTW 분석 요약")

    if overall_worst:
        st.markdown("**전체 운동에서 가장 차이가 큰 관절**")
        import pandas as pd
        st.dataframe(pd.DataFrame([{
            "관절": _FEATURE_KO.get(w["name"], w["name"]),
            "평균 차이": round(w["avg_diff"], 4),
            "심각도": "🔴 높음" if w["avg_diff"] > 0.1 else "🟡 중간" if w["avg_diff"] > 0.05 else "🟢 낮음",
        } for w in overall_worst]), width='stretch', hide_index=True)

    if phase_details:
        st.markdown("**Phase별 문제 요약**")
        import pandas as pd
        phase_rows = []
        for ph, detail in phase_details.items():
            wf_names = ", ".join(
                _FEATURE_KO.get(w["name"], w["name"])
                for w in detail.get("worst_features", [])
            )
            phase_rows.append({
                "구간":           _PHASE_KO.get(ph, ph),
                "DTW 유사도":     f"{detail.get('dtw_score', 0):.0%}",
                "속도":           {"fast":"⚡빠름","normal":"✅적절","slow":"🐢느림"}.get(detail.get("speed","normal"),""),
                "문제 프레임 비율": f"{detail.get('bad_frame_ratio',0):.0%}",
                "주요 문제 관절": wf_names or "-",
            })
        st.dataframe(pd.DataFrame(phase_rows), width='stretch', hide_index=True)

# ── AI 피드백 생성 ────────────────────────────────────────────
st.divider()
st.subheader("AI 피드백 생성")

import os
api_key = st.text_input(
    "OpenAI API Key (선택)",
    type="password",
    placeholder=".env에 OPENAI_API_KEY 설정 시 비워도 됩니다.",
    help="프로젝트 루트 .env 파일에 OPENAI_API_KEY=your_key 형식으로 저장하면 자동 로드됩니다.",
)

# 프롬프트 추가 입력 옵션
with st.expander("추가 요청사항 (선택)"):
    extra_notes = st.text_area(
        "AI에게 추가로 전달할 내용",
        placeholder="예: 초보자 기준으로 설명해줘 / 어깨 부상 이력이 있어 / 영어로 답해줘",
        height=80,
    )

col_gen, col_clear = st.columns([3, 1])
with col_gen:
    generate_btn = st.button("🤖 AI 피드백 생성", type="primary", width='stretch')
with col_clear:
    if st.button("초기화", width='stretch'):
        st.session_state.pop("feedback_result", None)
        st.session_state.pop("feedback_prompt", None)
        st.rerun()

if generate_btn:
    with st.spinner("AI 피드백 생성 중... (10~20초 소요)"):
        try:
            prompt = build_prompt(
                llm_context    = llm_ctx,
                posture_errors = posture_errors,
                avg_score      = avg_score,
                exercise_count = exercise_count,
                frame_scores   = res.get("frame_scores", []),
                error_frames   = error_frames,
                extra_notes    = extra_notes if extra_notes.strip() else None,
            )
            result = generate_feedback(
                prompt  = prompt,
                api_key = api_key.strip() or None,
            )
            if isinstance(result, dict):
                st.session_state["feedback_result"]      = result.get("feedback", "")
                st.session_state["highlighted_frames"]   = result.get("highlighted_frames", [])
            else:
                st.session_state["feedback_result"]      = result or ""
                st.session_state["highlighted_frames"]   = []
            st.session_state["feedback_prompt"] = prompt
        except Exception as e:
            st.error(f"오류 발생: {e}")

# ── 피드백 출력 ───────────────────────────────────────────────
if "feedback_result" in st.session_state:
    fb                 = st.session_state["feedback_result"]
    highlighted_frames = st.session_state.get("highlighted_frames", [])

    if fb:
        st.divider()
        st.subheader("📋 AI 피드백")
        st.markdown(fb)

        # ── GPT가 언급한 프레임 이미지 ────────────────────────
        if highlighted_frames:
            import cv2
            from collections import defaultdict as _dd

            frame_mapping     = (dtw_result or {}).get("frame_mapping", {})
            error_frames_data = res.get("error_frames", [])
            frame_scores_data = res.get("frame_scores", [])

            # (rep_idx, phase) → 프레임 목록 인덱싱
            err_index   = _dd(list)
            score_index = _dd(list)
            for ef in error_frames_data:
                err_index[(ef.get("rep_idx", 0), ef.get("phase", ""))].append(ef)
            for fs in frame_scores_data:
                score_index[(fs.get("rep_idx", 0), fs.get("phase", ""))].append(fs)

            st.divider()
            st.subheader("🖼️ GPT가 언급한 구간 프레임")

            for hf in highlighted_frames:
                rep_idx  = hf.get("rep_idx", 0)
                phase    = hf.get("phase", "")
                hf_type  = hf.get("type", "bad")   # "good" | "bad"
                reason   = hf.get("reason", "")
                icon     = "✅" if hf_type == "good" else "⚠️"
                label    = f"{icon} {rep_idx}번째 횟수 [{_PHASE_KO.get(phase, phase)}] — {reason}"

                with st.expander(label, expanded=True):
                    # 대표 프레임 선택
                    key = (rep_idx, phase)
                    if hf_type == "bad" and err_index[key]:
                        target = min(err_index[key], key=lambda x: x["score"])
                        img_path = target.get("img_path")
                        pts      = target.get("pts")
                        errors   = target.get("errors", [])
                    elif hf_type == "good" and score_index[key]:
                        target = max(score_index[key], key=lambda x: x["score"])
                        # frame_scores엔 img_path 없으므로 frame_idx로 keypoints에서 찾기
                        fidx   = target.get("frame_idx")
                        kp     = next((k for k in res.get("keypoints", []) if k.get("frame_idx") == fidx), None)
                        img_path = kp.get("img_path") if kp else None
                        pts      = None
                        errors   = []
                    else:
                        st.info("해당 구간 프레임 데이터 없음")
                        continue

                    if not img_path:
                        st.info("이미지 경로 없음")
                        continue

                    mapping = frame_mapping.get(img_path)
                    cols = st.columns(3) if mapping else st.columns(2)

                    with cols[0]:
                        st.caption("사용자 원본")
                        img = cv2.imread(img_path)
                        if img is not None:
                            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

                    with cols[1]:
                        st.caption("스켈레톤")
                        if pts:
                            try:
                                from utils.visualization import draw_skeleton_on_frame
                                skel = draw_skeleton_on_frame(img_path, pts)
                                if skel is not None:
                                    st.image(skel, use_container_width=True)
                            except Exception:
                                st.info("스켈레톤 없음")
                        else:
                            img2 = cv2.imread(img_path)
                            if img2 is not None:
                                st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), use_container_width=True)

                    if mapping and len(cols) > 2:
                        with cols[2]:
                            st.caption(f"레퍼런스 (#{mapping['ref_idx']})")
                            ref_img = cv2.imread(mapping["ref_img"])
                            if ref_img is not None:
                                st.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                    if errors:
                        for e in errors[:3]:
                            st.warning(e)

        # 피드백 다운로드
        st.divider()
        export = {
            "exercise_type":   exercise_type,
            "exercise_count":  exercise_count,
            "avg_score":       round(avg_score, 4),
            "dtw_score":       round(dtw_sc, 4) if dtw_sc is not None else None,
            "combined_score":  round(combined, 4),
            "top_errors":      Counter(posture_errors).most_common(5),
            "overall_worst":   overall_worst,
            "feedback":        fb,
        }
        st.download_button(
            "📥 피드백 저장 (JSON)",
            data=json.dumps(export, indent=2, ensure_ascii=False),
            file_name=f"{res.get('video_name','video')}_feedback.json",
            mime="application/json",
        )

        with st.expander("프롬프트 보기 (디버그)"):
            st.text(st.session_state.get("feedback_prompt", ""))
    else:
        st.error("피드백 생성에 실패했습니다. API 키를 확인하거나 잠시 후 다시 시도해주세요.")