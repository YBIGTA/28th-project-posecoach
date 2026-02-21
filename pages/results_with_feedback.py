"""
운동 영상 분석 결과 대시보드 (Phase 기반) + Gemini AI 피드백
app_phase.py에서 생성한 분석 결과를 시각화하고 AI 종합 피드백을 제공합니다.
"""
import streamlit as st
import pandas as pd
import json
import cv2
from pathlib import Path
import sys

# ---------- 1. 경로 설정 ----------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "preprocess" / "scripts"))

from utils.visualization import draw_skeleton_on_frame
from gemini_feedback import generate_feedback  # Gemini 피드백 모듈

st.set_page_config(page_title="AI Pose Coach | Analysis Report", page_icon="📊", layout="wide")

# ---------- 2. 스타일 ----------
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', sans-serif; }
    
    .stApp {
        background-color: #0e1117;
        color: #f5f1da;
    }

    .report-header {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(180deg, rgba(231, 227, 196, 0.05) 0%, rgba(14, 17, 23, 0) 100%);
        margin-bottom: 2rem;
    }
    
    .section-label {
        color: #e7e3c4;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2.5rem 0 1.5rem 0;
        border-left: 4px solid #e7e3c4;
        padding-left: 15px;
    }

    .metric-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
    }

    .grade-badge {
        font-size: 2.5rem;
        font-weight: 800;
        color: #e7e3c4;
        margin: 10px 0;
    }

    div.stButton > button {
        border-radius: 50px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    .chart-box {
        background: rgba(255, 255, 255, 0.02);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 20px;
    }
    
    .phase-info-tag {
        display: inline-block;
        padding: 4px 12px;
        background: #e7e3c4;
        color: #1c1c1c;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }

    /* AI 피드백 박스 */
    .ai-feedback-box {
        background: linear-gradient(135deg, rgba(231,227,196,0.07) 0%, rgba(100,120,200,0.05) 100%);
        border: 1px solid rgba(231, 227, 196, 0.25);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin: 1rem 0 2rem 0;
        line-height: 1.8;
    }

    .ai-feedback-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.2rem;
        font-size: 1.1rem;
        font-weight: 700;
        color: #e7e3c4;
    }

    .api-key-box {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 3. 결과 로드 ----------
if 'analysis_results' not in st.session_state:
    st.warning("⚠️ 분석 결과가 없습니다.")
    st.info("먼저 영상을 업로드하고 분석을 진행해주세요.")
    st.stop()

res = st.session_state["analysis_results"]
kp_list = res["keypoints"]
frame_scores = res.get("frame_scores", [])
error_frames = res.get("error_frames", [])

# ---------- 4. 헤더 ----------
st.markdown('<div class="report-header">', unsafe_allow_html=True)
st.markdown(f"# 📊 분석 리포트")
st.markdown(f"**{res.get('video_name', '영상')}** | {res.get('exercise_type', '')} 분석 결과")
st.markdown('</div>', unsafe_allow_html=True)

# 기본 메트릭
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("총 프레임", f"{res['total_frames']}개")
with col2:
    st.metric("키포인트 성공", f"{res['success_count']}개")
with col3:
    st.metric("추출 FPS", f"{res['fps']}fps")
with col4:
    st.metric("운동 횟수", f"{res.get('exercise_count', 0)}회")

st.divider()

# ---------- 5. 점수 요약 ----------
if not frame_scores:
    st.warning("분석된 프레임 점수 데이터가 없습니다.")
    st.stop()

scores = [fs["score"] for fs in frame_scores]
avg_score = sum(scores) / len(scores) if scores else 0

if avg_score >= 0.9:
    grade = "S"
elif avg_score >= 0.7:
    grade = "A"
elif avg_score >= 0.5:
    grade = "B"
else:
    grade = "C"

dtw_result = res.get("dtw_result")
dtw_active = dtw_result and dtw_result.get("overall_dtw_score") is not None

if dtw_active:
    dtw_score = dtw_result["overall_dtw_score"]
    combined_score = avg_score * 0.7 + dtw_score * 0.3

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f'<div class="metric-container"><p>{res.get("exercise_type")} 횟수</p><div class="grade-badge">{res.get("exercise_count", 0)}회</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-container"><p>평균 자세 점수</p><div class="grade-badge">{avg_score:.0%}</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-container"><p>DTW 유사도</p><div class="grade-badge">{dtw_score:.0%}</div></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown(f'<div class="metric-container"><p>종합 점수 (등급)</p><div class="grade-badge">{combined_score:.0%} ({grade})</div></div>', unsafe_allow_html=True)

    phase_dtw = dtw_result.get("phase_dtw_scores", {})
    phase_counts = dtw_result.get("phase_segment_counts", {})
    if phase_dtw:
        st.markdown('<div class="section-label">Phase별 DTW 유사도</div>', unsafe_allow_html=True)
        dtw_cols = st.columns(len(phase_dtw))
        for i, (phase, score) in enumerate(sorted(phase_dtw.items())):
            cnt = phase_counts.get(phase, 0)
            with dtw_cols[i]:
                st.markdown(
                    f'<div class="metric-container">'
                    f'<p>{phase}</p>'
                    f'<div class="grade-badge">{score:.0%}</div>'
                    f'<small style="color:#888;">{cnt}세그먼트</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
else:
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f'<div class="metric-container"><p>{res.get("exercise_type")} 횟수</p><div class="grade-badge">{res.get("exercise_count", 0)}회</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-container"><p>평균 점수</p><div class="grade-badge">{avg_score:.0%}</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-container"><p>등급</p><div class="grade-badge">{grade}</div></div>', unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════
# ---------- 6. 🤖 AI 종합 피드백 (Gemini) ----------
# ══════════════════════════════════════════════════════════════

st.markdown('<div class="section-label">🤖 AI 종합 피드백</div>', unsafe_allow_html=True)

# API 키 입력 영역
with st.expander("⚙️ Gemini API 설정", expanded=("gemini_api_key" not in st.session_state)):
    st.markdown('<div class="api-key-box">', unsafe_allow_html=True)
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.get("gemini_api_key", ""),
        type="password",
        placeholder="AIza...",
        help="Google AI Studio(https://aistudio.google.com)에서 발급받은 API 키를 입력하세요.",
    )
    if api_key_input:
        st.session_state["gemini_api_key"] = api_key_input

    col_temp, col_tokens = st.columns(2)
    with col_temp:
        temperature = st.slider(
            "Temperature (창의성)",
            min_value=0.0, max_value=1.0,
            value=st.session_state.get("gemini_temperature", 0.7),
            step=0.05,
            help="높을수록 다양한 표현, 낮을수록 일관된 표현",
        )
        st.session_state["gemini_temperature"] = temperature
    with col_tokens:
        max_tokens = st.slider(
            "최대 출력 길이 (토큰)",
            min_value=300, max_value=6000,
            value=st.session_state.get("gemini_max_tokens", 6000),
            step=100,
        )
        st.session_state["gemini_max_tokens"] = max_tokens
    st.markdown('</div>', unsafe_allow_html=True)

# 피드백 생성 버튼 & 결과
feedback_col1, feedback_col2 = st.columns([1, 4])
with feedback_col1:
    generate_btn = st.button(
        "✨ AI 피드백 생성",
        use_container_width=True,
        type="primary",
        disabled=not st.session_state.get("gemini_api_key", ""),
    )

with feedback_col2:
    if not st.session_state.get("gemini_api_key", ""):
        st.warning("위 설정에서 Gemini API 키를 입력하면 AI 피드백을 받을 수 있습니다.")

if generate_btn:
    with st.spinner("🤖 AI가 자세를 분석하고 피드백을 작성 중입니다..."):
        try:
            feedback_text = generate_feedback(
                analysis_results=res,
                api_key=st.session_state.get("gemini_api_key"),
                temperature=st.session_state.get("gemini_temperature", 0.7),
                max_output_tokens=st.session_state.get("gemini_max_tokens", 1200),
            )
            st.session_state["gemini_feedback"] = feedback_text
        except Exception as e:
            st.session_state["gemini_feedback"] = None
            st.error(f"❌ 피드백 생성 실패: {e}")

# 저장된 피드백 표시
if st.session_state.get("gemini_feedback"):
    st.markdown('<div class="ai-feedback-box">', unsafe_allow_html=True)
    st.markdown(
        '<div class="ai-feedback-header">🧠 트레이너 AI 종합 피드백</div>',
        unsafe_allow_html=True
    )
    st.markdown(st.session_state["gemini_feedback"])
    st.markdown('</div>', unsafe_allow_html=True)

    # 피드백 복사/다운로드 버튼
    dl_col1, dl_col2 = st.columns([1, 5])
    with dl_col1:
        st.download_button(
            "📋 피드백 저장",
            data=st.session_state["gemini_feedback"],
            file_name=f"{res.get('video_name', 'feedback')}_ai_feedback.md",
            mime="text/markdown",
            use_container_width=True,
        )

st.divider()

# ---------- 7. 시계열 차트 ----------
st.markdown('<div class="section-label">프레임별 자세 점수</div>', unsafe_allow_html=True)
chart_df = pd.DataFrame({
    "프레임": [fs["frame_idx"] for fs in frame_scores],
    "점수": [fs["score"] for fs in frame_scores],
    "Phase": [fs["phase"] for fs in frame_scores],
})

st.markdown('<div class="chart-box">', unsafe_allow_html=True)
st.line_chart(chart_df, x="프레임", y="점수", color="#e7e3c4")
st.markdown('</div>', unsafe_allow_html=True)

c_left, c_right = st.columns(2)

with c_left:
    st.markdown('<div class="chart-box"><b>⏱️ Phase 분포 (프레임 수)</b>', unsafe_allow_html=True)
    phase_counts = chart_df["Phase"].value_counts()
    st.bar_chart(phase_counts, color="#e7e3c4")
    st.markdown('</div>', unsafe_allow_html=True)

with c_right:
    st.markdown('<div class="chart-box"><b>🎯 Phase별 평균 자세 점수</b>', unsafe_allow_html=True)
    phase_avg_scores = chart_df.groupby("Phase")["점수"].mean()
    st.bar_chart(phase_avg_scores, color="#e7e3c4")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ---------- 8. 오류 프레임 브라우저 ----------
if error_frames:
    st.markdown('<div class="section-label">취약 구간 집중 리뷰</div>', unsafe_allow_html=True)
    
    error_options = [
        f"프레임 {ef['frame_idx']} [{ef['phase']}] - {ef['score']:.0%}"
        for ef in error_frames
    ]
    selected_idx = st.selectbox(
        "오류가 감지된 프레임을 선택하세요",
        range(len(error_options)),
        format_func=lambda i: error_options[i]
    )
    
    ef = error_frames[selected_idx]
    
    col_img, col_fb = st.columns([1, 1])
    
    with col_img:
        skel_img = draw_skeleton_on_frame(ef["img_path"], ef.get("pts"))
        if skel_img is not None:
            st.image(
                skel_img,
                use_container_width=True,
                caption=f"Error Detail : Frame {ef['frame_idx']}"
            )
        else:
            st.warning("이미지를 로드할 수 없습니다.")
    
    with col_fb:
        st.markdown(
            f'<div class="phase-info-tag">{ef["phase"].upper()} PHASE</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"#### 점수: {ef['score']:.0%}")
        
        for err_msg in ef["errors"]:
            st.error(f"⚠️ {err_msg}")
        
        if "details" in ef and ef["details"]:
            st.markdown("---")
            for k, v in ef["details"].items():
                icon = "✅" if v["status"] == "ok" else "❌"
                st.markdown(
                    f"{icon} **{k}**: {v['value']} <br>"
                    f"<small style='color:#888;'>{v['feedback']}</small>",
                    unsafe_allow_html=True
                )

st.divider()

# ---------- 9. 전체 프레임 브라우저 ----------
st.markdown('<div class="section-label">전체 구간 리뷰</div>', unsafe_allow_html=True)
st.caption("슬라이더를 움직여 각 프레임의 AI 분석 결과와 피드백을 실시간으로 확인하세요.")

f_idx = st.slider("프레임 탐색", 0, res["total_frames"] - 1, 0)

selected_f = kp_list[f_idx]
target_score = next(
    (fs for fs in frame_scores if fs["frame_idx"] == f_idx),
    None
)

if target_score:
    score_val = target_score['score']
    score_color = "#e7e3c4" if score_val >= 0.7 else "#FF4B4B"
    st.markdown(f"""
        <div style="display: flex; gap: 10px; margin-bottom: 15px;">
            <div class="phase-info-tag">{target_score["phase"].upper()} PHASE</div>
            <div class="phase-info-tag" style="background: {score_color}; color: #1c1c1c;">
                자세 점수: {score_val:.0%}
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="phase-info-tag" style="background:#444; color:#eee;">'
        '분석 제외 구간 (READY/FINISH)'
        '</div>',
        unsafe_allow_html=True
    )

b_col1, b_col2 = st.columns(2)

with b_col1:
    orig_img = cv2.imread(selected_f["img_path"])
    if orig_img is not None:
        st.image(
            cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption="Original Frame"
        )

with b_col2:
    skel_img = draw_skeleton_on_frame(selected_f["img_path"], selected_f["pts"])
    if skel_img is not None:
        st.image(skel_img, use_container_width=True, caption="Skeleton Visualization")
    else:
        st.info("이 프레임에서 키포인트를 검출하지 못했습니다.")

if target_score:
    st.markdown("##### 피드백")
    if target_score["errors"]:
        for err in target_score["errors"]:
            st.error(f"{err}")
    else:
        st.success("✅ 감지된 자세 오류 없음")
    
    if "details" in target_score and target_score["details"]:
        with st.expander("상세 수치 보기"):
            for check_name, detail in target_score["details"].items():
                icon = "✅" if detail["status"] == "ok" else "⚠️" if detail["status"] == "warning" else "❌"
                st.markdown(
                    f"{icon} **{check_name}**: {detail['value']}  \n"
                    f"→ {detail['feedback']}"
                )
else:
    st.info("운동이 활성화되지 않은 구간입니다. 준비 자세를 완료한 시점부터 피드백이 제공됩니다.")

st.divider()

# ---------- 10. 하단 액션 ----------
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    export_data = {
        "video": res["video_name"],
        "exercise_type": res.get("exercise_type", ""),
        "exercise_count": res.get("exercise_count", 0),
        "resolution": res["resolution"],
        "fps": res["fps"],
        "total_frames": res["total_frames"],
        "extracted_keypoints": res["success_count"],
        "avg_posture_score": round(avg_score, 2),
        "dtw_result": dtw_result,
        "combined_score": round(avg_score * 0.7 + dtw_result.get("overall_dtw_score", 0) * 0.3, 2) if dtw_active and dtw_result else None,
        "error_frame_count": len(error_frames),
        "ai_feedback": st.session_state.get("gemini_feedback", None),
        "phase_scores": [
            {
                "frame_idx": fs["frame_idx"],
                "phase": fs["phase"],
                "score": fs["score"],
                "errors": fs["errors"],
            }
            for fs in frame_scores
        ],
    }
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    st.download_button(
        "📥 JSON 리포트 다운로드",
        data=json_str,
        file_name=f"{res['video_name']}_analysis.json",
        mime="application/json",
        use_container_width=True
    )

with c2:
    if st.button("🔄 새로운 영상 분석", use_container_width=True):
        for key in ["analysis_results", "gemini_feedback"]:
            if key in st.session_state:
                del st.session_state[key]
        st.switch_page("pages/uploadvid.py")

with c3:
    if st.button("🏠 처음으로", use_container_width=True):
        st.switch_page("pages/home.py")

st.markdown(
    '<div style="text-align: center; color: #555; padding: 2rem;">'
    '© 2026 AI Pose Coach. All rights reserved.'
    '</div>',
    unsafe_allow_html=True
)
