"""
AIFit - 운동 기록 히스토리
종합 통계, 성장 추이 차트, 운동 목록
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.database import get_user_workouts, get_user_stats

st.set_page_config(page_title="AIFit - 운동 기록", page_icon="📈", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', sans-serif; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background-color: #0e1117;
        color: #f5f1da;
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
        background: linear-gradient(145deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(231,227,196,0.15);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(231,227,196,0.5);
    }
    .metric-container p {
        color: #888;
        font-size: 0.9rem;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .grade-badge {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e7e3c4;
        margin: 8px 0;
    }

    .chart-box {
        background: rgba(255,255,255,0.02);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.05);
        margin-bottom: 20px;
    }

    .workout-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(231,227,196,0.1);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 12px;
    }

    div.stButton > button {
        border-radius: 50px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 로그인 체크 ----------
user_id = st.session_state.get("user_id")
username = st.session_state.get("username", "")

if not user_id:
    st.warning("로그인이 필요합니다.")
    st.info("운동 기록을 확인하려면 로그인해주세요.")
    if st.button("로그인하기"):
        st.switch_page("pages/login.py")
    st.stop()

# ---------- 헤더 ----------
st.markdown(f"""
<div style="text-align:center; padding:2rem 0;">
    <h1 style="font-weight:300; letter-spacing:4px; color:#f5f1da;">
        {username}님의 운동 기록
    </h1>
</div>
""", unsafe_allow_html=True)

# ---------- 데이터 로드 ----------
stats = get_user_stats(user_id)
workouts = get_user_workouts(user_id)

if stats["total_workouts"] == 0:
    st.markdown("""
    <div style="text-align:center; padding:4rem 0; color:#888;">
        <p style="font-size:3rem;">📭</p>
        <p style="font-size:1.2rem;">아직 운동 기록이 없습니다.</p>
        <p>영상을 업로드하고 분석을 시작해보세요!</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        if st.button("운동 시작하기", use_container_width=True):
            st.switch_page("pages/uploadvid.py")
    with col2:
        if st.button("홈으로", use_container_width=True):
            st.switch_page("pages/home.py")
    st.stop()

# ---------- 종합 통계 카드 ----------
st.markdown('<div class="section-label">종합 통계</div>', unsafe_allow_html=True)

s1, s2, s3, s4, s5 = st.columns(5)
with s1:
    st.markdown(f'<div class="metric-container"><p>총 운동 횟수</p><div class="grade-badge">{stats["total_workouts"]}회</div></div>', unsafe_allow_html=True)
with s2:
    mins = stats["total_duration"] / 60
    st.markdown(f'<div class="metric-container"><p>총 운동 시간</p><div class="grade-badge">{mins:.1f}분</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown(f'<div class="metric-container"><p>평균 점수</p><div class="grade-badge">{stats["overall_avg_score"]:.0%}</div></div>', unsafe_allow_html=True)
with s4:
    st.markdown(f'<div class="metric-container"><p>총 운동 수</p><div class="grade-badge">{stats["total_reps"]}회</div></div>', unsafe_allow_html=True)
with s5:
    st.markdown(f'<div class="metric-container"><p>최다 운동</p><div class="grade-badge" style="font-size:1.5rem;">{stats["favorite_exercise"]}</div></div>', unsafe_allow_html=True)

st.divider()

# ---------- 성장 추이 차트 ----------
st.markdown('<div class="section-label">성장 추이</div>', unsafe_allow_html=True)

df = pd.DataFrame([
    {
        "날짜": w["created_at"][:10],
        "시간": w["created_at"],
        "점수": w["avg_score"],
        "운동": w["exercise_type"],
        "횟수": w["exercise_count"],
        "등급": w["grade"],
    }
    for w in workouts
])
df = df.sort_values("시간")

chart_left, chart_right = st.columns(2)

with chart_left:
    st.markdown('<div class="chart-box"><b>점수 추이</b>', unsafe_allow_html=True)
    fig_line = px.line(
        df, x="시간", y="점수", color="운동",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_line.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e7e3c4",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", range=[0, 1]),
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with chart_right:
    st.markdown('<div class="chart-box"><b>세션별 운동 횟수</b>', unsafe_allow_html=True)
    fig_bar = px.bar(
        df, x="시간", y="횟수", color="운동",
        color_discrete_sequence=["#e7e3c4", "#888"],
    )
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e7e3c4",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ---------- 운동 목록 ----------
st.markdown('<div class="section-label">운동 기록 목록</div>', unsafe_allow_html=True)

for w in workouts:
    # 등급 색상
    g = w["grade"]
    if "S" in g:
        gc = "#FFD700"
    elif "A" in g:
        gc = "#e7e3c4"
    elif "B" in g:
        gc = "#888"
    else:
        gc = "#FF4B4B"

    date_str = w["created_at"][:16].replace("T", " ")
    grip_info = f" ({w['grip_type']})" if w.get("grip_type") else ""
    dtw_info = f" | DTW: {w['dtw_score']:.0%}" if w.get("dtw_score") is not None else ""
    combined_info = f" | 종합: {w['combined_score']:.0%}" if w.get("combined_score") is not None else ""

    header = f"{date_str} — {w['exercise_type']}{grip_info} — {w['exercise_count']}회 — {w['avg_score']:.0%}"

    with st.expander(header):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"**영상:** {w['video_name']}")
            st.markdown(f"**운동:** {w['exercise_type']}{grip_info}")
            st.markdown(f"**횟수:** {w['exercise_count']}회")
        with col_b:
            st.markdown(f"**점수:** {w['avg_score']:.0%}{dtw_info}{combined_info}")
            st.markdown(f"**등급:** <span style='color:{gc};font-weight:700;'>{g}</span>", unsafe_allow_html=True)
            st.markdown(f"**오류 프레임:** {w['error_frame_count']}개")
        with col_c:
            st.markdown(f"**영상 길이:** {w['duration']:.1f}초")
            st.markdown(f"**FPS:** {w['fps']} | **프레임:** {w['total_frames']}")

        # 오류 상세
        if w.get("errors"):
            st.markdown("---")
            st.markdown("**주요 오류:**")
            for err in w["errors"]:
                st.markdown(f"- {err['error_msg']} ({err['count']}회)")

        # Phase 점수
        if w.get("phase_scores"):
            st.markdown("---")
            st.markdown("**Phase별 점수:**")
            pcols = st.columns(len(w["phase_scores"]))
            for col, ps in zip(pcols, w["phase_scores"]):
                with col:
                    st.metric(
                        ps["phase"].replace("_", " ").title(),
                        f"{ps['avg_score']:.0%}",
                        delta=f"{ps['frame_count']} frames",
                    )

st.divider()

# ---------- 하단 네비게이션 ----------
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🏋️ 새 운동 시작", use_container_width=True):
        st.switch_page("pages/uploadvid.py")
with c2:
    if st.button("🏠 홈으로", use_container_width=True):
        st.switch_page("pages/home.py")
with c3:
    if st.button("🚪 로그아웃", use_container_width=True):
        for key in ["user_id", "username", "guest_mode"]:
            st.session_state.pop(key, None)
        st.switch_page("pages/login.py")
