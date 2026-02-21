"""
AIFit - AI 기반 운동 자세 분석 시스템
메인 진입점 (Gemini AI 피드백 기능 추가)

변경사항:
  - gemini_api_key, gemini_feedback, gemini_temperature, gemini_max_tokens
    세션 상태 초기화 추가
  - pages/results.py → pages/results_with_feedback.py 로 라우팅 변경
"""
import streamlit as st
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="AIFit - AI 운동 자세 분석",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 기존 세션 상태 초기화 ─────────────────────────────────
if 'analysis_state' not in st.session_state:
    st.session_state.analysis_state = 'home'

if 'exercise_type' not in st.session_state:
    st.session_state.exercise_type = None

if 'uploaded_video' not in st.session_state:
    st.session_state.uploaded_video = None

if 'extract_fps' not in st.session_state:
    st.session_state.extract_fps = 2

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# ── Gemini 피드백 관련 세션 상태 초기화 ──────────────────
if 'gemini_api_key' not in st.session_state:
    # 환경변수에서 자동 로드 (없으면 빈 문자열 → UI에서 입력)
    import os
    st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

if 'gemini_feedback' not in st.session_state:
    st.session_state.gemini_feedback = None

if 'gemini_temperature' not in st.session_state:
    st.session_state.gemini_temperature = 0.7

if 'gemini_max_tokens' not in st.session_state:
    st.session_state.gemini_max_tokens = 6000

# ── 홈으로 리다이렉션 ────────────────────────────────────
st.switch_page("pages/home.py")
