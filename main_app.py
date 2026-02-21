"""
AIFit - AI 기반 운동 자세 분석 시스템
메인 진입점
"""
import streamlit as st
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from db.database import init_db

# DB 초기화
init_db()

# 페이지 설정
st.set_page_config(
    page_title="AIFit - AI 운동 자세 분석",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'analysis_state' not in st.session_state:
    st.session_state.analysis_state = 'home'

if 'exercise_type' not in st.session_state:
    st.session_state.exercise_type = None

if 'uploaded_video' not in st.session_state:
    st.session_state.uploaded_video = None

if 'extract_fps' not in st.session_state:
    st.session_state.extract_fps = 10

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Gemini 피드백 관련 세션 상태 초기화
import os
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
if 'gemini_feedback' not in st.session_state:
    st.session_state.gemini_feedback = None
if 'gemini_temperature' not in st.session_state:
    st.session_state.gemini_temperature = 0.7
if 'gemini_max_tokens' not in st.session_state:
    st.session_state.gemini_max_tokens = 6000

# 유저 세션 상태 초기화
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

if 'username' not in st.session_state:
    st.session_state.username = None

if 'guest_mode' not in st.session_state:
    st.session_state.guest_mode = False

# 미로그인 시 로그인 페이지로
if st.session_state.user_id is None and not st.session_state.guest_mode:
    st.switch_page("pages/login.py")
else:
    st.switch_page("pages/home.py")
