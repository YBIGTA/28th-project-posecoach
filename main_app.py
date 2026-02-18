"""
AIFit - AI 기반 운동 자세 분석 시스템
메인 진입점
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

# 세션 상태 초기화
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

# 홈으로 리다이렉션
st.switch_page("pages/home.py")