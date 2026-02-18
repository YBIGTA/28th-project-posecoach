"""
AI Pose Coach - Home
Minimal Fullscreen Hero Design
"""

import streamlit as st
from pathlib import Path
import base64

st.set_page_config(page_title="AI Pose Coach", page_icon="🏋️", layout="wide")

# ---------- 배경 이미지 base64 변환 ----------
def get_base64_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 배경 이미지 경로 (수정 가능)
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BG_IMAGE = BASE_DIR.parent / "assets" / "hero.png"

bg_base64 = get_base64_image(BG_IMAGE)

# ---------- Custom CSS ----------
st.markdown(f"""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    html, body, [class*="css"] {{
        font-family: 'Pretendard', sans-serif;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    html, body {{
        overflow: hidden;
    }}

    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.45);
        z-index: 0;
    }}

    /* 중앙 컨텐츠 섹션 */
    .hero {{
        position: relative;
        z-index: 2;
        padding-top: 25vh; /* 화면 상단에서의 위치 */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: #f5f1da;
    }}

    .hero-title {{
        font-size: 6rem;
        font-weight: 300;
        letter-spacing: 6px;
        margin-bottom: 0.5rem; /* 타이틀 아래 간격 줄임 */
    }}

    .hero-sub {{
        font-size: 1.3rem;
        letter-spacing: 2px;
        margin-bottom: 1.5rem; /* 서브타이틀과 버튼 사이 간격 대폭 줄임 */
    }}

    /* ⭐ Streamlit 실제 버튼 스타일링 ⭐ */
    div.stButton > button {{
        background-color: #e7e3c4 !important;
        color: #1c1c1c !important;
        border-radius: 50px !important; /* 버튼 둥글게 */
        padding: 0.7rem 3rem !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
        display: block;
        margin: 0 auto;
    }}

    div.stButton > button:hover {{
        background-color: #ffffff !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important;
    }}
    
    /* 버튼 컨테이너 위치 미세 조정 */
    .button-container {{
        margin-top: -10px; 
    }}
</style>

<div class="overlay"></div>

<div class="hero">
    <div class="hero-title">AI POSE COACH</div>
    <div class="hero-sub">AI 기반 운동 영상 분석 시스템</div>
</div>
""", unsafe_allow_html=True)


# ---------- 버튼 섹션 ----------
# col2의 폭을 조절하여 버튼의 가로 길이를 제한할 수 있습니다.
col1, col2, col3 = st.columns([1.5, 1, 1.5])

with col2:
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("시작하기", use_container_width=True):
        st.switch_page("pages/uploadvid.py")
    st.markdown('</div>', unsafe_allow_html=True)