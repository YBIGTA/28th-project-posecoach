"""
AI Pose Coach - Home
Minimal Fullscreen Hero Design
"""

import streamlit as st
from pathlib import Path
import base64

st.set_page_config(page_title="AI Pose Coach", page_icon="🏋️", layout="wide")

# ---------- 배경 이미지 base64 변환 (없으면 None) ----------
BASE_DIR = Path(__file__).resolve().parent
BG_IMAGE = BASE_DIR.parent / "assets" / "hero.png"

def get_base64_image(img_path: Path):
    """이미지를 base64로 인코딩. 파일이 없으면 None 반환."""
    try:
        with open(img_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

bg_base64 = get_base64_image(BG_IMAGE)

# ---------- 배경 스타일 결정 ----------
if bg_base64:
    # hero.png가 있으면 이미지 배경 사용
    bg_style = f'background-image: url("data:image/png;base64,{bg_base64}"); background-size: cover; background-position: center; background-attachment: fixed;'
else:
    # 없으면 어두운 그라디언트 폴백
    bg_style = "background: linear-gradient(135deg, #0a0c12 0%, #1a1f2e 40%, #0e1117 100%);"

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
        {bg_style}
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

    .hero {{
        position: relative;
        z-index: 2;
        padding-top: 25vh;
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
        margin-bottom: 0.5rem;
    }}

    .hero-sub {{
        font-size: 1.3rem;
        letter-spacing: 2px;
        margin-bottom: 1.5rem;
    }}

    div.stButton > button {{
        background-color: #e7e3c4 !important;
        color: #1c1c1c !important;
        border-radius: 50px !important;
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
col1, col2, col3 = st.columns([1.5, 1, 1.5])

with col2:
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("시작하기", use_container_width=True):
        st.switch_page("pages/uploadvid.py")
    st.markdown('</div>', unsafe_allow_html=True)
