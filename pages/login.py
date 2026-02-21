"""
AIFit - 로그인 / 회원가입 페이지
"""
import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.database import init_db
from db.auth import register_user, login_user

# DB 초기화
init_db()

st.set_page_config(page_title="AIFit - 로그인", page_icon="🔑", layout="wide")

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

    .login-header {
        text-align: center;
        padding: 3rem 0 1rem 0;
    }
    .login-header h1 {
        font-size: 3rem;
        font-weight: 300;
        letter-spacing: 4px;
        color: #f5f1da;
    }
    .login-header p {
        color: #888;
        letter-spacing: 2px;
    }

    .login-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(231,227,196,0.15);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    div.stButton > button {
        background-color: #e7e3c4 !important;
        color: #1c1c1c !important;
        border-radius: 50px !important;
        padding: 0.7rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        background-color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2) !important;
    }

    .guest-btn button {
        background-color: transparent !important;
        color: #888 !important;
        border: 1px solid #444 !important;
    }
    .guest-btn button:hover {
        color: #f5f1da !important;
        border-color: #e7e3c4 !important;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- 이미 로그인 상태면 홈으로 ----------
if st.session_state.get("user_id") is not None:
    st.switch_page("pages/home.py")

# ---------- 헤더 ----------
st.markdown("""
<div class="login-header">
    <h1>AI POSE COACH</h1>
    <p>로그인하여 운동 기록을 관리하세요</p>
</div>
""", unsafe_allow_html=True)

# ---------- 로그인/회원가입 폼 ----------
_, center, _ = st.columns([1, 1.2, 1])

with center:
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["로그인", "회원가입"])

    with tab_login:
        with st.form("login_form"):
            login_id = st.text_input("아이디", key="login_id")
            login_pw = st.text_input("비밀번호", type="password", key="login_pw")
            submitted = st.form_submit_button("로그인", use_container_width=True)

            if submitted:
                if not login_id or not login_pw:
                    st.error("아이디와 비밀번호를 입력해주세요.")
                else:
                    ok, msg, uid = login_user(login_id, login_pw)
                    if ok:
                        st.session_state["user_id"] = uid
                        st.session_state["username"] = login_id
                        st.success(msg)
                        st.switch_page("pages/home.py")
                    else:
                        st.error(msg)

    with tab_register:
        with st.form("register_form"):
            reg_id = st.text_input("아이디", key="reg_id")
            reg_pw = st.text_input("비밀번호", type="password", key="reg_pw")
            reg_pw2 = st.text_input("비밀번호 확인", type="password", key="reg_pw2")
            submitted = st.form_submit_button("회원가입", use_container_width=True)

            if submitted:
                if not reg_id or not reg_pw:
                    st.error("모든 항목을 입력해주세요.")
                elif reg_pw != reg_pw2:
                    st.error("비밀번호가 일치하지 않습니다.")
                else:
                    ok, msg = register_user(reg_id, reg_pw)
                    if ok:
                        st.success(msg + " 로그인해주세요.")
                    else:
                        st.error(msg)

    st.markdown('</div>', unsafe_allow_html=True)

    # 게스트 버튼
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="guest-btn">', unsafe_allow_html=True)
    if st.button("게스트로 시작 (기록 저장 안됨)", use_container_width=True):
        st.session_state["user_id"] = None
        st.session_state["username"] = "게스트"
        st.session_state["guest_mode"] = True
        st.switch_page("pages/home.py")
    st.markdown('</div>', unsafe_allow_html=True)
