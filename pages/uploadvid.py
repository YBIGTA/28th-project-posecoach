import streamlit as st
from pathlib import Path
import sys
import json
import random
import time
import cv2
import base64

# ---------- 1. 경로 및 초기 설정 ----------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "preprocess" / "scripts"))

FRAME_EXTRACT_FPS = 10
UPLOAD_VIDEO_DIR = ROOT / "data" / "uploads"
OUT_FRAMES_DIR = ROOT / "data" / "frames"
TARGET_RESOLUTION = (1920, 1080)
UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="AI Pose Coach | Settings", page_icon="⚙️", layout="wide")

# ---------- 2. Modern Dark CSS ----------
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', sans-serif; }
    .stApp { background-color: #0e1117; color: #f5f1da; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .page-title { font-size: 3rem; font-weight: 700; color: #f5f1da; text-align: center; margin-top: 2rem; letter-spacing: -1px; }
    .subtitle { text-align: center; color: #888; margin-bottom: 3rem; }
    .section-title { font-size: 1.4rem; font-weight: 600; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 10px; }
    .section-number { color: #e7e3c4; font-weight: 800; border-bottom: 2px solid #e7e3c4; }
    .exercise-option { background: rgba(255, 255, 255, 0.03); border-radius: 20px; padding: 2rem; text-align: center; transition: all 0.3s ease; border: 2px solid rgba(255,255,255,0.05); }
    .exercise-option.selected { border-color: #e7e3c4; background: rgba(231, 227, 196, 0.08); box-shadow: 0 0 20px rgba(231, 227, 196, 0.1); }
    div.stButton > button { border-radius: 50px !important; font-weight: 600 !important; transition: all 0.3s ease !important; }
    button[kind="primary"] { background-color: #e7e3c4 !important; color: #1c1c1c !important; border: none !important; height: 55px !important; font-size: 1.2rem !important; }
    .summary-card { background: rgba(255, 255, 255, 0.02); border-radius: 20px; padding: 2rem; border: 1px solid rgba(231, 227, 196, 0.2); }
</style>
""", unsafe_allow_html=True)

# ---------- 3. Header ----------
st.markdown('<div class="page-title"> SETTINGS</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">운동 종목을 선택하고 영상을 분석하세요</p>', unsafe_allow_html=True)

# ---------- 4. UI Layout ----------
col_main, col_side = st.columns([2, 1], gap="large")

with col_main:
    st.markdown('<div class="section-title"><span class="section-number">01</span> 운동 종류 선택</div>', unsafe_allow_html=True)
    ex_col1, ex_col2 = st.columns(2)
    current_ex = st.session_state.get('exercise_type')
    
    with ex_col1:
        selected = "selected" if current_ex == '푸시업' else ""
        st.markdown(f'<div class="exercise-option {selected}"><h3>🤸‍♂️ 푸시업</h3><p style="color:#888">Push-Up</p></div>', unsafe_allow_html=True)
        if st.button("푸시업 선택", key="sel_push", use_container_width=True):
            st.session_state.exercise_type = '푸시업'
            st.rerun()

    with ex_col2:
        selected = "selected" if current_ex == '풀업' else ""
        st.markdown(f'<div class="exercise-option {selected}"><h3>🏋️‍♂️ 풀업</h3><p style="color:#888">Pull-Up</p></div>', unsafe_allow_html=True)
        if st.button("풀업 선택", key="sel_pull", use_container_width=True):
            st.session_state.exercise_type = '풀업'
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span class="section-number">02</span> 영상 업로드</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("영상 업로드", type=["mp4", "mov", "avi", "webm"], label_visibility="collapsed")
    if uploaded_file:
        st.session_state.uploaded_video = uploaded_file
        st.video(uploaded_file)

with col_side:
    st.markdown('<div class="section-title"><span class="section-number">03</span> 분석 설정</div>', unsafe_allow_html=True)
    extract_fps = st.slider("추출 FPS", 2, 30, st.session_state.get('extract_fps', 10))
    st.session_state.extract_fps = extract_fps
    
    # ✅ DTW 레퍼런스 관리
    st.divider()
    st.markdown("**DTW 레퍼런스**")
    
    ex_type = st.session_state.get('exercise_type', '미선택')
    if ex_type != '미선택':
        ref_name = "reference_pushup.json" if ex_type == '푸시업' else "reference_pullup.json"
        ref_path = ROOT / "ds_modules" / ref_name
        
        if ref_path.exists():
            st.success(f"✅ {ex_type} 레퍼런스 등록됨")
            if st.button("🗑️ 레퍼런스 삭제", key="del_ref", use_container_width=True):
                ref_path.unlink()
                st.rerun()
        else:
            st.warning(f"⚠️ {ex_type} 레퍼런스 미등록")
        
        ref_video = st.file_uploader(
            "모범 영상 업로드",
            type=["mp4", "mov", "avi", "webm"],
            key="ref_uploader",
            help="완벽한 폼의 운동 영상을 업로드하세요"
        )
        
        if ref_video and st.button("📥 레퍼런스 생성", key="gen_ref", use_container_width=True):
            with st.spinner("레퍼런스 생성 중..."):
                tmp_video_path = UPLOAD_VIDEO_DIR / f"ref_{ref_video.name}"
                with open(tmp_video_path, "wb") as f:
                    f.write(ref_video.getbuffer())
                
                try:
                    from scripts.generate_reference import generate_reference
                    success = generate_reference(
                        str(tmp_video_path),
                        ex_type,
                        str(ref_path),
                        extract_fps
                    )
                    tmp_video_path.unlink(missing_ok=True)
                    
                    if success:
                        st.success("✅ 레퍼런스 생성 완료!")
                        st.rerun()
                    else:
                        st.error("❌ 레퍼런스 생성 실패")
                except Exception as e:
                    st.error(f"오류: {e}")
                    tmp_video_path.unlink(missing_ok=True)
    
    st.divider()
    
    vid_name = st.session_state.uploaded_video.name if st.session_state.get('uploaded_video') else '미업로드'
    display_name = (vid_name[:20] + '...') if len(vid_name) > 20 else vid_name

    st.markdown(f"""
    <div class="summary-card">
        <p style="color:#888; font-size:0.85rem; margin-bottom:15px; letter-spacing:1px;">설정 요약</p>
        <div style="margin-bottom: 8px;"><span style="color:#e7e3c4; font-weight:600;">종목:</span> <span style="color:#fff;">{ex_type}</span></div>
        <div style="margin-bottom: 8px;"><span style="color:#e7e3c4; font-weight:600;">파일:</span> <span style="color:#fff;">{display_name}</span></div>
        <div style="margin-bottom: 20px;"><span style="color:#e7e3c4; font-weight:600;">설정:</span> <span style="color:#fff;">{extract_fps} FPS</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    can_proceed = (ex_type != '미선택' and vid_name != '미업로드')
    start_analysis = st.button("🚀 분석 시작하기", type="primary", disabled=not can_proceed, use_container_width=True)
    
    if st.button("🏠 처음으로", use_container_width=True):
        st.switch_page("pages/home.py")

# ---------- 5. Analysis Logic ----------
if start_analysis and can_proceed:
    modal_placeholder = st.empty()
    
    workout_tips = [
        "💡 푸시업을 할 때는 코어에 힘을 주고 몸을 일직선으로 유지하세요!",
        "💡 풀업은 팔의 힘보다 등의 힘(광배근)을 사용한다는 느낌으로 당기세요.",
        "💡 운동 전 스트레칭은 부상 방지에 가장 효과적입니다.",
        "💡 근육 성장의 80%는 주방(식단)과 침대(휴식)에서 일어납니다.",
        "💡 호흡을 멈추지 마세요. 수축할 때 뱉고 이완할 때 마시는 것이 기본입니다."
    ]
    workout_memes = ["밈이나 명언"]
    
    selected_tip = random.choice(workout_tips)
    selected_meme = random.choice(workout_memes)

    try:
        from video_preprocess import extract_frames
        from extract_yolo_frames import process_single_frame
        from utils.keypoints import load_pose_model
        from utils.activity_segment import detect_active_frame_indices, resolve_activity_model_path
        from ds_modules import compute_virtual_keypoints, normalize_pts, KeypointSmoother, PushUpCounter, PullUpCounter
        from ds_modules.phase_detector import create_phase_detector, extract_phase_metric
        from ds_modules.posture_evaluator_phase import PushUpEvaluator, PullUpEvaluator
        from ds_modules.dtw_scorer import DTWScorer, extract_feature_vector  # ✅ DTW 추가

        uploaded = st.session_state.uploaded_video
        video_save_path = UPLOAD_VIDEO_DIR / uploaded.name
        with open(video_save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        
        video_stem = video_save_path.stem
        frames_dir = OUT_FRAMES_DIR / video_stem
        if frames_dir.exists():
            import shutil
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        steps = ["프레임 추출 중...", "관절 위치 추적 중...", "자세 분석 중..."]
        
        # Step 1
        with modal_placeholder.container():
            st.markdown(f"""
                <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(14, 17, 23, 0.95); z-index: 9999; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #f5f1da; padding: 20px;">
                    <div style="background: rgba(255,255,255,0.05); border: 1px solid #e7e3c4; border-radius: 30px; padding: 40px; text-align: center; max-width: 600px; box-shadow: 0 20px 50px rgba(0,0,0,0.5);">
                        <h2 style="color: #e7e3c4; margin-bottom: 20px;">🏋️ 분석 중...</h2>
                        <p style="font-size: 1.2rem; margin-bottom: 30px; font-style: italic;">"{selected_meme}"</p>
                        <div style="background: rgba(231, 227, 196, 0.1); padding: 20px; border-radius: 15px; margin-bottom: 30px;">
                            <p style="margin: 0; font-size: 1rem; color: #f5f1da;">{selected_tip}</p>
                        </div>
                        <p style="color: #888; font-size: 0.9rem; margin-bottom: 10px;">{steps[0]}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            cap = cv2.VideoCapture(str(video_save_path))
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_src_frames / src_fps if src_fps > 0 else 0
            src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            extract_frames(video_save_path, frames_dir, extract_fps, TARGET_RESOLUTION)
        
        # Step 2
        with modal_placeholder.container():
            st.markdown(f"""
                <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(14, 17, 23, 0.95); z-index: 9999; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #f5f1da; padding: 20px;">
                    <div style="background: rgba(255,255,255,0.05); border: 1px solid #e7e3c4; border-radius: 30px; padding: 40px; text-align: center; max-width: 600px;">
                        <h2 style="color: #e7e3c4; margin-bottom: 20px;">🏋️ 분석 중...</h2>
                        <p style="font-size: 1.2rem; margin-bottom: 30px; font-style: italic;">"{selected_meme}"</p>
                        <div style="background: rgba(231, 227, 196, 0.1); padding: 20px; border-radius: 15px; margin-bottom: 30px;">
                            <p style="margin: 0; font-size: 1rem; color: #f5f1da;">{selected_tip}</p>
                        </div>
                        <p style="color: #888; font-size: 0.9rem; margin-bottom: 10px;">{steps[1]}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            frame_files = sorted(f for f in frames_dir.iterdir() if f.suffix.lower() in {".jpg", ".png", ".jpeg"})
            pose_model = load_pose_model()
            all_keypoints = []
            success_count = 0
            for i, fpath in enumerate(frame_files):
                pts = process_single_frame(pose_model, fpath)
                if pts is not None:
                    success_count += 1
                all_keypoints.append({"frame_idx": i, "img_key": fpath.name, "img_path": str(fpath), "pts": pts})

        # Step 3
        with modal_placeholder.container():
            st.markdown(f"""
                <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(14, 17, 23, 0.95); z-index: 9999; display: flex; flex-direction: column; justify-content: center; align-items: center; color: #f5f1da; padding: 20px;">
                    <div style="background: rgba(255,255,255,0.05); border: 1px solid #e7e3c4; border-radius: 30px; padding: 40px; text-align: center; max-width: 600px;">
                        <h2 style="color: #e7e3c4; margin-bottom: 20px;">🏋️ 분석 중...</h2>
                        <p style="font-size: 1.2rem; margin-bottom: 30px; font-style: italic;">"{selected_meme}"</p>
                        <div style="background: rgba(231, 227, 196, 0.1); padding: 20px; border-radius: 15px; margin-bottom: 30px;">
                            <p style="margin: 0; font-size: 1rem; color: #f5f1da;">{selected_tip}</p>
                        </div>
                        <p style="color: #888; font-size: 0.9rem; margin-bottom: 10px;">{steps[2]}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            img_h, img_w = TARGET_RESOLUTION[1], TARGET_RESOLUTION[0]
            smoother = KeypointSmoother(window=3)
            exercise_type = st.session_state.exercise_type
            phase_detector = create_phase_detector(exercise_type)
            counter, evaluator = (PushUpCounter(), PushUpEvaluator()) if exercise_type == "푸시업" else (PullUpCounter(), PullUpEvaluator())
            
            # ✅ DTW 초기화
            ref_name = "reference_pushup.json" if exercise_type == "푸시업" else "reference_pullup.json"
            ref_path = ROOT / "ds_modules" / ref_name
            dtw_scorer = DTWScorer(str(ref_path), exercise_type)
            dtw_active = dtw_scorer.active
            
            frame_scores, error_frames = [], []
            total_frames = len(all_keypoints)  # ✅ 총 프레임 수
            use_ml_filter = True
            exercise_tag = "pushup" if isinstance(counter, PushUpCounter) else "pullup"
            model_path = resolve_activity_model_path(exercise_tag)
            active_frame_indices, filter_meta = detect_active_frame_indices(
                frame_files,
                extract_fps=extract_fps,
                use_ml=use_ml_filter,
                model_path=model_path,
                min_keep_ratio=0.35,
                return_details=True,
            )
            if not active_frame_indices:
                active_frame_indices = set(range(total_frames))
                filter_meta = {"method": "fallback_all", "reason": "no active frames selected"}
            
            # ✅ 수정된 부분: counter.update 호출 방식 변경
            for i, kp_data in enumerate(all_keypoints):
                pts = kp_data["pts"]
                flat = compute_virtual_keypoints(pts)
                smoothed = smoother.smooth(flat)
                npts = normalize_pts(smoothed, img_w, img_h) if smoothed else None
                phase_metric = extract_phase_metric(npts, exercise_type)
                current_phase = phase_detector.update(phase_metric) if phase_metric is not None else 'ready'
                
                # ✅ 영상 끝 10프레임이면 강제 종료
                if i >= total_frames - 10 and counter.is_active:
                    counter.is_active = False
                
                # ✅ 핵심 수정: is_active 체크 후 evaluator 실행
                if counter.is_active and i in active_frame_indices:
                    res = evaluator.evaluate(npts, phase=current_phase)
                    
                    # DTW 피처 축적
                    if dtw_active:
                        feat_vec = extract_feature_vector(npts, exercise_type)
                        dtw_scorer.accumulate(feat_vec, current_phase)
                    
                    # 카운터 업데이트 (파라미터 제거)
                    counter.update(npts, current_phase)
                    
                    frame_scores.append({
                        "frame_idx": i, 
                        "phase": current_phase, 
                        "score": res["score"], 
                        "errors": res["errors"],
                        "details": res["details"]
                    })
                    
                    if res["errors"] and res["errors"] != ["키포인트 없음"]:
                        error_frames.append({
                            "frame_idx": i, 
                            "img_path": kp_data["img_path"], 
                            "phase": current_phase, 
                            "score": res["score"], 
                            "errors": res["errors"], 
                            "details": res["details"], 
                            "pts": pts
                        })
                else:
                    # is_active 전환 체크용
                    counter.update(npts, current_phase)

        # ✅ DTW 결과 산출
        dtw_result = dtw_scorer.finalize() if dtw_active else None

        # ✅ 결과 저장
        st.session_state.analysis_results = {
            "video_name": video_stem, 
            "exercise_type": exercise_type, 
            "exercise_count": counter.count,
            "frame_scores": frame_scores, 
            "error_frames": error_frames, 
            "duration": round(duration, 1),
            "fps": extract_fps, 
            "keypoints": all_keypoints, 
            "total_frames": len(frame_files), 
            "analysis_target_frames": len(active_frame_indices),
            "filter_method": filter_meta.get("method", ""),
            "filter_reason": filter_meta.get("reason", ""),
            "filter_model_path": str(model_path),
            "success_count": success_count,
            "resolution": list(TARGET_RESOLUTION),
            "dtw_result": dtw_result,
            "dtw_active": dtw_active,
        }
        st.switch_page("pages/results.py")

    except Exception as e:
        modal_placeholder.empty()
        st.error(f"오류 발생: {e}")
