"""
운동 영상 분석 서버
영상 업로드 → 프레임 추출 → YOLO26n-pose 키포인트 추출 → 규칙 기반 자세 분석 → 피드백
"""
import sys
import json
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
import pandas as pd

# 기존 preprocess 모듈 import 경로 설정
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "preprocess"))
sys.path.insert(0, str(ROOT / "preprocess" / "scripts"))
sys.path.insert(0, str(ROOT))

from config import (
    UPLOAD_VIDEO_DIR, OUT_FRAMES_DIR, OUT_FRAMES_MP_DIR,
    FRAME_EXTRACT_FPS, TARGET_RESOLUTION,
)
from video_preprocess import extract_frames
from extract_yolo_frames import process_single_frame
from utils.keypoints import load_pose_model, COCO_KEYPOINT_MAP
from utils.visualization import draw_skeleton_on_frame

# 필요한 모듈들을 모두 가져옵니다. (없으면 ds_modules/__init__.py에 추가하거나 직접 파일에서 import)
from ds_modules import (
    compute_virtual_keypoints, normalize_pts,
    KeypointSmoother, 
    PushUpCounter, PullUpCounter,
    PushUpEvaluator, PullUpEvaluator, 
)

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(page_title="운동 영상 분석", layout="wide")
st.title("운동 영상 분석")

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.header("설정")
    exercise_type = st.selectbox("운동 종류", [
        "푸시업", "풀업"])
    extract_fps = st.slider("프레임 추출 FPS", min_value=1, max_value=3,
                            value=FRAME_EXTRACT_FPS)
    st.caption(f"해상도: {TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]} (FHD)")
    st.caption("모델: YOLO26n-pose (COCO 17 키포인트)")

# ── YOLO 모델 캐싱 로드 ──────────────────────────────────────
@st.cache_resource
def get_pose_model():
    return load_pose_model()

# ── 1. 영상 업로드 ────────────────────────────────────────────
st.header("1. 영상 업로드")
uploaded = st.file_uploader(
    "분석할 운동 영상을 업로드하세요 (MP4, AVI, MOV, MKV, WEBM)",
    type=["mp4", "avi", "mov", "mkv", "webm"],
)

if uploaded is not None:
    # 파일 크기 검증 (500MB)
    if uploaded.size > 500 * 1024 * 1024:
        st.error("파일 크기가 500MB를 초과합니다.")
        st.stop()

    st.video(uploaded)

    # 업로드 파일 저장
    UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_save_path = UPLOAD_VIDEO_DIR / uploaded.name
    with open(video_save_path, "wb") as f:
        f.write(uploaded.getbuffer())

    video_stem = video_save_path.stem
    frames_dir = OUT_FRAMES_DIR / video_stem

    # ── 2. 분석 실행 ──────────────────────────────────────────
    st.header("2. 분석 실행")

    if st.button("분석 시작", type="primary"):
        # Step 1: 프레임 추출
        st.subheader("Step 1: 프레임 추출")
        progress_frames = st.progress(0, text="프레임 추출 중...")

        cap = cv2.VideoCapture(str(video_save_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_src_frames / src_fps if src_fps > 0 else 0
        cap.release()

        extracted_count = extract_frames(
            video_save_path, frames_dir, extract_fps, TARGET_RESOLUTION
        )
        progress_frames.progress(100, text=f"프레임 추출 완료: {extracted_count}개")

        # Step 2: YOLO26n-pose 키포인트 추출
        st.subheader("Step 2: 키포인트 추출 (YOLO26n-pose)")
        progress_kp = st.progress(0, text="키포인트 추출 중...")

        frame_files = sorted(
            f for f in frames_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".png", ".jpeg"}
        )

        pose_model = get_pose_model()
        all_keypoints = []
        success_count = 0

        for i, fpath in enumerate(frame_files):
            pts = process_single_frame(pose_model, fpath)
            all_keypoints.append({
                "frame_idx": i,
                "img_key": fpath.name,
                "img_path": str(fpath),
                "pts": pts,
            })
            if pts is not None:
                success_count += 1

            pct = int((i + 1) / len(frame_files) * 100)
            progress_kp.progress(pct, text=f"키포인트 추출 중... {i+1}/{len(frame_files)}")

        progress_kp.progress(100, text=f"키포인트 추출 완료: {success_count}/{len(frame_files)}")

        # Step 3: 자세 분석
        st.subheader("Step 3: 자세 분석")
        progress_eval = st.progress(0, text="자세 분석 중...")

        # 프레임 해상도
        sample_img = cv2.imread(str(frame_files[0])) if frame_files else None
        if sample_img is not None:
            img_h, img_w = sample_img.shape[:2]
        else:
            img_w, img_h = TARGET_RESOLUTION

        # 카운터/평가기/스무더 초기화
        smoother = KeypointSmoother(window=3)
        
        # 선택된 운동에 따라 클래스 할당
        if exercise_type == "푸시업":
            counter = PushUpCounter()
            evaluator = PushUpEvaluator()
        elif exercise_type == "풀업":
            counter = PullUpCounter()
            evaluator = PullUpEvaluator()
        else:
            # 기본값 (혹은 에러 처리)
            counter = PushUpCounter()
            evaluator = PushUpEvaluator()

        frame_scores = []
        error_frames = []

        for i, kp_data in enumerate(all_keypoints):
            pts = kp_data["pts"]
            # 가상 키포인트 생성 → 스무딩 → 정규화
            flat = compute_virtual_keypoints(pts)
            smoothed = smoother.smooth(flat)
            npts = normalize_pts(smoothed, img_w, img_h) if smoothed else None

            # 1. 카운터 상태 업데이트 (준비 완료 여부 판별)
            counter.update(npts)

            # 2. 상태에 따른 처리 (Active일 때만 평가)
            status_text = "준비 중..."
            
            if counter.is_active:
                status_text = f"운동 중 (Count: {counter.count})"
                
                # 자세 평가 실행
                result = evaluator.evaluate(npts)

                # 점수 기록
                frame_scores.append({
                    "frame_idx": i,
                    "score": result["score"],
                    "errors": result["errors"],
                    "details": result["details"],
                })

                # 오류가 있는 프레임 별도 기록
                if result["errors"] and result["errors"] != ["키포인트 없음"]:
                    error_frames.append({
                        "frame_idx": i,
                        "img_key": kp_data["img_key"],
                        "img_path": kp_data["img_path"],
                        "score": result["score"],
                        "errors": result["errors"],
                        "details": result["details"],
                        "pts": pts,
                    })
            
            else:
                # 준비 중: 카운트다운 표시 등
                frames_left = max(0, counter.active_threshold - counter.ready_frames)
                if counter.ready_frames > 0:
                    status_text = f"준비 자세 유지 중... ({counter.ready_frames}/{counter.active_threshold})"
                else:
                    status_text = "대기 중 (준비 자세를 취해주세요)"

            pct = int((i + 1) / len(all_keypoints) * 100)
            progress_eval.progress(pct, text=f"[{status_text}] {i+1}/{len(all_keypoints)}")

        progress_eval.progress(100, text="자세 분석 완료!")

        # 결과를 session_state에 저장
        st.session_state["results"] = {
            "video_name": video_stem,
            "exercise_type": exercise_type,
            "total_frames": len(frame_files),
            "success_count": success_count,
            "fps": extract_fps,
            "resolution": list(TARGET_RESOLUTION),
            "src_resolution": [src_w, src_h],
            "duration": round(duration, 1),
            "keypoints": all_keypoints,
            "exercise_count": counter.count,
            "frame_scores": frame_scores,
            "error_frames": error_frames,
        }

        st.success("분석 완료!")

    # ── 3. 결과 대시보드 ──────────────────────────────────────
    if "results" in st.session_state:
        res = st.session_state["results"]
        kp_list = res["keypoints"]
        frame_scores = res.get("frame_scores", [])
        error_frames = res.get("error_frames", [])

        st.header("3. 분석 결과")

        # [경고] 분석된 프레임이 하나도 없는 경우 (준비 동작 인식 실패 등)
        if not frame_scores:
            st.warning("⚠️ 운동 동작이 감지되지 않았습니다.")
            st.info("""
            **가능한 원인:**
            1. 준비 자세(시작 자세)가 짧아서 인식되지 않았습니다.
            2. 설정된 운동 종류와 실제 영상이 다릅니다.
            3. 카메라에 전신이 나오지 않았습니다.
            
            👉 `exercise_counter.py`의 `active_threshold` 값을 더 낮추거나, 영상을 확인해주세요.
            """)

        # ── 메트릭 카드 ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 프레임", f"{res['total_frames']}개")
        c2.metric("키포인트 성공", f"{res['success_count']}개")
        c3.metric("추출 FPS", f"{res['fps']}fps")
        c4.metric("해상도", f"{res['resolution'][0]}x{res['resolution'][1]}")

        # ── 자세 분석 메트릭 (Step 3 결과) ──
        if frame_scores:
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{res.get('exercise_type', '운동')} 횟수",
                      f"{res.get('exercise_count', 0)}회")
            m2.metric("오류 프레임",
                      f"{len(error_frames)}개",
                      delta=f"{len(error_frames)}/{res['total_frames']}",
                      delta_color="inverse")

            scores = [fs["score"] for fs in frame_scores]
            avg_score = sum(scores) / len(scores) if scores else 0
            m3.metric("평균 자세 점수", f"{avg_score:.0%}")

            # ── 프레임별 점수 차트 ──
            st.divider()
            st.subheader("프레임별 자세 점수")
            chart_df = pd.DataFrame({
                "프레임": [fs["frame_idx"] for fs in frame_scores],
                "점수": scores,
            })
            st.line_chart(chart_df, x="프레임", y="점수")

            # ── 오류 프레임 브라우저 ──
            if error_frames:
                st.divider()
                st.subheader("오류 프레임 상세")

                error_options = [
                    f"프레임 {ef['frame_idx']} (점수: {ef['score']:.0%}) — {', '.join(ef['errors'][:2])}"
                    for ef in error_frames
                ]
                selected_idx = st.selectbox("오류 프레임 선택", range(len(error_options)),
                                            format_func=lambda i: error_options[i])

                ef = error_frames[selected_idx]

                col_img, col_fb = st.columns([1, 1])

                with col_img:
                    st.markdown("**스켈레톤 오버레이**")
                    if ef["pts"] is not None:
                        skel_img = draw_skeleton_on_frame(ef["img_path"], ef["pts"])
                        if skel_img is not None:
                            st.image(skel_img, use_container_width=True)
                        else:
                            st.warning("이미지를 로드할 수 없습니다.")
                    else:
                        st.info("키포인트 없음")

                with col_fb:
                    st.markdown("**피드백**")
                    st.markdown(f"**프레임**: {ef['frame_idx']}")
                    st.markdown(f"**자세 점수**: {ef['score']:.0%}")

                    for err_msg in ef["errors"]:
                        st.error(err_msg)

                    if ef["details"]:
                        st.markdown("**상세 수치**")
                        for check_name, detail in ef["details"].items():
                            icon = "✅" if detail["status"] == "ok" else "⚠️" if detail["status"] == "warning" else "❌"
                            st.markdown(
                                f"{icon} **{check_name}**: {detail['value']}  \n"
                                f"→ {detail['feedback']}"
                            )

        st.divider()

        # ── 프레임 선택 슬라이더 ──
        if res["total_frames"] > 0:
            st.subheader("프레임 브라우저")
            frame_idx = st.slider(
                "프레임 선택", 0, res["total_frames"] - 1, 0,
                format="프레임 %d",
            )

            selected = kp_list[frame_idx]
            img_path = selected["img_path"]
            pts = selected["pts"]

            # 2컬럼 레이아웃: 원본 | 스켈레톤
            col_orig, col_skel = st.columns(2)

            with col_orig:
                st.markdown("**원본 프레임**")
                orig_img = cv2.imread(img_path)
                if orig_img is not None:
                    st.image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB),
                             use_container_width=True)
                else:
                    st.warning("이미지를 로드할 수 없습니다.")

            with col_skel:
                st.markdown("**스켈레톤 오버레이**")
                if pts is not None:
                    skel_img = draw_skeleton_on_frame(img_path, pts)
                    if skel_img is not None:
                        st.image(skel_img, use_container_width=True)
                    else:
                        st.warning("이미지를 로드할 수 없습니다.")
                else:
                    st.info("이 프레임에서 키포인트를 검출하지 못했습니다.")

            # 해당 프레임 자세 점수 표시
            if frame_scores:
                # frame_scores는 active된 프레임만 들어있으므로, 현재 슬라이더의 frame_idx와 일치하는 것을 찾아야 함
                # 방법: frame_scores 리스트를 순회하거나 딕셔너리로 변환
                target_score = next((fs for fs in frame_scores if fs["frame_idx"] == frame_idx), None)
                
                if target_score:
                    st.markdown(f"**이 프레임 자세 점수**: {target_score['score']:.0%}")
                    if target_score["errors"]:
                        for err in target_score["errors"]:
                            st.warning(err)
                else:
                    st.info("이 프레임은 분석 대상(Active) 구간이 아닙니다. (준비 중 혹은 종료)")

            # 키포인트 테이블
            if pts is not None:
                with st.expander("키포인트 상세"):
                    rows = []
                    for name, pt in pts.items():
                        rows.append({
                            "관절명": name,
                            "x": pt["x"],
                            "y": pt["y"],
                            "confidence": pt["vis"],
                        })
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

            # JSON 다운로드
            st.divider()
            
            # 안전한 평균 점수 계산
            avg_score_val = 0
            if frame_scores:
                avg_score_val = sum(fs["score"] for fs in frame_scores) / len(frame_scores)

            export_data = {
                "video": res["video_name"],
                "exercise_type": res.get("exercise_type", ""),
                "exercise_count": res.get("exercise_count", 0),
                "resolution": res["resolution"],
                "fps": res["fps"],
                "total_frames": res["total_frames"],
                "extracted_keypoints": res["success_count"],
                "avg_posture_score": round(avg_score_val, 2) if frame_scores else None,
                "error_frame_count": len(error_frames),
                "frames": [
                    {
                        "frame_idx": kp["frame_idx"],
                        "img_key": kp["img_key"],
                        "pts": kp["pts"],
                    }
                    for kp in kp_list
                    if kp["pts"] is not None
                ],
            }
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="분석 결과 JSON 다운로드",
                data=json_str,
                file_name=f"{res['video_name']}_analysis.json",
                mime="application/json",
            )