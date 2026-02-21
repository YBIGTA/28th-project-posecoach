"""
운동 영상 분석 서버 (Phase 감지 버전)
"""
import sys
import json
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "preprocess"))
sys.path.insert(0, str(ROOT / "preprocess" / "scripts"))
sys.path.insert(0, str(ROOT))

from config import (
    UPLOAD_VIDEO_DIR, OUT_FRAMES_DIR,
    FRAME_EXTRACT_FPS, TARGET_RESOLUTION,
)
from video_preprocess import extract_frames
from extract_yolo_frames import process_single_frame
from utils.keypoints import load_pose_model
from utils.visualization import draw_skeleton_on_frame

from ds_modules import (
    compute_virtual_keypoints, normalize_pts,
    KeypointSmoother, PushUpCounter, PullUpCounter,
)
from ds_modules.phase_detector import create_phase_detector, extract_phase_metric
from ds_modules.posture_evaluator_phase import PushUpEvaluator, PullUpEvaluator
from ds_modules.dtw_scorer import DTWScorer, extract_feature_vector

_FEATURE_KO = {
    "elbow_L": "왼쪽 팔꿈치 각도", "elbow_R": "오른쪽 팔꿈치 각도",
    "back": "등(척추) 직선", "abd_L": "왼쪽 어깨 외전", "abd_R": "오른쪽 어깨 외전",
    "head_tilt": "고개 기울기", "hand_offset": "손 위치",
    "shoulder_packing": "어깨 패킹", "elbow_flare": "팔꿈치 벌림", "body_sway": "몸통 흔들림",
}
_SPEED_KO = {"fast": "⚡ 너무 빠름", "normal": "✅ 적절", "slow": "🐢 너무 느림"}
_PHASE_KO = {
    "top": "최고점(팔 펴기)", "bottom": "최저점(내려가기)",
    "descending": "내려가는 구간", "ascending": "올라오는 구간", "ready": "준비 자세",
}

st.set_page_config(page_title="운동 영상 분석", layout="wide")
st.title("운동 영상 분석 (Phase별 평가)")

@st.cache_resource
def get_pose_model():
    return load_pose_model()

# ── 사이드바 ─────────────────────────────────────────────────
with st.sidebar:
    st.header("설정")
    exercise_type = st.selectbox("운동 종류", ["푸시업", "풀업"])
    extract_fps   = st.slider("프레임 추출 FPS", 1, 30, value=FRAME_EXTRACT_FPS)
    st.caption(f"해상도: {TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]}")

    st.divider()
    st.subheader("DTW 레퍼런스")
    ref_name = "reference_pushup.json" if exercise_type == "푸시업" else "reference_pullup.json"
    ref_path = ROOT / "ds_modules" / ref_name

    if ref_path.exists():
        st.success(f"등록됨: {ref_name}")
        if st.button("레퍼런스 삭제", type="secondary"):
            ref_path.unlink()
            st.rerun()
    else:
        st.warning("레퍼런스 미등록")

    ref_video = st.file_uploader("모범 영상 업로드", type=["mp4","avi","mov","mkv","webm"], key="ref_uploader")
    if ref_video and st.button("레퍼런스 생성", type="primary"):
        with st.spinner("레퍼런스 생성 중..."):
            tmp_dir  = ROOT / "preprocess" / "data" / "uploads"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / f"ref_{ref_video.name}"
            tmp_path.write_bytes(ref_video.getbuffer())
            sys.path.insert(0, str(ROOT / "scripts"))
            from generate_reference import generate_reference
            ok = generate_reference(str(tmp_path), exercise_type, str(ref_path), extract_fps, pose_model=get_pose_model())
            tmp_path.unlink(missing_ok=True)
            if ok:
                st.success("완료!")
                get_pose_model.clear()
                st.rerun()
            else:
                st.error("실패. 영상을 확인해주세요.")


# ── 1. 영상 업로드 ────────────────────────────────────────────
st.header("1. 영상 업로드")
uploaded = st.file_uploader("분석할 운동 영상 (MP4/AVI/MOV/MKV/WEBM)", type=["mp4","avi","mov","mkv","webm"])

if uploaded is not None:
    if uploaded.size > 500 * 1024 * 1024:
        st.error("500MB 초과")
        st.stop()

    st.video(uploaded)
    UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_path = UPLOAD_VIDEO_DIR / uploaded.name
    video_path.write_bytes(uploaded.getbuffer())
    video_stem = video_path.stem
    frames_dir = OUT_FRAMES_DIR / video_stem

    st.header("2. 분석 실행")

    if st.button("분석 시작", type="primary"):
        import shutil
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Step 1
        st.subheader("Step 1: 프레임 추출")
        prog1 = st.progress(0, text="프레임 추출 중...")
        cap = cv2.VideoCapture(str(video_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration  = total_src / src_fps if src_fps > 0 else 0
        cap.release()
        n_ext = extract_frames(video_path, frames_dir, extract_fps, TARGET_RESOLUTION)
        prog1.progress(100, text=f"완료: {n_ext}개")

        # Step 2
        st.subheader("Step 2: 키포인트 추출")
        prog2 = st.progress(0, text="키포인트 추출 중...")
        frame_files = sorted(f for f in frames_dir.iterdir() if f.suffix.lower() in {".jpg",".png",".jpeg"})
        pose_model  = get_pose_model()
        all_kp, ok_count = [], 0
        for i, fpath in enumerate(frame_files):
            pts = process_single_frame(pose_model, fpath)
            all_kp.append({"frame_idx": i, "img_key": fpath.name, "img_path": str(fpath), "pts": pts})
            if pts:
                ok_count += 1
            prog2.progress(int((i+1)/len(frame_files)*100), text=f"{i+1}/{len(frame_files)}")
        prog2.progress(100, text=f"완료: {ok_count}/{len(frame_files)}")

        # Step 3
        st.subheader("Step 3: 자세 분석")
        prog3 = st.progress(0, text="분석 중...")

        sample = cv2.imread(str(frame_files[0])) if frame_files else None
        img_h, img_w = sample.shape[:2] if sample is not None else (TARGET_RESOLUTION[1], TARGET_RESOLUTION[0])

        smoother       = KeypointSmoother(window=3)
        phase_detector = create_phase_detector(exercise_type)
        counter        = PushUpCounter() if exercise_type == "푸시업" else PullUpCounter()
        evaluator      = PushUpEvaluator() if exercise_type == "푸시업" else PullUpEvaluator()
        dtw_scorer     = DTWScorer(str(ref_path), exercise_type)
        dtw_active     = dtw_scorer.active

        if dtw_active:
            st.info(f"DTW 활성화 ({ref_name})")

        frame_scores, error_frames = [], []

        for i, kp in enumerate(all_kp):
            pts      = kp["pts"]
            flat     = compute_virtual_keypoints(pts)
            smoothed = smoother.smooth(flat)
            npts     = normalize_pts(smoothed, img_w, img_h) if smoothed else None
            metric   = extract_phase_metric(npts, exercise_type)
            phase    = phase_detector.update(metric) if metric is not None else "ready"
            counter.update(npts, phase)

            if counter.is_active:
                result = evaluator.evaluate(npts, phase=phase)
                if dtw_active:
                    vec = extract_feature_vector(npts, exercise_type)
                    dtw_scorer.accumulate(vec, phase, img_path=kp["img_path"])

                frame_scores.append({
                    "frame_idx": i, "phase": phase, "rep_idx": counter.count,
                    "score": result["score"], "errors": result["errors"],
                    "details": result["details"], "weights_used": result.get("weights_used", {}),
                })
                if result["errors"] and result["errors"] != ["키포인트 없음"]:
                    error_frames.append({
                        "frame_idx": i, "phase": phase, "rep_idx": counter.count,
                        "img_key": kp["img_key"], "img_path": kp["img_path"],
                        "score": result["score"], "errors": result["errors"],
                        "details": result["details"], "pts": pts,
                    })
                status = f"운동 중 (Count:{counter.count}, Phase:{phase})"
            else:
                status = f"준비 중 ({counter.ready_frames}/{counter.active_threshold})" if counter.ready_frames > 0 else "대기 중"

            prog3.progress(int((i+1)/len(all_kp)*100), text=f"[{status}] {i+1}/{len(all_kp)}")

        prog3.progress(100, text="완료!")
        dtw_result = dtw_scorer.finalize() if dtw_active else None

        st.session_state["results"] = {
            "video_name": video_stem, "exercise_type": exercise_type,
            "total_frames": len(frame_files), "ok_count": ok_count,
            "fps": extract_fps, "resolution": list(TARGET_RESOLUTION),
            "duration": round(duration, 1), "keypoints": all_kp,
            "exercise_count": counter.count, "frame_scores": frame_scores,
            "error_frames": error_frames, "dtw_result": dtw_result, "dtw_active": dtw_active,
        }
        st.success("분석 완료!")

    # ── 3. 결과 ───────────────────────────────────────────────
    if "results" not in st.session_state:
        st.stop()

    res          = st.session_state["results"]
    kp_list      = res["keypoints"]
    frame_scores = res["frame_scores"]
    error_frames = res["error_frames"]
    dtw_result   = res.get("dtw_result")
    dtw_active   = res.get("dtw_active", False)

    st.header("3. 분석 결과")

    if not frame_scores:
        st.warning("운동 동작이 감지되지 않았습니다. 준비 자세 유지 시간, 운동 종류, 카메라 각도를 확인해주세요.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 프레임",     f"{res['total_frames']}개")
    c2.metric("키포인트 성공", f"{res['ok_count']}개")
    c3.metric("추출 FPS",      f"{res['fps']}fps")
    c4.metric("해상도",        f"{res['resolution'][0]}x{res['resolution'][1]}")

    st.divider()
    avg_score = sum(fs["score"] for fs in frame_scores) / len(frame_scores)

    if dtw_active and dtw_result and dtw_result.get("overall_dtw_score") is not None:
        dtw_sc   = dtw_result["overall_dtw_score"]
        combined = avg_score * 0.7 + dtw_sc * 0.3

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(f"{exercise_type} 횟수", f"{res['exercise_count']}회")
        m2.metric("평균 자세 점수",        f"{avg_score:.0%}")
        m3.metric("DTW 유사도",            f"{dtw_sc:.0%}")
        m4.metric("종합 점수 (70/30)",     f"{combined:.0%}")
        m5.metric("오류 프레임",           f"{len(error_frames)}개",
                  delta=f"{len(error_frames)}/{len(frame_scores)}", delta_color="inverse")

        phase_dtw  = dtw_result.get("phase_dtw_scores", {})
        phase_segs = dtw_result.get("phase_segment_counts", {})
        if phase_dtw:
            st.markdown("**페이즈별 DTW 유사도**")
            pcols = st.columns(len(phase_dtw))
            for col, (ph, sc) in zip(pcols, phase_dtw.items()):
                col.metric(f"{_PHASE_KO.get(ph, ph)} ({phase_segs.get(ph,0)}seg)", f"{sc:.0%}")

        # DTW 상세 분석
        llm_ctx       = dtw_result.get("llm_context", {})
        phase_detail  = llm_ctx.get("phase_details", {})
        overall_worst = llm_ctx.get("overall_worst_features", [])

        if phase_detail:
            st.divider()
            st.subheader("DTW 상세 분석")
            if overall_worst:
                st.info("🔍 전체 운동에서 가장 차이가 큰 관절: " + " · ".join(
                    f"**{_FEATURE_KO.get(w['name'], w['name'])}** (차이 {w['avg_diff']:.3f})"
                    for w in overall_worst
                ))
            tabs = st.tabs([_PHASE_KO.get(p, p) for p in phase_detail])
            for tab, (phase, detail) in zip(tabs, phase_detail.items()):
                with tab:
                    ca, cb, cc = st.columns(3)
                    ca.metric("DTW 유사도", f"{detail.get('dtw_score', 0):.0%}")
                    cb.metric("속도 패턴",  _SPEED_KO.get(detail.get("speed","normal"), ""))
                    bad = detail.get("bad_frame_ratio", 0.0)
                    cc.metric("문제 프레임 비율", f"{bad:.0%}",
                              delta="높음" if bad > 0.3 else "양호",
                              delta_color="inverse" if bad > 0.3 else "normal")
                    wf = detail.get("worst_features", [])
                    if wf:
                        st.markdown("**주요 문제 관절**")
                        st.dataframe(pd.DataFrame([{
                            "관절": _FEATURE_KO.get(w["name"], w["name"]),
                            "평균 차이": round(w["avg_diff"], 4),
                            "심각도": "🔴 높음" if w["avg_diff"] > 0.1 else "🟡 중간" if w["avg_diff"] > 0.05 else "🟢 낮음",
                        } for w in wf]), width='stretch', hide_index=True)
                    else:
                        st.success("이 구간에서 특이한 문제가 없습니다.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{exercise_type} 횟수", f"{res['exercise_count']}회")
        m2.metric("평균 자세 점수",        f"{avg_score:.0%}")
        m3.metric("오류 프레임",           f"{len(error_frames)}개",
                  delta=f"{len(error_frames)}/{len(frame_scores)}", delta_color="inverse")

    # 점수 차트
    st.divider()
    st.subheader("프레임별 자세 점수")
    chart_df = pd.DataFrame({
        "프레임": [fs["frame_idx"] for fs in frame_scores],
        "점수":   [fs["score"]     for fs in frame_scores],
        "Phase":  [fs["phase"]     for fs in frame_scores],
    })
    st.line_chart(chart_df, x="프레임", y="점수")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Phase 분포**")
        st.bar_chart(chart_df["Phase"].value_counts())
    with col2:
        st.markdown("**Phase별 평균 점수**")
        st.bar_chart(chart_df.groupby("Phase")["점수"].mean())

    # 오류 프레임
    if error_frames:
        st.divider()
        st.subheader("오류 프레임 상세")
        opts = [
            f"프레임 {ef['frame_idx']} [{ef['phase']}] ({ef['score']:.0%}) — {', '.join(ef['errors'][:2])}"
            for ef in error_frames
        ]
        sel = st.selectbox("오류 프레임 선택", range(len(opts)), format_func=lambda i: opts[i])
        ef  = error_frames[sel]
        ci, cf = st.columns(2)
        with ci:
            st.markdown("**스켈레톤 오버레이**")
            if ef["pts"]:
                skel = draw_skeleton_on_frame(ef["img_path"], ef["pts"])
                if skel is not None:
                    st.image(skel, width='stretch')
            else:
                st.info("키포인트 없음")
        with cf:
            st.markdown(f"**프레임** {ef['frame_idx']}  |  **Phase** `{ef['phase']}`  |  **점수** {ef['score']:.0%}")
            for err in ef["errors"]:
                st.error(err)
            if ef["details"]:
                st.markdown("**상세 수치**")
                for name, d in ef["details"].items():
                    icon = "✅" if d["status"]=="ok" else "⚠️" if d["status"]=="warning" else "❌"
                    st.markdown(f"{icon} **{name}**: {d['value']}  \n→ {d['feedback']}")

    # ── 프레임 브라우저 ────────────────────────────────────────
    st.divider()
    st.subheader("프레임 브라우저")

    frame_mapping = (dtw_result or {}).get("frame_mapping", {})
    frame_idx = st.slider("프레임 선택", 0, res["total_frames"]-1, 0, format="프레임 %d")
    selected  = kp_list[frame_idx]
    img_path  = selected["img_path"]
    pts       = selected["pts"]
    mapping   = frame_mapping.get(img_path)

    # 자세 정보 표시
    target = next((fs for fs in frame_scores if fs["frame_idx"] == frame_idx), None)
    if target:
        fb1, fb2, fb3 = st.columns(3)
        fb1.metric("Phase",     _PHASE_KO.get(target["phase"], target["phase"]))
        fb2.metric("자세 점수", f"{target['score']:.0%}")
        if mapping:
            fb3.metric("매핑 레퍼런스 프레임", f"#{mapping['ref_idx']}")
        if target["errors"]:
            for err in target["errors"]:
                st.warning(err)
        else:
            st.success("이 프레임에서 자세 오류가 없습니다.")
    else:
        st.info("이 프레임은 분석 대상 구간이 아닙니다.")

    st.markdown("---")

    # 이미지 비교
    if mapping:
        co, cs, cr = st.columns(3)
        with co:
            st.markdown("**사용자 원본**")
            img = cv2.imread(img_path)
            if img is not None:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width='stretch')
        with cs:
            st.markdown("**사용자 스켈레톤**")
            if pts:
                skel = draw_skeleton_on_frame(img_path, pts)
                if skel is not None:
                    st.image(skel, width='stretch')
            else:
                st.info("키포인트 없음")
        with cr:
            st.markdown(f"**레퍼런스 (#{mapping['ref_idx']})**")
            ref_img = cv2.imread(mapping["ref_img"])
            if ref_img is not None:
                st.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB), width='stretch')
            else:
                st.warning("레퍼런스 이미지 없음")
    else:
        co, cs = st.columns(2)
        with co:
            st.markdown("**원본 프레임**")
            img = cv2.imread(img_path)
            if img is not None:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width='stretch')
        with cs:
            st.markdown("**스켈레톤 오버레이**")
            if pts:
                skel = draw_skeleton_on_frame(img_path, pts)
                if skel is not None:
                    st.image(skel, width='stretch')
            else:
                st.info("키포인트 없음")

    if pts:
        with st.expander("키포인트 상세"):
            st.dataframe(pd.DataFrame([
                {"관절명": k, "x": v["x"], "y": v["y"], "confidence": v["vis"]}
                for k, v in pts.items()
            ]), width='stretch', hide_index=True)

    # ── 페이지 이동 ────────────────────────────────────────
    st.divider()
    col_fb, col_ag = st.columns(2)
    with col_fb:
        if st.button("🤖 AI 상세 피드백 받기", type="primary", use_container_width=True):
            st.switch_page("pages/feedback.py")
    with col_ag:
        if st.button("💬 AI Agent와 대화하기", use_container_width=True):
            st.switch_page("pages/agent.py")

    # JSON 다운로드 (numpy/raw 데이터 제외하고 직렬화 안전하게)
    st.divider()
    avg_val = sum(fs["score"] for fs in frame_scores) / len(frame_scores) if frame_scores else 0
    dtw_sc  = (dtw_result or {}).get("overall_dtw_score")
    export_dtw = None
    if dtw_result:
        export_dtw = {
            "overall_dtw_score":    dtw_result.get("overall_dtw_score"),
            "phase_dtw_scores":     dtw_result.get("phase_dtw_scores", {}),
            "phase_segment_counts": dtw_result.get("phase_segment_counts", {}),
            "llm_context":          dtw_result.get("llm_context", {}),
        }
    export = {
        "video": res["video_name"], "exercise_type": exercise_type,
        "exercise_count": res["exercise_count"],
        "avg_posture_score": round(avg_val, 4),
        "combined_score": round(avg_val*0.7 + dtw_sc*0.3, 4) if dtw_sc is not None else None,
        "dtw_result": export_dtw,
        "error_frame_count": len(error_frames),
        "phase_scores": [
            {"frame_idx": fs["frame_idx"], "phase": fs["phase"],
             "score": fs["score"], "errors": fs["errors"]}
            for fs in frame_scores
        ],
    }
    st.download_button(
        "분석 결과 JSON 다운로드",
        data=json.dumps(export, indent=2, ensure_ascii=False),
        file_name=f"{res['video_name']}_analysis.json",
        mime="application/json",
    )