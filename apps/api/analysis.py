from __future__ import annotations

import shutil
import sys
import time
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "preprocess" / "scripts"))

from video_preprocess import extract_frames  # type: ignore
from extract_yolo_frames import process_frame_batch  # type: ignore
from utils.keypoints import load_pose_model  # type: ignore
from utils.activity_segment import (  # type: ignore
    apply_pullup_rule_first_filter,
    apply_pushup_rule_first_filter,
    detect_active_frame_indices,
    resolve_activity_model_path,
)
from ds_modules import (  # type: ignore
    DTWScorer,
    KeypointSmoother,
    PullUpCounter,
    PullUpEvaluator,
    PushUpCounter,
    PushUpEvaluator,
    compute_virtual_keypoints,
    create_phase_detector,
    extract_feature_vector,
    extract_phase_metric,
    normalize_pts,
)
from utils.visualization import draw_skeleton_on_frame  # type: ignore


# --------------------
# constants
# --------------------
PUSHUP_KO = "푸시업"
PULLUP_KO = "풀업"

GRIP_OVERHAND = "오버핸드"
GRIP_UNDERHAND = "언더핸드"
GRIP_WIDE = "와이드"

NO_SPOT_ERROR = "스포트 없음"

ANALYSIS_RESOLUTION = (640, 360)
UPLOAD_VIDEO_DIR = ROOT / "data" / "uploads"
OUT_FRAMES_DIR = ROOT / "data" / "frames"

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


# --------------------
# helpers (urls, save)
# --------------------
def _local_path_to_static_url(p: str) -> Optional[str]:
    try:
        rel = Path(p).resolve().relative_to(OUT_FRAMES_DIR.resolve())
        return f"/static/frames/{rel.as_posix()}"
    except Exception:
        return None


def _frame_path_to_url(frame_path: str) -> Optional[str]:
    return _local_path_to_static_url(frame_path)


def save_skeleton_overlay(img_path: str, keypoints: Optional[dict], out_path: Path) -> Optional[str]:
    rgb = draw_skeleton_on_frame(img_path, keypoints)
    if rgb is None:
        return None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), bgr)
    return _local_path_to_static_url(str(out_path)) if ok else None


def save_skeleton_overlay_original_res(
    frame_bgr: np.ndarray,
    keypoints: Optional[dict],
    scale_x: float,
    scale_y: float,
    out_path: Path,
) -> Optional[str]:
    """원본 해상도 프레임 위에 스케일링된 키포인트로 스켈레톤을 그려 저장한다."""
    if frame_bgr is None:
        return None
    if keypoints is None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), frame_bgr)
        return _local_path_to_static_url(str(out_path)) if ok else None

    from utils.visualization import POSE_CONNECTIONS, JOINT_COLOR, CONNECTION_COLOR, JOINT_RADIUS, CONNECTION_THICKNESS, VIS_THRESHOLD

    img = frame_bgr.copy()

    # 연결선
    for joint_a, joint_b in POSE_CONNECTIONS:
        if joint_a not in keypoints or joint_b not in keypoints:
            continue
        pa, pb = keypoints[joint_a], keypoints[joint_b]
        if pa["vis"] < VIS_THRESHOLD or pb["vis"] < VIS_THRESHOLD:
            continue
        pt_a = (int(pa["x"] * scale_x), int(pa["y"] * scale_y))
        pt_b = (int(pb["x"] * scale_x), int(pb["y"] * scale_y))
        cv2.line(img, pt_a, pt_b, CONNECTION_COLOR, CONNECTION_THICKNESS)

    # 관절점
    for name, pt in keypoints.items():
        if pt["vis"] < VIS_THRESHOLD:
            continue
        center = (int(pt["x"] * scale_x), int(pt["y"] * scale_y))
        cv2.circle(img, center, JOINT_RADIUS, JOINT_COLOR, -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), img)
    return _local_path_to_static_url(str(out_path)) if ok else None


# --------------------
# canonicalize
# --------------------
def canonicalize_exercise_type(value: str) -> tuple[str, str]:
    normalized = (value or "").strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    push_aliases = {"pushup", "pushups", PUSHUP_KO}
    pull_aliases = {"pullup", "pullups", PULLUP_KO}

    if normalized in push_aliases:
        return "pushup", PUSHUP_KO
    if normalized in pull_aliases:
        return "pullup", PULLUP_KO
    raise ValueError("exercise_type은 pushup 또는 pullup이어야 합니다. (한글 라벨도 허용)")


def canonicalize_grip_type(value: Optional[str]) -> str:
    if not value:
        return GRIP_OVERHAND

    normalized = value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    mapping = {
        "overhand": GRIP_OVERHAND,
        "underhand": GRIP_UNDERHAND,
        "wide": GRIP_WIDE,
        GRIP_OVERHAND: GRIP_OVERHAND,
        GRIP_UNDERHAND: GRIP_UNDERHAND,
        GRIP_WIDE: GRIP_WIDE,
    }
    return mapping.get(normalized, GRIP_OVERHAND)


# --------------------
# model cache
# --------------------
@lru_cache(maxsize=1)
def get_pose_model():
    return load_pose_model()


# --------------------
# upload path
# --------------------
def build_upload_path(original_filename: str) -> Path:
    UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_filename).suffix.lower()
    suffix = suffix if suffix in SUPPORTED_VIDEO_EXTENSIONS else ".mp4"
    stem = Path(original_filename).stem or "upload"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)
    uniq = int(time.time() * 1000)
    return UPLOAD_VIDEO_DIR / f"{safe_stem}_{uniq}{suffix}"



def _generate_reference_json_realtime(
    reference_video_path: Path,
    exercise_ko: str,
    extract_fps: int,
    out_json_path: Path,
) -> bool:
    """
    ✅ 옵션 B: scripts/generate_reference.py를 실시간으로 실행해서
    reference JSON을 생성한다.
    """
    script_path = ROOT / "scripts" / "generate_reference.py"
    if not script_path.exists():
        # 스크립트 경로가 다르면 여기서 바로 실패함
        print(f"⚠ generate_reference.py 없음: {script_path}")
        return False

    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--video",
                str(reference_video_path),
                "--exercise",
                exercise_ko,          # ✅ scripts는 한글 choices=["푸시업","풀업"]
                "--output",
                str(out_json_path),
                "--fps",
                str(int(extract_fps)),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return out_json_path.exists() and out_json_path.stat().st_size > 10
    except subprocess.CalledProcessError as e:
        print("⚠ reference JSON 실시간 생성 실패")
        if e.stdout:
            print(e.stdout[-2000:])
        if e.stderr:
            print(e.stderr[-2000:])
        return False
    except Exception as e:
        print(f"⚠ reference JSON 실시간 생성 예외: {e}")
        return False


# --------------------
# main analysis
# --------------------
def run_video_analysis(
    video_path: Path,
    exercise_type: str,
    extract_fps: int,
    grip_type: Optional[str] = None,
    reference_video_path: Optional[Path] = None,  # DTW용 레퍼런스 영상
) -> dict:
    exercise_en, exercise_ko = canonicalize_exercise_type(exercise_type)
    grip_ko = canonicalize_grip_type(grip_type) if exercise_en == "pullup" else None

    if extract_fps <= 0:
        raise ValueError("extract_fps는 1 이상이어야 합니다.")

    OUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # --- 원본 비디오 메타 ---
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_src_frames / src_fps if src_fps and src_fps > 0 else 0.0
    cap.release()

    # --- 프레임 추출 경로 ---
    frames_dir = OUT_FRAMES_DIR / video_path.stem
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    overlays_dir = frames_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # --- 프레임 추출 (640×360 분석 해상도) ---
    extract_frames(video_path, frames_dir, extract_fps, ANALYSIS_RESOLUTION)
    frame_files = sorted(
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    # --- grayscale 프리로드 (activity filter 디스크 재읽기 제거) ---
    preloaded_grays: list = []
    for fpath in frame_files:
        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
            preloaded_grays.append(cv2.GaussianBlur(small, (5, 5), 0))
        else:
            preloaded_grays.append(None)

    # --- 키포인트 추출 (배치 YOLO 추론) ---
    pose_model = get_pose_model()
    batch_results = process_frame_batch(pose_model, frame_files, batch_size=8)

    all_keypoints: list[dict] = []
    success_count = 0

    for i, fpath in enumerate(frame_files):
        pts = batch_results[i]
        if pts is not None:
            success_count += 1
        all_keypoints.append(
            {
                "frame_idx": i,
                "img_key": fpath.name,
                "img_path": str(fpath),
                "img_url": _frame_path_to_url(str(fpath)),
                "pts": pts,
            }
        )

    img_w, img_h = ANALYSIS_RESOLUTION[0], ANALYSIS_RESOLUTION[1]
    smoother = KeypointSmoother(window=3)
    phase_detector = create_phase_detector(exercise_ko, fps=extract_fps)

    # --- 1) Motion/ML filtering ---
    model_path = resolve_activity_model_path(exercise_en)
    selected_indices, filter_meta = detect_active_frame_indices(
        frame_files=frame_files,
        extract_fps=extract_fps,
        use_ml=True,
        model_path=model_path,
        min_keep_ratio=0.35,
        return_details=True,
        preloaded_grays=preloaded_grays,
    )
    selected_indices = set(selected_indices)
    filtering = {
        "method": filter_meta.get("method", ""),
        "reason": filter_meta.get("reason", ""),
        "model_path": str(model_path),
    }

    if not selected_indices:
        selected_indices = set(range(len(frame_files)))
        filtering = {
            "method": "fallback_all",
            "reason": "활성 프레임이 선택되지 않아 전체 프레임을 사용합니다.",
            "model_path": str(model_path),
        }

    # --- 2) normalize + phase sequence ---
    npts_sequence: list[Optional[dict]] = []
    phase_sequence: list[str] = []

    for kp_data in all_keypoints:
        flat = compute_virtual_keypoints(kp_data["pts"])
        smoothed = smoother.smooth(flat)
        npts = normalize_pts(smoothed, img_w, img_h) if smoothed else None

        phase_metric = extract_phase_metric(npts, exercise_ko)
        current_phase = phase_detector.update(phase_metric) if phase_metric is not None else phase_detector.phase

        npts_sequence.append(npts)
        phase_sequence.append(current_phase)

    # --- 3) exercise-specific rule-first refinement ---
    if exercise_en == "pushup":
        rule_selected, rule_meta = apply_pushup_rule_first_filter(
            npts_sequence=npts_sequence,
            phase_sequence=phase_sequence,
            ml_selected_indices=selected_indices,
            min_keep_ratio=0.08,
        )
    else:
        rule_selected, rule_meta = apply_pullup_rule_first_filter(
            npts_sequence=npts_sequence,
            ml_selected_indices=selected_indices,
            min_keep_ratio=0.08,
        )

    if rule_selected:
        selected_indices = set(rule_selected)
        filtering = {
            "method": f"rule_first_{exercise_en}_ml_fallback",
            "reason": rule_meta.get("reason", ""),
            "model_path": str(model_path),
            "rule_active_frames": rule_meta.get("rule_active_frames", 0),
            "rule_rest_frames": rule_meta.get("rule_rest_frames", 0),
            "ml_fallback_frames": rule_meta.get("ml_fallback_frames", 0),
        }

    if not selected_indices:
        selected_indices = set(range(len(all_keypoints)))
        filtering = {
            "method": "none",
            "reason": "필터링 결과가 없어 모든 프레임을 사용했습니다.",
            "model_path": str(model_path),
        }

    for kp in all_keypoints:
        kp["selected_for_analysis"] = kp["frame_idx"] in selected_indices

    # --- counter/evaluator + dtw reference ---
    if exercise_en == "pushup":
        counter = PushUpCounter(fps=extract_fps)
        evaluator = PushUpEvaluator()
        ref_name = "reference_pushup.json"
    else:
        counter = PullUpCounter(fps=extract_fps)
        evaluator = PullUpEvaluator(grip_type=grip_ko or GRIP_OVERHAND)
        ref_name = "reference_pullup.json"

    # --- DTW 레퍼런스 경로 결정 ---
    default_ref_json_path = ROOT / "ds_modules" / ref_name

    # ✅ 실시간 reference_json 생성 (요청 전용 파일)
    runtime_ref_json_path = frames_dir / f"reference_runtime_{exercise_en}.json"
    use_ref_json_path = default_ref_json_path

    if reference_video_path and reference_video_path.exists():
        ok = _generate_reference_json_realtime(
            reference_video_path=reference_video_path,
            exercise_ko=exercise_ko,
            extract_fps=extract_fps,
            out_json_path=runtime_ref_json_path,
        )
        if ok:
            use_ref_json_path = runtime_ref_json_path
            print(f"🔥 DTW runtime reference 사용: {use_ref_json_path}")
        else:
            print("⚠ runtime reference 생성 실패 → 기본 reference JSON fallback")

    # --- DTW scorer init ---
    dtw_scorer = DTWScorer(str(use_ref_json_path), exercise_ko)
    dtw_active = bool(getattr(dtw_scorer, "active", False))

    # --- scoring loop (오버레이 없이 점수만 계산) ---
    frame_scores: list[dict] = []
    error_frames_pending: list[dict] = []  # 오버레이 생성 대기

    for i, kp_data in enumerate(all_keypoints):
        pts = kp_data["pts"]
        npts = npts_sequence[i]
        current_phase = phase_sequence[i]

        was_active = counter.is_active
        counter.update(npts, current_phase)
        is_analysis_active = was_active or counter.is_active

        if (not is_analysis_active) or (i not in selected_indices):
            continue

        eval_result = evaluator.evaluate(npts, phase=current_phase)

        if dtw_active:
            feat_vec = extract_feature_vector(npts, exercise_ko)
            dtw_scorer.accumulate(feat_vec, current_phase)

        errors = eval_result.get("errors", []) or []
        is_error = errors and errors != [NO_SPOT_ERROR]

        frame_scores.append(
            {
                "frame_idx": kp_data["frame_idx"],
                "img_url": _local_path_to_static_url(kp_data["img_path"]),
                "skeleton_url": None,  # 에러 프레임만 나중에 채움
                "phase": current_phase,
                "score": eval_result.get("score", 0.0),
                "errors": errors,
                "details": eval_result.get("details", None),
            }
        )

        if is_error:
            error_frames_pending.append(
                {
                    "frame_idx": kp_data["frame_idx"],
                    "img_path": kp_data["img_path"],
                    "img_url": _frame_path_to_url(kp_data["img_path"]),
                    "phase": current_phase,
                    "score": eval_result.get("score", 0.0),
                    "errors": errors,
                    "details": eval_result.get("details", None),
                    "pts": pts,
                    "score_list_idx": len(frame_scores) - 1,
                }
            )

    # --- 에러 프레임만 원본 해상도로 스켈레톤 오버레이 생성 ---
    scale_x = src_w / img_w
    scale_y = src_h / img_h
    error_frames: list[dict] = []

    if error_frames_pending:
        # 에러 프레임의 원본 비디오 타임스탬프 계산 및 seek
        error_frame_indices = {ef["frame_idx"] for ef in error_frames_pending}
        # 프레임 인덱스 → 원본 비디오의 프레임 번호 매핑
        frame_interval = src_fps / extract_fps if src_fps > 0 else 1.0

        cap = cv2.VideoCapture(str(video_path))
        orig_frames: dict[int, np.ndarray] = {}

        for ef in sorted(error_frames_pending, key=lambda x: x["frame_idx"]):
            fidx = ef["frame_idx"]
            src_frame_num = int(fidx * frame_interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_frame_num)
            ret, frame = cap.read()
            if ret:
                orig_frames[fidx] = frame
        cap.release()

        for ef in error_frames_pending:
            fidx = ef["frame_idx"]
            orig_frame = orig_frames.get(fidx)
            overlay_path = overlays_dir / f"frame_{fidx:06d}_skeleton.jpg"
            skeleton_url = save_skeleton_overlay_original_res(
                orig_frame, ef["pts"], scale_x, scale_y, overlay_path,
            )
            # frame_scores에도 skeleton_url 반영
            frame_scores[ef["score_list_idx"]]["skeleton_url"] = skeleton_url

            error_frames.append(
                {
                    "frame_idx": fidx,
                    "img_path": ef["img_path"],
                    "img_url": ef["img_url"],
                    "skeleton_url": skeleton_url,
                    "phase": ef["phase"],
                    "score": ef["score"],
                    "errors": ef["errors"],
                    "details": ef["details"],
                    "pts": ef["pts"],
                }
            )

    # --- finalize rep if active ---
    if counter.is_active:
        if len(counter.visited_phases & counter.required_sequence) >= counter.min_required:
            counter.count += 1
        counter.is_active = False

    dtw_result = dtw_scorer.finalize() if dtw_active else None

    return {
        "video_name": video_path.stem,
        "exercise_type": exercise_ko,
        "exercise_type_en": exercise_en,
        "grip_type": grip_ko,
        "exercise_count": counter.count,
        "frame_scores": frame_scores,
        "error_frames": error_frames,
        "duration": round(float(duration), 1),
        "fps": int(extract_fps),
        "keypoints": all_keypoints,
        "total_frames": len(frame_files),
        "analyzed_frame_count": len(selected_indices),
        "scored_frame_count": len(frame_scores),
        "filtered_out_count": max(0, len(frame_files) - len(selected_indices)),
        "filtering": filtering,
        "selected_frame_indices": sorted(selected_indices),
        "success_count": success_count,
        "resolution": list(ANALYSIS_RESOLUTION),
        "original_resolution": [src_w, src_h],
        "dtw_result": dtw_result,
        "dtw_active": bool(dtw_active),
        # 디버깅용: 실제 사용된 레퍼런스 JSON 경로
        "dtw_reference_json": str(use_ref_json_path),
    }