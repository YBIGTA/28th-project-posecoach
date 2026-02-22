from __future__ import annotations

import shutil
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "preprocess" / "scripts"))

from video_preprocess import extract_frames  # type: ignore
from extract_yolo_frames import process_single_frame  # type: ignore
from utils.keypoints import load_pose_model
from utils.activity_segment import (
    apply_pullup_rule_first_filter,
    apply_pushup_rule_first_filter,
    detect_active_frame_indices,
    resolve_activity_model_path,
)
from ds_modules import (
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

from pathlib import Path
from utils.visualization import draw_skeleton_on_frame


PUSHUP_KO = "푸시업"
PULLUP_KO = "풀업"

GRIP_OVERHAND = "오버핸드"
GRIP_UNDERHAND = "언더핸드"
GRIP_WIDE = "와이드"

NO_SPOT_ERROR = "스포트 없음"

TARGET_RESOLUTION = (1920, 1080)
UPLOAD_VIDEO_DIR = ROOT / "data" / "uploads"
OUT_FRAMES_DIR = ROOT / "data" / "frames"

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}

def save_skeleton_overlay(img_path: Path, keypoints: Optional[dict], out_path: Path) -> bool:
    rgb = draw_skeleton_on_frame(img_path, keypoints)
    if rgb is None:
        return False
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), bgr)

def _local_path_to_static_url(p: str) -> Optional[str]:
    """
    OUT_FRAMES_DIR 하위에 저장된 어떤 이미지든 /static/frames/... URL로 바꿔준다.
    (원본 프레임이든, overlays든 다 동일 규칙)
    """
    try:
        rel = Path(p).resolve().relative_to(OUT_FRAMES_DIR.resolve())
        return f"/static/frames/{rel.as_posix()}"
    except Exception:
        return None

def _frame_path_to_url(frame_path: str) -> Optional[str]:
    try:
        rel = Path(frame_path).resolve().relative_to(OUT_FRAMES_DIR.resolve())
        return f"/static/frames/{rel.as_posix()}"
    except Exception:
        return None


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


@lru_cache(maxsize=1)
def get_pose_model():
    return load_pose_model()


def build_upload_path(original_filename: str) -> Path:
    UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_filename).suffix.lower()
    suffix = suffix if suffix in SUPPORTED_VIDEO_EXTENSIONS else ".mp4"
    stem = Path(original_filename).stem or "upload"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)
    uniq = int(time.time() * 1000)
    return UPLOAD_VIDEO_DIR / f"{safe_stem}_{uniq}{suffix}"


def _extract_reference_sequences(
    reference_video_path: Path,
    exercise_ko: str,
    extract_fps: int,
) -> dict[str, list[np.ndarray]]:
    """
    레퍼런스 영상에서 phase별 feature vector 시퀀스를 추출한다.
    DTWScorer.reference 형식과 동일하게 반환:
        { phase_name: [np.ndarray, ...] }
    """
    img_w, img_h = TARGET_RESOLUTION

    ref_frames_dir = OUT_FRAMES_DIR / (reference_video_path.stem + "_ref")
    if ref_frames_dir.exists():
        shutil.rmtree(ref_frames_dir)
    ref_frames_dir.mkdir(parents=True, exist_ok=True)

    extract_frames(reference_video_path, ref_frames_dir, extract_fps, TARGET_RESOLUTION)

    ref_frame_files = sorted(
        f for f in ref_frames_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not ref_frame_files:
        return {}

    pose_model = get_pose_model()
    ref_smoother = KeypointSmoother(window=3)
    ref_phase_detector = create_phase_detector(exercise_ko, fps=extract_fps)

    ref_sequences: dict[str, list[np.ndarray]] = {}

    for fpath in ref_frame_files:
        pts = process_single_frame(pose_model, fpath)
        flat = compute_virtual_keypoints(pts)
        smoothed = ref_smoother.smooth(flat)
        npts = normalize_pts(smoothed, img_w, img_h) if smoothed else None

        phase_metric = extract_phase_metric(npts, exercise_ko)
        phase = (
            ref_phase_detector.update(phase_metric)
            if phase_metric is not None
            else ref_phase_detector.phase
        )

        feat_vec = extract_feature_vector(npts, exercise_ko)
        if feat_vec is not None:
            ref_sequences.setdefault(phase, []).append(feat_vec)

    return ref_sequences


def run_video_analysis(
    video_path: Path,
    exercise_type: str,
    extract_fps: int,
    grip_type: Optional[str] = None,
    reference_video_path: Optional[Path] = None,  # ← 레퍼런스 영상 (DTW용)
) -> dict:
    exercise_en, exercise_ko = canonicalize_exercise_type(exercise_type)
    grip_ko = canonicalize_grip_type(grip_type) if exercise_en == "pullup" else None

    if extract_fps <= 0:
        raise ValueError("extract_fps는 1 이상이어야 합니다.")

    OUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_src_frames / src_fps if src_fps > 0 else 0
    cap.release()

    frames_dir = OUT_FRAMES_DIR / video_path.stem
    
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    overlays_dir = frames_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    extract_frames(video_path, frames_dir, extract_fps, TARGET_RESOLUTION)

    frame_files = sorted(
        f for f in frames_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    pose_model = get_pose_model()
    all_keypoints = []
    success_count = 0
    for i, fpath in enumerate(frame_files):
        pts = process_single_frame(pose_model, fpath)
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

    img_h, img_w = TARGET_RESOLUTION[1], TARGET_RESOLUTION[0]
    smoother = KeypointSmoother(window=3)
    phase_detector = create_phase_detector(exercise_ko, fps=extract_fps)

    # 1) Motion/ML filtering first.
    model_path = resolve_activity_model_path(exercise_en)
    selected_indices, filter_meta = detect_active_frame_indices(
        frame_files=frame_files,
        extract_fps=extract_fps,
        use_ml=True,
        model_path=model_path,
        min_keep_ratio=0.35,
        return_details=True,
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

    # 2) Build normalized sequence once.
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

    # 3) Exercise-specific rule-first refinement.
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

    if exercise_en == "pushup":
        counter = PushUpCounter(fps=extract_fps)
        evaluator = PushUpEvaluator()
        ref_name = "reference_pushup.json"
    else:
        counter = PullUpCounter(fps=extract_fps)
        evaluator = PullUpEvaluator(grip_type=grip_ko or GRIP_OVERHAND)
        ref_name = "reference_pullup.json"

    # ── DTWScorer 초기화 ──────────────────────────────────────
    # 우선순위: 레퍼런스 영상 > JSON fallback
    ref_json_path = ROOT / "ds_modules" / ref_name
    dtw_scorer = DTWScorer(str(ref_json_path), exercise_ko)

    if reference_video_path and reference_video_path.exists():
        # 레퍼런스 영상에서 직접 feature sequence 추출
        ref_sequences = _extract_reference_sequences(
            reference_video_path=reference_video_path,
            exercise_ko=exercise_ko,
            extract_fps=extract_fps,
        )
        if ref_sequences:
            # DTWScorer의 reference를 영상 기반으로 교체
            dtw_scorer.reference = ref_sequences
            dtw_scorer.active = True
            dtw_scorer._phase_scores = {}   # 누적 버퍼 초기화
            dtw_scorer._current_phase = None
            dtw_scorer._current_segment = []
        # ref_sequences가 비어있으면 JSON fallback 그대로 사용

    dtw_active = dtw_scorer.active
    # ─────────────────────────────────────────────────────────

    frame_scores = []
    error_frames = []

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

        overlay_path = overlays_dir / f"frame_{kp_data['frame_idx']:06d}_skeleton.jpg"
        rgb = draw_skeleton_on_frame(kp_data["img_path"], kp_data["pts"])
        
        skeleton_url = None
        if rgb is not None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overlay_path), bgr)
            skeleton_url = _local_path_to_static_url(str(overlay_path))
        
        frame_scores.append(
    {
        "frame_idx": kp_data["frame_idx"],
        "img_url": _local_path_to_static_url(kp_data["img_path"]),
        "skeleton_url": skeleton_url,  # ✅ 핵심
        "phase": current_phase,
        "score": eval_result["score"],
        "errors": eval_result["errors"],
        "details": eval_result["details"],
    }
)

        errors = eval_result.get("errors", [])
        if errors and errors != [NO_SPOT_ERROR]:
            error_frames.append(
                {
                    "frame_idx": kp_data["frame_idx"],
                    "img_path": kp_data["img_path"],
                    "img_url": _frame_path_to_url(kp_data["img_path"]),
                    "skeleton_url": skeleton_url,
                    "phase": current_phase,
                    "score": eval_result["score"],
                    "errors": errors,
                    "details": eval_result["details"],
                    "pts": pts,
                }
            )

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
        "duration": round(duration, 1),
        "fps": extract_fps,
        "keypoints": all_keypoints,
        "total_frames": len(frame_files),
        "analyzed_frame_count": len(selected_indices),
        "scored_frame_count": len(frame_scores),
        "filtered_out_count": max(0, len(frame_files) - len(selected_indices)),
        "filtering": filtering,
        "selected_frame_indices": sorted(selected_indices),
        "success_count": success_count,
        "resolution": list(TARGET_RESOLUTION),
        "dtw_result": dtw_result,
        "dtw_active": dtw_active,
    }