"""
Motion-segment frame filtering.
Uses ML inference when available, falls back to rule-based filtering.
"""
from pathlib import Path

import cv2
import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _ROOT / "data" / "models"
_DEFAULT_MODEL_IN_DATA = _ROOT / "data" / "models" / "activity_filter.pkl"
_DEFAULT_MODEL_IN_ROOT = _ROOT / "activity_filter.pkl"
DEFAULT_MODEL_PATH = _DEFAULT_MODEL_IN_DATA if _DEFAULT_MODEL_IN_DATA.exists() else _DEFAULT_MODEL_IN_ROOT

_PUSHUP_MODEL_CANDIDATES = (
    "activity_filter_pushup.pkl",
    "pushup_activity_filter.pkl",
)
_PULLUP_MODEL_CANDIDATES = (
    "activity_filter_pullup.pkl",
    "pullup_activity_filter.pkl",
)


def resolve_activity_model_path(exercise_tag=None):
    """
    Resolve exercise-specific activity-filter model path when available.
    Falls back to DEFAULT_MODEL_PATH if no matching file exists.
    """
    tag = str(exercise_tag or "").strip().lower()
    candidates = ()
    if tag == "pushup":
        candidates = _PUSHUP_MODEL_CANDIDATES
    elif tag == "pullup":
        candidates = _PULLUP_MODEL_CANDIDATES

    for name in candidates:
        path_in_model_dir = _MODEL_DIR / name
        if path_in_model_dir.exists():
            return path_in_model_dir
        path_in_root = _ROOT / name
        if path_in_root.exists():
            return path_in_root

    return DEFAULT_MODEL_PATH


def _safe_imread(path, flags):
    # Handle non-ASCII paths on Windows more robustly.
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, flags)
            if img is not None:
                return img
    except Exception:
        pass
    return cv2.imread(str(path), flags)


def _load_gray_small(img_path, size=(160, 90)):
    img = _safe_imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0)


def _fill_short_gaps(flags, max_gap):
    filled = flags[:]
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            i += 1
            continue
        start = i
        while i < n and not flags[i]:
            i += 1
        end = i - 1
        left_active = (start - 1) >= 0 and flags[start - 1]
        right_active = i < n and flags[i]
        if left_active and right_active and (end - start + 1) <= max_gap:
            for j in range(start, i):
                filled[j] = True
    return filled


def _active_segments_from_flags(flags):
    segments = []
    in_seg = False
    seg_start = 0
    for idx, active in enumerate(flags):
        if active and not in_seg:
            seg_start = idx
            in_seg = True
        if not active and in_seg:
            segments.append((seg_start, idx - 1))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(flags) - 1))
    return segments


def _merge_nearby_segments(segments, max_gap):
    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if (start - prev_end - 1) <= max_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _flags_to_selected_indices(
    flags,
    extract_fps,
    padding_seconds,
    min_segment_seconds,
    merge_gap_seconds=0.0,
):
    if not flags:
        return set()

    min_seg = max(1, int(round(extract_fps * min_segment_seconds)))
    pad = max(0, int(round(extract_fps * padding_seconds)))
    merge_gap = max(0, int(round(extract_fps * merge_gap_seconds)))

    segments = _active_segments_from_flags(flags)
    segments = [(start, end) for start, end in segments if (end - start + 1) >= min_seg]
    if merge_gap > 0:
        segments = _merge_nearby_segments(segments, max_gap=merge_gap)

    selected = set()
    n = len(flags)
    for start, end in segments:
        left = max(0, start - pad)
        right = min(n - 1, end + pad)
        selected.update(range(left, right + 1))

    return selected


def _extract_base_features(frame_files):
    n = len(frame_files)
    feats = np.zeros((n, 6), dtype=np.float32)
    if n == 0:
        return feats

    first_idx = -1
    prev = None
    for idx, fpath in enumerate(frame_files):
        prev = _load_gray_small(fpath)
        if prev is not None:
            first_idx = idx
            break

    if first_idx < 0 or prev is None:
        return feats

    prev_edges = cv2.Canny(prev, 40, 120)

    for idx in range(first_idx + 1, n):
        cur = _load_gray_small(frame_files[idx])
        if cur is None:
            continue

        diff = cv2.absdiff(cur, prev)
        mean_diff = float(np.mean(diff) / 255.0)
        std_diff = float(np.std(diff) / 255.0)
        change_ratio = float(np.mean(diff > 18))

        cur_edges = cv2.Canny(cur, 40, 120)
        edge_diff = cv2.absdiff(cur_edges, prev_edges)
        edge_change_ratio = float(np.mean(edge_diff > 0))

        intensity_std = float(np.std(cur) / 255.0)
        texture_var = float(min(cv2.Laplacian(cur, cv2.CV_64F).var() / 1000.0, 1.0))

        feats[idx] = np.array(
            [mean_diff, std_diff, change_ratio, edge_change_ratio, intensity_std, texture_var],
            dtype=np.float32,
        )
        prev = cur
        prev_edges = cur_edges

    return feats


def build_feature_matrix(frame_files, temporal_window=5):
    """Build shared feature matrix for training/inference."""
    base = _extract_base_features(frame_files)
    if base.shape[0] == 0:
        return base

    n, m = base.shape
    roll_mean = np.zeros((n, m), dtype=np.float32)
    roll_std = np.zeros((n, m), dtype=np.float32)
    delta = np.zeros((n, m), dtype=np.float32)
    for idx in range(n):
        start = max(0, idx - temporal_window + 1)
        window = base[start : idx + 1]
        roll_mean[idx] = np.mean(window, axis=0)
        roll_std[idx] = np.std(window, axis=0)
        if idx > 0:
            delta[idx] = base[idx] - base[idx - 1]

    return np.concatenate([base, roll_mean, roll_std, delta], axis=1)


def _rule_based_indices(
    frame_files,
    extract_fps,
    base_motion_threshold=0.01,
    percentile=60,
    quantile_scale=0.45,
    padding_seconds=2.0,
    gap_fill_seconds=2.0,
    min_segment_seconds=1.0,
    min_keep_ratio=0.08,
):
    if not frame_files:
        return set()

    feats = _extract_base_features(frame_files)
    motion_scores = feats[:, 0]

    valid_scores = motion_scores[1:]
    if valid_scores.size == 0:
        return set(range(len(frame_files)))

    dyn_threshold = float(np.percentile(valid_scores, percentile)) * quantile_scale
    threshold = max(base_motion_threshold, dyn_threshold)
    moving = [score >= threshold for score in motion_scores]
    moving = _fill_short_gaps(moving, max_gap=max(1, int(round(extract_fps * gap_fill_seconds))))

    selected = _flags_to_selected_indices(
        moving,
        extract_fps=extract_fps,
        padding_seconds=padding_seconds,
        min_segment_seconds=min_segment_seconds,
    )
    min_keep = max(1, int(np.ceil(len(frame_files) * min_keep_ratio)))
    if len(selected) < min_keep:
        return set(range(len(frame_files)))
    return selected


def _motion_activity_flags(
    feats,
    extract_fps,
    base_motion_threshold=0.01,
    percentile=65,
    quantile_scale=0.50,
    gap_fill_seconds=0.5,
):
    if feats.shape[0] == 0:
        return [], 0.0, np.asarray([], dtype=np.float32)

    motion_scores = feats[:, 0]
    valid_scores = motion_scores[1:]
    if valid_scores.size == 0:
        return [True] * len(motion_scores), 0.0, motion_scores

    dyn_threshold = float(np.percentile(valid_scores, percentile)) * quantile_scale
    threshold = max(base_motion_threshold, dyn_threshold)
    moving = [score >= threshold for score in motion_scores]
    moving = _fill_short_gaps(moving, max_gap=max(1, int(round(extract_fps * gap_fill_seconds))))
    return moving, threshold, motion_scores


def _smooth_probs(probs, window=5):
    if len(probs) == 0 or window <= 1:
        return probs
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(probs, kernel, mode="same")


def _safe_predict_proba(estimator, feats):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(feats)[:, 1]
    if hasattr(estimator, "decision_function"):
        logits = estimator.decision_function(feats)
        return 1.0 / (1.0 + np.exp(-logits))
    return estimator.predict(feats).astype(float)


def _ml_based_indices(
    frame_files,
    extract_fps,
    model_path,
    min_keep_ratio=0.08,
):
    try:
        import joblib
    except ImportError:
        return None, "joblib not installed"

    model_file = Path(model_path)
    if not model_file.exists():
        return None, f"model file missing: {model_file}"

    try:
        model_pkg = joblib.load(model_file)
    except Exception as e:
        return None, f"failed to load model: {e}"

    estimator = model_pkg.get("model", model_pkg) if isinstance(model_pkg, dict) else model_pkg
    on_threshold = float(model_pkg.get("on_threshold", 0.56)) if isinstance(model_pkg, dict) else 0.56
    off_threshold = float(model_pkg.get("off_threshold", 0.42)) if isinstance(model_pkg, dict) else 0.42
    smooth_window = int(model_pkg.get("smooth_window", 5)) if isinstance(model_pkg, dict) else 5
    padding_seconds = float(model_pkg.get("padding_seconds", 1.5)) if isinstance(model_pkg, dict) else 1.5
    min_segment_seconds = float(model_pkg.get("min_segment_seconds", 1.0)) if isinstance(model_pkg, dict) else 1.0
    max_padding_seconds = float(model_pkg.get("max_padding_seconds", 0.8)) if isinstance(model_pkg, dict) else 0.8
    ml_gap_fill_seconds = float(model_pkg.get("ml_gap_fill_seconds", 0.5)) if isinstance(model_pkg, dict) else 0.5
    merge_gap_seconds = float(model_pkg.get("merge_gap_seconds", 0.25)) if isinstance(model_pkg, dict) else 0.25
    rescue_prob_threshold = (
        float(model_pkg.get("rescue_prob_threshold", max(0.20, off_threshold - 0.03)))
        if isinstance(model_pkg, dict)
        else max(0.20, off_threshold - 0.03)
    )
    motion_percentile = float(model_pkg.get("motion_percentile", 65)) if isinstance(model_pkg, dict) else 65
    motion_quantile_scale = float(model_pkg.get("motion_quantile_scale", 0.50)) if isinstance(model_pkg, dict) else 0.50

    feats = build_feature_matrix(frame_files)
    if feats.shape[0] == 0:
        return set(), ""

    try:
        probs = _safe_predict_proba(estimator, feats)
    except Exception as e:
        return None, f"failed to run model: {e}"

    probs = _smooth_probs(np.asarray(probs, dtype=np.float32), window=max(1, smooth_window))

    ml_flags = []
    active = False
    for p in probs:
        if not active and p >= on_threshold:
            active = True
        elif active and p <= off_threshold:
            active = False
        ml_flags.append(active)

    motion_flags, _, _ = _motion_activity_flags(
        feats[:, :6],
        extract_fps=extract_fps,
        percentile=motion_percentile,
        quantile_scale=motion_quantile_scale,
        gap_fill_seconds=0.5,
    )
    flags = []
    for is_ml, is_motion, p in zip(ml_flags, motion_flags, probs):
        rescue = bool(is_motion and (p >= rescue_prob_threshold))
        flags.append(bool(is_ml or rescue))

    flags = _fill_short_gaps(flags, max_gap=max(1, int(round(extract_fps * ml_gap_fill_seconds))))
    core_selected = _flags_to_selected_indices(
        flags,
        extract_fps=extract_fps,
        padding_seconds=0.0,
        min_segment_seconds=min_segment_seconds,
        merge_gap_seconds=merge_gap_seconds,
    )
    padded_selected = _flags_to_selected_indices(
        flags,
        extract_fps=extract_fps,
        padding_seconds=min(padding_seconds, max_padding_seconds),
        min_segment_seconds=min_segment_seconds,
        merge_gap_seconds=merge_gap_seconds,
    )
    selected = set()
    for idx in padded_selected:
        if idx in core_selected:
            selected.add(idx)
            continue
        # Trim low-confidence/low-motion padding tails that often map to rest frames.
        if motion_flags[idx] or probs[idx] >= off_threshold:
            selected.add(idx)

    min_keep = max(1, int(np.ceil(len(frame_files) * min_keep_ratio)))
    if len(selected) < min_keep:
        return None, "selected frame ratio too small"

    coverage = len(selected) / float(len(frame_files))
    if coverage >= 0.90:
        p10, p90 = np.percentile(probs, [10, 90])
        contrast = float(p90 - p10)
        if contrast < 0.15:
            return None, f"ML low-contrast over-selection (coverage={coverage:.0%}, contrast={contrast:.3f})"

    return selected, ""


def apply_pullup_rule_first_filter(
    npts_sequence,
    ml_selected_indices,
    on_frames=2,
    off_frames=2,
    active_margin=0.03,
    rest_margin=0.12,
    min_keep_ratio=0.05,
):
    """
    Pull-up 전용 하이브리드 필터.
    1) 규칙(state machine)으로 명확한 active/rest를 우선 판정
    2) 규칙이 애매한 구간만 ML 선택 결과로 fallback
    """
    n = len(npts_sequence)
    if n == 0:
        return set(), {
            "rule_applied": True,
            "rule_active_frames": 0,
            "rule_rest_frames": 0,
            "ml_fallback_frames": 0,
            "reason": "no input frames",
        }

    ml_set = set(ml_selected_indices or set())
    selected = set()

    state = "unknown"
    on_streak = 0
    off_streak = 0
    rule_active_frames = 0
    rule_rest_frames = 0
    ml_fallback_frames = 0

    for idx, npts in enumerate(npts_sequence):
        active_signal = False
        rest_signal = False
        if npts is None:
            rest_signal = True
        else:
            try:
                wrist_y = (npts["Left Wrist"][1] + npts["Right Wrist"][1]) / 2.0
                shoulder_y = (npts["Left Shoulder"][1] + npts["Right Shoulder"][1]) / 2.0
                active_signal = wrist_y <= (shoulder_y + active_margin)
                rest_signal = wrist_y >= (shoulder_y + rest_margin)
            except Exception:
                # 키포인트가 불완전하면 규칙 판정을 보류하고 ML로 fallback
                active_signal = False
                rest_signal = False

        if active_signal:
            on_streak += 1
            off_streak = 0
        elif rest_signal:
            off_streak += 1
            on_streak = 0
        else:
            # 애매한 프레임에서 상태 흔들림을 줄이기 위해 완만히 감쇠
            on_streak = max(0, on_streak - 1)
            off_streak = max(0, off_streak - 1)

        if off_streak >= off_frames:
            state = "rest"
        elif on_streak >= on_frames:
            state = "active"

        if state == "active":
            selected.add(idx)
            rule_active_frames += 1
        elif state == "rest":
            rule_rest_frames += 1
        else:
            if idx in ml_set:
                selected.add(idx)
                ml_fallback_frames += 1

    min_keep = max(1, int(np.ceil(n * min_keep_ratio)))
    if len(selected) < min_keep:
        # 규칙이 과도하게 보수적인 경우 ML 결과로 복귀
        selected = set(ml_set)
        reason = "rule too strict; fell back to ml set"
    else:
        reason = ""

    details = {
        "rule_applied": True,
        "rule_active_frames": int(rule_active_frames),
        "rule_rest_frames": int(rule_rest_frames),
        "ml_fallback_frames": int(ml_fallback_frames),
        "reason": reason,
    }
    return selected, details


def _joint_angle(a, b, c):
    """Return angle ABC in degrees."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 180.0
    cos_val = float(np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def apply_pushup_rule_first_filter(
    npts_sequence,
    phase_sequence,
    ml_selected_indices,
    on_frames=2,
    off_frames=4,
    hold_frames=12,
    still_top_frames=12,
    motion_eps_deg=1.5,
    min_keep_ratio=0.05,
):
    """
    Push-up specific hybrid filter.
    1) Rule-based temporal state machine classifies active/rest first.
    2) ML selection is used only when rule state is still ambiguous.
    """
    n = len(npts_sequence)
    if n == 0:
        return set(), {
            "rule_applied": True,
            "rule_active_frames": 0,
            "rule_rest_frames": 0,
            "ml_fallback_frames": 0,
            "reason": "no input frames",
        }

    if not phase_sequence or len(phase_sequence) != n:
        phase_sequence = ["ready"] * n

    ml_set = set(ml_selected_indices or set())
    selected = set()

    state = "unknown"
    on_streak = 0
    off_streak = 0
    active_hold = 0
    still_top_streak = 0
    prev_elbow_angle = None

    rule_active_frames = 0
    rule_rest_frames = 0
    ml_fallback_frames = 0

    for idx, (npts, phase) in enumerate(zip(npts_sequence, phase_sequence)):
        active_signal = False
        rest_signal = False

        if npts is None:
            rest_signal = True
            prev_elbow_angle = None
            active_hold = max(0, active_hold - 1)
            still_top_streak = 0
        else:
            try:
                elbow_l = _joint_angle(npts["Left Shoulder"], npts["Left Elbow"], npts["Left Wrist"])
                elbow_r = _joint_angle(npts["Right Shoulder"], npts["Right Elbow"], npts["Right Wrist"])
                elbow_angle = (elbow_l + elbow_r) / 2.0
            except Exception:
                elbow_angle = prev_elbow_angle if prev_elbow_angle is not None else 180.0

            elbow_delta = abs(elbow_angle - prev_elbow_angle) if prev_elbow_angle is not None else 0.0
            prev_elbow_angle = elbow_angle

            moving_phase = phase in ("descending", "ascending", "bottom")
            if moving_phase:
                active_hold = hold_frames
            else:
                active_hold = max(0, active_hold - 1)

            if phase == "top" and elbow_delta <= motion_eps_deg:
                still_top_streak += 1
            else:
                still_top_streak = 0

            active_signal = moving_phase or (active_hold > 0 and phase != "ready")
            rest_signal = (phase == "ready") or (
                phase == "top" and still_top_streak >= still_top_frames and active_hold == 0
            )

        if active_signal and not rest_signal:
            on_streak += 1
            off_streak = 0
        elif rest_signal and not active_signal:
            off_streak += 1
            on_streak = 0
        else:
            on_streak = max(0, on_streak - 1)
            off_streak = max(0, off_streak - 1)

        if off_streak >= off_frames:
            state = "rest"
        elif on_streak >= on_frames:
            state = "active"

        if state == "active":
            selected.add(idx)
            rule_active_frames += 1
        elif state == "rest":
            rule_rest_frames += 1
        else:
            if idx in ml_set:
                selected.add(idx)
                ml_fallback_frames += 1

    min_keep = max(1, int(np.ceil(n * min_keep_ratio)))
    if len(selected) < min_keep:
        selected = set(ml_set)
        reason = "rule too strict; fell back to ml set"
    else:
        reason = ""

    details = {
        "rule_applied": True,
        "rule_active_frames": int(rule_active_frames),
        "rule_rest_frames": int(rule_rest_frames),
        "ml_fallback_frames": int(ml_fallback_frames),
        "reason": reason,
    }
    return selected, details


def detect_active_frame_indices(
    frame_files,
    extract_fps,
    use_ml=True,
    model_path=DEFAULT_MODEL_PATH,
    min_keep_ratio=0.08,
    return_details=False,
):
    """Return indices selected for downstream posture analysis."""
    if not frame_files:
        result = set()
        if return_details:
            return result, {"method": "rule", "reason": "no input frames"}
        return result

    details = {"method": "rule", "reason": ""}
    if use_ml:
        selected_ml, reason = _ml_based_indices(
            frame_files=frame_files,
            extract_fps=extract_fps,
            model_path=model_path,
            min_keep_ratio=min_keep_ratio,
        )
        if selected_ml is not None:
            details = {"method": "ml", "reason": ""}
            if return_details:
                return selected_ml, details
            return selected_ml
        details = {"method": "rule", "reason": reason}

    selected_rule = _rule_based_indices(
        frame_files=frame_files,
        extract_fps=extract_fps,
        min_keep_ratio=min_keep_ratio,
    )
    if return_details:
        return selected_rule, details
    return selected_rule
