"""
활동 프레임(운동/비운동) 이진 분류기 학습 스크립트.
모델 후보를 비교하고 validation 기준으로 임계값을 자동 튜닝한다.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.activity_segment import build_feature_matrix


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _resolve_frame_labels(df):
    """
    labels.csv를 frame_path,label 형태로 정규화한다.
    """
    if "label" not in df.columns:
        raise ValueError("labels.csv에 label 컬럼이 필요합니다.")

    # 0/1만 유효 라벨로 사용하고, -1/NaN/기타 값은 자동 제외한다.
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("유효 라벨(0/1)이 없습니다. labels.csv를 확인하세요.")
    df["label"] = df["label"].astype(int)

    if "frame_path" in df.columns:
        labels = {}
        for _, row in df.iterrows():
            p = Path(str(row["frame_path"])).expanduser()
            p = p if p.is_absolute() else (Path.cwd() / p)
            labels[str(p.resolve())] = int(row["label"])
        return labels

    if {"video_dir", "frame_idx", "label"}.issubset(df.columns):
        labels = {}
        frame_cache = {}
        for _, row in df.iterrows():
            vdir = Path(str(row["video_dir"])).expanduser()
            vdir = vdir if vdir.is_absolute() else (Path.cwd() / vdir)
            vdir = vdir.resolve()

            key = str(vdir)
            if key not in frame_cache:
                frame_cache[key] = sorted(
                    f for f in vdir.iterdir() if f.suffix.lower() in IMAGE_EXTS
                )
            frames = frame_cache[key]
            idx = int(row["frame_idx"])
            if 0 <= idx < len(frames):
                labels[str(frames[idx].resolve())] = int(row["label"])
        return labels

    raise ValueError("labels.csv는 frame_path,label 또는 video_dir,frame_idx,label 형식이어야 합니다.")


def _build_dataset(frame_label_map):
    video_dirs = sorted({str(Path(p).parent) for p in frame_label_map.keys()})
    X, y, groups, orders = [], [], [], []

    for vdir_str in video_dirs:
        vdir = Path(vdir_str)
        frame_files = sorted(f for f in vdir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if not frame_files:
            continue

        feats = build_feature_matrix(frame_files)
        for idx, fpath in enumerate(frame_files):
            key = str(fpath.resolve())
            if key not in frame_label_map:
                continue
            X.append(feats[idx])
            y.append(int(frame_label_map[key]))
            groups.append(vdir_str)
            orders.append(idx)

    if not X:
        raise ValueError("학습 데이터가 비어 있습니다. labels.csv와 프레임 경로를 확인하세요.")
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        np.asarray(groups),
        np.asarray(orders, dtype=np.int32),
    )


def _safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        logits = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-logits))
    return model.predict(X).astype(float)


def _find_best_threshold(y_true, y_prob, target_recall=0.85):
    candidates = np.linspace(0.12, 0.88, 77)
    best = None
    for t in candidates:
        pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        pre = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        ba = balanced_accuracy_score(y_true, pred)

        meets = rec >= target_recall
        score = (1.5 * f1) + (0.5 * ba) + (0.2 * rec)
        row = {
            "threshold": float(t),
            "score": float(score),
            "meets_recall": bool(meets),
            "f1": float(f1),
            "precision": float(pre),
            "recall": float(rec),
            "balanced_acc": float(ba),
        }
        if best is None:
            best = row
            continue

        # 우선순위: target_recall 충족 > 높은 score > 높은 f1
        if row["meets_recall"] and not best["meets_recall"]:
            best = row
        elif row["meets_recall"] == best["meets_recall"]:
            if row["score"] > best["score"] + 1e-9:
                best = row
            elif abs(row["score"] - best["score"]) <= 1e-9 and row["f1"] > best["f1"]:
                best = row
    return best


def _split_indices(X, y, groups, orders, test_size, random_state):
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, valid_idx = next(splitter.split(X, y, groups=groups))
        return train_idx, valid_idx, "group"

    # 그룹이 1개면 시간 순 hold-out으로 과적합 검증을 한다.
    n = len(y)
    split = max(1, min(n - 1, int(round(n * (1.0 - test_size)))))
    sort_idx = np.argsort(orders)
    train_idx = sort_idx[:split]
    valid_idx = sort_idx[split:]

    # 한쪽 클래스만 나오는 경우 stratified random으로 폴백한다.
    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[valid_idx])) < 2:
        rng = np.random.default_rng(random_state)
        all_idx = np.arange(n)
        pos_idx = all_idx[y == 1]
        neg_idx = all_idx[y == 0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        p_split = max(1, int(round(len(pos_idx) * (1.0 - test_size))))
        n_split = max(1, int(round(len(neg_idx) * (1.0 - test_size))))
        train_idx = np.concatenate([pos_idx[:p_split], neg_idx[:n_split]])
        valid_idx = np.array(sorted(set(all_idx) - set(train_idx.tolist())))
        return train_idx, valid_idx, "stratified_random"

    return train_idx, valid_idx, "time_holdout"


def _candidate_models(random_state):
    return {
        "logreg": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1200, class_weight="balanced")),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
    }


def train_and_save(args):
    labels_df = pd.read_csv(args.labels)
    frame_label_map = _resolve_frame_labels(labels_df)
    X, y, groups, orders = _build_dataset(frame_label_map)

    if len(np.unique(y)) < 2:
        raise ValueError("label 클래스가 2개(0/1) 모두 필요합니다.")

    train_idx, valid_idx, split_mode = _split_indices(
        X=X,
        y=y,
        groups=groups,
        orders=orders,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]

    candidates = _candidate_models(args.random_state)
    records = []
    best_pack = None

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        valid_prob = _safe_predict_proba(model, X_valid)
        thr_pack = _find_best_threshold(y_valid, valid_prob, target_recall=args.target_recall)
        threshold = thr_pack["threshold"]
        valid_pred = (valid_prob >= threshold).astype(int)

        rec = {
            "name": name,
            "threshold": threshold,
            "f1": f1_score(y_valid, valid_pred, zero_division=0),
            "precision": precision_score(y_valid, valid_pred, zero_division=0),
            "recall": recall_score(y_valid, valid_pred, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y_valid, valid_pred),
            "acc": accuracy_score(y_valid, valid_pred),
            "score": thr_pack["score"],
            "meets_recall": thr_pack["meets_recall"],
        }
        try:
            rec["roc_auc"] = roc_auc_score(y_valid, valid_prob)
        except Exception:
            rec["roc_auc"] = np.nan
        records.append(rec)

        if best_pack is None:
            best_pack = (rec, model)
            continue
        prev = best_pack[0]
        if rec["meets_recall"] and not prev["meets_recall"]:
            best_pack = (rec, model)
        elif rec["meets_recall"] == prev["meets_recall"]:
            if rec["score"] > prev["score"] + 1e-9:
                best_pack = (rec, model)
            elif abs(rec["score"] - prev["score"]) <= 1e-9 and rec["f1"] > prev["f1"]:
                best_pack = (rec, model)

    best_rec, best_model = best_pack
    on_threshold = float(best_rec["threshold"])
    off_threshold = float(max(0.05, on_threshold - args.hysteresis_gap))

    # 선택된 모델을 전체 데이터로 재학습한다.
    final_model = clone(best_model)
    final_model.fit(X, y)

    print(f"=== Validation Summary (split={split_mode}) ===")
    for rec in sorted(records, key=lambda r: (-int(r['meets_recall']), -r["score"])):
        auc_text = "-" if np.isnan(rec["roc_auc"]) else f"{rec['roc_auc']:.4f}"
        print(
            f"[{rec['name']}] t={rec['threshold']:.3f} "
            f"f1={rec['f1']:.4f} p={rec['precision']:.4f} r={rec['recall']:.4f} "
            f"ba={rec['balanced_acc']:.4f} auc={auc_text}"
        )

    print("\n=== Best Model ===")
    print(
        f"name={best_rec['name']}, threshold={on_threshold:.3f}, "
        f"f1={best_rec['f1']:.4f}, precision={best_rec['precision']:.4f}, "
        f"recall={best_rec['recall']:.4f}, balanced_acc={best_rec['balanced_acc']:.4f}"
    )

    # best 모델의 validation 분류 리포트
    best_valid_prob = _safe_predict_proba(best_model, X_valid)
    best_valid_pred = (best_valid_prob >= on_threshold).astype(int)
    print("\n=== Validation Report (Best) ===")
    print(classification_report(y_valid, best_valid_pred, digits=4))

    model_pkg = {
        "model": final_model,
        "feature_version": "motion_v4",
        "selected_model": best_rec["name"],
        "on_threshold": on_threshold,
        "off_threshold": off_threshold,
        "smooth_window": args.smooth_window,
        "padding_seconds": args.padding_seconds,
        "min_segment_seconds": args.min_segment_seconds,
        "hysteresis_gap": args.hysteresis_gap,
        "validation": {
            "split_mode": split_mode,
            "f1": float(best_rec["f1"]),
            "precision": float(best_rec["precision"]),
            "recall": float(best_rec["recall"]),
            "balanced_acc": float(best_rec["balanced_acc"]),
            "acc": float(best_rec["acc"]),
            "threshold": float(on_threshold),
            "target_recall": float(args.target_recall),
        },
    }

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pkg, output_path)

    print(f"\n학습 샘플 수: {len(y)}")
    print(f"활동(1): {int(np.sum(y == 1))}, 비활동(0): {int(np.sum(y == 0))}")
    print(f"저장 완료: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="활동 프레임 필터 모델 학습")
    parser.add_argument("--labels", required=True, help="labels.csv 경로")
    parser.add_argument(
        "--output",
        default="data/models/activity_filter.pkl",
        help="학습 모델 저장 경로",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument("--hysteresis-gap", type=float, default=0.12)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--padding-seconds", type=float, default=1.8)
    parser.add_argument("--min-segment-seconds", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    train_and_save(parse_args())