"""
AIFit - SQLite 데이터베이스 관리
테이블 생성, 운동 기록 CRUD
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "posecoach.db"


def get_connection() -> sqlite3.Connection:
    """SQLite 연결 반환 (WAL 모드)"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """테이블 생성 (IF NOT EXISTS)"""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at    TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS workouts (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL REFERENCES users(id),
            created_at       TEXT NOT NULL DEFAULT (datetime('now','localtime')),
            video_name       TEXT NOT NULL,
            exercise_type    TEXT NOT NULL,
            grip_type        TEXT,
            exercise_count   INTEGER NOT NULL DEFAULT 0,
            duration         REAL NOT NULL DEFAULT 0,
            fps              INTEGER NOT NULL,
            total_frames     INTEGER NOT NULL,
            avg_score        REAL NOT NULL,
            grade            TEXT NOT NULL,
            dtw_active       INTEGER NOT NULL DEFAULT 0,
            dtw_score        REAL,
            combined_score   REAL,
            error_frame_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS workout_errors (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            workout_id INTEGER NOT NULL REFERENCES workouts(id) ON DELETE CASCADE,
            error_msg  TEXT NOT NULL,
            count      INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS workout_phase_scores (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            workout_id  INTEGER NOT NULL REFERENCES workouts(id) ON DELETE CASCADE,
            phase       TEXT NOT NULL,
            avg_score   REAL NOT NULL,
            frame_count INTEGER NOT NULL DEFAULT 0
        );
    """)
    conn.commit()
    conn.close()


def save_workout(user_id: int, analysis_results: dict) -> int:
    """분석 결과를 DB에 저장하고 workout id 반환"""
    res = analysis_results
    frame_scores = res.get("frame_scores", [])

    # 평균 점수
    scores = [fs["score"] for fs in frame_scores]
    avg_score = sum(scores) / len(scores) if scores else 0

    # 등급
    if avg_score >= 0.9:
        grade = "S CLASS"
    elif avg_score >= 0.7:
        grade = "A CLASS"
    elif avg_score >= 0.5:
        grade = "B CLASS"
    else:
        grade = "C CLASS"

    # DTW
    dtw_result = res.get("dtw_result")
    dtw_active = res.get("dtw_active", False)
    dtw_score = None
    combined_score = None
    if dtw_active and dtw_result and dtw_result.get("overall_dtw_score") is not None:
        dtw_score = dtw_result["overall_dtw_score"]
        combined_score = avg_score * 0.7 + dtw_score * 0.3

    # 오류 집계
    error_frames = res.get("error_frames", [])
    error_counter: dict[str, int] = {}
    for ef in error_frames:
        for msg in ef.get("errors", []):
            error_counter[msg] = error_counter.get(msg, 0) + 1

    # Phase별 점수 집계
    phase_data: dict[str, dict] = {}
    for fs in frame_scores:
        phase = fs["phase"]
        if phase not in phase_data:
            phase_data[phase] = {"total_score": 0.0, "count": 0}
        phase_data[phase]["total_score"] += fs["score"]
        phase_data[phase]["count"] += 1

    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO workouts
               (user_id, video_name, exercise_type, grip_type,
                exercise_count, duration, fps, total_frames,
                avg_score, grade, dtw_active, dtw_score,
                combined_score, error_frame_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                res.get("video_name", ""),
                res.get("exercise_type", ""),
                res.get("grip_type"),
                res.get("exercise_count", 0),
                res.get("duration", 0),
                res.get("fps", 0),
                res.get("total_frames", 0),
                round(avg_score, 4),
                grade,
                1 if dtw_active else 0,
                round(dtw_score, 4) if dtw_score is not None else None,
                round(combined_score, 4) if combined_score is not None else None,
                len(error_frames),
            ),
        )
        workout_id = cur.lastrowid

        # 오류 저장
        for msg, cnt in error_counter.items():
            conn.execute(
                "INSERT INTO workout_errors (workout_id, error_msg, count) VALUES (?, ?, ?)",
                (workout_id, msg, cnt),
            )

        # Phase별 점수 저장
        for phase, data in phase_data.items():
            conn.execute(
                "INSERT INTO workout_phase_scores (workout_id, phase, avg_score, frame_count) VALUES (?, ?, ?, ?)",
                (workout_id, phase, round(data["total_score"] / data["count"], 4), data["count"]),
            )

        conn.commit()
        return workout_id
    finally:
        conn.close()


def get_user_workouts(user_id: int) -> list[dict]:
    """유저의 운동 기록 목록 반환 (최신순)"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM workouts WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()

    workouts = []
    for row in rows:
        w = dict(row)
        # 오류 목록
        w["errors"] = [
            dict(r)
            for r in conn.execute(
                "SELECT error_msg, count FROM workout_errors WHERE workout_id = ?",
                (w["id"],),
            ).fetchall()
        ]
        # Phase 점수
        w["phase_scores"] = [
            dict(r)
            for r in conn.execute(
                "SELECT phase, avg_score, frame_count FROM workout_phase_scores WHERE workout_id = ?",
                (w["id"],),
            ).fetchall()
        ]
        workouts.append(w)

    conn.close()
    return workouts


def get_user_stats(user_id: int) -> dict:
    """유저의 종합 통계 반환"""
    conn = get_connection()

    row = conn.execute(
        """SELECT
               COUNT(*) as total_workouts,
               COALESCE(SUM(duration), 0) as total_duration,
               COALESCE(AVG(avg_score), 0) as overall_avg_score,
               COALESCE(SUM(exercise_count), 0) as total_reps
           FROM workouts WHERE user_id = ?""",
        (user_id,),
    ).fetchone()

    # 최다 운동 종류
    fav_row = conn.execute(
        """SELECT exercise_type, COUNT(*) as cnt
           FROM workouts WHERE user_id = ?
           GROUP BY exercise_type ORDER BY cnt DESC LIMIT 1""",
        (user_id,),
    ).fetchone()

    conn.close()

    return {
        "total_workouts": row["total_workouts"],
        "total_duration": round(row["total_duration"], 1),
        "overall_avg_score": round(row["overall_avg_score"], 4),
        "total_reps": row["total_reps"],
        "favorite_exercise": fav_row["exercise_type"] if fav_row else "-",
    }
