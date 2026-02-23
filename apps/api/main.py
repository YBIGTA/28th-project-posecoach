from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from apps.api.analysis import (
    SUPPORTED_VIDEO_EXTENSIONS,
    build_upload_path,
    run_video_analysis,
)
from db.auth import login_user, register_user
from db.database import get_user_stats, get_user_workouts, init_db, save_workout
from gemini_feedback import generate_feedback
from apps.api.report_router import report_router

ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = ROOT / "apps" / "web" / "dist"

app = FastAPI(
    title="PoseCoach API",
    version="0.1.0",
    description="Backend API for auth, workout history, and pose analysis.",
)

app.include_router(report_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frames_dir = ROOT / "data" / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/frames", StaticFiles(directory=str(frames_dir)), name="frames")


class AuthRequest(BaseModel):
    username: str = Field(min_length=2)
    password: str = Field(min_length=4)


class SaveWorkoutRequest(BaseModel):
    analysis_results: dict


class GeminiFeedbackRequest(BaseModel):
    analysis_results: dict
    api_key: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=6000, ge=128, le=8192)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/auth/register")
def register(payload: AuthRequest) -> dict:
    ok, message = register_user(payload.username, payload.password)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    login_ok, _, user_id = login_user(payload.username, payload.password)
    return {
        "success": True,
        "message": message,
        "user_id": user_id if login_ok else None,
        "username": payload.username,
    }


@app.post("/auth/login")
def login(payload: AuthRequest) -> dict:
    ok, message, user_id = login_user(payload.username, payload.password)
    if not ok or user_id is None:
        raise HTTPException(status_code=401, detail=message)
    return {
        "success": True,
        "message": message,
        "user_id": user_id,
        "username": payload.username,
    }


@app.get("/workouts/{user_id}")
def workouts(user_id: int) -> dict:
    return {"workouts": get_user_workouts(user_id)}


@app.get("/workouts/{user_id}/stats")
def workout_stats(user_id: int) -> dict:
    return get_user_stats(user_id)


@app.post("/workouts/{user_id}")
def create_workout(user_id: int, payload: SaveWorkoutRequest) -> dict:
    workout_id = save_workout(user_id, payload.analysis_results)
    return {"workout_id": workout_id}


@app.post("/analysis/feedback")
def create_gemini_feedback(payload: GeminiFeedbackRequest) -> dict:
    try:
        feedback = generate_feedback(
            analysis_results=payload.analysis_results,
            api_key=payload.api_key,
            temperature=payload.temperature,
            max_output_tokens=payload.max_output_tokens,
        )
        return {"feedback": feedback}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"피드백 생성 중 오류가 발생했습니다: {e}") from e


@app.post("/analysis")
async def analyze_video(
    video: UploadFile = File(...),
    reference_video: Optional[UploadFile] = File(None),  # DTW용 레퍼런스 영상
    exercise_type: str = Form(...),
    extract_fps: int = Form(10),
    grip_type: Optional[str] = Form(None),
    save_result: bool = Form(False),
    user_id: Optional[int] = Form(None),
) -> dict:
    if extract_fps < 1 or extract_fps > 30:
        raise HTTPException(status_code=400, detail="extract_fps는 1~30 사이여야 합니다.")

    filename = video.filename or "upload.mp4"
    ext = Path(filename).suffix.lower()
    if ext and ext not in SUPPORTED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다: {ext}. 지원 형식: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}",
        )

    save_path = build_upload_path(filename)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ref_save_path: Optional[Path] = None

    try:
        # 사용자 영상 저장
        with save_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)

        # 레퍼런스 영상 저장 (있을 때만)
        if reference_video and reference_video.filename:
            ref_ext = Path(reference_video.filename).suffix.lower()
            if ref_ext and ref_ext not in SUPPORTED_VIDEO_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"레퍼런스 영상 형식 불지원: {ref_ext}")

            ref_save_path = build_upload_path(reference_video.filename)
            ref_save_path.parent.mkdir(parents=True, exist_ok=True)
            with ref_save_path.open("wb") as f:
                shutil.copyfileobj(reference_video.file, f)

        results = run_video_analysis(
            video_path=save_path,
            exercise_type=exercise_type,
            extract_fps=extract_fps,
            grip_type=grip_type,
            reference_video_path=ref_save_path,
        )

        workout_id = None
        if save_result:
            if user_id is None:
                raise HTTPException(status_code=400, detail="save_result=true일 때 user_id가 필요합니다.")
            workout_id = save_workout(user_id, results)

        return {"analysis_results": results, "saved_workout_id": workout_id}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {e}") from e
    finally:
        try:
            video.file.close()
        except Exception:
            pass
        if reference_video:
            try:
                reference_video.file.close()
            except Exception:
                pass


# ── React SPA 서빙 (HF Spaces / 프로덕션) ──────────────────────
# dist 폴더가 있을 때만 활성화 (로컬 개발 시에는 Vite dev server 사용)
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="spa-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> FileResponse:
        return FileResponse(str(DIST_DIR / "index.html"))