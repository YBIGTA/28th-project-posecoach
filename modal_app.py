import modal

# ── 1. 이미지 정의 ─────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    # 시스템 패키지
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "ffmpeg",
        "curl",
    )
    # Node.js 20 (프론트엔드 빌드용)
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y --no-install-recommends nodejs",
    )
    # Python 패키지
    .pip_install(
        "fastapi[standard]",
        "uvicorn",
        "ultralytics",
        "opencv-python-headless",
        "numpy",
        "pandas",
        "scipy",
        "fastdtw",
        "python-multipart",
        "bcrypt",
        "plotly",
        "python-dotenv",
        "requests",
        "reportlab>=4.0",
        "aiofiles",
        "joblib",
        "scikit-learn",
        "pydantic",
        "jinja2",
        "google-generativeai",
    )
    # reportlab 설치 확인 (캐시 무효화 + 보장)
    .run_commands("python -c 'import reportlab; print(reportlab.__version__)'")
    # ── 프론트엔드 빌드 (copy=True 필요: 빌드 스텝 전에 파일이 있어야 함) ──
    .add_local_file("./apps/web/package.json", remote_path="/root/apps/web/package.json", copy=True)
    .add_local_file("./apps/web/package-lock.json", remote_path="/root/apps/web/package-lock.json", copy=True)
    .add_local_file("./apps/web/tsconfig.json", remote_path="/root/apps/web/tsconfig.json", copy=True)
    .add_local_file("./apps/web/vite.config.ts", remote_path="/root/apps/web/vite.config.ts", copy=True)
    .add_local_file("./apps/web/index.html", remote_path="/root/apps/web/index.html", copy=True)
    .add_local_dir("./apps/web/src", remote_path="/root/apps/web/src", copy=True)
    .run_commands(
        "cd /root/apps/web && npm ci && VITE_API_BASE_URL='' npm run build",
    )
    # 주의: /root/data 에 mkdir 하면 안 됨 (볼륨 마운트 충돌)
    # ── 런타임 파일 (마운트 - 마지막에 배치, copy 불필요) ──
    .add_local_dir("./apps/api", remote_path="/root/apps/api")
    .add_local_dir("./db", remote_path="/root/db")
    .add_local_dir("./ds_modules", remote_path="/root/ds_modules")
    .add_local_dir("./utils", remote_path="/root/utils")
    .add_local_dir("./preprocess", remote_path="/root/preprocess")
    .add_local_dir("./scripts", remote_path="/root/scripts")
    .add_local_dir("./apps/assets", remote_path="/root/assets")
    .add_local_file("./yolo26n-pose.pt", remote_path="/root/yolo26n-pose.pt")
    .add_local_file("./activity_filter.pkl", remote_path="/root/activity_filter.pkl")
    .add_local_file("./gemini_feedback.py", remote_path="/root/gemini_feedback.py")
)

# ── 2. 앱 & 볼륨 ──────────────────────────────────────────────
app = modal.App("posecoach-app")
volume = modal.Volume.from_name("posecoach-db-volume", create_if_missing=True)


# ── 3. FastAPI 서빙 ───────────────────────────────────────────
@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/data": volume},
    timeout=600,
    scaledown_window=300,
    env={
        "PYTHONPATH": "/root:/root/apps/api:/root/preprocess/scripts:/root/utils",
    },
)
@modal.asgi_app()
def fastapi_app():
    import os
    os.chdir("/root")
    # 볼륨 위에 필요한 하위 디렉터리 생성 (이미지 빌드가 아닌 런타임에서)
    os.makedirs("/root/data/uploads", exist_ok=True)
    os.makedirs("/root/data/frames", exist_ok=True)
    os.makedirs("/root/data/models", exist_ok=True)
    from apps.api.main import app as web
    return web
