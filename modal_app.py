import modal
import os

# 1. 서버 환경 정의 (필요한 라이브러리 전부 추가)
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "fastapi[standard]", 
        "ultralytics", 
        "opencv-python", 
        "pandas", 
        "python-multipart",
        "jinja2",
        "bcrypt",      # <-- 로그인 에러 해결을 위해 추가!
        "pybase64"     # 혹시 모를 이미지 처리용 추가
    )
    # 모든 폴더를 서버로 복사
    .add_local_dir("./apps", remote_path="/root/apps")
    .add_local_dir("./db", remote_path="/root/db")
    .add_local_dir("./ds_modules", remote_path="/root/ds_modules")
    .add_local_dir("./utils", remote_path="/root/utils")
    .add_local_dir("./preprocess", remote_path="/root/preprocess")
    # 모델 및 필수 파일 복사
    .add_local_file("./yolo26n-pose.pt", remote_path="/root/yolo26n-pose.pt")
    .add_local_file("./activity_filter.pkl", remote_path="/root/activity_filter.pkl")
    .add_local_file("./gemini_feedback.py", remote_path="/root/gemini_feedback.py")
)

app = modal.App("posecoach-app")
volume = modal.Volume.from_name("posecoach-db-volume", create_if_missing=True)

# 2. 백엔드 실행 함수
@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/data": volume},
    timeout=600,
    # 파이썬이 모든 폴더를 인식할 수 있게 길을 터줍니다 (PYTHONPATH)
    env={
        "PYTHONPATH": "/root:/root/apps/api:/root/preprocess/scripts:/root/utils"
    }
)
@modal.asgi_app()
def fastapi_app():
    from apps.api.main import app as backend_app
    return backend_app