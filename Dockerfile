# ── Stage 1: React 빌드 ──────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /build/web
COPY apps/web/package*.json ./
RUN npm ci

COPY apps/web/ ./
# 빌드 시 API는 같은 서버(상대경로)에서 서빙
ENV VITE_API_BASE_URL=""
RUN npm run build


# ── Stage 2: Python 앱 ───────────────────────────────────────
FROM python:3.11-slim

# OpenCV / ffmpeg 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 의존성 먼저 (캐시 활용)
COPY requirements.txt ./
# 서버 환경: GUI 불필요한 headless 버전 사용
RUN pip install --no-cache-dir \
        opencv-python-headless \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        google-generativeai \
        pydantic \
        "reportlab>=4.0" \
        aiofiles \
        joblib \
        scikit-learn

# 소스 복사
COPY . .

# React 빌드 결과물 복사
COPY --from=frontend-builder /build/web/dist ./apps/web/dist

# 데이터 디렉터리 생성 (HF Spaces persistent storage 마운트 위치)
RUN mkdir -p /app/data/uploads /app/data/frames /app/data

# HF Spaces는 7860 포트 사용
EXPOSE 7860

CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "7860"]