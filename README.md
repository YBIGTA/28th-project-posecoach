---
title: PoseCoach
emoji: 🏋️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# PoseCoach - AI 기반 운동 자세 분석 시스템

YOLO Pose 모델과 DTW(Dynamic Time Warping) 알고리즘을 활용하여 운동 영상을 분석하고, 자세 교정 피드백을 제공하는 웹 애플리케이션입니다.

**Frontend**: React + TypeScript | **Backend**: FastAPI | **AI**: YOLO v8 Pose + Gemini

## 주요 기능

- **운동 자세 분석**: 푸시업, 풀업(오버핸드/언더핸드/와이드) 영상을 프레임 단위로 분석
- **실시간 키포인트 추출**: YOLO v8 Pose 모델로 20개 관절 좌표 검출
- **Phase 기반 평가**: 운동 구간(Phase)별 자세 점수 및 오류 피드백
- **DTW 유사도 비교**: 레퍼런스 영상과의 폼 유사도 분석 (선택)
- **AI 종합 피드백 (Gemini)**: 분석 결과를 기반으로 Gemini LLM이 전문 트레이너 수준의 자세 교정 피드백을 생성
- **운동 기록 관리**: SQLite 기반 유저별 운동 기록 저장 및 성장 추이 확인
- **PDF 리포트**: 분석 결과를 PDF로 다운로드

## 기술 스택

| 구분 | 기술 |
|------|------|
| Frontend | React 18 + TypeScript + Tailwind CSS |
| Backend | FastAPI + uvicorn |
| Pose Detection | Ultralytics YOLO v8 Pose |
| Motion Analysis | DTW (fastdtw), Phase Detection |
| LLM 피드백 | Google Gemini API |
| 데이터베이스 | SQLite |
| 인증 | bcrypt |

## 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/your-repo/28th-project-posecoach.git
cd 28th-project-posecoach
```

### 2. 백엔드 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 프론트엔드 의존성 설치

```bash
cd apps/web
npm install
```

### 4. 실행 (개발)

```bash
# 터미널 1 — 백엔드
uvicorn apps.api.main:app --reload --port 8000

# 터미널 2 — 프론트엔드
cd apps/web && npm run dev
```

### 5. Docker로 한 번에 실행

```bash
docker build -t posecoach .
docker run -p 7860:7860 posecoach
# → http://localhost:7860
```

## 프로젝트 구조

```
28th-project-posecoach/
├── Dockerfile               # HF Spaces / Docker 배포용
├── gemini_feedback.py       # Gemini API 기반 AI 피드백 생성
├── requirements.txt
│
├── apps/
│   ├── api/
│   │   ├── main.py          # FastAPI 앱 진입점 (라우터, 정적 파일 서빙)
│   │   ├── analysis.py      # 영상 분석 파이프라인
│   │   └── report_router.py # PDF 리포트 생성
│   └── web/                 # React 프론트엔드
│       └── src/
│           ├── pages/       # Home, UploadVideo, Result, MyPage 등
│           └── lib/api.ts   # API 클라이언트
│
├── db/
│   ├── database.py          # SQLite 연결, 스키마, CRUD
│   └── auth.py              # bcrypt 기반 유저 인증
│
├── ds_modules/              # AI 분석 모듈
│   ├── phase_detector.py    # 운동 Phase 검출
│   ├── exercise_counter.py  # 반복 횟수 카운팅
│   ├── posture_evaluator_phase.py  # 자세 평가 (Soft Scoring + Cohen's d)
│   └── dtw_scorer.py        # DTW 유사도 스코어링
│
├── utils/                   # 키포인트 로딩, 시각화
├── preprocess/              # 영상 전처리 스크립트
└── data/                    # 업로드 영상, 프레임, DB 저장
```

## 사용 방법

1. **회원가입/로그인** — 계정을 만들어 운동 기록을 관리하거나, 게스트로 시작
2. **운동 선택** — 푸시업 또는 풀업(그립 타입 선택) 지정
3. **영상 업로드** — MP4, MOV, AVI, WEBM 형식 지원
4. **분석 설정** — FPS 조절, 레퍼런스 영상 등록(DTW 비교용)
5. **결과 확인** — 등급(S/A/B/C), 횟수, Phase별 점수, 오류 프레임 리뷰
6. **AI 피드백** — Gemini AI가 전문 트레이너 수준의 종합 피드백 제공
7. **운동 기록** — 마이페이지에서 성장 추이 확인

## 등급 기준

| 등급 | 종합 점수 (avg×0.7 + DTW×0.3) |
|------|-------------------------------|
| S CLASS | 90% 이상 |
| A CLASS | 70% 이상 |
| B CLASS | 50% 이상 |
| C CLASS | 50% 미만 |
