---
title: PoseCoach
emoji: 🏋️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# PoseCoach - AI 운동 자세 분석 시스템

YOLO Pose + DTW 알고리즘 기반 운동 영상 자세 분석 웹 애플리케이션 (React + FastAPI)

An AI-powered exercise form analysis web app built with React and FastAPI, using YOLO Pose and DTW algorithms.

---

## 주요 기능 / Features

- **운동 자세 분석** — 푸시업, 풀업(오버핸드/언더핸드/와이드) 영상을 프레임 단위로 분석
- **실시간 키포인트 추출** — YOLO Pose 모델로 17개 관절 좌표 검출
- **Phase 기반 평가** — 운동 구간(Phase)별 자세 점수 및 오류 피드백
- **DTW 유사도 비교** — 레퍼런스 영상과의 폼 유사도 분석
- **AI 종합 피드백 (Gemini)** — Gemini LLM 기반 전문 트레이너 수준의 자세 교정 피드백
- **PDF 리포트 내보내기** — 분석 결과 + AI 피드백을 PDF로 다운로드
- **운동 기록 관리** — SQLite 기반 유저별 운동 기록 저장 및 성장 추이 확인
- **프레임 네비게이터** — 오류 프레임을 스켈레톤 오버레이와 함께 리뷰

## 기술 스택 / Tech Stack

| 구분 | 기술 |
|------|------|
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui |
| Backend | FastAPI, Uvicorn |
| Pose Detection | Ultralytics YOLO Pose |
| Motion Analysis | DTW (fastdtw), Phase Detection |
| LLM 피드백 | Google Gemini API |
| PDF 리포트 | ReportLab |
| 데이터베이스 | SQLite |
| 인증 | bcrypt |

## 배포 환경 / Deployment

| 구분 | 내용 |
|------|------|
| 메인 배포 | Modal.com (서버리스 GPU 클라우드) |
| GPU | NVIDIA T4 |
| 런타임 | Python 3.11 (Debian Slim) |
| 웹서버 | FastAPI + Uvicorn |
| 스토리지 | Modal Volume (영구 저장) |

## 프로젝트 구조 / Project Structure

```
28th-project-posecoach/
├── apps/
│   ├── api/                    # FastAPI 백엔드
│   │   ├── main.py             # API 진입점 (라우팅, CORS)
│   │   ├── analysis.py         # 영상 분석 파이프라인
│   │   └── report_router.py    # PDF 리포트 생성 엔드포인트
│   ├── assets/                 # 폰트 파일 (NotoSansKR)
│   └── web/                    # React 프론트엔드
│       ├── src/
│       │   ├── pages/          # Home, Login, SelectExercise, SelectGrip, UploadVideo, Result, MyPage
│       │   ├── components/     # UI 컴포넌트 (shadcn/ui)
│       │   └── lib/            # API 클라이언트, 인증 유틸
│       └── vite.config.ts
│
├── ds_modules/                 # 분석 모듈
│   ├── angle_utils.py          # 관절 각도 계산
│   ├── coord_filter.py         # 키포인트 스무딩
│   ├── phase_detector.py       # 운동 Phase 검출
│   ├── exercise_counter.py     # 반복 횟수 카운팅
│   ├── posture_evaluator_phase.py  # 자세 평가
│   ├── compute_cohens_d.py     # 통계 유틸
│   ├── dtw_scorer.py           # DTW 유사도 스코어링
│   ├── weights_pushup.json     # 푸시업 평가 가중치
│   └── weights_pullup.json     # 풀업 평가 가중치
│
├── db/
│   ├── database.py             # SQLite 스키마, CRUD
│   └── auth.py                 # bcrypt 기반 유저 인증
│
├── utils/
│   ├── keypoints.py            # Pose 모델 로딩
│   ├── activity_segment.py     # 활동 구간 분리
│   └── visualization.py        # 스켈레톤 시각화
│
├── gemini_feedback.py          # Gemini API 피드백 생성
├── modal_app.py                # Modal.com 배포 설정
├── preprocess/                 # 데이터 전처리 스크립트
├── scripts/                    # 레퍼런스 데이터 생성 스크립트
├── data/                       # 업로드 영상, 프레임, DB
└── requirements.txt
```

## 설치 및 실행 / Getting Started

### 1. 저장소 클론

```bash
git clone https://github.com/YBIGTA/28th-project-posecoach.git
cd 28th-project-posecoach
```

### 2. 백엔드 실행 (터미널 1)

```bash
pip install -r requirements.txt
uvicorn apps.api.main:app --reload
```

백엔드는 기본적으로 `http://localhost:8000`에서 실행됩니다.

### 3. 프론트엔드 실행 (터미널 2)

```bash
cd apps/web
npm install
npm run dev
```

프론트엔드는 기본적으로 `http://localhost:5173`에서 실행됩니다.

### 4. Gemini API 키 설정 (선택)

AI 피드백 기능을 사용하려면 [Google AI Studio](https://aistudio.google.com)에서 API 키를 발급받으세요.

```bash
# .env 파일에 설정
echo "GEMINI_API_KEY=AIza..." > .env
```

또는 결과 페이지에서 직접 입력할 수 있습니다.

### 5. Modal 배포 (GPU 서버)

```bash
pip install modal
modal setup   # 최초 1회 로그인
modal deploy modal_app.py
```

## 사용 방법 / Usage

1. **회원가입/로그인** — 계정 생성 또는 게스트 모드로 시작
2. **운동 선택** — 푸시업 또는 풀업(그립 타입 선택)
3. **영상 업로드** — MP4, MOV, AVI, WEBM 지원. 레퍼런스 영상 등록(DTW 비교용) 가능
4. **분석 결과 확인** — 등급(S/A/B/C), 운동 횟수, Phase별 점수, 오류 프레임 네비게이터
5. **AI 피드백** — Gemini 기반 전문 트레이너 수준 피드백 생성
6. **PDF 리포트** — 분석 결과 + AI 피드백을 PDF로 다운로드
7. **마이페이지** — 운동 기록 히스토리 및 성장 추이 확인

## API 엔드포인트 / Endpoints

| Method | Path | 설명 |
|--------|------|------|
| `GET` | `/health` | 서버 상태 확인 |
| `POST` | `/auth/register` | 회원가입 |
| `POST` | `/auth/login` | 로그인 |
| `GET` | `/workouts/{user_id}` | 운동 기록 조회 |
| `GET` | `/workouts/{user_id}/stats` | 운동 통계 조회 |
| `POST` | `/workouts/{user_id}` | 운동 기록 저장 |
| `POST` | `/analysis` | 영상 분석 (multipart form) |
| `POST` | `/analysis/feedback` | Gemini AI 피드백 생성 |
| `POST` | `/analysis/report` | PDF 리포트 다운로드 |

## 등급 기준 / Grading

| 등급 / Grade | 평균 자세 점수 / Average Score |
|:---:|:---:|
| S CLASS | 90% 이상 |
| A CLASS | 70% 이상 |
| B CLASS | 50% 이상 |
| C CLASS | 50% 미만 |

## 파이프라인 최적화 / Pipeline Optimization

분석 정확도에는 영향 없이 다음 최적화가 적용되어 있습니다.

| 최적화 | 설명 |
|--------|------|
| **640x360 분석 해상도** | YOLO는 내부적으로 640x640으로 리사이즈하므로 FHD 업스케일 불필요. `normalize_pts()`가 [0,1] 정규화 좌표를 생성하여 해상도 무관 |
| **YOLO 배치 추론** | 프레임을 batch_size=8로 묶어 한 번에 추론. 단일 프레임 루프 대비 15-25% 속도 향상 |
| **Activity Filter 메모리 캐시** | grayscale 썸네일을 미리 계산하여 디스크 재읽기 제거 |
| **에러 프레임만 오버레이** | 스켈레톤 오버레이는 에러가 검출된 프레임에만 생성. 오버레이 단계 80-95% 속도 향상 |
| **원본 해상도 스켈레톤** | 에러 프레임의 스켈레톤 오버레이는 원본 비디오에서 해당 프레임을 직접 읽어 원본 해상도로 생성 |

### 예상 성능 / Expected Performance

| 환경 | 30초 영상 (3fps, ~90프레임) |
|------|---------------------------|
| GPU T4 (Modal.com) | ~30초-1분 |
