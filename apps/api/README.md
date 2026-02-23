# PoseCoach API

## Run

```bash
uvicorn apps.api.main:app --reload
```

## Main endpoints

- `GET /health`
- `POST /auth/register`
- `POST /auth/login`
- `GET /workouts/{user_id}`
- `GET /workouts/{user_id}/stats`
- `POST /workouts/{user_id}`
- `POST /analysis` (multipart form)
- `POST /analysis/feedback`
- `POST /analysis/report`

`/analysis` form fields:

- `video` (file, required)
- `reference_video` (file, optional — for DTW comparison)
- `exercise_type` (`pushup`/`pullup`; Korean labels are also accepted)
- `extract_fps` (1-30, default: `10`)
- `grip_type` (`overhand`, `underhand`, `wide` or Korean labels; pull-up only)
- `save_result` (`true`/`false`)
- `user_id` (required when `save_result=true`)

`/analysis/feedback` JSON body:

- `analysis_results` (dict, required)
- `api_key` (string, optional — Gemini API key)
- `temperature` (float, 0.0-1.0, default: `0.7`)
- `max_output_tokens` (int, 128-8192, default: `6000`)

`/analysis/report` JSON body:

- `analysis_results` (dict, required)
- `ai_feedback` (string, optional — pre-generated feedback from frontend)
- `gemini_api_key` (string, optional — for server-side feedback generation)
- `generate_feedback` (bool, default: `true` — auto-generate feedback if not provided)
