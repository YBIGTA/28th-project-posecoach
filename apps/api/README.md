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

`/analysis` form fields:

- `video` (file, required)
- `exercise_type` (`pushup`/`pullup`; Korean labels are also accepted)
- `extract_fps` (1-30, default: `10`)
- `grip_type` (`overhand`, `underhand`, `wide` or Korean labels; pull-up only)
- `save_result` (`true`/`false`)
- `user_id` (required when `save_result=true`)
