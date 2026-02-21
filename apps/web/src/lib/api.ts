export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

export type AnalysisResults = {
  video_name: string;
  exercise_type: string;
  exercise_type_en?: "pushup" | "pullup" | string;
  grip_type?: string | null;
  exercise_count: number;
  frame_scores: Array<{
    frame_idx: number;
    img_url?: string | null;
    phase: string;
    score: number;
    errors: string[];
    details?: Record<
      string,
      {
        status: string;
        value: string;
        feedback: string;
      }
    >;
  }>;
  error_frames: Array<{
    frame_idx: number;
    img_url?: string | null;
    phase: string;
    score: number;
    errors: string[];
    details?: Record<
      string,
      {
        status: string;
        value: string;
        feedback: string;
      }
    >;
  }>;
  keypoints?: Array<{
    frame_idx: number;
    img_url?: string | null;
    pts?: Record<
      string,
      {
        x: number;
        y: number;
        z?: number;
        vis: number;
      }
    > | null;
    selected_for_analysis?: boolean;
  }>;
  duration: number;
  fps: number;
  total_frames: number;
  analyzed_frame_count?: number;
  filtered_out_count?: number;
  filtering?: {
    method?: string;
    reason?: string;
    model_path?: string;
    rule_active_frames?: number;
    rule_rest_frames?: number;
    ml_fallback_frames?: number;
  };
  selected_frame_indices?: number[];
  success_count: number;
  dtw_active?: boolean;
  dtw_result?: {
    overall_dtw_score?: number | null;
    phase_dtw_scores?: Record<string, number>;
  } | null;
};

export type AnalyzeVideoResponse = {
  analysis_results: AnalysisResults;
  saved_workout_id?: number | null;
};

export type GeminiFeedbackInput = {
  analysisResults: AnalysisResults;
  apiKey?: string;
  temperature?: number;
  maxOutputTokens?: number;
};

export type UserStats = {
  total_workouts: number;
  total_duration: number;
  overall_avg_score: number;
  total_reps: number;
  favorite_exercise: string;
};

export type WorkoutRecord = {
  id: number;
  user_id: number;
  created_at: string;
  video_name: string;
  exercise_type: string;
  grip_type?: string | null;
  exercise_count: number;
  duration: number;
  fps: number;
  total_frames: number;
  avg_score: number;
  grade: string;
  dtw_active: number;
  dtw_score?: number | null;
  combined_score?: number | null;
  error_frame_count: number;
  errors?: Array<{
    error_msg: string;
    count: number;
  }>;
  phase_scores?: Array<{
    phase: string;
    avg_score: number;
    frame_count: number;
  }>;
};

export type AnalyzeVideoInput = {
  videoFile: File;
  exerciseType: "pushup" | "pullup";
  gripType?: string;
  extractFps?: number;
  saveResult?: boolean;
  userId?: number;
};

function normalizeGrip(gripType?: string): string | undefined {
  if (!gripType) return undefined;
  const value = gripType.trim().toLowerCase();
  if (value === "neutral") return "overhand";
  return value;
}

export async function analyzeVideo(input: AnalyzeVideoInput): Promise<AnalyzeVideoResponse> {
  const formData = new FormData();
  formData.append("video", input.videoFile);
  formData.append("exercise_type", input.exerciseType);
  formData.append("extract_fps", String(input.extractFps ?? 10));

  const grip = normalizeGrip(input.gripType);
  if (grip) formData.append("grip_type", grip);

  if (input.saveResult) {
    formData.append("save_result", "true");
    if (typeof input.userId === "number") {
      formData.append("user_id", String(input.userId));
    }
  } else {
    formData.append("save_result", "false");
  }

  const response = await fetch(`${API_BASE_URL}/analysis`, {
    method: "POST",
    body: formData,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail ?? "영상 분석에 실패했습니다.";
    throw new Error(typeof detail === "string" ? detail : "영상 분석에 실패했습니다.");
  }

  return payload as AnalyzeVideoResponse;
}

export async function fetchUserStats(userId: number): Promise<UserStats> {
  const response = await fetch(`${API_BASE_URL}/workouts/${userId}/stats`);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail ?? "통계 조회에 실패했습니다.";
    throw new Error(typeof detail === "string" ? detail : "통계 조회에 실패했습니다.");
  }
  return payload as UserStats;
}

export async function fetchUserWorkouts(userId: number): Promise<WorkoutRecord[]> {
  const response = await fetch(`${API_BASE_URL}/workouts/${userId}`);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail ?? "운동 기록 조회에 실패했습니다.";
    throw new Error(typeof detail === "string" ? detail : "운동 기록 조회에 실패했습니다.");
  }

  const workouts = Array.isArray(payload?.workouts) ? payload.workouts : [];
  return workouts as WorkoutRecord[];
}

export async function generateGeminiFeedback(input: GeminiFeedbackInput): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/analysis/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      analysis_results: input.analysisResults,
      api_key: input.apiKey,
      temperature: input.temperature ?? 0.7,
      max_output_tokens: input.maxOutputTokens ?? 6000,
    }),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail ?? "AI 피드백 생성에 실패했습니다.";
    throw new Error(typeof detail === "string" ? detail : "AI 피드백 생성에 실패했습니다.");
  }

  const text = payload?.feedback;
  if (typeof text !== "string" || !text.trim()) {
    throw new Error("AI 피드백 응답이 비어 있습니다.");
  }
  return text;
}
