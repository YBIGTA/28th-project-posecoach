export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

const PROTOTYPE_GEMINI_API_KEY = "AIzaSyCNVNNZkO0s7lJx6AQYZ8WMJUuxX-FzBAw";
const ENV_GEMINI_API_KEY = (import.meta.env.VITE_GEMINI_API_KEY as string | undefined) ?? "";

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
    details?: Record<string, { status: string; value: string; feedback: string }>;
  }>;
  error_frames: Array<{
    frame_idx: number;
    img_url?: string | null;
    phase: string;
    score: number;
    errors: string[];
    details?: Record<string, { status: string; value: string; feedback: string }>;
  }>;
  keypoints?: Array<{
    frame_idx: number;
    img_url?: string | null;
    pts?: Record<string, { x: number; y: number; z?: number; vis: number }> | null;
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
  errors?: Array<{ error_msg: string; count: number }>;
  phase_scores?: Array<{ phase: string; avg_score: number; frame_count: number }>;
};

export type AnalyzeVideoInput = {
  videoFile: File;
  referenceFile?: File;           // ← 추가
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

  // 레퍼런스 영상 (DTW용) - 백엔드 필드명 확인 필요 시 "reference_video"로 변경
  if (input.referenceFile) {
    formData.append("reference_video", input.referenceFile);
  }

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
  const normalizeKey = (value?: string): string | undefined => {
    if (!value) return undefined;
    const key = value.trim();
    if (!key || key === "PASTE_YOUR_GEMINI_API_KEY_HERE") return undefined;
    return key;
  };

  const resolvedApiKey =
    normalizeKey(input.apiKey) ||
    normalizeKey(PROTOTYPE_GEMINI_API_KEY) ||
    normalizeKey(ENV_GEMINI_API_KEY) ||
    undefined;

  const request = async (apiKey?: string) => {
    const response = await fetch(`${API_BASE_URL}/analysis/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        analysis_results: {
          video_name: input.analysisResults.video_name,
          exercise_type: input.analysisResults.exercise_type,
          exercise_count: input.analysisResults.exercise_count,
          frame_scores: input.analysisResults.frame_scores,
          error_frames: input.analysisResults.error_frames,
          duration: input.analysisResults.duration,
          fps: input.analysisResults.fps,
          total_frames: input.analysisResults.total_frames,
          dtw_active: input.analysisResults.dtw_active,
          dtw_result: input.analysisResults.dtw_result,
        },
        api_key: apiKey,
        temperature: input.temperature ?? 0.7,
        max_output_tokens: input.maxOutputTokens ?? 6000,
      }),
    });
    const payload = await response.json().catch(() => ({}));
    return { response, payload };
  };

  let { response, payload } = await request(resolvedApiKey);

  if (!response.ok && resolvedApiKey) {
    const detail = String(payload?.detail ?? "");
    const looksLikeKeyIssue =
      detail.includes("API 키") ||
      detail.toLowerCase().includes("api key") ||
      detail.includes("403") ||
      detail.includes("reported as leaked");
    if (looksLikeKeyIssue) {
      const fallback = await request(undefined);
      response = fallback.response;
      payload = fallback.payload;
    }
  }

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