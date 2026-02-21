import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Home,
  Sparkles,
  XCircle,
} from "lucide-react";

import { API_BASE_URL, generateGeminiFeedback, type AnalysisResults } from "../lib/api";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Progress } from "../components/ui/progress";

interface FeedbackItem {
  id: string;
  type: "good" | "warning" | "error";
  title: string;
  description: string;
}

type ResultPageState = {
  exercise?: "pushup" | "pullup";
  grip?: string;
  analysisResults?: AnalysisResults;
  savedWorkoutId?: number | null;
};

type PosePoint = {
  x: number;
  y: number;
  z?: number;
  vis: number;
};

type PosePoints = Record<string, PosePoint> | null | undefined;

const COCO_SKELETON: Array<[string, string]> = [
  ["Nose", "Left Eye"], ["Nose", "Right Eye"],
  ["Left Eye", "Left Ear"], ["Right Eye", "Right Ear"],
  ["Left Shoulder", "Right Shoulder"],
  ["Left Shoulder", "Left Elbow"], ["Left Elbow", "Left Wrist"],
  ["Right Shoulder", "Right Elbow"], ["Right Elbow", "Right Wrist"],
  ["Left Shoulder", "Left Hip"], ["Right Shoulder", "Right Hip"],
  ["Left Hip", "Right Hip"],
  ["Left Hip", "Left Knee"], ["Left Knee", "Left Ankle"],
  ["Right Hip", "Right Knee"], ["Right Knee", "Right Ankle"],
];

const VIS_THRESHOLD = 0.5;
const PHASE_LABEL: Record<string, string> = {
  ready: "준비",
  top: "상단",
  descending: "하강",
  bottom: "하단",
  ascending: "상승",
};

function getIcon(type: FeedbackItem["type"]) {
  if (type === "good") return <CheckCircle2 className="w-5 h-5 text-green-600" />;
  if (type === "warning") return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
  return <XCircle className="w-5 h-5 text-red-600" />;
}

function getBadgeVariant(type: FeedbackItem["type"]) {
  if (type === "good") return "default";
  if (type === "warning") return "secondary";
  return "destructive";
}

function buildFeedbackItems(results: AnalysisResults): FeedbackItem[] {
  const counts = new Map<string, number>();
  for (const ef of results.error_frames ?? []) {
    for (const msg of ef.errors ?? []) {
      counts.set(msg, (counts.get(msg) ?? 0) + 1);
    }
  }

  const rankedErrors = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
  const items: FeedbackItem[] = rankedErrors.map(([msg, count], idx) => ({
    id: `err-${idx}`,
    type: count >= 5 ? "error" : "warning",
    title: `자세 오류 (${count}프레임)`,
    description: msg,
  }));

  if (items.length === 0) {
    items.push({
      id: "good-0",
      type: "good",
      title: "양호한 동작",
      description: "활성 구간에서 큰 자세 오류가 감지되지 않았습니다.",
    });
  }

  items.unshift({
    id: "summary-0",
    type: "good",
    title: "반복 횟수",
    description: `감지된 반복 횟수: ${results.exercise_count}회`,
  });

  return items;
}

function scoreColor(score: number): string {
  if (score >= 80) return "rgb(22, 163, 74)";
  if (score >= 60) return "rgb(234, 179, 8)";
  return "rgb(220, 38, 38)";
}

function toImageUrl(imgUrl?: string | null): string | null {
  if (!imgUrl) return null;
  if (imgUrl.startsWith("http://") || imgUrl.startsWith("https://")) return imgUrl;
  return `${API_BASE_URL}${imgUrl}`;
}

function toFilteringMethodLabel(method?: string): string {
  if (!method) return "알 수 없음";
  if (method === "rule") return "규칙 기반";
  if (method === "ml_fallback") return "규칙 기반 + ML 보완";
  if (method === "none") return "필터링 없음";
  return method;
}

function toFilteringReasonLabel(reason?: string): string | undefined {
  if (!reason) return undefined;
  if (reason === "no input frames") return "입력 프레임이 없습니다.";
  if (reason === "joblib not installed") return "joblib가 설치되어 있지 않습니다.";
  if (reason.startsWith("model file missing:")) return `모델 파일이 없습니다: ${reason.replace("model file missing:", "").trim()}`;
  if (reason.startsWith("failed to load model:")) return `모델 로드에 실패했습니다: ${reason.replace("failed to load model:", "").trim()}`;
  if (reason.startsWith("failed to run model:")) return `모델 추론에 실패했습니다: ${reason.replace("failed to run model:", "").trim()}`;
  if (reason === "selected frame ratio too small") return "선택 프레임 비율이 너무 작습니다.";
  if (reason.startsWith("ML low-contrast over-selection")) return "ML 결과 대비가 낮아 과도 선택으로 판단되었습니다.";
  return reason;
}

function toPhaseLabel(phase?: string): string {
  if (!phase) return "-";
  return PHASE_LABEL[phase] ?? phase;
}

function SkeletonPreview({
  imageUrl,
  keypoints,
  showOverlay,
}: {
  imageUrl: string | null;
  keypoints: PosePoints;
  showOverlay: boolean;
}) {
  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const img = imageRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const draw = () => {
      const width = img.naturalWidth || img.width || 1;
      const height = img.naturalHeight || img.height || 1;
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      if (!showOverlay || !keypoints) return;

      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgb(255,255,0)";
      for (const [a, b] of COCO_SKELETON) {
        const pa = keypoints[a];
        const pb = keypoints[b];
        if (!pa || !pb) continue;
        if (pa.vis < VIS_THRESHOLD || pb.vis < VIS_THRESHOLD) continue;
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
      }

      ctx.fillStyle = "rgb(0,255,0)";
      for (const point of Object.values(keypoints)) {
        if (point.vis < VIS_THRESHOLD) continue;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    if (img.complete) draw();
    else img.onload = draw;

    return () => {
      if (img) img.onload = null;
    };
  }, [imageUrl, keypoints, showOverlay]);

  if (!imageUrl) {
    return (
      <div className="rounded-lg border bg-slate-100 text-sm text-slate-600 p-10 text-center">
        프레임 이미지를 불러올 수 없습니다.
      </div>
    );
  }

  return (
    <div className="relative rounded-lg overflow-hidden border bg-black">
      <img ref={imageRef} src={imageUrl} alt="스켈레톤 미리보기" className="w-full h-auto block" />
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
    </div>
  );
}

export function Result() {
  const navigate = useNavigate();
  const location = useLocation();
  const state = (location.state ?? {}) as ResultPageState;

  const exercise = state.exercise ?? "pushup";
  const grip = state.grip;
  const analysis = state.analysisResults;
  const savedWorkoutId = state.savedWorkoutId ?? null;

  const [showOverlay, setShowOverlay] = useState(true);
  const [frameIndex, setFrameIndex] = useState<number>(0);
  const [geminiLoading, setGeminiLoading] = useState(false);
  const [geminiError, setGeminiError] = useState<string | null>(null);
  const [geminiFeedback, setGeminiFeedback] = useState<string | null>(null);
  const autoGeminiRequestedRef = useRef<string>("");

  if (!analysis) {
    return (
      <div className="modern-shell min-h-screen w-full p-8 flex items-center justify-center">
        <Card className="max-w-xl w-full glass-card">
          <CardContent className="p-8 text-center space-y-4">
            <h1 className="text-2xl font-bold">분석 결과가 없습니다</h1>
            <p className="text-soft">영상을 업로드한 뒤 분석을 실행해 주세요.</p>
            <Button className="modern-primary-btn" onClick={() => navigate("/select-exercise")}>분석 시작으로 이동</Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  const avgScoreRaw =
    analysis.frame_scores && analysis.frame_scores.length > 0
      ? analysis.frame_scores.reduce((acc, item) => acc + item.score, 0) / analysis.frame_scores.length
      : 0;
  const overallScore = Math.round(avgScoreRaw * 100);

  const feedbackItems = buildFeedbackItems(analysis);
  const goodCount = feedbackItems.filter((item) => item.type === "good").length;
  const warningCount = feedbackItems.filter((item) => item.type === "warning").length;
  const errorCount = feedbackItems.filter((item) => item.type === "error").length;

  const dtwOverall = analysis.dtw_result?.overall_dtw_score;
  const dtwPhase = analysis.dtw_result?.phase_dtw_scores ?? {};

  const keypointFrames = useMemo(
    () => [...(analysis.keypoints ?? [])].sort((a, b) => a.frame_idx - b.frame_idx),
    [analysis.keypoints],
  );

  const scoreByFrame = useMemo(
    () => new Map((analysis.frame_scores ?? []).map((row) => [row.frame_idx, row])),
    [analysis.frame_scores],
  );

  const keypointByFrame = useMemo(
    () => new Map(keypointFrames.map((row) => [row.frame_idx, row])),
    [keypointFrames],
  );

  const selectedFrameSet = useMemo(() => {
    if (analysis.selected_frame_indices && analysis.selected_frame_indices.length > 0) {
      return new Set(analysis.selected_frame_indices);
    }
    const selectedFromPayload = keypointFrames
      .filter((row) => row.selected_for_analysis)
      .map((row) => row.frame_idx);
    if (selectedFromPayload.length > 0) return new Set(selectedFromPayload);
    return new Set((analysis.frame_scores ?? []).map((row) => row.frame_idx));
  }, [analysis.selected_frame_indices, analysis.frame_scores, keypointFrames]);

  const maxFrameIdx = Math.max(0, analysis.total_frames - 1);

  useEffect(() => {
    const initial = analysis.selected_frame_indices?.[0] ?? 0;
    setFrameIndex(Math.min(Math.max(0, initial), maxFrameIdx));
  }, [analysis.video_name, analysis.selected_frame_indices, maxFrameIdx]);

  const currentKeypoint = keypointByFrame.get(frameIndex);
  const currentScore = scoreByFrame.get(frameIndex);
  const currentImageUrl = toImageUrl(currentKeypoint?.img_url ?? currentScore?.img_url);
  const currentSelected = selectedFrameSet.has(frameIndex);

  const handleGenerateGeminiFeedback = useCallback(async () => {
    if (geminiLoading) return;
    setGeminiLoading(true);
    setGeminiError(null);
    try {
      const feedback = await generateGeminiFeedback({
        analysisResults: {
          video_name: analysis.video_name,
          exercise_type: analysis.exercise_type,
          exercise_type_en: analysis.exercise_type_en,
          grip_type: analysis.grip_type,
          exercise_count: analysis.exercise_count,
          frame_scores: analysis.frame_scores,
          error_frames: analysis.error_frames,
          duration: analysis.duration,
          fps: analysis.fps,
          total_frames: analysis.total_frames,
          analyzed_frame_count: analysis.analyzed_frame_count,
          filtered_out_count: analysis.filtered_out_count,
          filtering: analysis.filtering,
          selected_frame_indices: analysis.selected_frame_indices,
          success_count: analysis.success_count,
          dtw_active: analysis.dtw_active,
          dtw_result: analysis.dtw_result,
        },
      });
      setGeminiFeedback(feedback);
    } catch (e) {
      setGeminiError(e instanceof Error ? e.message : "AI 피드백 생성에 실패했습니다.");
    } finally {
      setGeminiLoading(false);
    }
  }, [analysis, geminiLoading]);

  const autoGeminiKey = useMemo(
    () => `${analysis.video_name}|${analysis.exercise_count}|${analysis.total_frames}|${analysis.filtered_out_count ?? -1}`,
    [analysis.video_name, analysis.exercise_count, analysis.total_frames, analysis.filtered_out_count],
  );

  useEffect(() => {
    if (autoGeminiRequestedRef.current === autoGeminiKey) return;
    autoGeminiRequestedRef.current = autoGeminiKey;
    void handleGenerateGeminiFeedback();
  }, [autoGeminiKey, handleGenerateGeminiFeedback]);

  return (
    <div className="modern-shell min-h-screen w-full p-6 md:p-8">
      <div className="mx-auto max-w-6xl">
        <div className="text-center mb-12 animate-rise">
          <span className="hero-chip mb-4">
            <Sparkles className="w-3.5 h-3.5 mr-1" />
            분석 리포트
          </span>
          
          <h1 className="text-4xl md:text-5xl font-bold mb-2">분석 완료</h1>
          <p className="text-soft">
            {exercise === "pullup" ? "풀업" : "푸시업"} 결과
            {grip && <span> ({grip})</span>}
          </p>
          {savedWorkoutId ? (
            <p className="text-sm text-emerald-700 dark:text-emerald-300 mt-2">운동 기록으로 저장됨 (#{savedWorkoutId})</p>
          ) : (
            <p className="text-sm text-soft mt-2">로그인하지 않아 기록에 저장되지 않았습니다.</p>
          )}
        </div>

        <Card className="mb-8 glass-card animate-rise delay-1">
          <CardContent className="p-8">
            <div className="text-center mb-6">
              <div className="text-6xl font-bold mb-2" style={{ color: scoreColor(overallScore) }}>
                {overallScore}
                <span className="text-3xl">%</span>
              </div>
              <p className="text-soft">종합 점수</p>
            </div>
            <Progress value={overallScore} className="h-3" />

            <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mt-8">
              <div className="text-center p-4 rounded-xl border border-slate-700 bg-slate-900/72">
                <div className="text-2xl font-bold text-cyan-300">{analysis.exercise_count}</div>
                <p className="text-sm text-slate-100">반복 횟수</p>
              </div>
              <div className="text-center p-4 rounded-xl border border-slate-700 bg-slate-900/72">
                <div className="text-2xl font-bold text-slate-100">{analysis.total_frames}</div>
                <p className="text-sm text-slate-100">추출 프레임</p>
              </div>
              <div className="text-center p-4 rounded-xl border border-slate-700 bg-slate-900/72">
                <div className="text-2xl font-bold text-indigo-300">{analysis.analyzed_frame_count ?? 0}</div>
                <p className="text-sm text-slate-100">분석 프레임</p>
              </div>
              <div className="text-center p-4 rounded-xl border border-slate-700 bg-slate-900/72">
                <div className="flex items-center justify-center mb-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 mr-1" />
                  <span className="text-2xl font-bold text-emerald-400">{goodCount}</span>
                </div>
                <p className="text-sm text-slate-100">양호</p>
              </div>
              <div className="text-center p-4 rounded-xl border border-slate-700 bg-slate-900/72">
                <div className="flex items-center justify-center mb-2">
                  <AlertTriangle className="w-5 h-5 text-amber-400 mr-1" />
                  <span className="text-2xl font-bold text-amber-400">{warningCount}</span>
                </div>
                <p className="text-sm text-slate-100">주의</p>
              </div>
              <div className="text-center p-4 rounded-xl border border-slate-700 bg-slate-900/72">
                <div className="flex items-center justify-center mb-2">
                  <XCircle className="w-5 h-5 text-red-400 mr-1" />
                  <span className="text-2xl font-bold text-red-400">{errorCount}</span>
                </div>
                <p className="text-sm text-slate-100">오류</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {analysis.dtw_active ? (
          <Card className="mb-8 glass-card animate-rise delay-1">
            <CardHeader>
              <CardTitle>DTW 유사도</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl bg-slate-100/70 dark:bg-slate-800/60 border border-slate-200/70 dark:border-slate-700">
                  <div className="text-sm text-soft mb-1">종합 DTW 점수</div>
                  <div className="text-3xl font-bold">
                    {typeof dtwOverall === "number" ? `${Math.round(dtwOverall * 100)}%` : "없음"}
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-slate-100/70 dark:bg-slate-800/60 border border-slate-200/70 dark:border-slate-700">
                  <div className="text-sm text-soft mb-2">Phase별 점수</div>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(dtwPhase).length === 0 ? (
                      <span className="text-sm text-soft">단계 점수 없음</span>
                    ) : (
                      Object.entries(dtwPhase).map(([phase, val]) => (
                        <Badge key={phase} variant="secondary">
                          {toPhaseLabel(phase)}: {Math.round(val * 100)}%
                        </Badge>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ) : null}

        <Card className="mb-8 glass-card animate-rise delay-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-2xl md:text-3xl font-extrabold tracking-tight">
              <span aria-hidden>📋</span>
              상세 피드백
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {feedbackItems.map((item) => (
                <div key={item.id} className="flex items-start gap-4 p-4 rounded-xl border border-slate-700 bg-slate-900/55">
                  <div className="flex-shrink-0 mt-0.5">{getIcon(item.type)}</div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-semibold">{item.title}</h3>
                      <Badge variant={getBadgeVariant(item.type)}>
                        {item.type === "good" ? "양호" : item.type === "warning" ? "주의" : "오류"}
                      </Badge>
                    </div>
                    <p className="text-soft text-sm">{item.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="mb-8 glass-card animate-rise delay-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-2xl md:text-3xl font-extrabold tracking-tight">
              <span aria-hidden>🤖</span>
              트레이너의 종합 리포트
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <Button
                  type="button"
                  onClick={handleGenerateGeminiFeedback}
                  disabled={geminiLoading}
                  className="modern-primary-btn"
                >
                  {geminiLoading ? "자동 생성 중..." : "AI 종합 피드백 다시 생성"}
                </Button>
                {geminiFeedback ? (
                  <Button type="button" variant="outline" className="modern-outline-btn" onClick={() => setGeminiFeedback(null)}>
                    지우기
                  </Button>
                ) : null}
              </div>

              {geminiError ? (
                <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  {geminiError}
                </div>
              ) : null}

              {geminiFeedback ? (
                <div className="rounded-lg border border-slate-700 bg-slate-900/65 px-5 py-5 text-slate-100">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      h2: (props) => <h2 className="mt-6 mb-3 text-xl md:text-2xl font-bold text-slate-50" {...props} />,
                      h3: (props) => <h3 className="mt-5 mb-2 text-lg md:text-xl font-semibold text-slate-100" {...props} />,
                      p: (props) => <p className="mb-3 text-sm md:text-base leading-7 text-slate-200" {...props} />,
                      ul: (props) => <ul className="mb-4 list-disc pl-5 space-y-1 text-sm md:text-base text-slate-200" {...props} />,
                      ol: (props) => <ol className="mb-4 list-decimal pl-5 space-y-1 text-sm md:text-base text-slate-200" {...props} />,
                      li: (props) => <li className="leading-7" {...props} />,
                      strong: (props) => <strong className="font-extrabold text-slate-50" {...props} />,
                      hr: (props) => <hr className="my-5 border-slate-700" {...props} />,
                    }}
                  >
                    {geminiFeedback}
                  </ReactMarkdown>
                </div>
              ) : (
                <p className="text-sm text-soft">아직 생성된 AI 종합 피드백이 없습니다.</p>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="mb-8 glass-card animate-rise delay-3">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-2xl md:text-3xl font-extrabold tracking-tight">
              <span aria-hidden>🎬</span>
              프레임 탐색기
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2 mb-4">
              <Badge variant="secondary">프레임 {frameIndex}</Badge>
              {currentScore ? <Badge>{toPhaseLabel(currentScore.phase)}</Badge> : null}
              {currentScore ? <Badge variant="secondary">점수: {Math.round(currentScore.score * 100)}%</Badge> : null}
              {!currentScore && currentSelected ? (
                <Badge variant="secondary">준비/종료 구간 (점수 없음)</Badge>
              ) : null}
              {!currentSelected ? (
                <Badge className="bg-red-600 text-white hover:bg-red-600">필터링 제외 프레임 (휴식 구간)</Badge>
              ) : null}
            </div>

            <div className="flex flex-wrap items-center gap-2 mb-4">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="modern-outline-btn"
                disabled={frameIndex <= 0}
                onClick={() => setFrameIndex((prev) => Math.max(0, prev - 1))}
              >
                <ChevronLeft className="w-4 h-4 mr-1" />
                이전
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="modern-outline-btn"
                disabled={frameIndex >= maxFrameIdx}
                onClick={() => setFrameIndex((prev) => Math.min(maxFrameIdx, prev + 1))}
              >
                다음
                <ChevronRight className="w-4 h-4 ml-1" />
              </Button>
              <Button type="button" variant="outline" size="sm" className="modern-outline-btn" onClick={() => setShowOverlay((prev) => !prev)}>
                {showOverlay ? "오버레이 숨기기" : "오버레이 보기"}
              </Button>
            </div>

            <input
              type="range"
              min={0}
              max={maxFrameIdx}
              value={frameIndex}
              onChange={(e) => setFrameIndex(Number(e.target.value))}
              className="w-full mb-6 accent-cyan-600"
            />

            <div>
              <p className="text-sm text-soft mb-2">스켈레톤 오버레이</p>
              <SkeletonPreview imageUrl={currentImageUrl} keypoints={currentKeypoint?.pts} showOverlay={showOverlay} />
            </div>

            {currentScore ? (
              <div className="mt-6 space-y-2">
                <h3 className="font-semibold">프레임 피드백</h3>
                {currentScore.errors && currentScore.errors.length > 0 ? (
                  currentScore.errors.map((err, idx) => (
                    <p key={`${currentScore.frame_idx}-${idx}`} className="text-sm text-red-700 dark:text-red-300">
                      - {err}
                    </p>
                  ))
                ) : (
                  <p className="text-sm text-emerald-700 dark:text-emerald-300">이 프레임에서 자세 오류가 감지되지 않았습니다.</p>
                )}
              </div>
            ) : (
              <p className="mt-6 text-sm text-soft">
                {currentSelected
                  ? "활성 분석 구간에 포함되지만 점수 산정 Phase 바깥 프레임입니다."
                  : "활동성 필터링으로 제외된 프레임입니다. (휴식/저움직임 구간)"}
              </p>
            )}
          </CardContent>
        </Card>

        <div className="flex justify-center gap-4">
          <Button size="lg" variant="outline" className="modern-outline-btn px-8" onClick={() => navigate("/")}>
            <Home className="w-4 h-4 mr-2" />
            홈
          </Button>
          <Button
            size="lg"
            className="modern-primary-btn px-8"
            onClick={() => navigate("/select-exercise")}
          >
            다시 분석하기
          </Button>
        </div>
      </div>
    </div>
  );
}
