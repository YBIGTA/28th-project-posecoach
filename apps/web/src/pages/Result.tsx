import { useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Dumbbell } from "lucide-react";
import { Button } from "../components/ui/button";
import { getSession } from "../lib/auth";
import { generateGeminiFeedback } from "../lib/api";

type FrameScore = {
  frame_idx: number; img_url?: string | null; skeleton_url?: string | null; phase: string;
  score: number; errors: string[];
  details?: Record<string, { status: string; value: string; feedback: string }>;
};
type ErrorFrame = FrameScore;
type KeypointFrame = {
  frame_idx: number;
  img_url?: string | null;
  pts?: Record<string, { x: number; y: number; z?: number; vis: number }> | null;
  selected_for_analysis?: boolean;
};
type DtwResult = {
  overall_dtw_score?: number;
  phase_dtw_scores?: Record<string, number>;
  phase_segment_counts?: Record<string, number>;
  worst_joints?: Record<string, number>;
};
type AnalysisResults = {
  video_name: string; exercise_type: string; exercise_count: number;
  frame_scores: FrameScore[]; error_frames: ErrorFrame[];
  keypoints?: KeypointFrame[];
  duration: number; fps: number; total_frames: number;
  analyzed_frame_count?: number;
  filtered_out_count?: number;
  selected_frame_indices?: number[];
  filtering?: {
    method?: string;
    reason?: string;
    model_path?: string;
    rule_active_frames?: number;
    rule_rest_frames?: number;
    ml_fallback_frames?: number;
  };
  dtw_result?: DtwResult; dtw_active: boolean;
};

const API = (import.meta as any).env?.VITE_API_BASE_URL ?? "http://localhost:8000";
const pct = (v: number) => `${Math.round(v * 100)}%`;
const ALL_PHASE = "__ALL_PHASE__";

const STEPS = ["운동 선택", "그립 선택", "결과 확인"];

const PHASE_COLOR: Record<string, string> = {
  top: "rgba(91,143,255,0.7)", bottom: "rgba(255,107,53,0.7)",
  ascending: "rgba(200,241,53,0.7)", descending: "rgba(200,241,53,0.55)",
  ready: "#444", finish: "#444",
};

function StepBar({ current }: { current: number }) {
  return (
    <div className="flex items-center py-6">
      {STEPS.map((s, i) => (
        <div key={s} className={`flex items-center ${i < STEPS.length - 1 ? "flex-1" : ""}`}>
          <div className="flex items-center gap-2">
            <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[12px] font-semibold
              ${i < current ? "bg-[#c8f135]/10 border border-[#c8f135]/40 text-[#c8f135]"
              : i === current ? "bg-[#c8f135] text-black"
              : "bg-white/5 border border-white/10 text-white/30"}`}>
              {i < current ? "✓" : i + 1}
            </div>
            <span className={`text-[12px] tracking-wide ${i === current ? "text-[#c8f135]" : "text-white/40"}`}>{s}</span>
          </div>
          {i < STEPS.length - 1 && (
            <div className={`flex-1 h-px mx-4 ${i < current ? "bg-[#c8f135]/40" : "bg-white/10"}`} />
          )}
        </div>
      ))}
    </div>
  );
}

function Chip({ children, color = "#c8f135" }: { children: React.ReactNode; color?: string }) {
  return (
    <span style={{ fontFamily: "DM Mono, monospace", fontSize: 9, borderRadius: 4, padding: "3px 8px",
      background: `${color}15`, border: `1px solid ${color}45`, color, display: "inline-block" }}>
      {children}
    </span>
  );
}

function MiniBarChart({ items }: { items: { label: string; pct: number; val: string; color?: string }[] }) {
  return (
    <div className="flex flex-col gap-4 py-2">
      {items.map(it => (
        <div key={it.label} className="flex items-center gap-3">
          <span className="text-right shrink-0 text-white/40" style={{ fontFamily: "DM Mono, monospace", fontSize: 12, width: 72 }}>{it.label}</span>
          <div className="flex-1 h-6 bg-white/5 rounded overflow-hidden">
            <div style={{ height: "100%", width: `${it.pct}%`, background: it.color ?? "rgba(200,241,53,0.6)", transition: "width .4s", borderRadius: 3 }} />
          </div>
          <span className="shrink-0 text-white/40" style={{ fontFamily: "DM Mono, monospace", fontSize: 12, width: 36 }}>{it.val}</span>
        </div>
      ))}
    </div>
  );
}

export function Result() {
  const navigate = useNavigate();
  const location = useLocation();
  const session = useMemo(() => getSession(), []);
  const { analysisResults: res, exercise, grip } =
    (location.state ?? {}) as { analysisResults?: AnalysisResults; exercise?: string; grip?: string };

  // ── 모든 useState는 조건문보다 위에 ──
  const [activeTab, setActiveTab] = useState<"phase" | "review" | "ai">("phase");
  const [selErrIdx, setSelErrIdx] = useState(0);
  const [frameIdx, setFrameIdx]   = useState(0);
  const [selPhase, setSelPhase]   = useState<string>(ALL_PHASE);
  const [geminiKey, setGeminiKey] = useState("");
  const [feedback, setFeedback]   = useState<string | null>(null);
  const [fbLoading, setFbLoading] = useState(false);
  const [fbError, setFbError]     = useState<string | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  if (!res) {
    return (
      <div className="min-h-screen w-full bg-[#0a0a0a] text-white flex items-center justify-center px-6 py-10">
        <div className="w-full max-w-[1400px] rounded-[30px] border border-white/10 bg-[#0f1116]/80 backdrop-blur-xl shadow-[0_30px_80px_rgba(0,0,0,0.6)] p-20 text-center">
          <div className="text-2xl font-extrabold mb-6">분석 결과가 없습니다</div>
          <Button className="bg-[#c8f135] text-black hover:bg-[#b4da30]" onClick={() => navigate("/select-exercise")}>← 분석 시작하기</Button>
        </div>
      </div>
    );
  }

  const frame_scores = res.frame_scores ?? [];
  const error_frames = res.error_frames ?? [];
  const dtw_result   = res.dtw_result ?? undefined;
  const dtw_active   = res.dtw_active ?? false;
  const avgScore = frame_scores.length ? frame_scores.reduce((a, b) => a + b.score, 0) / frame_scores.length : 0;
  const dtw      = dtw_active && dtw_result?.overall_dtw_score != null ? dtw_result.overall_dtw_score : null;
  const combined = dtw != null ? avgScore * 0.7 + dtw * 0.3 : avgScore;
  let grade = "C", gradeColor = "#ff6b35";
  if (combined >= 0.9) { grade = "S"; gradeColor = "#c8f135"; }
  else if (combined >= 0.7) { grade = "A"; gradeColor = "#5b8fff"; }
  else if (combined >= 0.5) { grade = "B"; gradeColor = "rgba(255,255,255,0.6)"; }

  const byPhase: Record<string, number[]> = {};
  frame_scores.forEach(f => { byPhase[f.phase] = [...(byPhase[f.phase] ?? []), f.score]; });
  const phaseAvg = Object.entries(byPhase).map(([p, sc]) => ({ phase: p, avg: sc.reduce((a, b) => a + b, 0) / sc.length }));
  const phaseCnt = Object.entries(byPhase).map(([p, sc]) => ({ phase: p, cnt: sc.length }));
  const maxCnt   = Math.max(...phaseCnt.map(p => p.cnt), 1);

  const totalFrames = Math.max(res.total_frames ?? 0, 0);
  const allFrameIndices = Array.from({ length: totalFrames }, (_, idx) => idx);
  const scoreByFrame = new Map(frame_scores.map(frame => [frame.frame_idx, frame] as const));
  const keypointByFrame = new Map((res.keypoints ?? []).map(frame => [frame.frame_idx, frame] as const));
  const selectedFrameSet = (() => {
    const explicit = res.selected_frame_indices ?? [];
    if (explicit.length > 0) {
      return new Set(explicit);
    }

    const payloadSelected = (res.keypoints ?? [])
      .filter(frame => frame.selected_for_analysis)
      .map(frame => frame.frame_idx);
    if (payloadSelected.length > 0) {
      return new Set(payloadSelected);
    }

    return new Set(frame_scores.map(frame => frame.frame_idx));
  })();

  const visibleFrameIndices = selPhase === ALL_PHASE
    ? allFrameIndices
    : allFrameIndices.filter((idx) => scoreByFrame.get(idx)?.phase === selPhase);

  const safeFrameCursor = visibleFrameIndices.length > 0
    ? Math.min(frameIdx, visibleFrameIndices.length - 1)
    : 0;
  const maxVisibleCursor = Math.max(visibleFrameIndices.length - 1, 0);
  const canGoPrev = safeFrameCursor > 0;
  const canGoNext = safeFrameCursor < maxVisibleCursor;
  const selectedFrameIdx = visibleFrameIndices[safeFrameCursor] ?? 0;
  const selFrame = scoreByFrame.get(selectedFrameIdx);
  const selFrameKeypoint = keypointByFrame.get(selectedFrameIdx);
  const selFrameIncluded = selectedFrameSet.has(selectedFrameIdx);
  const selFrameEvaluated = Boolean(selFrame);
  const selFrameImageUrl = selFrame?.img_url ?? selFrameKeypoint?.img_url;
  const jumpToCursor = (nextCursor: number) => {
    setFrameIdx(Math.min(Math.max(nextCursor, 0), maxVisibleCursor));
  };
  const jumpBy = (delta: number) => jumpToCursor(safeFrameCursor + delta);
  const selErr = error_frames.length > 0 ? error_frames[selErrIdx] : undefined;

  const errCount: Record<string, number> = {};
  error_frames.forEach(ef => ef.errors.forEach(e => { errCount[e] = (errCount[e] ?? 0) + 1; }));
  const topErrors = Object.entries(errCount).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const maxErrCnt = topErrors[0]?.[1] ?? 1;

  const genFeedback = async () => {
    if (fbLoading) return;
    setFbLoading(true); setFbError(null);
    try {
      const text = await generateGeminiFeedback({
        analysisResults: {
          video_name: res.video_name,
          exercise_type: res.exercise_type,
          exercise_count: res.exercise_count,
          frame_scores: res.frame_scores,
          error_frames: res.error_frames,
          duration: res.duration,
          fps: res.fps,
          total_frames: res.total_frames,
          dtw_active: res.dtw_active,
          dtw_result: res.dtw_result,
        },
        apiKey: geminiKey || undefined,
      });
      setFeedback(text);
    } catch (e) { setFbError(e instanceof Error ? e.message : "오류"); }
    finally { setFbLoading(false); }
  };

  const exportJson = () => {
    const blob = new Blob([JSON.stringify({ ...res, ai_feedback: feedback }, null, 2)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = `${res.video_name}_analysis.json`; a.click();
  };

  const downloadReport = async () => {
    if (reportLoading) return;
    setReportLoading(true);
    try {
      const r = await fetch(`${API}/analysis/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ analysis_results: res, ai_feedback: feedback }),
      });
      if (!r.ok) throw new Error("리포트 생성 실패");
      const blob = await r.blob();
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement("a");
      a.href     = url;
      a.download = `posecoach_report_${res.video_name}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert("PDF 리포트 생성에 실패했습니다.");
    } finally {
      setReportLoading(false);
    }
  };

  const tabCls = (on: boolean) =>
    `px-5 py-3 text-[16px] tracking-wide border-b-2 transition-all cursor-pointer bg-transparent border-0 outline-none
    ${on ? "text-[#c8f135] border-[#c8f135]" : "text-white/30 border-transparent hover:text-white/60"}`;

  return (
    <div className="min-h-screen w-full bg-[#0a0a0a] text-white px-6 py-10">
      <div className="w-full max-w-[1400px] mx-auto rounded-[30px] border border-white/10 bg-[#0f1116]/80 backdrop-blur-xl shadow-[0_30px_80px_rgba(0,0,0,0.6)] overflow-hidden">

        {/* ── HEADER ── */}
        <header className="flex items-center justify-between px-8 py-6 border-b border-white/10 backdrop-blur-xl bg-black/40">
          <button onClick={() => navigate("/")} className="flex items-center gap-2 font-extrabold tracking-widest text-white">
            <Dumbbell className="w-5 h-5 text-[#c8f135]" />
            POSECOACH
          </button>
          <div className="flex items-center gap-3">
            <Button variant="outline" size="sm"
              className="h-8 px-3 border-white/10 text-white/60 hover:text-[#c8f135] hover:border-[#c8f135]/40 hover:bg-[#c8f135]/10"
              onClick={() => navigate("/")}>
              ← 뒤로
            </Button>
            {session && (
              <div className="h-8 px-3 rounded-full border border-[#c8f135]/30 bg-[#c8f135]/10 text-[#c8f135] flex items-center justify-center text-xs tracking-wider">
                {session.username}
              </div>
            )}
          </div>
        </header>

        {/* ── CONTENT ── */}
        <div className="px-10 py-10">

          <StepBar current={2} />

          {/* ── 메트릭 5카드 ── */}
          <div className="grid grid-cols-5 gap-4 mb-10">
            {[
              { label: "운동 횟수", val: `${res.exercise_count}회`, color: "#c8f135" },
              { label: "평균 자세 점수", val: pct(avgScore), color: "#c8f135" },
              { label: "DTW 유사도", val: dtw != null ? pct(dtw) : "비활성", color: "#5b8fff" },
              { label: "오류 프레임", val: `${error_frames.length}개`, color: "#ff6b35" },
              { label: "종합 점수", val: pct(combined), color: gradeColor, sub: `${grade} GRADE` },
            ].map(m => (
              <div key={m.label} className="rounded-2xl border border-white/10 bg-white/5 p-6 flex flex-col gap-3">
                <div className="text-white/50 text-sm" style={{ fontFamily: "DM Mono, monospace" }}>{m.label}</div>
                <div className="font-extrabold text-3xl leading-none" style={{ color: m.color }}>{m.val}</div>
                {m.sub && <div className="text-[9px] text-white/20" style={{ fontFamily: "DM Mono, monospace" }}>{m.sub}</div>}
              </div>
            ))}
          </div>

          {/* ── 탭 ── */}
          <div className="flex border-b border-white/10 mb-8">
            <button className={tabCls(activeTab === "phase")}  style={{ fontFamily: "DM Mono, monospace" }} onClick={() => setActiveTab("phase")}>Phase 분석</button>
            <button className={tabCls(activeTab === "review")} style={{ fontFamily: "DM Mono, monospace" }} onClick={() => setActiveTab("review")}>취약구간 리뷰</button>
            <button className={tabCls(activeTab === "ai")}     style={{ fontFamily: "DM Mono, monospace" }} onClick={() => setActiveTab("ai")}>AI 피드백</button>
          </div>

          {/* ══ TAB A — Phase 분석 ══ */}
          {activeTab === "phase" && (
            <div className="flex flex-col gap-7">
              <div className="grid grid-cols-2 gap-5">
                <div className="rounded-xl border border-white/8 bg-black/30 overflow-hidden flex flex-col">
                  <div className="px-5 py-4 border-b border-white/8 flex items-center gap-2 text-sm text-white/50" style={{ fontFamily: "DM Mono, monospace" }}>
                    <div className="w-3 h-3 rounded-full bg-[#5b8fff]" /> Phase 분포
                  </div>
                  <div className="p-6 flex-1 flex flex-col justify-center">
                    <MiniBarChart items={phaseCnt.map(p => ({ label: p.phase, pct: (p.cnt / maxCnt) * 100, val: `${p.cnt}f`, color: "rgba(91,143,255,0.6)" }))} />
                  </div>
                </div>
                <div className="rounded-xl border border-white/8 bg-black/30 overflow-hidden flex flex-col">
                  <div className="px-5 py-4 border-b border-white/8 flex items-center gap-2 text-sm text-white/50" style={{ fontFamily: "DM Mono, monospace" }}>
                    <div className="w-3 h-3 rounded-full bg-[#ff6b35]" /> Phase별 평균 점수
                  </div>
                  <div className="p-6 flex-1 flex flex-col justify-center">
                    <MiniBarChart items={phaseAvg.map(p => ({ label: p.phase, pct: p.avg * 100, val: pct(p.avg), color: p.avg < 0.6 ? "rgba(255,107,53,0.7)" : "rgba(200,241,53,0.6)" }))} />
                  </div>
                </div>
              </div>

              {/* DTW 패널 */}
              {dtw != null && dtw_result?.phase_dtw_scores && (
                <div className="rounded-xl border border-[#5b8fff]/20 bg-[#5b8fff]/5 overflow-hidden">
                  <div className="px-5 py-4 border-b border-[#5b8fff]/15 text-l text-[#5b8fff]" style={{ fontFamily: "DM Mono, monospace" }}>
                    ≋ DTW 모범 동작 대비 분석
                  </div>
                  <div className="p-6 flex flex-col gap-6">
                    <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${Object.keys(dtw_result.phase_dtw_scores).length}, 1fr)` }}>
                      {Object.entries(dtw_result.phase_dtw_scores).map(([p, sc]) => (
                        <div key={p} className="rounded-xl border border-white/8 bg-white/5 p-6 flex flex-col gap-3">
                          <div className="text-sm text-white/40" style={{ fontFamily: "DM Mono, monospace" }}>{p}</div>
                          <div className="text-4xl font-extrabold" style={{ color: sc >= 0.7 ? "#5b8fff" : "#ff6b35" }}>{pct(sc)}</div>
                          <div className="text-xs text-white/30" style={{ fontFamily: "DM Mono, monospace" }}>{dtw_result.phase_segment_counts?.[p] ?? 0} segs</div>
                        </div>
                      ))}
                    </div>
                    {dtw_result.worst_joints && (
                      <div className="flex flex-col gap-3">
                        <div className="text-sm text-white/40" style={{ fontFamily: "DM Mono, monospace" }}>주요 차이 관절 (모범 대비)</div>
                        {Object.entries(dtw_result.worst_joints).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([j, v]) => {
                          const maxV = Math.max(...Object.values(dtw_result.worst_joints!));
                          return (
                            <div key={j} className="flex items-center gap-4 rounded-xl bg-white/5 px-5 py-4">
                              <span className="flex-1 text-sm text-white/50" style={{ fontFamily: "DM Mono, monospace" }}>{j}</span>
                              <div className="w-40 h-2.5 bg-white/10 rounded-full overflow-hidden">
                                <div style={{ height: "100%", width: `${(v / maxV) * 100}%`, background: v / maxV > 0.6 ? "#ff6b35" : "#5b8fff", borderRadius: 9999 }} />
                              </div>
                              <span className="text-sm text-white/40 w-12 text-right" style={{ fontFamily: "DM Mono, monospace" }}>{v.toFixed(3)}</span>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ══ TAB B — 취약구간 리뷰 ══ */}
          {activeTab === "review" && (
            <div className="flex flex-col gap-7">
              <div className="grid gap-4" style={{ gridTemplateColumns: "200px 1fr" }}>
                <div className="flex flex-col gap-2">
                  {topErrors.length === 0 && (
                    <div className="text-white/30 text-xs p-4" style={{ fontFamily: "DM Mono, monospace" }}>감지된 오류 없음 ✅</div>
                  )}
                  {topErrors.map(([err, cnt], i) => (
                    <button key={err} onClick={() => setSelErrIdx(i)}
                      className={`rounded-xl p-4 flex items-center gap-3 text-left transition-all w-full cursor-pointer border
                        ${selErrIdx === i ? "bg-[#ff6b35]/8 border-[#ff6b35]/30" : "bg-white/3 border-white/8 hover:border-white/20"}`}>
                      <div className="w-8 h-8 rounded-lg bg-white/5 shrink-0 flex items-center justify-center text-xs">⚠️</div>
                      <div className="flex-1 min-w-0">
                        <div className="text-[9px] text-white/70 mb-1 truncate" style={{ fontFamily: "DM Mono, monospace" }}>{err}</div>
                        <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                          <div style={{ height: "100%", width: `${(cnt / maxErrCnt) * 100}%`, background: "#ff6b35", borderRadius: 9999 }} />
                        </div>
                      </div>
                      <span className="text-[9px] text-white/30 shrink-0" style={{ fontFamily: "DM Mono, monospace" }}>{cnt}회</span>
                    </button>
                  ))}
                </div>

                {selErr && (
                  <div className="flex flex-col gap-3">
                    <div className="flex gap-2 flex-wrap">
                      <Chip color="#c8f135">{selErr.phase}</Chip>
                      {selErr.errors.map((e, i) => <Chip key={i} color="#ff6b35">❌ {e}</Chip>)}
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="rounded-xl border border-white/8 overflow-hidden bg-white/3">
                        <div className="px-3 py-2 border-b border-white/8 flex justify-between text-[9px] text-white/30" style={{ fontFamily: "DM Mono, monospace" }}>
                          <span>Original</span><span>frame #{selErr.frame_idx}</span>
                        </div>
                        {selErr.img_url
                          ? <img src={`${API}${selErr.img_url}`} alt="original" className="w-full block" />
                          : <div className="aspect-[4/3] flex items-center justify-center text-white/20 text-xs">이미지 없음</div>}
                      </div>
                      <div className="rounded-xl border border-[#c8f135]/15 overflow-hidden bg-white/3">
                        <div className="px-3 py-2 border-b border-[#c8f135]/10 flex justify-between text-[9px]" style={{ fontFamily: "DM Mono, monospace" }}>
                          <span className="text-[#c8f135]/60">Skeleton</span><span className="text-[#ff6b35]">오류 강조</span>
                        </div>
                        {selErr.skeleton_url
  ? <img src={`${API}${selErr.skeleton_url}`} alt="skeleton" className="w-full block" />
  : <div className="aspect-[4/3] flex items-center justify-center text-white/20 text-xs">스켈레톤 없음</div>}
                      </div>
                    </div>
                    {selErr.details && (
                      <div className="flex flex-col gap-1">
                        {Object.entries(selErr.details).map(([k, v]) => (
                          <div key={k} className="flex items-center gap-3 rounded-lg bg-white/3 px-3 py-2">
                            <span className="text-xs w-3">{v.status === "ok" ? "✅" : v.status === "warning" ? "⚠️" : "❌"}</span>
                            <span className="flex-1 text-[9px] text-white/40" style={{ fontFamily: "DM Mono, monospace" }}>{k}</span>
                            <span className="text-[9px] text-white/30" style={{ fontFamily: "DM Mono, monospace" }}>{v.value}</span>
                            <div className="w-14 h-1 bg-white/10 rounded-full overflow-hidden">
                              <div style={{ height: "100%", borderRadius: 9999, width: v.status === "ok" ? "80%" : v.status === "warning" ? "50%" : "25%", background: v.status === "ok" ? "#c8f135" : v.status === "warning" ? "#f5a623" : "#ff6b35" }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* 전체구간 리뷰 */}
              <div className="border-t border-white/8 pt-8 flex flex-col gap-5">
                <div className="text-[10px] text-[#c8f135] tracking-widest" style={{ fontFamily: "DM Mono, monospace" }}>🔍 전체구간 리뷰</div>
                <div className="rounded-xl border border-white/8 bg-white/3 p-5 flex flex-col gap-4">
                  <div className="flex justify-between items-center text-[9px] text-white/30" style={{ fontFamily: "DM Mono, monospace" }}>
                    <span>프레임 내비게이터</span>
                    <span className="text-[#c8f135]">frame #{selectedFrameIdx} / {Math.max(totalFrames - 1, 0)}</span>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => jumpBy(-1)}
                      disabled={!canGoPrev}
                      className={`text-[10px] rounded-md px-3 py-1.5 border ${canGoPrev ? "border-white/20 text-white/70 hover:border-white/40" : "border-white/10 text-white/20 cursor-not-allowed"}`}
                      style={{ fontFamily: "DM Mono, monospace" }}
                    >
                      ← 이전
                    </button>
                    <button
                      type="button"
                      onClick={() => jumpBy(1)}
                      disabled={!canGoNext}
                      className={`text-[10px] rounded-md px-3 py-1.5 border ${canGoNext ? "border-white/20 text-white/70 hover:border-white/40" : "border-white/10 text-white/20 cursor-not-allowed"}`}
                      style={{ fontFamily: "DM Mono, monospace" }}
                    >
                      다음 →
                    </button>
                  </div>

                  <input
                    type="range"
                    min={0}
                    max={maxVisibleCursor}
                    value={safeFrameCursor}
                    onChange={e => jumpToCursor(+e.target.value)}
                    className="w-full accent-[#c8f135] cursor-pointer"
                  />
                  <div className="flex justify-between text-[8px] text-white/30" style={{ fontFamily: "DM Mono, monospace" }}>
                    <span>0</span>
                    <span>{Math.max(totalFrames - 1, 0)}</span>
                  </div>

                  <div className="relative">
                    <div className="h-6 bg-white/5 rounded-lg overflow-hidden relative">
                      {visibleFrameIndices.map((frameNo, i) => {
                        const frameScore = scoreByFrame.get(frameNo);
                        const included = selectedFrameSet.has(frameNo);
                        const frameColor = frameScore
                          ? (PHASE_COLOR[frameScore.phase] ?? "#444").replace("0.7", "0.5").replace("0.55", "0.4")
                          : "rgba(255,255,255,0.08)";
                        const background = included ? frameColor : "rgba(255,107,53,0.25)";
                        const width = visibleFrameIndices.length > 0 ? `${(1 / visibleFrameIndices.length) * 100}%` : "0%";
                        const left = visibleFrameIndices.length > 0 ? `${(i / visibleFrameIndices.length) * 100}%` : "0%";
                        return (
                          <div
                            key={frameNo}
                            style={{ position: "absolute", top: 0, left, width, height: "100%", background }}
                          />
                        );
                      })}
                      {visibleFrameIndices.length > 0 && (
                        <div
                          style={{
                            position: "absolute",
                            top: 0,
                            left: `${(safeFrameCursor / Math.max(visibleFrameIndices.length - 1, 1)) * 100}%`,
                            transform: "translateX(-1px)",
                            width: 2,
                            height: "100%",
                            background: "#c8f135",
                          }}
                        />
                      )}
                    </div>
                  </div>
                  <div className="flex gap-4 flex-wrap">
                    {Object.entries(PHASE_COLOR).filter(([p]) => byPhase[p]).map(([p, c]) => (
                      <div key={p} className="flex items-center gap-1 text-[8px] text-white/30" style={{ fontFamily: "DM Mono, monospace" }}>
                        <div style={{ width: 8, height: 8, borderRadius: 2, background: c }} />{p}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[9px] text-white/30" style={{ fontFamily: "DM Mono, monospace" }}>구간 선택</span>
                  {[ALL_PHASE, ...Object.keys(byPhase)].map(p => (
                    <button key={p} onClick={() => { setSelPhase(p); setFrameIdx(0); }}
                      className={`text-[9px] rounded-full px-3 py-1 border transition-all cursor-pointer
                        ${selPhase === p ? "bg-[#c8f135]/10 border-[#c8f135]/40 text-[#c8f135]" : "bg-transparent border-white/10 text-white/30 hover:border-white/30"}`}
                      style={{ fontFamily: "DM Mono, monospace" }}>
                      {p === ALL_PHASE ? "전체" : p}
                    </button>
                  ))}
                </div>

                <div className="flex items-center gap-3 rounded-xl border border-white/8 bg-white/3 px-4 py-3">
                  <Chip color={selFrame ? "#c8f135" : "#999"}>
                    {selFrame?.phase ?? "평가 없음"}
                  </Chip>
                  {selFrame ? (
                    <Chip color={selFrame.score < 0.7 ? "#ff6b35" : "#5b8fff"}>자세 점수: {pct(selFrame.score)}</Chip>
                  ) : (
                    <Chip color="#999">점수 없음</Chip>
                  )}
                  <Chip color={selFrameEvaluated ? "#5b8fff" : "#ff6b35"}>
                    {selFrameEvaluated ? "평가 포함" : "평가 제외"}
                  </Chip>
                  <span className="flex-1" />
                  <span className="text-[9px] text-white/20" style={{ fontFamily: "DM Mono, monospace" }}>
                    frame #{selectedFrameIdx}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {[
                    { label: "사용자 원본", sub: `frame #${selectedFrameIdx}`, borderCls: "border-white/8", headerCls: "text-white/30", imgFilter: undefined, imgUrl: selFrameImageUrl, placeholder: "원본 이미지" },
                    { label: "스켈레톤 오버레이", sub: "오류 강조", borderCls: "border-[#c8f135]/15", headerCls: "text-[#c8f135]/50", imgUrl: selFrame?.skeleton_url, placeholder: "스켈레톤" },
                  ].map(col => (
                    <div key={col.label} className={`rounded-xl border ${col.borderCls} overflow-hidden`}>
                      <div className={`px-3 py-2 border-b ${col.borderCls} flex justify-between text-[9px] ${col.headerCls} bg-black/20`} style={{ fontFamily: "DM Mono, monospace" }}>
                        <span>{col.label}</span><span className="text-white/20">{col.sub}</span>
                      </div>
                      <div className="bg-black flex items-center justify-center">
                        {col.imgUrl
                          ? <img src={`${API}${col.imgUrl}`} alt={col.label} className="w-full h-auto object-contain" style={col.imgFilter ? { filter: col.imgFilter } : undefined} />
                          : <span className="text-[9px] text-white/20" style={{ fontFamily: "DM Mono, monospace" }}>{col.placeholder}</span>}
                      </div>
                    </div>
                  ))}
                </div>

                {selFrame && (selFrame.errors ?? []).length > 0 && (
                  <div className="flex flex-col gap-2">
                    {selFrame.errors.map((e, i) => (
                      <div key={i} className="rounded-lg bg-[#ff6b35]/8 border border-[#ff6b35]/20 px-4 py-2 text-[10px] text-[#ff6b35]" style={{ fontFamily: "DM Mono, monospace" }}>⚠ {e}</div>
                    ))}
                  </div>
                )}
                {selFrame && (selFrame.errors ?? []).length === 0 && (
                  <div className="rounded-lg bg-[#c8f135]/5 border border-[#c8f135]/15 px-4 py-2 text-[10px] text-[#c8f135]/70" style={{ fontFamily: "DM Mono, monospace" }}>✅ 감지된 자세 오류 없음</div>
                )}
                {!selFrame && (
                  <div className="rounded-lg bg-white/5 border border-white/10 px-4 py-2 text-[10px] text-white/60" style={{ fontFamily: "DM Mono, monospace" }}>
                    {selFrameIncluded
                      ? "필터에는 포함되었지만 평가 대상 phase가 아니라 점수가 없습니다."
                      : "이 프레임은 평가에서 제외되었습니다. (필터링된 휴식/비활성 구간)"}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ══ TAB C — AI 피드백 ══ */}
          {activeTab === "ai" && (
            <div className="flex flex-col gap-7">
              <div className="rounded-xl border border-white/8 bg-white/3 p-6">
                <div className="text-[9px] text-white/30 mb-3" style={{ fontFamily: "DM Mono, monospace" }}>// GEMINI API KEY</div>
                <div className="flex gap-3">
                  <input type="password" value={geminiKey} onChange={e => setGeminiKey(e.target.value)} placeholder="AIza..."
                    className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-sm text-white placeholder-white/20 outline-none focus:border-[#c8f135]/40 transition-colors" />
                  <Button disabled={fbLoading} onClick={genFeedback} className="bg-[#c8f135] text-black hover:bg-[#b4da30] px-6 text-sm whitespace-nowrap">
                    {fbLoading ? "생성 중..." : "AI 피드백 생성"}
                  </Button>
                </div>
                {fbError && <div className="mt-3 text-[10px] text-[#ff6b35]" style={{ fontFamily: "DM Mono, monospace" }}>⚠ {fbError}</div>}
              </div>
              {feedback ? (
                <div className="rounded-xl border border-[#c8f135]/15 bg-gradient-to-br from-[#c8f135]/4 to-[#5b8fff]/4 p-8">
                  <div className="text-[9px] text-[#c8f135]/60 mb-4 tracking-widest" style={{ fontFamily: "DM Mono, monospace" }}>🤖 AI 트레이너 종합 피드백</div>
                  <div className="text-sm text-white/60 leading-relaxed whitespace-pre-wrap" style={{ fontFamily: "Inter, sans-serif", fontWeight: 300 }}>{feedback}</div>
                </div>
              ) : (
                <div className="text-center py-16 text-[10px] text-white/20" style={{ fontFamily: "DM Mono, monospace" }}>
                  Gemini API Key를 입력하고 AI 피드백을 생성하세요
                </div>
              )}
            </div>
          )}

          {/* ── 하단 액션 ── */}
          <div className="flex gap-3 mt-10 pt-8 border-t border-white/8">
            <Button className="flex-1 bg-[#c8f135] text-black hover:bg-[#b4da30] text-xs" disabled={reportLoading} onClick={downloadReport}>
              {reportLoading ? "리포트 생성 중..." : "⬇️ 피드백 리포트 받기"}
            </Button>
            <Button variant="outline" size="sm" className="border-white/10 text-white/40 hover:text-white text-xs" onClick={exportJson}>↓ JSON</Button>
            <Button variant="outline" size="sm" className="border-white/10 text-white/40 hover:text-white text-xs" onClick={() => navigate("/select-exercise")}>🔄 새 분석</Button>
            <Button variant="outline" size="sm" className="border-white/10 text-white/40 hover:text-white text-xs" onClick={() => navigate("/")}>🏠 홈</Button>
          </div>

        </div>
      </div>
    </div>
  );
}
