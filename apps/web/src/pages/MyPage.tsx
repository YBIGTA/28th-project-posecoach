import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Dumbbell } from "lucide-react";
import { Button } from "../components/ui/button";
import { fetchUserStats, fetchUserWorkouts, type UserStats, type WorkoutRecord } from "../lib/api";
import { clearSession, getSession } from "../lib/auth";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend
} from "recharts";

const PHASE_LABEL: Record<string, string> = {
  ready: "준비", top: "상단", descending: "하강", bottom: "하단", ascending: "상승",
};

const toPercent = (v: number | null | undefined) =>
  typeof v === "number" && !Number.isNaN(v) ? `${Math.round(v * 100)}%` : "-";

const toPhaseLabel = (p?: string) => (p ? (PHASE_LABEL[p] ?? p) : "-");

const formatDateFull = (v: string) => {
  const d = new Date(v.replace(" ", "T"));
  if (Number.isNaN(d.getTime())) return v;
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
};

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload || payload.length === 0) return null;
  const data = payload[0].payload;

  return (
    <div style={{
      background: "rgba(10,10,10,0.95)",
      padding: "10px 14px",
      borderRadius: 12,
      border: "1px solid rgba(255,255,255,0.1)",
      fontFamily: "DM Mono, monospace",
      fontSize: 12,
      color: "white",
    }}>
      <div style={{ color: "rgba(255,255,255,0.6)", marginBottom: "4px" }}>
        {data.fullDate}
      </div>
      {data.pushup !== null && (
        <div style={{ color: "#c8f135", marginTop: "2px" }}>
          푸시업 : {data.pushup}점
        </div>
      )}
      {data.pullup !== null && (
        <div style={{ color: "#5b8fff", marginTop: "2px" }}>
          풀업 : {data.pullup}점
        </div>
      )}
    </div>
  );
};

type Tab = "profile" | "history";

const gradeColor = (g: string) => {
  if (g.includes("S")) return "#c8f135";
  if (g.includes("A")) return "#5b8fff";
  if (g.includes("B")) return "rgba(255,255,255,0.5)";
  return "#ff6b35";
};

export function MyPage() {
  const navigate = useNavigate();
  const session = useMemo(() => getSession(), []);
  const [tab, setTab] = useState<Tab>("profile");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<UserStats | null>(null);
  const [workouts, setWorkouts] = useState<WorkoutRecord[]>([]);

  const [visibleCount, setVisibleCount] = useState(8);
  const [modalWorkout, setModalWorkout] = useState<WorkoutRecord | null>(null);

  // ✅ 차트 관련 상태 (운동 종류 & X축 모드)
  const [selectedExercise, setSelectedExercise] = useState<"all" | "pushup" | "pullup">("all");
  const [xMode, setXMode] = useState<"date" | "time">("date");

  useEffect(() => {
    if (!session) { setLoading(false); return; }
    let mounted = true;
    Promise.all([fetchUserStats(session.user_id), fetchUserWorkouts(session.user_id)])
      .then(([s, w]) => { if (!mounted) return; setStats(s); setWorkouts(w); })
      .catch(e => { if (!mounted) return; setError(e instanceof Error ? e.message : "오류 발생"); })
      .finally(() => { if (!mounted) return; setLoading(false); });
    return () => { mounted = false; };
  }, [session]);

  if (!session) {
    return (
      <div className="min-h-screen w-full bg-[#0a0a0a] text-white flex items-center justify-center px-6 py-10">
        <div className="w-full max-w-[560px] rounded-[30px] border border-white/10 bg-[#0f1116]/80 backdrop-blur-xl shadow-[0_30px_80px_rgba(0,0,0,0.6)] p-16 text-center">
          <div className="text-3xl font-extrabold mb-4">로그인이 필요합니다</div>
          <div className="text-white/30 text-base mb-10" style={{ fontFamily: "DM Mono, monospace" }}>
            마이페이지는 로그인 후 이용 가능합니다
          </div>
          <div className="flex gap-4 justify-center">
            <Button variant="outline" className="border-white/10 text-white/60 hover:text-white text-base px-6 py-6" onClick={() => navigate("/")}>
              홈으로
            </Button>
            <Button className="bg-[#c8f135] text-black hover:bg-[#b4da30] text-base px-6 py-6" onClick={() => navigate("/login")}>
              로그인하기
            </Button>
          </div>
        </div>
      </div>
    );
  }

  const initials = session.username.slice(0, 2).toUpperCase();

  const tabCls = (on: boolean) =>
    `px-7 py-4 text-[17px] tracking-wide border-b-2 transition-all cursor-pointer bg-transparent border-0 outline-none
    ${on ? "text-[#c8f135] border-[#c8f135]" : "text-white/35 border-transparent hover:text-white/70"}`;

  const smallActionBtn = "text-sm px-6 py-6";
  const smallActionBtnOutline = "text-sm px-6 py-6";

  // ✅ 개선된 추이 데이터 생성 로직
  const trendData = useMemo(() => {
    if (!workouts || workouts.length === 0) return [];

    const grouped: Record<string, any> = {};
    const sortedWorkouts = [...workouts].reverse();

    sortedWorkouts.forEach(w => {
      const d = new Date(w.created_at.replace(" ", "T"));
      
      // 1. 내부 그룹핑을 위한 안전한 고유 키 생성 (다른 날짜의 같은 시간이 겹치지 않도록)
      const dateStr = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
      const timeStr = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
      
      const groupKey = xMode === "date" ? dateStr : `${dateStr} ${timeStr}`;

      // 2. x축에 실제로 보여줄 라벨 포맷 설정
      const displayLabel = xMode === "date"
        ? `${String(d.getMonth() + 1).padStart(2, "0")}/${String(d.getDate()).padStart(2, "0")}`
        : timeStr;

      if (!grouped[groupKey]) {
        grouped[groupKey] = {
          date: displayLabel,
          fullDate: w.created_at, // 툴팁에서 정확한 시간 확인용
          pushup: null,
          pullup: null,
        };
      }

      // 같은 그룹핑 조건 안에서 덮어쓰거나 추가
      if (w.exercise_type === "푸시업") {
        grouped[groupKey].pushup = Math.round((w.avg_score ?? 0) * 100);
      }
      if (w.exercise_type === "풀업") {
        grouped[groupKey].pullup = Math.round((w.avg_score ?? 0) * 100);
      }
    });

    return Object.values(grouped).slice(-12);
  }, [workouts, xMode]);

  const visibleWorkouts = useMemo(() => workouts.slice(0, visibleCount), [workouts, visibleCount]);

  return (
    <>
      <div className="min-h-screen w-full bg-[#0a0a0a] text-white px-6 py-10">
        <div className="w-full max-w-[1400px] mx-auto rounded-[30px] border border-white/10 bg-[#0f1116]/80 backdrop-blur-xl shadow-[0_30px_80px_rgba(0,0,0,0.6)] overflow-hidden">

          <header className="flex items-center justify-between px-10 py-7 border-b border-white/10 bg-black/40">
            <button
              onClick={() => navigate("/")}
              className="flex items-center gap-3 font-extrabold tracking-widest text-white text-lg"
            >
              <Dumbbell className="w-6 h-6 text-[#c8f135]" />
              POSECOACH
            </button>
            <div className="flex items-center gap-3">
              <Button variant="outline" size="sm" className="h-10 px-4 border-white/10 text-white/70 hover:text-[#c8f135] hover:border-[#c8f135]/40 hover:bg-[#c8f135]/10 text-sm" onClick={() => navigate("/")}>
                ← 홈
              </Button>
              <div className="h-10 px-4 rounded-full border border-[#c8f135]/30 bg-[#c8f135]/10 text-[#c8f135] flex items-center justify-center text-sm tracking-wider">
                {session.username}
              </div>
            </div>
          </header>

          <div className="px-10 py-10">
            <div className="flex items-center gap-7 pb-9 mb-9 border-b border-white/8">
              <div className="w-20 h-20 rounded-full bg-[#c8f135]/10 border-2 border-[#c8f135]/40 flex items-center justify-center text-3xl font-extrabold text-[#c8f135] shrink-0">
                {initials}
              </div>

              <div className="flex-1">
                <div className="text-[34px] font-extrabold mb-2">{session.username}</div>
                <div className="text-white/30 text-[14px] mb-4" style={{ fontFamily: "DM Mono, monospace" }}>
                  마지막 운동: {workouts[0] ? formatDateFull(workouts[0].created_at).slice(0, 10) : "기록 없음"}
                </div>

                <div className="flex gap-2 flex-wrap">
                  {[
                    "🏋️ PoseCoach 멤버",
                    ...(stats && stats.total_workouts >= 5 ? ["🏆 꾸준한 운동가"] : []),
                    ...(stats && stats.total_workouts >= 10 ? ["💪 분석 마스터"] : []),
                  ].map(b => (
                    <span key={b} className="text-[14px] px-4 py-2 rounded-full bg-[#c8f135]/8 border border-[#c8f135]/20 text-[#c8f135]/80" style={{ fontFamily: "DM Mono, monospace" }}>
                      {b}
                    </span>
                  ))}
                </div>
              </div>

              <Button variant="outline" size="sm" className="border-red-500/20 text-red-400/70 hover:text-red-300 hover:border-red-300/40 text-sm h-10 px-4" onClick={() => { clearSession(); navigate("/"); }}>
                로그아웃
              </Button>
            </div>

            {!loading && stats && (
              <div className="grid grid-cols-4 gap-5 mb-10">
                {[
                  { label: "총 세션", val: stats.total_workouts, sub: "회 분석 완료" },
                  { label: "평균 점수", val: toPercent(stats.overall_avg_score), sub: "전체 평균" },
                  { label: "총 운동 수", val: stats.total_reps, sub: "회 누적" },
                  { label: "최다 운동", val: stats.favorite_exercise || "-", sub: "가장 많이 분석" },
                ].map(s => (
                  <div key={s.label} className="rounded-2xl border border-white/10 bg-white/5 p-7 flex flex-col gap-3">
                    <span className="text-[13px] text-white/35 uppercase tracking-wider" style={{ fontFamily: "DM Mono, monospace" }}>{s.label}</span>
                    <span className="text-4xl font-extrabold text-[#c8f135] leading-none">{s.val}</span>
                    <span className="text-[13px] text-white/25" style={{ fontFamily: "DM Mono, monospace" }}>{s.sub}</span>
                  </div>
                ))}
              </div>
            )}

            <div className="flex border-b border-white/10 mb-10">
              <button className={tabCls(tab === "profile")} style={{ fontFamily: "DM Mono, monospace" }} onClick={() => setTab("profile")}>프로필</button>
              <button className={tabCls(tab === "history")} style={{ fontFamily: "DM Mono, monospace" }} onClick={() => setTab("history")}>운동 기록</button>
            </div>

            {loading && <div className="text-center py-20 text-white/20 text-2xl" style={{ fontFamily: "DM Mono, monospace" }}>데이터를 불러오는 중...</div>}
            {error && <div className="rounded-2xl bg-[#ff6b35]/8 border border-[#ff6b35]/20 px-6 py-5 text-[#ff6b35] text-lg mb-6" style={{ fontFamily: "DM Mono, monospace" }}>⚠ {error}</div>}

            {/* ═══ TAB: 프로필 ═══ */}
            {!loading && tab === "profile" && (
              <div className="grid grid-cols-2 gap-10">
                
                <div className="col-span-2">
                  <div className="flex items-center justify-between mb-5 pb-4 border-b border-white/8">
                    <div className="text-[14px] text-white/35 uppercase tracking-widest" style={{ fontFamily: "DM Mono, monospace" }}>
                      운동별 분석 추이
                    </div>
                    
                    {trendData.length > 0 && (
                      <div className="flex items-center gap-4">
                        {/* ✅ X축 토글 버튼 */}
                        <div className="flex gap-2">
                          {[
                            { key: "date", label: "날짜" },
                            { key: "time", label: "시간" },
                          ].map(btn => (
                            <button
                              key={btn.key}
                              onClick={() => setXMode(btn.key as any)}
                              className={`px-4 py-1.5 text-xs rounded-full border transition-all
                                ${xMode === btn.key ? "bg-[#5b8fff] text-white border-[#5b8fff] font-bold" : "border-white/20 text-white/60 hover:text-white hover:bg-white/5"}`}
                            >
                              {btn.label}
                            </button>
                          ))}
                        </div>

                        <div className="w-px h-5 bg-white/20" />

                        {/* ✅ 종목 토글 버튼 */}
                        <div className="flex gap-2">
                          {[
                            { key: "all", label: "전체" },
                            { key: "pushup", label: "푸시업" },
                            { key: "pullup", label: "풀업" },
                          ].map(btn => (
                            <button
                              key={btn.key}
                              onClick={() => setSelectedExercise(btn.key as any)}
                              className={`px-4 py-1.5 text-xs rounded-full border transition-all
                                ${selectedExercise === btn.key ? "bg-[#c8f135] text-black border-[#c8f135] font-bold" : "border-white/20 text-white/60 hover:text-white hover:bg-white/5"}`}
                            >
                              {btn.label}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {trendData.length === 0 ? (
                    <div className="text-white/20 text-2xl py-6" style={{ fontFamily: "DM Mono, monospace" }}>
                      운동 기록이 없습니다
                    </div>
                  ) : (
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-6 h-[260px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={trendData}>
                          <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                          <XAxis dataKey="date" stroke="rgba(255,255,255,0.35)" />
                          <YAxis stroke="rgba(255,255,255,0.35)" domain={[0, 100]} />
                          <Tooltip content={<CustomTooltip />} />
                          <Legend wrapperStyle={{ fontSize: "12px", fontFamily: "DM Mono, monospace", paddingTop: "10px" }} />

                          {(selectedExercise === "all" || selectedExercise === "pushup") && (
                            <Line
                              name="푸시업"
                              type="monotone"
                              dataKey="pushup"
                              stroke="#c8f135"
                              strokeWidth={3}
                              dot={{ r: 4 }}
                              activeDot={{ r: 6 }}
                              connectNulls={true}
                            />
                          )}

                          {(selectedExercise === "all" || selectedExercise === "pullup") && (
                            <Line
                              name="풀업"
                              type="monotone"
                              dataKey="pullup"
                              stroke="#5b8fff"
                              strokeWidth={3}
                              dot={{ r: 4 }}
                              activeDot={{ r: 6 }}
                              connectNulls={true}
                            />
                          )}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                <div>
                  <div className="text-[14px] text-white/35 uppercase tracking-widest mb-5 pb-4 border-b border-white/8" style={{ fontFamily: "DM Mono, monospace" }}>
                    계정 설정
                  </div>
                  <div className="flex flex-col gap-3">
                    {[["아이디", session.username], ["비밀번호", "••••••••"], ["계정 유형", "일반 회원"]].map(([k, v]) => (
                      <div key={k} className="rounded-2xl border border-white/8 bg-white/3 px-6 py-5 flex items-center justify-between">
                        <span className="text-[16px] text-white/45" style={{ fontFamily: "DM Mono, monospace" }}>{k}</span>
                        <span className={`text-[16px] ${k === "계정 유형" ? "text-[#c8f135]/80" : "text-white/70"}`} style={{ fontFamily: "DM Mono, monospace" }}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="text-[14px] text-white/35 uppercase tracking-widest mb-5 pb-4 border-b border-white/8" style={{ fontFamily: "DM Mono, monospace" }}>
                    운동별 누적 기록
                  </div>
                  {workouts.length > 0 ? (() => {
                    const cnt: Record<string, number> = {};
                    const reps: Record<string, number> = {};
                    workouts.forEach(w => {
                      cnt[w.exercise_type] = (cnt[w.exercise_type] || 0) + 1;
                      reps[w.exercise_type] = (reps[w.exercise_type] || 0) + w.exercise_count;
                    });
                    const maxCnt = Math.max(...Object.values(cnt));
                    return (
                      <div className="flex flex-col gap-3">
                        {Object.entries(cnt).map(([ex, c]) => (
                          <div key={ex} className="rounded-2xl border border-white/8 bg-white/3 px-6 py-5 flex items-center gap-4">
                            <span className="text-[16px] text-[#c8f135] w-16 shrink-0" style={{ fontFamily: "DM Mono, monospace" }}>{ex}</span>
                            <div className="flex-1 h-2.5 bg-white/10 rounded-full overflow-hidden">
                              <div style={{ height: "100%", width: `${(c / maxCnt) * 100}%`, background: "rgba(200,241,53,0.6)", borderRadius: 9999 }} />
                            </div>
                            <span className="text-[15px] text-white/35 whitespace-nowrap" style={{ fontFamily: "DM Mono, monospace" }}>
                              {c}세션 · {reps[ex]}회
                            </span>
                          </div>
                        ))}
                      </div>
                    );
                  })() : (
                    <div className="text-white/20 text-2xl py-6" style={{ fontFamily: "DM Mono, monospace" }}>운동 기록이 없습니다</div>
                  )}
                </div>

                <div className="col-span-2 flex gap-3 pt-6 border-t border-white/8">
                  <Button className={`bg-[#c8f135] text-black hover:bg-[#b4da30] ${smallActionBtn}`} onClick={() => navigate("/select-exercise")}>새 운동 시작</Button>
                  <Button variant="outline" className={`border-white/10 text-white/60 hover:text-white ${smallActionBtnOutline}`} onClick={() => navigate("/")}>홈으로</Button>
                </div>
              </div>
            )}

            {/* ═══ TAB: 운동 기록 ═══ */}
            {!loading && tab === "history" && (
              <div>
                {workouts.length === 0 ? (
                  <div className="text-center py-24">
                    <div className="text-6xl mb-5">📭</div>
                    <div className="text-white/25 text-base mb-8" style={{ fontFamily: "DM Mono, monospace" }}>아직 운동 기록이 없습니다</div>
                    <Button className="bg-[#c8f135] text-black hover:bg-[#b4da30] text-base px-8 py-6" onClick={() => navigate("/select-exercise")}>운동 시작하기 →</Button>
                  </div>
                ) : (
                  <div className="flex flex-col gap-8">
                    <div className="grid grid-cols-4 gap-4">
                      {visibleWorkouts.map(w => {
                        const gc = gradeColor(w.grade);
                        return (
                          <button
                            key={w.id}
                            onClick={() => setModalWorkout(w)}
                            className="relative aspect-square rounded-2xl border bg-white/5 p-6 text-left overflow-hidden transition-all border-white/10 hover:bg-white/10 hover:border-white/20"
                          >
                            <div className="absolute top-4 right-4 px-3 py-1 text-sm font-extrabold rounded-lg border" style={{ color: gc, background: `${gc}18`, borderColor: `${gc}50` }}>{w.grade}</div>
                            <div className="h-full w-full flex flex-col justify-center items-center text-center gap-2">
                              <div className="text-[#c8f135] text-[15px]" style={{ fontFamily: "DM Mono, monospace" }}>{w.exercise_type}{w.grip_type ? ` · ${w.grip_type}` : ""}</div>
                              <div className="text-4xl font-extrabold">{w.exercise_count}회</div>
                              <div className="text-white/40 text-[13px]" style={{ fontFamily: "DM Mono, monospace" }}>평균 {toPercent(w.avg_score)}</div>
                              <div className="text-white/25 text-[12px]" style={{ fontFamily: "DM Mono, monospace" }}>{formatDateFull(w.created_at)}</div>
                            </div>
                            <div className="absolute inset-0 bg-black/60 opacity-0 hover:opacity-100 transition-all flex items-center justify-center text-white text-sm tracking-widest">자세히 보기</div>
                          </button>
                        );
                      })}
                    </div>
                    {visibleCount < workouts.length && (
                      <div className="flex justify-center">
                        <Button variant="outline" className="border-white/10 text-white/60 hover:text-white" onClick={() => setVisibleCount(v => v + 8)}>더보기</Button>
                      </div>
                    )}
                    <div className="flex gap-3 pt-6 border-t border-white/8">
                      <Button className={`bg-[#c8f135] text-black hover:bg-[#b4da30] ${smallActionBtn}`} onClick={() => navigate("/select-exercise")}>새 운동 시작</Button>
                      <Button variant="outline" className={`border-white/10 text-white/60 hover:text-white ${smallActionBtnOutline}`} onClick={() => navigate("/")}>홈으로</Button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {modalWorkout && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50" onClick={() => setModalWorkout(null)}>
          <div className="bg-[#111] w-[900px] max-h-[90vh] overflow-y-auto rounded-2xl p-8 border border-white/10 relative" onClick={(e) => e.stopPropagation()}>
            <button onClick={() => setModalWorkout(null)} className="absolute top-4 right-4 text-white/40 hover:text-white text-xl">✕</button>
            <div className="text-2xl font-extrabold mb-6">{modalWorkout.exercise_type}{modalWorkout.grip_type ? ` · ${modalWorkout.grip_type}` : ""}</div>
            
            <div className="grid grid-cols-3 gap-6 mb-8 text-center">
              <div className="bg-white/5 p-6 rounded-xl">
                <div className="text-white/40 text-sm mb-2">평균 점수</div>
                <div className="text-3xl font-bold text-[#c8f135]">{toPercent(modalWorkout.avg_score)}</div>
              </div>
              <div className="bg-white/5 p-6 rounded-xl">
                <div className="text-white/40 text-sm mb-2">DTW 점수</div>
                <div className="text-3xl font-bold text-[#5b8fff]">{modalWorkout.dtw_score ? Math.round(modalWorkout.dtw_score * 100) : "-"}</div>
              </div>
              <div className="bg-white/5 p-6 rounded-xl border border-[#c8f135]/40">
                <div className="text-white/40 text-sm mb-2">Combined Score</div>
                <div className="text-4xl font-extrabold text-[#c8f135]">{modalWorkout.combined_score ? Math.round(modalWorkout.combined_score * 100) : "-"}</div>
              </div>
            </div>

            {modalWorkout.errors && modalWorkout.errors.length > 0 && (
              <div className="mb-8">
                <div className="text-white/40 text-sm mb-4">주요 오류</div>
                <div className="flex flex-col gap-3">
                  {modalWorkout.errors.map((e, i) => (
                    <div key={i} className="bg-[#ff6b35]/10 border border-[#ff6b35]/30 px-5 py-3 rounded-xl text-[#ff6b35]">
                      ⚠ {e.error_msg} ({e.count}회)
                    </div>
                  ))}
                </div>
              </div>
            )}

            {modalWorkout.phase_scores && modalWorkout.phase_scores.length > 0 && (
              <div>
                <div className="text-white/40 text-sm mb-4">Phase별 점수</div>
                <div className="flex gap-4 flex-wrap">
                  {modalWorkout.phase_scores.map((p, i) => (
                    <div key={i} className="bg-white/5 border border-white/10 px-6 py-4 rounded-xl text-center">
                      <div className="text-white/40 text-xs mb-1">{toPhaseLabel(p.phase)}</div>
                      <div className="text-xl font-bold text-[#c8f135]">{toPercent(p.avg_score)}</div>
                      <div className="text-white/30 text-xs">{p.frame_count} frames</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}