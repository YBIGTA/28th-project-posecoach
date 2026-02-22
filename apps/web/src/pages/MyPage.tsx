import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Dumbbell } from "lucide-react";
import { Button } from "../components/ui/button";
import { fetchUserStats, fetchUserWorkouts, type UserStats, type WorkoutRecord } from "../lib/api";
import { clearSession, getSession } from "../lib/auth";

const PHASE_LABEL: Record<string, string> = {
  ready: "준비", top: "상단", descending: "하강", bottom: "하단", ascending: "상승",
};
const toPercent = (v: number | null | undefined) =>
  typeof v === "number" && !Number.isNaN(v) ? `${Math.round(v * 100)}%` : "-";
const toPhaseLabel = (p?: string) => (p ? (PHASE_LABEL[p] ?? p) : "-");
const formatDate = (v: string) => {
  const d = new Date(v.replace(" ", "T"));
  if (Number.isNaN(d.getTime())) return v;
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
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
  const [expandedId, setExpandedId] = useState<number | null>(null);

  useEffect(() => {
    if (!session) { setLoading(false); return; }
    let mounted = true;
    Promise.all([fetchUserStats(session.user_id), fetchUserWorkouts(session.user_id)])
      .then(([s, w]) => { if (!mounted) return; setStats(s); setWorkouts(w); })
      .catch(e => { if (!mounted) return; setError(e instanceof Error ? e.message : "오류 발생"); })
      .finally(() => { if (!mounted) return; setLoading(false); });
    return () => { mounted = false; };
  }, [session]);

  // 미로그인
  if (!session) {
    return (
      <div className="min-h-screen w-full bg-[#0a0a0a] text-white flex items-center justify-center px-6 py-10">
        <div className="w-full max-w-[560px] rounded-[30px] border border-white/10 bg-[#0f1116]/80 backdrop-blur-xl shadow-[0_30px_80px_rgba(0,0,0,0.6)] p-16 text-center">
          <div className="text-3xl font-extrabold mb-4">로그인이 필요합니다</div>
          <div className="text-white/30 text-base mb-10" style={{ fontFamily: "DM Mono, monospace" }}>
            마이페이지는 로그인 후 이용 가능합니다
          </div>
          <div className="flex gap-4 justify-center">
            <Button
              variant="outline"
              className="border-white/10 text-white/60 hover:text-white text-base px-6 py-6"
              onClick={() => navigate("/")}
            >
              홈으로
            </Button>
            <Button
              className="bg-[#c8f135] text-black hover:bg-[#b4da30] text-base px-6 py-6"
              onClick={() => navigate("/login")}
            >
              로그인하기
            </Button>
          </div>
        </div>
      </div>
    );
  }

  const initials = session.username.slice(0, 2).toUpperCase();

  // ✅ 탭 텍스트(프로필/운동 기록) 조금 더 크게
  const tabCls = (on: boolean) =>
    `px-7 py-4 text-[17px] tracking-wide border-b-2 transition-all cursor-pointer bg-transparent border-0 outline-none
    ${on ? "text-[#c8f135] border-[#c8f135]" : "text-white/35 border-transparent hover:text-white/70"}`;

  // ✅ 하단 버튼은 글자만 조금 작게
  const smallActionBtn = "text-sm px-6 py-6";
  const smallActionBtnOutline = "text-sm px-6 py-6";

  return (
    <div className="min-h-screen w-full bg-[#0a0a0a] text-white px-6 py-10">
      <div className="w-full max-w-[1400px] mx-auto rounded-[30px] border border-white/10 bg-[#0f1116]/80 backdrop-blur-xl shadow-[0_30px_80px_rgba(0,0,0,0.6)] overflow-hidden">

        {/* ── HEADER ── */}
        <header className="flex items-center justify-between px-10 py-7 border-b border-white/10 bg-black/40">
          <button
            onClick={() => navigate("/")}
            className="flex items-center gap-3 font-extrabold tracking-widest text-white text-lg"
          >
            <Dumbbell className="w-6 h-6 text-[#c8f135]" />
            POSECOACH
          </button>
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              className="h-10 px-4 border-white/10 text-white/70 hover:text-[#c8f135] hover:border-[#c8f135]/40 hover:bg-[#c8f135]/10 text-sm"
              onClick={() => navigate("/")}
            >
              ← 홈
            </Button>
            <div className="h-10 px-4 rounded-full border border-[#c8f135]/30 bg-[#c8f135]/10 text-[#c8f135] flex items-center justify-center text-sm tracking-wider">
              {session.username}
            </div>
          </div>
        </header>

        {/* ── CONTENT ── */}
        <div className="px-10 py-10">

          {/* 프로필 헤더 */}
          <div className="flex items-center gap-7 pb-9 mb-9 border-b border-white/8">
            <div className="w-20 h-20 rounded-full bg-[#c8f135]/10 border-2 border-[#c8f135]/40 flex items-center justify-center text-3xl font-extrabold text-[#c8f135] shrink-0">
              {initials}
            </div>

            <div className="flex-1">
              {/* ✅ 프로필 이름 조금 더 크게 */}
              <div className="text-[34px] font-extrabold mb-2">{session.username}</div>
              <div className="text-white/30 text-[14px] mb-4" style={{ fontFamily: "DM Mono, monospace" }}>
                // 마지막 운동: {workouts[0] ? formatDate(workouts[0].created_at).slice(0, 10) : "기록 없음"}
              </div>

              <div className="flex gap-2 flex-wrap">
                {[
                  "🏋️ PoseCoach 멤버",
                  ...(stats && stats.total_workouts >= 5 ? ["🏆 꾸준한 운동가"] : []),
                  ...(stats && stats.total_workouts >= 10 ? ["💪 분석 마스터"] : []),
                ].map(b => (
                  <span
                    key={b}
                    className="text-[14px] px-4 py-2 rounded-full bg-[#c8f135]/8 border border-[#c8f135]/20 text-[#c8f135]/80"
                    style={{ fontFamily: "DM Mono, monospace" }}
                  >
                    {b}
                  </span>
                ))}
              </div>
            </div>

            <Button
              variant="outline"
              size="sm"
              className="border-red-500/20 text-red-400/70 hover:text-red-300 hover:border-red-300/40 text-sm h-10 px-4"
              onClick={() => { clearSession(); navigate("/"); }}
            >
              로그아웃
            </Button>
          </div>

          {/* 종합 스탯 카드 */}
          {!loading && stats && (
            <div className="grid grid-cols-4 gap-5 mb-10">
              {[
                { label: "총 세션", val: stats.total_workouts, sub: "회 분석 완료" },
                { label: "평균 점수", val: toPercent(stats.overall_avg_score), sub: "전체 평균" },
                { label: "총 운동 수", val: stats.total_reps, sub: "회 누적" },
                { label: "최다 운동", val: stats.favorite_exercise || "-", sub: "가장 많이 분석" },
              ].map(s => (
                <div key={s.label} className="rounded-2xl border border-white/10 bg-white/5 p-7 flex flex-col gap-3">
                  <span className="text-[13px] text-white/35 uppercase tracking-wider" style={{ fontFamily: "DM Mono, monospace" }}>
                    {s.label}
                  </span>
                  <span className="text-4xl font-extrabold text-[#c8f135] leading-none">{s.val}</span>
                  <span className="text-[13px] text-white/25" style={{ fontFamily: "DM Mono, monospace" }}>
                    {s.sub}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* 탭 */}
          <div className="flex border-b border-white/10 mb-10">
            <button className={tabCls(tab === "profile")} style={{ fontFamily: "DM Mono, monospace" }} onClick={() => setTab("profile")}>
              프로필
            </button>
            <button className={tabCls(tab === "history")} style={{ fontFamily: "DM Mono, monospace" }} onClick={() => setTab("history")}>
              운동 기록
            </button>
          </div>

          {loading && (
            <div className="text-center py-20 text-white/20 text-2xl" style={{ fontFamily: "DM Mono, monospace" }}>
              데이터를 불러오는 중...
            </div>
          )}
          {error && (
            <div className="rounded-2xl bg-[#ff6b35]/8 border border-[#ff6b35]/20 px-6 py-5 text-[#ff6b35] text-lg mb-6" style={{ fontFamily: "DM Mono, monospace" }}>
              ⚠ {error}
            </div>
          )}

          {/* ═══ TAB: 프로필 ═══ */}
          {!loading && tab === "profile" && (
            <div className="grid grid-cols-2 gap-10">
              {/* 계정 설정 */}
              <div>
                {/* ✅ 섹션 타이틀 조금 더 크게 */}
                <div className="text-[14px] text-white/35 uppercase tracking-widest mb-5 pb-4 border-b border-white/8" style={{ fontFamily: "DM Mono, monospace" }}>
                  계정 설정
                </div>
                <div className="flex flex-col gap-3">
                  {[["아이디", session.username], ["비밀번호", "••••••••"], ["계정 유형", "일반 회원"]].map(([k, v]) => (
                    <div key={k} className="rounded-2xl border border-white/8 bg-white/3 px-6 py-5 flex items-center justify-between">
                      {/* ✅ 항목 텍스트 조금 더 크게 */}
                      <span className="text-[16px] text-white/45" style={{ fontFamily: "DM Mono, monospace" }}>{k}</span>
                      <span className={`text-[16px] ${k === "계정 유형" ? "text-[#c8f135]/80" : "text-white/70"}`} style={{ fontFamily: "DM Mono, monospace" }}>
                        {v}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* 운동별 누적 */}
              <div>
                {/* ✅ 섹션 타이틀 조금 더 크게 */}
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
                          {/* ✅ 운동명/요약 텍스트 조금 더 크게 */}
                          <span className="text-[16px] text-[#c8f135] w-16 shrink-0" style={{ fontFamily: "DM Mono, monospace" }}>
                            {ex}
                          </span>
                          <div className="flex-1 h-2.5 bg-white/10 rounded-full overflow-hidden">
                            <div
                              style={{
                                height: "100%",
                                width: `${(c / maxCnt) * 100}%`,
                                background: "rgba(200,241,53,0.6)",
                                borderRadius: 9999,
                              }}
                            />
                          </div>
                          <span className="text-[15px] text-white/35 whitespace-nowrap" style={{ fontFamily: "DM Mono, monospace" }}>
                            {c}세션 · {reps[ex]}회
                          </span>
                        </div>
                      ))}
                    </div>
                  );
                })() : (
                  <div className="text-white/20 text-2xl py-6" style={{ fontFamily: "DM Mono, monospace" }}>
                    운동 기록이 없습니다
                  </div>
                )}
              </div>

              {/* 하단 버튼 */}
              <div className="col-span-2 flex gap-3 pt-6 border-t border-white/8">
                <Button
                  className={`bg-[#c8f135] text-black hover:bg-[#b4da30] ${smallActionBtn}`}
                  onClick={() => navigate("/select-exercise")}
                >
                  새 운동 시작
                </Button>
                <Button
                  variant="outline"
                  className={`border-white/10 text-white/60 hover:text-white ${smallActionBtnOutline}`}
                  onClick={() => navigate("/")}
                >
                  홈으로
                </Button>
              </div>
            </div>
          )}

          {/* ═══ TAB: 운동 기록 ═══ */}
          {!loading && tab === "history" && (
            <div>
              {workouts.length === 0 ? (
                <div className="text-center py-24">
                  <div className="text-6xl mb-5">📭</div>
                  <div className="text-white/25 text-base mb-8" style={{ fontFamily: "DM Mono, monospace" }}>
                    아직 운동 기록이 없습니다
                  </div>
                  <Button className="bg-[#c8f135] text-black hover:bg-[#b4da30] text-base px-8 py-6" onClick={() => navigate("/select-exercise")}>
                    운동 시작하기 →
                  </Button>
                </div>
              ) : (
                <div className="flex flex-col gap-4">
                  {workouts.map(w => {
                    const isOpen = expandedId === w.id;
                    const gc = gradeColor(w.grade);
                    return (
                      <div key={w.id} className="rounded-2xl border border-white/8 bg-white/3 overflow-hidden">
                        {/* 카드 헤더 */}
                        <button
                          onClick={() => setExpandedId(isOpen ? null : w.id)}
                          className="w-full bg-transparent border-0 cursor-pointer flex items-center gap-5 px-7 py-6 text-left"
                        >
                          <div
                            className="text-lg font-extrabold shrink-0 px-4 py-2 rounded-xl border"
                            style={{ color: gc, background: `${gc}18`, borderColor: `${gc}50` }}
                          >
                            {w.grade}
                          </div>

                          <div className="flex-1 flex flex-col gap-2">
                            {/* ✅ 운동 기록 텍스트 조금 더 크게 */}
                            <div className="text-[16px] text-white/75" style={{ fontFamily: "DM Mono, monospace" }}>
                              {formatDate(w.created_at)} — {w.exercise_type}{w.grip_type ? ` · ${w.grip_type}` : ""} — {w.exercise_count}회
                            </div>
                            <div className="text-[15px] text-white/38" style={{ fontFamily: "DM Mono, monospace" }}>
                              평균 점수 {toPercent(w.avg_score)} · 오류 {w.error_frame_count}프레임 · {Math.round(w.duration)}초
                            </div>
                          </div>

                          <span className="text-white/40 text-2xl shrink-0" style={{ fontFamily: "DM Mono, monospace" }}>
                            {isOpen ? "▲" : "▼"}
                          </span>
                        </button>

                        {/* 펼침 */}
                        {isOpen && (
                          <div className="border-t border-white/8 px-7 py-7 flex flex-col gap-7">
                            <div className="grid grid-cols-3 gap-6">
                              {[
                                ["운동", w.exercise_type + (w.grip_type ? ` · ${w.grip_type}` : "")],
                                ["횟수", `${w.exercise_count}회`],
                                ["점수", toPercent(w.avg_score)],
                                ["영상", w.video_name],
                                ["FPS", String(w.fps)],
                                ["길이", `${Math.round(w.duration)}초`],
                              ].map(([k, v]) => (
                                <div key={k}>
                                  <div className="text-[13px] text-white/30 uppercase tracking-wider mb-2" style={{ fontFamily: "DM Mono, monospace" }}>
                                    {k}
                                  </div>
                                  <div className="text-[17px] text-white/82" style={{ fontFamily: "DM Mono, monospace" }}>
                                    {v}
                                  </div>
                                </div>
                              ))}
                            </div>

                            {w.errors && w.errors.length > 0 && (
                              <div>
                                <div className="text-[13px] text-white/30 uppercase tracking-wider mb-4" style={{ fontFamily: "DM Mono, monospace" }}>
                                  주요 오류
                                </div>
                                <div className="flex flex-col gap-3">
                                  {w.errors.map((e, i) => (
                                    <div
                                      key={i}
                                      className="rounded-xl bg-[#ff6b35]/8 border border-[#ff6b35]/20 px-5 py-4 text-[15px] text-[#ff6b35]"
                                      style={{ fontFamily: "DM Mono, monospace" }}
                                    >
                                      ⚠ {e.error_msg} ({e.count}회)
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {w.phase_scores && w.phase_scores.length > 0 && (
                              <div>
                                <div className="text-[13px] text-white/30 uppercase tracking-wider mb-4" style={{ fontFamily: "DM Mono, monospace" }}>
                                  Phase별 점수
                                </div>
                                <div className="flex gap-4 flex-wrap">
                                  {w.phase_scores.map((p, i) => (
                                    <div key={i} className="rounded-2xl border border-white/8 bg-white/5 px-6 py-5 flex flex-col gap-2">
                                      <span className="text-[13px] text-white/35" style={{ fontFamily: "DM Mono, monospace" }}>
                                        {toPhaseLabel(p.phase)}
                                      </span>
                                      <span className="text-2xl font-extrabold text-[#c8f135]">{toPercent(p.avg_score)}</span>
                                      <span className="text-[13px] text-white/25" style={{ fontFamily: "DM Mono, monospace" }}>
                                        {p.frame_count} frames
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}

                  <div className="flex gap-3 pt-6 border-t border-white/8">
                    <Button
                      className={`bg-[#c8f135] text-black hover:bg-[#b4da30] ${smallActionBtn}`}
                      onClick={() => navigate("/select-exercise")}
                    >
                      새 운동 시작
                    </Button>
                    <Button
                      variant="outline"
                      className={`border-white/10 text-white/60 hover:text-white ${smallActionBtnOutline}`}
                      onClick={() => navigate("/")}
                    >
                      홈으로
                    </Button>
                  </div>
                </div>
              )}
            </div>
          )}

        </div>
      </div>
    </div>
  );
}