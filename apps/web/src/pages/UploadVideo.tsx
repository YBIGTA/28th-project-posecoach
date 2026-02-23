import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Dumbbell } from "lucide-react";
import { analyzeVideo } from "../lib/api";
import { getSession } from "../lib/auth";
import { Button } from "../components/ui/button";

type RouteState = { exercise?: "pushup" | "pullup"; grip?: string };

const STEPS = ["운동 선택", "영상 업로드", "결과 확인"];

const TIPS = [
  "운동 많이 된다",
  "오늘 스트레스 많이 받을거야",
  "그런 스트레스도 필요하다",
  "진짜 도움 많이 되고 있어",
  "말 안하지만 지금 스트레스 되게 받는다",
  "오늘 자기 전에 생각 많이 날거야",
  "한판 쉴래? 근데 남들은 안쉬어",
  "상대 세게 나온다",
  "스트롱 스트롱",
  "굿 파트너",
  "예술이다 예술",
  "대화가 된다",
  "올라잇",
  "Light weight baby",
];

function StepBar({ current }: { current: number }) {
  return (
    <div className="flex items-center py-6">
      {STEPS.map((s, i) => (
        <div key={s} className={`flex items-center ${i < STEPS.length - 1 ? "flex-1" : ""}`}>
          <div className="flex items-center gap-2">
            <div
              className={`w-6 h-6 rounded-full flex items-center justify-center text-[12px] font-semibold
              ${
                i < current
                  ? "bg-[#c8f135]/10 border border-[#c8f135]/40 text-[#c8f135]"
                  : i === current
                  ? "bg-[#c8f135] text-black"
                  : "bg-white/5 border border-white/10 text-white/30"
              }`}
            >
              {i < current ? "✓" : i + 1}
            </div>
            <span className={`${i === current ? "text-[#c8f135]" : "text-white/40"} text-[12px] tracking-wide`}>
              {s}
            </span>
          </div>
          {i < STEPS.length - 1 && (
            <div className={`flex-1 h-px mx-4 ${i < current ? "bg-[#c8f135]/40" : "bg-white/10"}`} />
          )}
        </div>
      ))}
    </div>
  );
}

export function UploadVideo() {
  const navigate = useNavigate();
  const location = useLocation();
  const session = useMemo(() => getSession(), []);
  const { exercise = "pushup", grip } = (location.state ?? {}) as RouteState;

  const [mainFile, setMainFile] = useState<File | null>(null);
  const [mainUrl, setMainUrl] = useState<string | null>(null);
  const [refFile, setRefFile] = useState<File | null>(null);
  const [refUrl, setRefUrl] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [tipIdx, setTipIdx] = useState(0);

  // ✅ FPS 사용자 설정 (default=10)
  const [fps, setFps] = useState<number>(10);

  useEffect(() => {
    if (!mainFile) {
      setMainUrl(null);
      return;
    }
    const url = URL.createObjectURL(mainFile);
    setMainUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [mainFile]);

  useEffect(() => {
    if (!refFile) {
      setRefUrl(null);
      return;
    }
    const url = URL.createObjectURL(refFile);
    setRefUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [refFile]);

  useEffect(() => {
    if (!analyzing) return;
    setTipIdx(0);
    const t = setInterval(() => setTipIdx((p) => (p + 1) % TIPS.length), 2500);
    return () => clearInterval(t);
  }, [analyzing]);

  const handleAnalyze = async () => {
    if (!mainFile || analyzing) return;
    setAnalyzing(true);

    const safeFps = Number.isFinite(fps) ? Math.min(30, Math.max(10, Math.round(fps))) : 10;

    const result = await analyzeVideo({
      videoFile: mainFile,
      referenceFile: refFile ?? undefined,
      exerciseType: exercise,
      gripType: grip,
      extractFps: safeFps, // ✅ 사용자 설정 FPS
      saveResult: !!session,
      userId: session?.user_id,
    });

    navigate("/result", {
      state: {
        exercise,
        grip,
        analysisResults: result.analysis_results,
      },
    });
  };

  const exLabel = exercise === "pullup" ? "풀업" : "푸시업";

  return (
    <div className="min-h-screen w-full bg-[#0a0a0a] text-white flex items-center justify-center px-6 py-10">
      <div className="w-full max-w-[1400px] rounded-[30px] border border-white/10 bg-[#0f1116]/80 backdrop-blur-xl shadow-[0_30px_80px_rgba(0,0,0,0.6)] overflow-hidden">
        {/* ── HEADER (Home 통일) ── */}
        <header className="flex items-center justify-between px-8 py-6 border-b border-white/10 backdrop-blur-xl bg-black/40">
          <button
            onClick={() => navigate("/")}
            className="flex items-center gap-2 font-extrabold tracking-widest text-white"
          >
            <Dumbbell className="w-5 h-5 text-[#c8f135]" />
            POSECOACH
          </button>

          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              className="h-8 px-3 border-white/10 text-white/60 hover:text-[#c8f135] hover:border-[#c8f135]/40 hover:bg-[#c8f135]/10"
              onClick={() => navigate("/select-exercise")}
            >
              ← 뒤로
            </Button>

            {session && (
              <Button
                variant="outline"
                size="sm"
                className="h-8 px-3 border-[#c8f135]/40 text-[#c8f135] hover:bg-[#c8f135]/10"
                onClick={() => navigate("/mypage")}
              >
                마이페이지
              </Button>
            )}
          </div>
        </header>

        {/* CONTENT */}
        <div className="px-10 py-12">
          <StepBar current={1} />

          {/* 타이틀 */}
          <div className="text-center mt-10 mb-8">
            <div className="text-[#c8f135] text-xs tracking-widest mb-2">STEP 2</div>
            <h1 className="text-[clamp(2rem,4vw,3rem)] font-extrabold">영상 업로드</h1>
            <p className="text-white/40 text-sm mt-3">분석을 위한 영상을 업로드하세요.</p>
          </div>

          {/* ✅ FPS 설정 + ✅ 활성 배지 (설정 요약 박스 제거하고 여기로 이동/확대) */}
          <div className="max-w-[1000px] mx-auto mb-12">
            {/* FPS 컨트롤 */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-6 mb-6">
              <div className="flex items-center justify-between gap-6 flex-wrap">
                <div className="flex flex-col gap-1">
                  <div className="text-sm font-semibold text-white/80">FPS 설정</div>
                  <div className="text-xs text-white/40">프레임 추출 속도 (1~30, 기본 10)</div>
                </div>

                <div className="flex items-center gap-3">
                  <input
                    type="number"
                    min={1}
                    max={30}
                    value={fps}
                    onChange={(e) => setFps(Number(e.target.value))}
                    className="w-20 bg-black/30 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 outline-none focus:border-[#c8f135]/40"
                  />
                  <span className="text-white/40 text-sm">fps</span>
                </div>
              </div>

              <div className="mt-5">
                <input
                  type="range"
                  min={1}
                  max={30}
                  value={fps}
                  onChange={(e) => setFps(Number(e.target.value))}
                  className="w-full accent-[#c8f135]"
                />
                <div className="flex justify-between text-[11px] text-white/25 mt-2">
                  <span>1</span>
                  <span>10</span>
                  <span>20</span>
                  <span>30</span>
                </div>
              </div>
            </div>

            {/* 활성 배지 (크기 업) */}
            <div className="flex justify-center gap-3 flex-wrap">
              <span className="text-sm px-5 py-2 rounded-xl bg-[#c8f135]/10 border border-[#c8f135]/30 text-[#c8f135]">
                {exLabel} {grip && `· ${grip}`}
              </span>

              <span className="text-sm px-5 py-2 rounded-xl bg-white/5 border border-white/10 text-white/70">
                FPS · <span className="text-[#c8f135] font-semibold">{Math.round(fps)}</span>
              </span>

              {refFile && (
                <span className="text-sm px-5 py-2 rounded-xl bg-[#5b8fff]/10 border border-[#5b8fff]/30 text-[#5b8fff]">
                  ≋ DTW 활성
                </span>
              )}

              <span className="text-sm px-5 py-2 rounded-xl bg-white/5 border border-white/10 text-white/60">
                기록 저장 · {session ? "자동 저장" : "비로그인"}
              </span>
            </div>
          </div>

          {/* 2열 업로드 */}
          <div className="grid md:grid-cols-2 gap-10 max-w-[1200px] mx-auto mb-14">
            {/* 사용자 영상 */}
            <div className="rounded-2xl border border-white/10 bg-white/5 p-8">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <div className="font-bold text-lg mb-1">사용자 영상</div>
                  <div className="text-xs text-white/40">분석 대상 (필수)</div>
                </div>
                <span className="text-[12px] px-3 py-1 rounded-md bg-[#ff6b35]/10 border border-[#ff6b35]/30 text-[#ff6b35]">
                  필수
                </span>
              </div>
              {!mainFile ? (
                <label className="cursor-pointer block text-center py-20 border-2 border-dashed border-white/10 rounded-xl hover:border-[#c8f135]/40 transition text-white/40 text-sm">
                  영상 업로드
                  <input
                    type="file"
                    hidden
                    accept="video/*"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) setMainFile(f);
                    }}
                  />
                </label>
              ) : (
                <div className="space-y-4">
                  <video src={mainUrl ?? undefined} controls className="rounded-xl w-full" />

                  <div className="flex justify-center gap-3">
                    <label className="cursor-pointer px-4 py-2 text-xs rounded-lg border border-white/10 bg-white/5 hover:border-[#c8f135]/40 hover:bg-[#c8f135]/10 transition text-white/70">
                      파일 변경
                      <input
                        type="file"
                        hidden
                        accept="video/*"
                        onChange={(e) => {
                          const f = e.target.files?.[0];
                          if (f) setMainFile(f);
                        }}
                      />
                    </label>

                    <button
                      onClick={() => {
                        setMainFile(null);
                        setMainUrl(null);
                      }}
                      className="px-4 py-2 text-xs rounded-lg border border-red-400/30 bg-red-400/10 text-red-400 hover:bg-red-400/20 transition"
                    >
                      삭제
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* 레퍼런스 영상 */}
            <div className="rounded-2xl border border-[#5b8fff]/30 bg-[#5b8fff]/5 p-8">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <div className="font-bold text-lg mb-1">레퍼런스 영상</div>
                  <div className="text-xs text-white/40">DTW 비교용 (선택)</div>
                </div>
                <span className="text-[12px] px-3 py-1 rounded-md bg-[#5b8fff]/10 border border-[#5b8fff]/30 text-[#5b8fff]">
                  선택
                </span>
              </div>
              {!refFile ? (
                <label className="cursor-pointer block text-center py-20 border-2 border-dashed border-[#5b8fff]/30 rounded-xl hover:border-[#5b8fff]/60 transition text-white/40 text-sm">
                  모범 동작 업로드
                  <input
                    type="file"
                    hidden
                    accept="video/*"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) setRefFile(f);
                    }}
                  />
                </label>
              ) : (
                <div className="space-y-4">
                  <video src={refUrl ?? undefined} controls className="rounded-xl w-full" />

                  <div className="flex justify-center gap-3">
                    <label className="cursor-pointer px-4 py-2 text-xs rounded-lg border border-[#5b8fff]/30 bg-[#5b8fff]/10 hover:border-[#5b8fff]/60 transition text-[#5b8fff]">
                      파일 변경
                      <input
                        type="file"
                        hidden
                        accept="video/*"
                        onChange={(e) => {
                          const f = e.target.files?.[0];
                          if (f) setRefFile(f);
                        }}
                      />
                    </label>

                    <button
                      onClick={() => {
                        setRefFile(null);
                        setRefUrl(null);
                      }}
                      className="px-4 py-2 text-xs rounded-lg border border-red-400/30 bg-red-400/10 text-red-400 hover:bg-red-400/20 transition"
                    >
                      삭제
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* 분석 버튼 */}
          <div className="flex justify-center">
            <Button
              disabled={!mainFile || analyzing}
              onClick={handleAnalyze}
              className="bg-[#c8f135] text-black hover:bg-[#b4da30] px-16 py-6 text-sm"
            >
              분석 시작하기
            </Button>
          </div>
        </div>
      </div>

      {/* 분석 중 오버레이 */}
      {analyzing && (
        <div className="fixed inset-0 bg-black/95 flex items-center justify-center z-50">
          <div className="bg-[#111] border border-[#c8f135]/30 rounded-2xl p-12 text-center max-w-md w-[90%]">
            <div className="text-[#c8f135] text-2xl mb-4 animate-pulse">분석 중...</div>
            <div className="text-white/40 text-sm mb-6">포즈 추출 및 자세 평가를 진행하고 있습니다.</div>

            {/* ✅ TIPS 폰트 키움 */}
            <div className="text-white/70 text-lg font-semibold">{TIPS[tipIdx]}</div>
          </div>
        </div>
      )}
    </div>
  );
}