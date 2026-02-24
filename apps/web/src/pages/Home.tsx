import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowRight,
  Dumbbell,
  Sparkles,
} from "lucide-react";
import { Button } from "../components/ui/button";
import { clearSession, getSession } from "../lib/auth";

export function Home() {
  const navigate = useNavigate();
  const session = useMemo(() => getSession(), []);

  return (
    <div className="min-h-screen w-full bg-[#0a0a0a] text-white">
      <div className="mx-auto max-w-[1400px] overflow-hidden rounded-[30px] border border-white/10 bg-[#0f1116]/80 backdrop-blur-xl shadow-[0_30px_80px_rgba(0,0,0,0.6)]">
        
        {/* ── HEADER ── */}
        <header className="flex items-center justify-between px-8 py-6 border-b border-white/10 backdrop-blur-xl bg-black/40">
          <button
            onClick={() => navigate("/")}
            className="flex items-center gap-2 font-extrabold tracking-widest text-white"
          >
            <Dumbbell className="w-5 h-5 text-[#c8f135]" />
            POSECOACH
          </button>

          {session ? (
            <div className="flex items-center gap-3 text-xs">
              <span className="text-white/40">
                {session.username}
              </span>

              <Button
                variant="outline"
                className="border-[#c8f135]/40 text-[#c8f135] hover:bg-[#c8f135]/10"
                onClick={() => navigate("/mypage")}
              >
                마이페이지
              </Button>

              <Button
                variant="outline"
                className="border-red-400/30 text-red-400 hover:bg-red-400/10"
                onClick={() => {
                  clearSession();
                  window.location.reload();
                }}
              >
                로그아웃
              </Button>
            </div>
          ) : (
            <Button
              variant="outline"
              className="border-[#c8f135]/40 text-[#c8f135] hover:bg-[#c8f135]/10"
              onClick={() => navigate("/login")}
            >
              로그인
            </Button>
          )}
        </header>

        {/* ── HERO ── */}
        <section className="px-10 py-20 relative overflow-hidden">
          {/* 배경 라임 글로우 */}
          <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-[radial-gradient(circle,rgba(200,241,53,0.07)_0%,transparent_70%)] pointer-events-none" />

          <div className="grid md:grid-cols-2 items-center gap-16 relative z-10">
            {/* LEFT TEXT */}
            <div>
              <div className="inline-flex items-center gap-2 px-4 py-1 text-xs border border-[#c8f135]/30 bg-[#c8f135]/10 rounded-full text-[#c8f135] mb-6">
                <Sparkles className="w-3 h-3" />
                AI 자세 코칭
              </div>

              <h1 className="text-[clamp(3rem,8vw,6rem)] font-extrabold leading-[1.0] tracking-tight">
                POSE <span className="text-[#c8f135]">COACH</span>
              </h1>

              <p className="mt-6 text-white/50 max-w-lg">
                운동 영상을 업로드하면 프레임 단위 자세 점수와 AI 피드백을 제공합니다.
              </p>

              <div className="mt-10 flex gap-4 flex-wrap">
                <Button
                  size="lg"
                  className="bg-[#c8f135] text-black hover:bg-[#b4da30] px-10"
                  onClick={() => navigate("/select-exercise")}
                >
                  분석 시작하기
                  <ArrowRight className="ml-2 w-4 h-4" />
                </Button>

                {session && (
                  <Button
                    variant="outline"
                    size="lg"
                    className="border-[#c8f135]/40 text-[#c8f135] hover:bg-[#c8f135]/10"
                    onClick={() => navigate("/mypage")}
                  >
                    지난 기록 보기
                  </Button>
                )}
              </div>
            </div>

            {/* RIGHT IMAGE */}
            <div className="relative min-h-[360px] md:min-h-[460px]">
              <div className="absolute inset-0 rounded-3xl border border-white/10 bg-[#141821]" />
              <img
                src="https://images.unsplash.com/photo-1601422407692-ec4eeec1d9b3?q=80&w=1170&auto=format&fit=crop"
                alt="운동 랜딩 비주얼"
                className="absolute inset-3 h-[calc(100%-1.5rem)] w-[calc(100%-1.5rem)] rounded-2xl object-cover"
              />
              <div className="absolute inset-3 rounded-2xl bg-gradient-to-t from-black/70 via-black/30 to-transparent" />
            </div>
          </div>
        </section>

        {/* ── SCROLL FEATURE SECTIONS ── */}
        <section className="px-10 pb-24 border-t border-white/10">
          <div className="pt-16 pb-8">
            <h2 className="text-3xl font-bold">POINTS</h2>
            <p className="mt-3 text-white/45 text-sm">
              스크롤하면서 핵심 기능을 한 번에 이해할 수 있어요.
            </p>
          </div>

          <div className="flex flex-col gap-16">
            {[
              {
                eyebrow: "AI 동작 인식",
                title: "AI 동작 인식을 통한 \n 실시간 운동 피드백",
                desc:
                  "AI 동작 인식을 통해 운동마다 실시간으로 자세 피드백을 알려줍니다.\n" +
                  "쉽고 정확하게 운동에만 집중하세요.",
                img: "https://cdn.aitimes.com/news/photo/202204/144267_150053_2131.jpg",
              },
              {
                eyebrow: "동작 흐름 분석",
                title: "DTW 기반 유사도 분석",
                desc:
                  "단순한 동작 수행을 넘어, 정교한 DTW 알고리즘이\n" +
                  "모범 자세와의 흐름 차이를 미세하게 분석합니다.\n" +
                  "나의 움직임 궤적이 프로와 얼마나 일치하는지 확인하세요.",
                img: "https://images.unsplash.com/photo-1598971457999-ca4ef48a9a71?q=80&w=1170&auto=format&fit=crop",
              },
              {
                eyebrow: "AI 리포트",
                title: "나만의 AI 퍼스널 리포트",
                desc:
                  "프레임 단위의 정밀한 점수 산정은 물론,\n" +
                  "AI 트레이너가 분석한 취약 구간 교정법과\n" +
                  "성장을 위한 연습 우선순위를 한눈에 정리해 드립니다.",
                img: "https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=1170&auto=format&fit=crop",
              },
            ].map((f, idx) => (
              <ScrollFeatureRow key={f.eyebrow} feature={f} flip={idx % 2 === 1} />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}

// ── 보조 컴포넌트는 메인 컴포넌트 외부로 분리 ──
function ScrollFeatureRow({
  feature,
  flip,
}: {
  feature: { eyebrow: string; title: string; desc: string; img: string };
  flip?: boolean;
}) {
  return (
    <div className="rounded-[28px] border border-white/10 bg-[#0f1116]/70 overflow-hidden">
      <div
        className={`grid md:grid-cols-2 gap-0 items-stretch ${
          flip ? "md:[&>div:first-child]:order-2" : ""
        }`}
      >
        {/* TEXT */}
        <div className="p-10 md:p-14 flex flex-col justify-center">
          <div className="inline-flex items-center gap-2 px-4 py-1 text-xs border border-[#c8f135]/30 bg-[#c8f135]/10 rounded-full text-[#c8f135] w-fit">
            {feature.eyebrow}
          </div>

          <h3 className="mt-6 text-[clamp(2.2rem,4vw,3.4rem)] font-extrabold leading-[1.08] whitespace-pre-line">
            {feature.title}
          </h3>

          <p className="mt-5 text-white/45 text-[15px] leading-7 whitespace-pre-line max-w-xl">
            {feature.desc}
          </p>

          <div className="mt-8 flex gap-3">
            <div className="h-1.5 w-20 rounded-full bg-[#c8f135]/60" />
            <div className="h-1.5 w-10 rounded-full bg-white/10" />
            <div className="h-1.5 w-6 rounded-full bg-white/10" />
          </div>
        </div>

        {/* MEDIA */}
        <div className="relative min-h-[320px] md:min-h-[520px] bg-black">
          <img
            src={feature.img}
            alt={feature.eyebrow}
            className="absolute inset-0 w-full h-full object-cover opacity-90"
          />
          <div className="absolute inset-0 bg-gradient-to-l from-black/70 via-black/20 to-transparent" />
          <div className="absolute inset-6 rounded-3xl border border-white/10 shadow-[0_30px_80px_rgba(0,0,0,0.55)]" />
        </div>
      </div>
    </div>
  );
}