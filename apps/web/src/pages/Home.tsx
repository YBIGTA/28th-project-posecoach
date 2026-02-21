import { useMemo } from "react";
import { useNavigate } from "react-router-dom";

import { ArrowRight, Dumbbell, PlayCircle, Sparkles, Trophy } from "lucide-react";

import { Button } from "../components/ui/button";
import { clearSession, getSession } from "../lib/auth";

export function Home() {
  const navigate = useNavigate();
  const session = useMemo(() => getSession(), []);

  return (
    <div className="modern-shell min-h-screen w-full px-0 py-6 md:px-0 md:py-8">
      <div className="mx-auto max-w-[1400px] overflow-hidden rounded-[30px] border border-slate-700/50 bg-[#0f1116] text-slate-100 shadow-[0_30px_90px_rgba(0,0,0,0.45)]">
        <header className="flex flex-wrap items-center justify-between gap-3 px-6 py-5 md:px-9 md:py-7">
          <button
            type="button"
            onClick={() => navigate("/")}
            className="inline-flex items-center gap-2 text-sm font-extrabold tracking-[0.12em] text-white"
          >
            <Dumbbell className="w-4 h-10 text-orange-400" />
            POSECOACH
          </button>

          <nav className="hidden lg:flex items-center gap-6 text-[11px] uppercase tracking-wide text-slate-400">
            <span>Workouts</span>
            <span>Programs</span>
            <span>Insights</span>
            <span>Community</span>
          </nav>

          {session ? (
            <div className="flex items-center gap-2">
              <span className="hidden sm:block text-xs text-slate-300">{session.username}</span>
              <Button size="sm" variant="outline" className="modern-outline-btn h-8 px-3" onClick={() => navigate("/mypage")}>
                마이페이지
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="modern-outline-btn h-8 px-3"
                onClick={() => {
                  clearSession();
                  window.location.reload();
                }}
              >
                로그아웃
              </Button>
            </div>
          ) : (
            <Button size="sm" variant="outline" className="modern-outline-btn h-8 px-3" onClick={() => navigate("/login")}>
              로그인
            </Button>
          )}
        </header>

        <section className="px-6 pb-8 md:px-9 md:pb-11">
          <div className="grid items-end gap-9 lg:grid-cols-[1.03fr_0.97fr]">
            <div className="animate-rise">
              <span className="hero-chip mb-6">
                <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                AI 자세 코칭
              </span>

              <h1 className="text-[clamp(2.5rem,8vw,6rem)] leading-[0.95] font-extrabold tracking-tight text-white">
                운동을
                <br />
                함께 완성해요
              </h1>

              <p className="mt-6 max-w-xl text-sm md:text-base text-slate-300">
                푸시업과 풀업 영상을 업로드하면 프레임 단위 자세 점수, 반복 횟수, 교정 피드백을 한 번에 제공합니다.
              </p>

              <div className="mt-8 flex flex-wrap gap-3">
                <Button size="lg" className="modern-primary-btn rounded-lg px-8 py-6 text-sm md:text-base" onClick={() => navigate("/select-exercise")}>
                  분석 시작하기
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
                {session ? (
                  <Button size="lg" variant="outline" className="modern-outline-btn rounded-lg px-7 py-6" onClick={() => navigate("/mypage")}>
                    지난 기록 보기
                  </Button>
                ) : null}
              </div>

            
            </div>

            <div className="relative min-h-[360px] md:min-h-[460px] animate-rise delay-2">
              <div className="absolute inset-0 rounded-3xl border border-slate-700/60 bg-[#141821]" />
              <img
                src="https://images.unsplash.com/photo-1601422407692-ec4eeec1d9b3?q=80&w=1170&auto=format&fit=crop"
                alt="운동 랜딩 비주얼"
                className="absolute inset-3 h-[calc(100%-1.5rem)] w-[calc(100%-1.5rem)] rounded-2xl object-cover"
              />
              <div className="absolute inset-3 rounded-2xl bg-gradient-to-t from-black/75 via-black/30 to-black/5" />

              <div className="absolute right-5 top-7 rounded-xl border border-orange-300/30 bg-orange-400 px-4 py-3 text-white shadow-lg">
                <p className="text-[11px] font-semibold">진행률</p>
                <p className="text-3xl font-extrabold leading-none">4.9</p>
              </div>

              <div className="absolute left-6 top-[54%] rounded-xl border border-slate-600 bg-slate-900/90 px-4 py-3 text-slate-100 shadow-lg">
                <p className="flex items-center gap-2 text-lg font-bold">
                  <PlayCircle className="h-4 w-4 text-orange-400" />
                  350+
                </p>
                <p className="text-[11px] text-slate-300">Video Tutorial</p>
              </div>

              <div className="absolute right-5 bottom-7 rounded-xl border border-violet-400/30 bg-violet-600 px-4 py-3 text-white shadow-lg">
                <p className="text-2xl font-extrabold leading-none">500+</p>
                <p className="text-[11px]">Workout Clips</p>
              </div>
            </div>
          </div>

          <div className="mt-12 border-t border-slate-700/70 pt-8">
            <h2 className="text-3xl font-bold text-white">무엇부터 시작할까요?</h2>
            <p className="mt-2 text-sm text-slate-400">
              운동 루틴을 선택하고, 분석 리포트로 자세를 빠르게 개선하세요.
            </p>

            <div className="mt-6 grid grid-cols-1 gap-3 md:grid-cols-4">
              <article className="rounded-xl border border-slate-700 bg-[#2a2f40] p-4">
                <h3 className="text-sm font-bold text-white">운동 분석</h3>
                <p className="mt-2 text-xs text-slate-300">프레임 단위 점수와 자세 오류를 확인합니다.</p>
                <ArrowRight className="mt-4 h-4 w-4 text-slate-300" />
              </article>
              <article className="rounded-xl border border-slate-700 bg-[#2a2f40] p-4">
                <h3 className="text-sm font-bold text-white">운동 프로그램</h3>
                <p className="mt-2 text-xs text-slate-300">반복 횟수와 운동 시간을 기준으로 루틴을 관리합니다.</p>
                <ArrowRight className="mt-4 h-4 w-4 text-slate-300" />
              </article>
              <article className="rounded-xl border border-slate-700 bg-[#2a2f40] p-4">
                <h3 className="text-sm font-bold text-white">피드백 리포트</h3>
                <p className="mt-2 text-xs text-slate-300">AI 코멘트와 단계별 점수를 함께 제공합니다.</p>
                <ArrowRight className="mt-4 h-4 w-4 text-slate-300" />
              </article>
              <article className="rounded-xl border border-slate-700 bg-[#3b4155] p-4">
                <h3 className="flex items-center gap-1 text-sm font-bold text-white">
                  <Trophy className="h-4 w-4 text-orange-300" />
                  PRO 분석
                </h3>
                <p className="mt-2 text-xs text-slate-200">DTW 유사도와 과거 기록 비교로 고급 분석을 진행합니다.</p>
                <ArrowRight className="mt-4 h-4 w-4 text-slate-100" />
              </article>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
