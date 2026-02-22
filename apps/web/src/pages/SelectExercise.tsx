// ─────────────────────────────────────────
// SelectExercise.tsx (Home 스타일 통일 버전)
// ─────────────────────────────────────────

import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Dumbbell } from "lucide-react";
import { Button } from "../components/ui/button";
import { getSession } from "../lib/auth";

const STEPS = ["운동 선택", "영상 업로드", "결과 확인"];

function StepBar({ current }: { current: number }) {
  return (
    <div className="flex items-center py-6">
      {STEPS.map((s, i) => (
        <div
          key={s}
          className={`flex items-center ${
            i < STEPS.length - 1 ? "flex-1" : ""
          }`}
        >
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

            <span
              className={`text-[12px] tracking-wide ${
                i === current
                  ? "text-[#c8f135]"
                  : "text-white/40"
              }`}
            >
              {s}
            </span>
          </div>

          {i < STEPS.length - 1 && (
            <div
              className={`flex-1 h-px mx-4 ${
                i < current ? "bg-[#c8f135]/40" : "bg-white/10"
              }`}
            />
          )}
        </div>
      ))}
    </div>
  );
}

export function SelectExercise() {
  const navigate = useNavigate();
  const session = useMemo(() => getSession(), []);
  const [selected, setSelected] =
    useState<"pushup" | "pullup" | null>(null);

  const handleNext = () => {
    if (!selected) return;
    if (selected === "pullup") navigate("/select-grip");
    else navigate("/upload-video", { state: { exercise: selected } });
  };

  const exercises = [
    {
      id: "pullup" as const,
      name: "풀업",
      sub: "Pull-Up",
      desc: "광배근과 이두근 등 상체 후면",
      image: "https://images.unsplash.com/photo-1597452329152-52f9eee96576?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTB8fHB1bGx1cHxlbnwwfHwwfHx8MA%3D%3D",
      color: "#5b8fff",
    },
    {
      id: "pushup" as const,
      name: "푸시업",
      sub: "Push-Up",
      desc: "대흉근과 삼두근, 전면 삼각근",
      image: "https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
      color: "#c8f135",
    },
  ];

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
              onClick={() => navigate("/")}
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

        {/* ── CONTENT ── */}
        <div className="px-10 py-12">

          <StepBar current={0} />

          <div className="text-center mt-10 mb-12">
            <div className="text-[#c8f135] text-xs tracking-widest mb-2">
              STEP 1
            </div>
            <h1 className="text-[clamp(2rem,4vw,3rem)] font-extrabold">
              운동 종목 선택
            </h1>
            <p className="text-white/40 text-sm mt-3">
              분석할 운동을 선택하면 다음 단계로 자동 연결됩니다.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6 max-w-[800px] mx-auto mb-14">
            {exercises.map((ex) => (
              <div
              key={ex.id}
              onClick={() => setSelected(ex.id)}
              className={`rounded-2xl cursor-pointer transition-all duration-300 overflow-hidden
                ${
                  selected === ex.id
                    ? "bg-[#c8f135]/5 border border-[#c8f135]/40 shadow-[0_0_30px_rgba(200,241,53,0.12)]"
                    : "bg-white/5 border border-white/10 hover:border-[#c8f135]/30"
                }
              `}
            >
              {/* 이미지 영역 */}
              <div className="relative h-[220px]">
                <img
                  src={ex.image}
                  alt={ex.name}
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent" />
              </div>
            
              {/* 텍스트 영역 */}
              <div className="p-6">
                <div className="text-xl font-extrabold mb-1">
                  {ex.name}
                </div>
            
                <div className="text-xs tracking-wider mb-3 text-[#c8f135]">
                  {ex.sub}
                </div>
            
                <div className="text-white/40 text-sm">
                  {ex.desc}
                </div>
              </div>
            </div>
            ))}
          </div>

          <div className="flex justify-center">
            <Button
              disabled={!selected}
              onClick={handleNext}
              className="bg-[#c8f135] text-black hover:bg-[#b4da30] px-14 py-6 text-sm"
            >
              다음 →
            </Button>
          </div>

        </div>
      </div>
    </div>
  );
}