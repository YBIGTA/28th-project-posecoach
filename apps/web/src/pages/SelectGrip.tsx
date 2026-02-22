// ─────────────────────────────────────────
// SelectGrip.tsx (Home 통일 디자인 버전)
// ─────────────────────────────────────────

import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Dumbbell } from "lucide-react";
import { Button } from "../components/ui/button";
import { getSession } from "../lib/auth";

type GripId = "overhand" | "underhand" | "wide";

const STEPS = ["운동 선택", "영상 업로드", "결과 확인"];

function StepBar({ current }: { current: 0 | 1 | 2 }) {
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

            <span className={`text-[12px] tracking-wide ${i === current ? "text-[#c8f135]" : "text-white/40"}`}>
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

export function SelectGrip() {
  const navigate = useNavigate();
  const session = useMemo(() => getSession(), []);
  const [selected, setSelected] = useState<GripId | null>(null);

  const grips = [
    {
      id: "overhand" as const,
      name: "오버핸드",
      desc: "기본 풀업 그립 · 표준",
      image:
        "https://www.hsefitness.com/wp-content/uploads/2024/12/Pull-Up-Alternatives-Exercises-You-Can-Do-Without-a-Bar.jpg.webp",
    },
    {
      id: "underhand" as const,
      name: "언더핸드",
      desc: "친업 스타일 그립",
      image:
        "https://media.istockphoto.com/id/1297630176/ko/%EC%82%AC%EC%A7%84/%EC%B9%9C%EC%97%85.jpg?s=170667a&w=0&k=20&c=fu5FFYsaGEvXlaqeIDE5rSjz-lLKT-mi4xgZkplE0iw=",
    },
    {
      id: "wide" as const,
      name: "와이드",
      desc: "어깨보다 넓은 간격",
      image:
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQrqNGvaOUgWawWr-BS6WEHRI7Z-DX49HL1ng&s",
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

        {/* ── CONTENT ── */}
        <div className="px-10 py-12">

          <StepBar current={0} />

          <div className="text-center mt-10 mb-12">
            <div className="text-[#c8f135] text-xs tracking-widest mb-2">
              STEP 1-2
            </div>
            <h1 className="text-[clamp(2rem,4vw,3rem)] font-extrabold">
              그립 타입 선택
            </h1>
            <p className="text-white/40 text-sm mt-3">
              풀업 그립 유형을 선택하세요.
            </p>
          </div>

          {/* ── GRIP CARDS ── */}
          <div className="grid md:grid-cols-3 gap-6 max-w-[1000px] mx-auto mb-14">
            {grips.map((g) => (
              <div
                key={g.id}
                onClick={() => setSelected(g.id)}
                className={`rounded-2xl cursor-pointer transition-all duration-300 overflow-hidden
                  ${
                    selected === g.id
                      ? "bg-[#c8f135]/5 border border-[#c8f135]/40 shadow-[0_0_30px_rgba(200,241,53,0.12)]"
                      : "bg-white/5 border border-white/10 hover:border-[#c8f135]/30"
                  }
                `}
              >
                {/* 이미지 */}
                <div className="relative h-[200px]">
                  <img
                    src={g.image}
                    alt={g.name}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent" />
                </div>

                {/* 텍스트 */}
                <div className="p-6 text-center">
                  <div className="text-lg font-extrabold mb-2">
                    {g.name}
                  </div>

                  <div className="text-white/40 text-sm">
                    {g.desc}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* ── NEXT BUTTON ── */}
          <div className="flex justify-center">
            <Button
              disabled={!selected}
              onClick={() =>
                navigate("/upload-video", {
                  state: { exercise: "pullup", grip: selected },
                })
              }
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