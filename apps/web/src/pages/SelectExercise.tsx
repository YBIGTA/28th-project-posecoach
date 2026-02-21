import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { ArrowDown, ArrowLeft, ArrowRight, ArrowUp, Sparkles } from "lucide-react";

import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";

export function SelectExercise() {
  const navigate = useNavigate();
  const [selectedExercise, setSelectedExercise] = useState<"pushup" | "pullup" | null>(null);

  const handleNext = () => {
    if (!selectedExercise) return;
    if (selectedExercise === "pullup") {
      navigate("/select-grip");
      return;
    }
    navigate("/upload-reference", { state: { exercise: selectedExercise } });
  };

  return (
    <div className="modern-shell min-h-screen w-full p-8">
      <div className="mx-auto max-w-6xl">
        <Button variant="ghost" className="mb-8 modern-outline-btn" onClick={() => navigate("/")}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          뒤로
        </Button>

        <div className="text-center mb-12 animate-rise">
          <span className="hero-chip mb-4">
            <Sparkles className="w-3.5 h-3.5 mr-1" />
            루틴 설정
          </span>
          <h1 className="text-4xl md:text-5xl font-bold mb-4">운동 종목 선택</h1>
          <p className="text-soft">분석할 운동을 선택하면 다음 단계로 자동 연결됩니다.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <Card
            className={`glass-card overflow-hidden cursor-pointer transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl ${
              selectedExercise === "pullup" ? "ring-2 ring-cyan-500 shadow-2xl" : ""
            }`}
            onClick={() => setSelectedExercise("pullup")}
          >
            <CardContent className="p-5">
              <div className="relative aspect-video bg-slate-200 rounded-xl mb-5 overflow-hidden">
                <img
                  src="https://images.unsplash.com/photo-1597452329152-52f9eee96576?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTB8fHB1bGx1cHxlbnwwfHwwfHx8MA%3D%3D"
                  alt="풀업"
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-slate-900/55 via-transparent to-transparent" />
              </div>
              <div className="flex items-center justify-between px-1">
                <div>
                  <h3 className="text-2xl font-bold mb-2">풀업</h3>
                  <p className="text-soft">광배근과 이두근 등 상체 후면</p>
                </div>
                <ArrowUp className="w-8 h-8 text-cyan-600" />
              </div>
            </CardContent>
          </Card>

          <Card
            className={`glass-card overflow-hidden cursor-pointer transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl ${
              selectedExercise === "pushup" ? "ring-2 ring-cyan-500 shadow-2xl" : ""
            }`}
            onClick={() => setSelectedExercise("pushup")}
          >
            <CardContent className="p-5">
              <div className="relative aspect-video bg-slate-200 rounded-xl mb-5 overflow-hidden">
                <img
                  src="https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                  alt="푸시업"
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-slate-900/55 via-transparent to-transparent" />
              </div>
              <div className="flex items-center justify-between px-1">
                <div>
                  <h3 className="text-2xl font-bold mb-2">푸시업</h3>
                  <p className="text-soft">대흉근과 삼두근, 전면 삼각근</p>
                </div>
                <ArrowDown className="w-8 h-8 text-cyan-600" />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="flex justify-center mt-12">
          <Button
            size="lg"
            className="modern-primary-btn px-16 py-6 text-lg rounded-xl"
            disabled={!selectedExercise}
            onClick={handleNext}
          >
            다음
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </div>
    </div>
  );
}
