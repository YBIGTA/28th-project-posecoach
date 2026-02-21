import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { ArrowDown, ArrowLeft, ArrowUp } from "lucide-react";

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
    <div className="min-h-screen w-full flex items-center justify-center bg-slate-50 dark:bg-slate-900 p-8">
      <div className="max-w-5xl w-full">
        <Button variant="ghost" className="mb-8" onClick={() => navigate("/")}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          뒤로
        </Button>

        <h1 className="text-4xl font-bold text-center mb-4">운동 종목 선택</h1>
        <p className="text-center text-gray-600 mb-12">분석할 운동을 선택하세요.</p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <Card
            className={`cursor-pointer transition-all hover:shadow-xl ${
              selectedExercise === "pullup" ? "ring-4 ring-blue-600 shadow-xl" : ""
            }`}
            onClick={() => setSelectedExercise("pullup")}
          >
            <CardContent className="p-8">
              <div className="aspect-video bg-slate-200 rounded-lg mb-6 overflow-hidden">
                <img
                  src="https://images.unsplash.com/photo-1516208962313-9d183d94f577?q=80&w=1074&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                  alt="풀업"
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-2xl font-bold mb-2">풀업</h3>
                  <p className="text-gray-600">상체 당기기 동작</p>
                </div>
                <ArrowUp className="w-8 h-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>

          <Card
            className={`cursor-pointer transition-all hover:shadow-xl ${
              selectedExercise === "pushup" ? "ring-4 ring-blue-600 shadow-xl" : ""
            }`}
            onClick={() => setSelectedExercise("pushup")}
          >
            <CardContent className="p-8">
              <div className="aspect-video bg-slate-200 rounded-lg mb-6 overflow-hidden">
                <img
                  src="https://images.unsplash.com/photo-1571019614242-c5c5dee9f50b?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                  alt="푸시업"
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-2xl font-bold mb-2">푸시업</h3>
                  <p className="text-gray-600">상체 밀기 동작</p>
                </div>
                <ArrowDown className="w-8 h-8 text-blue-600" />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="flex justify-center mt-12">
          <Button
            size="lg"
            className="bg-blue-600 hover:bg-blue-700 text-white px-16 py-6 text-lg"
            disabled={!selectedExercise}
            onClick={handleNext}
          >
            다음
          </Button>
        </div>
      </div>
    </div>
  );
}
