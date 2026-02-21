import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { ArrowLeft } from "lucide-react";

import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";

type GripId = "overhand" | "underhand" | "wide";

export function SelectGrip() {
  const navigate = useNavigate();
  const [selectedGrip, setSelectedGrip] = useState<GripId | null>(null);

  const grips: Array<{ id: GripId; name: string; description: string }> = [
    { id: "overhand", name: "오버핸드", description: "기본 풀업 그립" },
    { id: "underhand", name: "언더핸드", description: "친업 스타일 그립" },
    { id: "wide", name: "와이드", description: "어깨보다 넓은 손 간격" },
  ];

  const handleNext = () => {
    if (!selectedGrip) return;
    navigate("/upload-reference", { state: { exercise: "pullup", grip: selectedGrip } });
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-slate-50 dark:bg-slate-900 p-8">
      <div className="max-w-4xl w-full">
        <Button variant="ghost" className="mb-8" onClick={() => navigate("/select-exercise")}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          뒤로
        </Button>

        <h1 className="text-4xl font-bold text-center mb-4">그립 선택</h1>
        <p className="text-center text-gray-600 mb-12">풀업 그립 유형을 선택하세요.</p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {grips.map((grip) => (
            <Card
              key={grip.id}
              className={`cursor-pointer transition-all hover:shadow-lg ${
                selectedGrip === grip.id ? "ring-4 ring-blue-600 shadow-xl" : ""
              }`}
              onClick={() => setSelectedGrip(grip.id)}
            >
              <CardContent className="p-6">
                <h3 className="text-xl font-bold mb-2">{grip.name}</h3>
                <p className="text-gray-600 text-sm">{grip.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="flex justify-center mt-12">
          <Button
            size="lg"
            className="bg-blue-600 hover:bg-blue-700 text-white px-16 py-6 text-lg"
            disabled={!selectedGrip}
            onClick={handleNext}
          >
            다음
          </Button>
        </div>
      </div>
    </div>
  );
}
