import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { ArrowLeft, CheckCircle2, Loader2, Upload, Video } from "lucide-react";

import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { analyzeVideo } from "../lib/api";
import { getSession } from "../lib/auth";

type UploadVideoState = {
  exercise?: "pushup" | "pullup";
  grip?: string;
  referenceFile?: File;
};

export function UploadVideo() {
  const navigate = useNavigate();
  const location = useLocation();
  const routeState = (location.state ?? {}) as UploadVideoState;

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [messageIndex, setMessageIndex] = useState(0);

  const exercise = routeState.exercise ?? "pushup";
  const grip = routeState.grip;
  const session = getSession();

  const rotatingMessages = [
    "팁: 코어에 힘을 주고 몸통을 일직선으로 유지하세요.",
    "\"오늘의 1%가 내일의 100%를 만든다.\"",
    "팁: 반동보다 통제된 속도로 움직여야 자세가 좋아집니다.",
    "\"꾸준함은 재능을 이긴다.\"",
    "팁: 내려갈 때 들이마시고, 올라올 때 내쉬세요.",
    "\"완벽보다 반복이 강하다.\"",
  ];

  useEffect(() => {
    if (!isAnalyzing) return;
    setMessageIndex(0);
    const timer = window.setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % rotatingMessages.length);
    }, 2300);
    return () => window.clearInterval(timer);
  }, [isAnalyzing]);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith("video/")) return;
    setSelectedFile(file);
    setErrorMessage(null);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  const handleAnalyze = async () => {
    if (!selectedFile || isAnalyzing) return;

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const payload = await analyzeVideo({
        videoFile: selectedFile,
        exerciseType: exercise,
        gripType: grip,
        extractFps: 10,
        saveResult: !!session,
        userId: session?.user_id,
      });

      navigate("/result", {
        state: {
          exercise,
          grip,
          referenceFile: routeState.referenceFile,
          videoFile: selectedFile,
          analysisResults: payload.analysis_results,
          savedWorkoutId: payload.saved_workout_id ?? null,
        },
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : "분석에 실패했습니다";
      setErrorMessage(message);
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-slate-50 dark:bg-slate-900 p-8">
      <div className="max-w-3xl w-full">
        <Button variant="ghost" className="mb-8" onClick={() => navigate(-1)} disabled={isAnalyzing}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          뒤로
        </Button>

        <div className="mb-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
            <Video className="w-8 h-8 text-green-600" />
          </div>
          <h1 className="text-4xl font-bold mb-4">운동 영상 업로드</h1>
          <p className="text-gray-600 mb-2">분석할 영상을 업로드하세요.</p>
          <p className="text-sm text-gray-500">
            선택한 운동: <span className="font-semibold">{exercise === "pullup" ? "풀업" : "푸시업"}</span>
            {grip && <span> - {grip}</span>}
          </p>
        </div>

        {isAnalyzing ? (
          <Card>
            <CardContent className="p-16">
              <div className="flex flex-col items-center text-center">
                <Loader2 className="w-16 h-16 text-blue-600 animate-spin mb-6" />
                <h2 className="text-2xl font-bold mb-2">영상 분석 중...</h2>
                <p className="text-gray-600">포즈 추출과 자세 평가를 진행하고 있습니다.</p>
                <div className="mt-6 rounded-lg border border-blue-100 bg-blue-50 px-5 py-4 max-w-xl w-full">
                  <p className="text-sm font-semibold text-blue-700 mb-1">운동 팁/명언</p>
                  <p className="text-sm text-slate-700 whitespace-pre-wrap">
                    {rotatingMessages[messageIndex]}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        ) : (
          <>
            <Card>
              <CardContent className="p-8">
                <div
                  className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
                    isDragging
                      ? "border-green-600 bg-green-50"
                      : selectedFile
                        ? "border-green-600 bg-green-50"
                        : "border-gray-300 hover:border-green-400"
                  }`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  {selectedFile ? (
                    <div className="flex flex-col items-center">
                      <CheckCircle2 className="w-16 h-16 text-green-600 mb-4" />
                      <p className="text-lg font-semibold mb-2">{selectedFile.name}</p>
                      <p className="text-gray-600 mb-4">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                      <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
                        다른 파일 선택
                      </Button>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center">
                      <Upload className="w-16 h-16 text-gray-400 mb-4" />
                      <p className="text-lg font-semibold mb-2">영상을 드래그하거나 클릭해 업로드하세요</p>
                      <p className="text-gray-500 mb-4">MP4, MOV, AVI, WEBM</p>
                      <Button onClick={() => fileInputRef.current?.click()}>파일 선택</Button>
                    </div>
                  )}

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={handleFileInputChange}
                  />
                </div>
              </CardContent>
            </Card>

            {errorMessage && (
              <p className="mt-4 text-center text-sm text-red-600">
                {errorMessage}
              </p>
            )}

            <div className="flex justify-center mt-8">
              <Button
                size="lg"
                className="bg-green-600 hover:bg-green-700 text-white px-16 py-6 text-lg"
                disabled={!selectedFile}
                onClick={handleAnalyze}
              >
                분석 시작
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
