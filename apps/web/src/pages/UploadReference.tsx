import { useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { ArrowLeft, CheckCircle2, Upload, Video } from "lucide-react";

import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";

type UploadReferenceState = {
  exercise?: "pushup" | "pullup";
  grip?: string;
};

export function UploadReference() {
  const navigate = useNavigate();
  const location = useLocation();
  const routeState = (location.state ?? {}) as UploadReferenceState;

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const exercise = routeState.exercise ?? "pushup";
  const grip = routeState.grip;

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith("video/")) return;
    setSelectedFile(file);
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

  const handleNext = () => {
    navigate("/upload-video", {
      state: {
        exercise,
        grip,
        referenceFile: selectedFile,
      },
    });
  };

  return (
    <div className="modern-shell min-h-screen w-full flex items-center justify-center p-8">
      <div className="max-w-3xl w-full">
        <Button variant="ghost" className="mb-8 modern-outline-btn" onClick={() => navigate(-1)}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          뒤로
        </Button>

        <div className="mb-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-full mb-4">
            <Video className="w-8 h-8 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold mb-4">레퍼런스 영상 업로드 (선택)</h1>
          <p className="text-soft mb-2">건너뛰고 바로 운동 영상 업로드로 진행할 수 있습니다.</p>
          <p className="text-sm text-soft">
            선택한 운동: <span className="font-semibold">{exercise === "pullup" ? "풀업" : "푸시업"}</span>
            {grip && <span> - {grip}</span>}
          </p>
        </div>

        <Card className="glass-card">
          <CardContent className="p-8">
            <div
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-all ${
                isDragging
                  ? "border-blue-600 bg-blue-50/80 dark:bg-blue-900/20"
                  : selectedFile
                    ? "border-green-600 bg-green-50/80 dark:bg-green-900/20"
                    : "border-gray-300 dark:border-slate-700 hover:border-blue-400"
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {selectedFile ? (
                <div className="flex flex-col items-center">
                  <CheckCircle2 className="w-16 h-16 text-green-600 mb-4" />
                  <p className="text-lg font-semibold mb-2">{selectedFile.name}</p>
                  <p className="text-soft mb-4">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                  <Button variant="outline" className="modern-outline-btn" onClick={() => fileInputRef.current?.click()}>
                    다른 파일 선택
                  </Button>
                </div>
              ) : (
                <div className="flex flex-col items-center">
                  <Upload className="w-16 h-16 text-gray-400 dark:text-slate-500 mb-4" />
                  <p className="text-lg font-semibold mb-2">레퍼런스 영상을 드래그하거나 클릭해 업로드하세요</p>
                  <p className="text-soft mb-4">MP4, MOV, AVI, WEBM</p>
                  <Button className="modern-primary-btn" onClick={() => fileInputRef.current?.click()}>파일 선택</Button>
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

        <div className="flex justify-center gap-4 mt-8">
          <Button variant="outline" size="lg" className="modern-outline-btn px-10" onClick={handleNext}>
            건너뛰기
          </Button>
          <Button
            size="lg"
            className="modern-primary-btn px-16 py-6 text-lg"
            onClick={handleNext}
          >
            다음
          </Button>
        </div>
      </div>
    </div>
  );
}
