import { useNavigate } from "react-router-dom";
import { useMemo } from "react";

import { Dumbbell } from "lucide-react";

import { Button } from "../components/ui/button";
import { clearSession, getSession } from "../lib/auth";

export function Home() {
  const navigate = useNavigate();
  const session = useMemo(() => getSession(), []);

  return (
    <div className="min-h-screen w-full flex items-center justify-center relative overflow-hidden">
      <div
        className="absolute inset-0 bg-cover bg-center"
        style={{
          backgroundImage:
            "url('https://images.unsplash.com/photo-1534438327276-14e5300c3a48?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')",
        }}
      >
        <div className="absolute inset-0 bg-black/65" />
      </div>

      <div className="absolute top-6 right-6 z-20">
        {session ? (
          <div className="flex items-center gap-3">
            <span className="text-white/90 text-sm">안녕하세요, {session.username}</span>
            <Button
              variant="outline"
              className="bg-white/10 border-white/20 text-white hover:bg-white/20"
              onClick={() => navigate("/mypage")}
            >
              마이페이지
            </Button>
            <Button
              variant="outline"
              className="bg-white/10 border-white/20 text-white hover:bg-white/20"
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
            className="bg-white/10 border-white/20 text-white hover:bg-white/20"
            onClick={() => navigate("/login")}
          >
            로그인
          </Button>
        )}
      </div>

      <div className="relative z-10 text-center px-8 max-w-3xl text-white">
        <div className="mb-8 flex justify-center">
          <div className="w-20 h-20 bg-blue-600 rounded-full flex items-center justify-center shadow-xl">
            <Dumbbell className="w-10 h-10 text-white" />
          </div>
        </div>

        <h1 className="text-5xl md:text-6xl font-bold mb-4">PoseCoach</h1>
        <p className="text-xl text-gray-200 mb-12">
          운동 영상을 바탕으로 푸시업/풀업 자세를 AI가 분석합니다.
        </p>

        <Button
          size="lg"
          className="bg-blue-600 hover:bg-blue-700 text-white px-12 py-6 text-lg rounded-full shadow-lg"
          onClick={() => navigate("/select-exercise")}
        >
          시작하기
        </Button>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-white">
          <div>
            <div className="text-2xl font-bold mb-2">자세 분석</div>
            <p className="text-gray-300">프레임 단위 자세 점수화</p>
          </div>
          <div>
            <div className="text-2xl font-bold mb-2">반복 횟수 카운트</div>
            <p className="text-gray-300">자동 반복 감지</p>
          </div>
          <div>
            <div className="text-2xl font-bold mb-2">피드백</div>
            <p className="text-gray-300">실행 가능한 자세 교정 가이드</p>
          </div>
        </div>
      </div>
    </div>
  );
}
