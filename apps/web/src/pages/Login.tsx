import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { login, register } from "../lib/auth";

type Mode = "login" | "register";

export function Login() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<Mode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!username.trim() || !password.trim()) {
      setError("아이디와 비밀번호를 입력해 주세요.");
      return;
    }

    if (mode === "register" && password !== confirmPassword) {
      setError("비밀번호가 일치하지 않습니다.");
      return;
    }

    setLoading(true);
    try {
      if (mode === "login") await login(username.trim(), password);
      else await register(username.trim(), password);
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "인증에 실패했습니다.");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-[#0a0a0a] text-white flex items-center justify-center px-6">
      
      {/* 🔥 Home이랑 동일한 outer shell 유지 */}
      <div className="w-full max-w-[900px] rounded-[30px] border border-white/10 bg-[#0f1116]/80 backdrop-blur-xl shadow-[0_30px_80px_rgba(0,0,0,0.6)] p-10">

        {/* ── 로고 */}
        <div className="text-center mb-10">
          <div className="text-2xl font-extrabold tracking-widest">
            POSE <span className="text-[#c8f135]">COACH</span>
          </div>
        </div>

        {/* ── 탭 */}
        <div className="flex border-b border-white/10 mb-8">
          {(["login", "register"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => {
                setMode(m);
                setError(null);
              }}
              className={`flex-1 pb-3 text-sm transition-all ${
                mode === m
                  ? "text-[#c8f135] border-b-2 border-[#c8f135]"
                  : "text-white/40 border-b-2 border-transparent hover:text-white/70"
              }`}
            >
              {m === "login" ? "로그인" : "회원가입"}
            </button>
          ))}
        </div>

        {/* ── 폼 */}
        <form
          onSubmit={onSubmit}
          className="flex flex-col gap-5 max-w-[420px] mx-auto"
        >
          <input
            className="bg-[#1a1f28] border border-white/10 rounded-xl px-4 py-3 text-sm outline-none focus:border-[#c8f135] transition"
            placeholder="아이디"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={loading}
            autoComplete="username"
          />

          <input
            type="password"
            className="bg-[#1a1f28] border border-white/10 rounded-xl px-4 py-3 text-sm outline-none focus:border-[#c8f135] transition"
            placeholder="비밀번호"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={loading}
            autoComplete={
              mode === "login" ? "current-password" : "new-password"
            }
          />

          {mode === "register" && (
            <input
              type="password"
              className="bg-[#1a1f28] border border-white/10 rounded-xl px-4 py-3 text-sm outline-none focus:border-[#c8f135] transition"
              placeholder="비밀번호 확인"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              disabled={loading}
              autoComplete="new-password"
            />
          )}

          {error && (
            <div className="text-red-400 text-sm bg-red-400/10 border border-red-400/20 rounded-lg px-3 py-2">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="mt-2 bg-[#c8f135] text-black font-semibold py-3 rounded-xl hover:bg-[#b4da30] transition"
          >
            {loading
              ? "처리 중..."
              : mode === "login"
              ? "로그인"
              : "회원가입"}
          </button>
        </form>

        {/* ── 구분선 */}
        <div className="flex items-center gap-4 my-8 max-w-[420px] mx-auto">
          <div className="flex-1 h-px bg-white/10" />
          <span className="text-xs text-white/30 tracking-widest">또는</span>
          <div className="flex-1 h-px bg-white/10" />
        </div>

        {/* ── 게스트 버튼 */}
        <div className="max-w-[420px] mx-auto">
          <button
            onClick={() => navigate("/")}
            className="w-full border border-white/10 text-white/50 py-3 rounded-xl hover:border-[#c8f135]/40 hover:text-[#c8f135] transition"
          >
            게스트로 시작하기 (기록 저장 안됨)
          </button>
        </div>

      </div>
    </div>
  );
}