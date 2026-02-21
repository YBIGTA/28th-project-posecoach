import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ArrowLeft } from "lucide-react";

import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { login, register } from "../lib/auth";

type Mode = "login" | "register";

export function Login() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<Mode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);

    if (!username.trim() || !password.trim()) {
      setErrorMessage("아이디와 비밀번호를 입력해 주세요.");
      return;
    }
    if (mode === "register" && password !== confirmPassword) {
      setErrorMessage("비밀번호 확인이 일치하지 않습니다.");
      return;
    }

    setLoading(true);
    try {
      if (mode === "login") {
        await login(username.trim(), password);
      } else {
        await register(username.trim(), password);
      }
      navigate("/");
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "인증에 실패했습니다.");
      setLoading(false);
    }
  };

  return (
    <div className="modern-shell min-h-screen w-full flex items-center justify-center p-8">
      <div className="max-w-xl w-full">
        <Button variant="ghost" className="mb-6 modern-outline-btn" onClick={() => navigate("/")}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          뒤로
        </Button>

        <Card className="glass-card">
          <CardContent className="p-8">
            <h1 className="text-3xl font-bold text-center mb-2">계정</h1>
            <p className="text-center text-soft mb-8">로그인하면 분석 기록을 저장할 수 있습니다.</p>

            <div className="grid grid-cols-2 gap-2 mb-6 bg-slate-100/70 dark:bg-slate-800/70 rounded-lg p-1">
              <button
                type="button"
                className={`rounded-md py-2 text-sm font-semibold transition-colors ${
                  mode === "login" ? "bg-slate-900 text-slate-100 shadow" : "text-slate-400"
                }`}
                onClick={() => setMode("login")}
              >
                로그인
              </button>
              <button
                type="button"
                className={`rounded-md py-2 text-sm font-semibold transition-colors ${
                  mode === "register" ? "bg-slate-900 text-slate-100 shadow" : "text-slate-400"
                }`}
                onClick={() => setMode("register")}
              >
                회원가입
              </button>
            </div>

            <form className="space-y-4" onSubmit={onSubmit}>
              <div>
                <label className="text-sm font-medium text-slate-200 block mb-1">아이디</label>
                <input
                  className="w-full rounded-md border border-slate-600 bg-slate-900/70 px-3 py-2 outline-none focus:ring-2 focus:ring-orange-500"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  disabled={loading}
                  autoComplete="username"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-slate-200 block mb-1">비밀번호</label>
                <input
                  type="password"
                  className="w-full rounded-md border border-slate-600 bg-slate-900/70 px-3 py-2 outline-none focus:ring-2 focus:ring-orange-500"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={loading}
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                />
              </div>

              {mode === "register" && (
                <div>
                  <label className="text-sm font-medium text-slate-200 block mb-1">비밀번호 확인</label>
                  <input
                    type="password"
                    className="w-full rounded-md border border-slate-600 bg-slate-900/70 px-3 py-2 outline-none focus:ring-2 focus:ring-orange-500"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    disabled={loading}
                    autoComplete="new-password"
                  />
                </div>
              )}

              {errorMessage && <p className="text-sm text-red-600">{errorMessage}</p>}

              <Button className="w-full modern-primary-btn" disabled={loading} type="submit">
                {loading ? "처리 중..." : mode === "login" ? "로그인" : "계정 만들기"}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
