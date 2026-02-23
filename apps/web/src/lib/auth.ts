import { API_BASE_URL } from "./api";

const SESSION_KEY = "posecoach_session";

export type AuthSession = {
  user_id: number;
  username: string;
};

type AuthResponse = {
  success: boolean;
  message: string;
  user_id?: number | null;
  username?: string;
};

function parseSession(raw: string | null): AuthSession | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as AuthSession;
    if (typeof parsed?.user_id === "number" && typeof parsed?.username === "string") {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

export function getSession(): AuthSession | null {
  return parseSession(window.localStorage.getItem(SESSION_KEY));
}

export function setSession(session: AuthSession): void {
  window.localStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

export function clearSession(): void {
  window.localStorage.removeItem(SESSION_KEY);
}

async function postAuth(path: string, username: string, password: string): Promise<AuthResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail ?? "인증 요청에 실패했습니다.";
    throw new Error(typeof detail === "string" ? detail : "인증 요청에 실패했습니다.");
  }
  return payload as AuthResponse;
}

export async function login(username: string, password: string): Promise<AuthSession> {
  const res = await postAuth("/auth/login", username, password);
  if (typeof res.user_id !== "number" || !res.username) {
    throw new Error("로그인 응답이 올바르지 않습니다.");
  }
  const session = { user_id: res.user_id, username: res.username };
  setSession(session);
  return session;
}

export async function register(username: string, password: string): Promise<AuthSession> {
  const res = await postAuth("/auth/register", username, password);
  if (typeof res.user_id !== "number" || !res.username) {
    throw new Error("회원가입은 완료되었지만 로그인 세션을 생성하지 못했습니다.");
  }
  const session = { user_id: res.user_id, username: res.username };
  setSession(session);
  return session;
}
