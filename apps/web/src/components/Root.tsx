import { useEffect, useState } from "react";
import { Home, Moon, Sun } from "lucide-react";
import { Outlet, useNavigate } from "react-router-dom";

import { Button } from "./ui/button";

const THEME_KEY = "posecoach_theme";

type ThemeMode = "light" | "dark";

function getInitialTheme(): ThemeMode {
  if (typeof window === "undefined") return "light";
  const saved = window.localStorage.getItem(THEME_KEY);
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function Root() {
  const navigate = useNavigate();
  const [theme, setTheme] = useState<ThemeMode>(getInitialTheme);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "dark") root.classList.add("dark");
    else root.classList.remove("dark");
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  return (
    <div className="min-h-screen w-full bg-background text-foreground">
      <Outlet />
      <div className="fixed right-4 bottom-4 z-[100] flex flex-col gap-2">
        <Button type="button" variant="secondary" className="shadow-md" onClick={() => navigate("/")}>
          <Home className="w-4 h-4" />
          홈
        </Button>
        <Button
          type="button"
          variant="secondary"
          className="shadow-md"
          onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
        >
          {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          {theme === "dark" ? "라이트 모드" : "다크 모드"}
        </Button>
      </div>
    </div>
  );
}
