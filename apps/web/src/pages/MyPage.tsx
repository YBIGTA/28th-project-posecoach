import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { ArrowLeft, Clock3, Dumbbell, History, Trophy } from "lucide-react";

import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { fetchUserStats, fetchUserWorkouts, type UserStats, type WorkoutRecord } from "../lib/api";
import { getSession } from "../lib/auth";

const PHASE_LABEL: Record<string, string> = {
  ready: "준비",
  top: "상단",
  descending: "하강",
  bottom: "하단",
  ascending: "상승",
};

function toPercent(v: number | null | undefined): string {
  if (typeof v !== "number" || Number.isNaN(v)) return "-";
  return `${Math.round(v * 100)}%`;
}

function toPhaseLabel(phase?: string): string {
  if (!phase) return "-";
  return PHASE_LABEL[phase] ?? phase;
}

function formatDate(value: string): string {
  const date = new Date(value.replace(" ", "T"));
  if (Number.isNaN(date.getTime())) return value;
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
}

export function MyPage() {
  const navigate = useNavigate();
  const session = useMemo(() => getSession(), []);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<UserStats | null>(null);
  const [workouts, setWorkouts] = useState<WorkoutRecord[]>([]);

  useEffect(() => {
    if (!session) {
      setLoading(false);
      return;
    }

    let mounted = true;
    Promise.all([
      fetchUserStats(session.user_id),
      fetchUserWorkouts(session.user_id),
    ])
      .then(([statsData, workoutData]) => {
        if (!mounted) return;
        setStats(statsData);
        setWorkouts(workoutData);
      })
      .catch((e) => {
        if (!mounted) return;
        setError(e instanceof Error ? e.message : "마이페이지 데이터를 불러오지 못했습니다.");
      })
      .finally(() => {
        if (!mounted) return;
        setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, [session]);

  if (!session) {
    return (
      <div className="min-h-screen w-full bg-slate-50 dark:bg-slate-900 p-8 flex items-center justify-center">
        <Card className="max-w-xl w-full">
          <CardContent className="p-8 text-center space-y-4">
            <h1 className="text-2xl font-bold">로그인이 필요합니다</h1>
            <p className="text-gray-600">마이페이지는 로그인 후 이용할 수 있습니다.</p>
            <div className="flex justify-center gap-3">
              <Button variant="outline" onClick={() => navigate("/")}>홈으로</Button>
              <Button onClick={() => navigate("/login")}>로그인하기</Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen w-full bg-slate-50 dark:bg-slate-900 p-8">
      <div className="max-w-6xl mx-auto">
        <Button variant="ghost" className="mb-6" onClick={() => navigate("/")}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          홈으로
        </Button>

        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">마이페이지</h1>
          <p className="text-gray-600">{session.username}님의 운동 통계와 과거 분석 기록입니다.</p>
        </div>

        {loading ? (
          <Card>
            <CardContent className="p-10 text-center text-gray-600">데이터를 불러오는 중...</CardContent>
          </Card>
        ) : null}

        {error ? (
          <Card className="border-red-200 bg-red-50 mb-6">
            <CardContent className="p-5 text-red-700">{error}</CardContent>
          </Card>
        ) : null}

        {!loading && !error ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
              <Card>
                <CardContent className="p-4 text-center">
                  <History className="w-5 h-5 mx-auto mb-2 text-slate-600" />
                  <p className="text-xs text-gray-500">총 운동 횟수</p>
                  <p className="text-2xl font-bold">{stats?.total_workouts ?? 0}</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <Clock3 className="w-5 h-5 mx-auto mb-2 text-slate-600" />
                  <p className="text-xs text-gray-500">총 운동 시간</p>
                  <p className="text-2xl font-bold">{Math.round(stats?.total_duration ?? 0)}초</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <Trophy className="w-5 h-5 mx-auto mb-2 text-slate-600" />
                  <p className="text-xs text-gray-500">평균 점수</p>
                  <p className="text-2xl font-bold">{toPercent(stats?.overall_avg_score)}</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <Dumbbell className="w-5 h-5 mx-auto mb-2 text-slate-600" />
                  <p className="text-xs text-gray-500">총 반복 횟수</p>
                  <p className="text-2xl font-bold">{stats?.total_reps ?? 0}회</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-4 text-center">
                  <p className="text-xs text-gray-500 mb-2">가장 많이 한 운동</p>
                  <p className="text-2xl font-bold">{stats?.favorite_exercise ?? "-"}</p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>과거 분석 기록</CardTitle>
              </CardHeader>
              <CardContent>
                {workouts.length === 0 ? (
                  <p className="text-gray-600">저장된 운동 기록이 없습니다.</p>
                ) : (
                  <div className="space-y-4">
                    {workouts.map((w) => (
                      <div key={w.id} className="rounded-lg border p-4 bg-white">
                        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                          <div className="font-semibold">{formatDate(w.created_at)}</div>
                          <div className="flex flex-wrap gap-2">
                            <Badge>{w.exercise_type}</Badge>
                            <Badge variant="secondary">{toPercent(w.avg_score)}</Badge>
                            <Badge variant="outline">{w.grade}</Badge>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-sm text-gray-700">
                          <div>영상: {w.video_name}</div>
                          <div>반복: {w.exercise_count}회</div>
                          <div>길이: {Math.round(w.duration)}초</div>
                          <div>FPS: {w.fps}</div>
                          <div>오류 프레임: {w.error_frame_count}개</div>
                          <div>그립: {w.grip_type ?? "-"}</div>
                        </div>

                        <details className="mt-3">
                          <summary className="cursor-pointer text-sm text-blue-700">상세 보기</summary>
                          <div className="mt-3 space-y-2 text-sm">
                            <div>
                              <p className="font-medium mb-1">오류 요약</p>
                              {(w.errors?.length ?? 0) === 0 ? (
                                <p className="text-gray-600">오류 없음</p>
                              ) : (
                                <ul className="list-disc pl-5">
                                  {(w.errors ?? []).map((err, idx) => (
                                    <li key={`${w.id}-err-${idx}`}>{err.error_msg} ({err.count}회)</li>
                                  ))}
                                </ul>
                              )}
                            </div>
                            <div>
                              <p className="font-medium mb-1">단계별 평균 점수</p>
                              {(w.phase_scores?.length ?? 0) === 0 ? (
                                <p className="text-gray-600">단계 데이터 없음</p>
                              ) : (
                                <ul className="list-disc pl-5">
                                  {(w.phase_scores ?? []).map((p, idx) => (
                                    <li key={`${w.id}-phase-${idx}`}>
                                      {toPhaseLabel(p.phase)}: {toPercent(p.avg_score)} ({p.frame_count}프레임)
                                    </li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          </div>
                        </details>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </>
        ) : null}
      </div>
    </div>
  );
}
