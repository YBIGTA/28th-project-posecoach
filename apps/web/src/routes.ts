import { createBrowserRouter } from "react-router-dom";
import { Root } from "./components/Root";
import { Home } from "./pages/Home";
import { SelectExercise } from "./pages/SelectExercise";
import { SelectGrip } from "./pages/SelectGrip";
import { UploadVideo } from "./pages/UploadVideo";
import { Result } from "./pages/Result";
import { Login } from "./pages/Login";
import { MyPage } from "./pages/MyPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Root,
    children: [
      { index: true, Component: Home },
      { path: "login", Component: Login },
      { path: "select-exercise", Component: SelectExercise },
      { path: "select-grip", Component: SelectGrip },
      { path: "upload-video", Component: UploadVideo },
      { path: "result", Component: Result },
      { path: "mypage", Component: MyPage },
    ],
  },
]);

