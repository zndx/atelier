import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Suspense, lazy } from "react";
import { DatasetProvider } from "./state/DatasetContext";
import { ToastProvider } from "./ui/Toast";
import { AppShell } from "./layout/AppShell";
import { Spinner } from "./ui/Feedback";
import Operate from "./screens/Operate";

const Status = lazy(() => import("./screens/Status"));
const Settings = lazy(() => import("./screens/Settings"));
const Agents = lazy(() => import("./screens/Agents"));
const EmbeddingsIndex = lazy(() => import("./screens/EmbeddingsIndex"));
const EmbeddingsDetail = lazy(() => import("./screens/EmbeddingsDetail"));
const Workflows = lazy(() => import("./screens/Workflows"));
const OverwatchReport = lazy(() => import("./screens/OverwatchReport"));

function Fallback() {
  return (
    <div className="flex h-full items-center justify-center py-24">
      <Spinner />
    </div>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <DatasetProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<AppShell />}>
              <Route index element={<Operate />} />
              <Route
                path="status"
                element={
                  <Suspense fallback={<Fallback />}>
                    <Status />
                  </Suspense>
                }
              />
              <Route
                path="settings"
                element={
                  <Suspense fallback={<Fallback />}>
                    <Settings />
                  </Suspense>
                }
              />
              <Route
                path="agents"
                element={
                  <Suspense fallback={<Fallback />}>
                    <Agents />
                  </Suspense>
                }
              />
              <Route
                path="embeddings"
                element={
                  <Suspense fallback={<Fallback />}>
                    <EmbeddingsIndex />
                  </Suspense>
                }
              />
              <Route
                path="embeddings/:datasetId"
                element={
                  <Suspense fallback={<Fallback />}>
                    <EmbeddingsDetail />
                  </Suspense>
                }
              />
              <Route
                path="workflows"
                element={
                  <Suspense fallback={<Fallback />}>
                    <Workflows />
                  </Suspense>
                }
              />
              <Route
                path="overwatch/:runId"
                element={
                  <Suspense fallback={<Fallback />}>
                    <OverwatchReport />
                  </Suspense>
                }
              />
            </Route>
          </Routes>
        </BrowserRouter>
      </DatasetProvider>
    </ToastProvider>
  );
}
