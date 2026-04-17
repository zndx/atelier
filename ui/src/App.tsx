import { ConfigProvider, Spin, theme } from "antd";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Suspense, lazy } from "react";
import Layout from "./components/Layout";
import { DatasetProvider } from "./contexts/DatasetContext";
import Landing from "./pages/Landing";

const Agents = lazy(() => import("./pages/Agents"));
const Embeddings = lazy(() => import("./pages/Embeddings"));
const EmbeddingsIndex = lazy(() => import("./pages/EmbeddingsIndex"));
const Status = lazy(() => import("./pages/Status"));
const TerminalPage = lazy(() => import("./pages/TerminalPage"));
const Workflows = lazy(() => import("./pages/Workflows"));
const OverwatchReport = lazy(() => import("./pages/OverwatchReport"));

function App() {
  return (
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: "#1890ff",
          borderRadius: 6,
        },
      }}
    >
      <DatasetProvider>
      <BrowserRouter>
        <Routes>
          <Route
            path="/"
            element={
              <Layout>
                <Landing />
              </Layout>
            }
          />
          <Route
            path="/agents"
            element={
              <Layout>
                <Suspense
                  fallback={
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        height: "calc(100vh - 128px)",
                      }}
                    >
                      <Spin size="large" />
                    </div>
                  }
                >
                  <Agents />
                </Suspense>
              </Layout>
            }
          />
          <Route
            path="/status"
            element={
              <Layout>
                <Suspense
                  fallback={
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        height: "calc(100vh - 128px)",
                      }}
                    >
                      <Spin size="large" />
                    </div>
                  }
                >
                  <Status />
                </Suspense>
              </Layout>
            }
          />
          <Route
            path="/workflows"
            element={
              <Layout fullHeight>
                <Suspense
                  fallback={
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        height: "calc(100vh - 128px)",
                      }}
                    >
                      <Spin size="large" />
                    </div>
                  }
                >
                  <Workflows />
                </Suspense>
              </Layout>
            }
          />
          <Route
            path="/terminal"
            element={
              <Layout fullHeight>
                <Suspense
                  fallback={
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        height: "calc(100vh - 128px)",
                      }}
                    >
                      <Spin size="large" />
                    </div>
                  }
                >
                  <TerminalPage />
                </Suspense>
              </Layout>
            }
          />
          <Route
            path="/embeddings"
            element={
              <Layout>
                <Suspense
                  fallback={
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        height: "calc(100vh - 128px)",
                      }}
                    >
                      <Spin size="large" />
                    </div>
                  }
                >
                  <EmbeddingsIndex />
                </Suspense>
              </Layout>
            }
          />
          <Route
            path="/embeddings/:datasetId"
            element={
              <Layout fullHeight>
                <Suspense
                  fallback={
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        height: "calc(100vh - 128px)",
                      }}
                    >
                      <Spin size="large" />
                    </div>
                  }
                >
                  <Embeddings />
                </Suspense>
              </Layout>
            }
          />
          <Route
            path="/overwatch/:runId"
            element={
              <Layout>
                <Suspense
                  fallback={
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        height: "calc(100vh - 128px)",
                      }}
                    >
                      <Spin size="large" />
                    </div>
                  }
                >
                  <OverwatchReport />
                </Suspense>
              </Layout>
            }
          />
        </Routes>
      </BrowserRouter>
      </DatasetProvider>
    </ConfigProvider>
  );
}

export default App;
