// Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
//
// This file contains material proprietary to Cloudera, Inc., and is provided
// to authorized licensees solely for use in connection with the Cloudera AI
// (CAI) Application from which it was obtained.  It may not be copied,
// modified, redistributed, or used in any other manner without the express
// written consent of Cloudera, Inc.

import { Card, Col, Row, Spin, Statistic, Tag, Typography } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  WarningOutlined,
  BookOutlined,
  ClusterOutlined,
  CodeOutlined,
  DatabaseOutlined,
  DotChartOutlined,
  ExperimentOutlined,
  ToolOutlined,
} from "@ant-design/icons";
import {
  lazy,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { Link } from "react-router-dom";
import type { AgentInfo } from "../types/canvas";
import { useDataset } from "../contexts/DatasetContext";

const Terminal = lazy(() => import("../components/Terminal"));

const { Title, Paragraph } = Typography;

interface StatusSummary {
  grpc: { ok: boolean; version?: string };
  postgres: { ok: boolean };
  qdrant: { ok: boolean };
  connected: boolean;
  degraded?: boolean;
  config?: { app_display_name?: string };
}

// State machine for the Service Status indicator.
//
//   mount → connecting (yellow, no value yet)
//   poll succeeds → connected (green) / degraded (yellow)
//   connected/degraded → disconnected (red) on first failed poll
//   disconnected for >= DISCONNECTED_ALARM_MS → connecting (yellow,
//                                              soften to "still trying")
//   subsequent failures while connecting → stay connecting
//
// The 10-second alarm window lets a brief gateway hiccup register as
// a loud red signal, while a longer outage softens to the "we are
// polling, the system is trying to reach the engine" state.  Avoids
// the thrashing UX of flipping between Disconnected and Connecting on
// every poll cycle while the gateway is genuinely unreachable.
type ConnectionState =
  | "connecting"
  | "connected"
  | "classifying"
  | "degraded"
  | "disconnected";
const DISCONNECTED_ALARM_MS = 10_000;

// Pipeline considered "in flight" iff the FSM is in any non-terminal
// state — same predicate Status.tsx uses for its Start/Stop button.
const PIPELINE_TERMINAL_STATES = new Set(["IDLE", "CONVERGED", "ERROR"]);

function Landing() {
  const [status, setStatus] = useState<StatusSummary | null>(null);
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("connecting");
  const disconnectedSinceRef = useRef<number | null>(null);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [termCount, setTermCount] = useState<number | null>(null);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const { activeDatasetId, activeSourceId, datasets, sources } = useDataset();

  useEffect(() => {
    // Poll /api/status periodically so a transient first-load failure
    // (Vite up before the gateway, gRPC slow to bind, etc.) recovers
    // automatically without the operator hitting refresh.  The Status
    // page already polls /api/fsm/status on a 5s cadence; the Landing
    // page's Service Status indicator matches that pattern and drives
    // the ConnectionState machine described above.
    const fetchStatus = () =>
      fetch("/api/status")
        .then((r) => r.json())
        .then((data: StatusSummary) => {
          setStatus(data);
          disconnectedSinceRef.current = null;
          setConnectionState(data.degraded ? "degraded" : "connected");
        })
        .catch(() => {
          setStatus(null);
          setConnectionState((prev) => {
            const now = Date.now();
            if (prev === "connected" || prev === "degraded") {
              // First failure after a healthy poll — fire the alarm.
              disconnectedSinceRef.current = now;
              return "disconnected";
            }
            if (prev === "disconnected") {
              const since = disconnectedSinceRef.current ?? now;
              return now - since >= DISCONNECTED_ALARM_MS
                ? "connecting"
                : "disconnected";
            }
            // Already 'connecting' (initial mount or aged-out outage).
            // Stay there — no thrashing back to Disconnected.
            return "connecting";
          });
        });
    fetchStatus();
    const statusInterval = setInterval(fetchStatus, 5000);

    // Poll FSM state on the same cadence — when the pipeline is
    // running the Service Status card swaps in a "Classifying"
    // spinner instead of the bare "Connected" indicator.  Failures
    // are silent: a missing FSM endpoint just means we fall back
    // to plain Connected.
    const fetchFsm = () =>
      fetch("/api/fsm/status")
        .then((r) => r.json())
        .then((data) => {
          const fsmState = data?.state ?? "IDLE";
          setPipelineRunning(!PIPELINE_TERMINAL_STATES.has(fsmState));
        })
        .catch(() => setPipelineRunning(false));
    fetchFsm();
    const fsmInterval = setInterval(fetchFsm, 5000);

    fetch("/api/agents")
      .then((r) => r.json())
      .then((data) => setAgents(data.agents || []))
      .catch(() => setAgents([]));

    const vocabParams = activeSourceId
      ? `?source_id=${encodeURIComponent(activeSourceId)}`
      : "";
    fetch(`/api/vocabulary/stats${vocabParams}`)
      .then((r) => r.json())
      .then((data) => setTermCount(data.terms ?? null))
      .catch(() => setTermCount(null));

    return () => {
      clearInterval(statusInterval);
      clearInterval(fsmInterval);
    };
  }, [activeSourceId]);

  const connectionDisplay: Record<
    ConnectionState,
    { label: string; color: string; icon: ReactNode }
  > = {
    connecting: {
      label: "Connecting",
      color: "#faad14",
      icon: <SyncOutlined spin />,
    },
    connected: {
      label: "Connected",
      color: "#52c41a",
      icon: <CheckCircleOutlined />,
    },
    classifying: {
      label: "Classifying",
      color: "#52c41a",
      icon: <SyncOutlined spin />,
    },
    degraded: {
      label: "Degraded",
      color: "#faad14",
      icon: <WarningOutlined />,
    },
    disconnected: {
      label: "Disconnected",
      color: "#ff4d4f",
      icon: <CloseCircleOutlined />,
    },
  };
  // Promote ``connected → classifying`` while the pipeline is in
  // flight; degraded/disconnected/connecting still take precedence
  // because a misbehaving service is more important to surface than
  // a busy one.
  const displayState: ConnectionState =
    connectionState === "connected" && pipelineRunning
      ? "classifying"
      : connectionState;
  const conn = connectionDisplay[displayState];

  const skillCount = useMemo(
    () => agents.reduce((n, a) => n + (a.tool_ids?.length || 0), 0),
    [agents],
  );

  // Entities = tables (datasets) + columns (row_count per dataset).
  // Aligns with Apache Atlas where both hive_table and hive_column are
  // first-class entity types that can be independently tagged with Terms.
  const entityCount = useMemo(
    () =>
      datasets.reduce((n, d) => n + (d.row_count || 0), 0) + datasets.length,
    [datasets],
  );

  // Operator-overridable display name (defaults to "Atelier" upstream;
  // CAI deployments can rebrand via ATELIER_APP_DISPLAY_NAME in the
  // SOPS-encrypted dotenv).  Falls back if /api/status hasn't returned
  // yet so the page never flashes blank.
  const appName = status?.config?.app_display_name ?? "Atelier";

  return (
    <>
      <Title level={2}>{appName}</Title>
      <Paragraph type="secondary">
        Agentic classification workbench powered by the Claude Agent SDK with
        interactive embedding visualization and adaptive keystone-agent
        orchestration.
      </Paragraph>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Link to="/status">
            <Card hoverable>
              <Statistic
                title="Service Status"
                value={conn.label}
                prefix={conn.icon}
                valueStyle={{ color: conn.color }}
              />
              {status?.grpc?.version && (
                <Tag color="blue" style={{ marginTop: 8 }}>
                  v{status.grpc.version}
                </Tag>
              )}
              {status?.degraded && (
                <div style={{ marginTop: 4, fontSize: 11, color: "#8c8c8c" }}>
                  {[
                    status.postgres?.ok ? null : "postgres",
                    status.qdrant?.ok ? null : "qdrant",
                  ]
                    .filter(Boolean)
                    .join(", ")}{" "}
                  unreachable
                </div>
              )}
            </Card>
          </Link>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Skills"
              value={skillCount}
              prefix={<ToolOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Entities"
              value={entityCount}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Terms"
              value={termCount ?? "—"}
              prefix={<BookOutlined />}
              suffix={
                sources.length > 1 ? (
                  <span style={{ fontSize: 12, color: "#8c8c8c" }}>
                    {sources.length} sources
                  </span>
                ) : null
              }
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} md={12} lg={8}>
          <Link to="/agents">
            <Card
              title="Agents"
              extra={<ExperimentOutlined />}
              hoverable
              style={{ height: "100%" }}
            >
              <Paragraph type="secondary">
                Define and orchestrate keystone agents using the Claude Agent SDK.
                Agents adapt as classification workflows evolve.
              </Paragraph>
            </Card>
          </Link>
        </Col>
        <Col xs={24} md={12} lg={8}>
          <Link to={activeDatasetId ? `/embeddings/${activeDatasetId}` : "/embeddings"}>
            <Card
              title="Embeddings"
              extra={<DotChartOutlined />}
              hoverable
              style={{ height: "100%" }}
            >
              <Paragraph type="secondary">
                Interactive visualization of classification embeddings.
                Explore results from the signals pipeline.
              </Paragraph>
              {(() => {
                const active = datasets.find((d) => d.id === activeDatasetId);
                return active ? (
                  <div style={{ marginTop: 12 }}>
                    <Tag color="blue">
                      {active.name} ({active.row_count?.toLocaleString() ?? "—"} entities)
                    </Tag>
                  </div>
                ) : null;
              })()}
            </Card>
          </Link>
        </Col>
        <Col xs={24} md={12} lg={8}>
          <Link to="/workflows">
            <Card
              title="Workflows"
              extra={<ClusterOutlined />}
              hoverable
              style={{ height: "100%" }}
            >
              <Paragraph type="secondary">
                Orchestration canvas — situational awareness for Claude's
                dynamic keystone agent coordination.
              </Paragraph>
              <Tag color="blue" style={{ marginTop: 8 }}>
                Preview
              </Tag>
            </Card>
          </Link>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col span={24}>
          <Card
            title={
              <span>
                <CodeOutlined style={{ marginRight: 8 }} />
                Terminal
              </span>
            }
            extra={
              <Tag
                color="geekblue"
                style={{ margin: 0, fontSize: 11 }}
              >
                Claude Agent SDK
              </Tag>
            }
            styles={{
              body: {
                padding: 0,
                background: "#0d1117",
                borderRadius: "0 0 6px 6px",
              },
            }}
          >
            <Suspense
              fallback={
                <div
                  style={{
                    height: 340,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "#0d1117",
                  }}
                >
                  <Spin />
                </div>
              }
            >
              <Terminal style={{ height: 340 }} />
            </Suspense>
          </Card>
        </Col>
      </Row>
    </>
  );
}

export default Landing;
