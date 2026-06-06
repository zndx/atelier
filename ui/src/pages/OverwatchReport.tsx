import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import {
  Button,
  Card,
  message,
  Segmented,
  Space,
  Spin,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import {
  CopyOutlined,
  DislikeOutlined,
  EyeOutlined,
  LikeOutlined,
} from "@ant-design/icons";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

const { Title, Text } = Typography;

type ViewMode = "rendered" | "source";

export default function OverwatchReport() {
  const { runId } = useParams<{ runId: string }>();
  const [report, setReport] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("rendered");
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);

  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    fetch(`/api/overwatch/report/${encodeURIComponent(runId)}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.ok) {
          setReport(data.report);
        } else {
          setError(data.error || "Report not found");
        }
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [runId]);

  const handleCopy = () => {
    if (report) {
      navigator.clipboard.writeText(report).then(
        () => message.success("Copied to clipboard"),
        () => message.error("Failed to copy"),
      );
    }
  };

  const handleFeedback = (type: "up" | "down") => {
    setFeedback(type);
    message.info(
      type === "up"
        ? "Thanks! Positive feedback recorded."
        : "Thanks — we'll improve the analysis.",
    );
    // Future: POST feedback to /api/overwatch/feedback/{runId}
  };

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: 80 }}>
        <Spin size="large" />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: 24 }}>
        <Card>
          <Text type="danger">{error}</Text>
        </Card>
      </div>
    );
  }

  return (
    <div style={{ padding: "0" }}>
      <div style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 12 }}>
        <Title level={4} style={{ margin: 0 }}>
          Overwatch Report
        </Title>
        <Tag color="purple">{runId?.slice(0, 8)}</Tag>
      </div>

      <Card
        extra={
          <Space>
            <Segmented
              size="small"
              value={viewMode}
              onChange={(v) => setViewMode(v as ViewMode)}
              options={[
                { label: "Rendered", value: "rendered", icon: <EyeOutlined /> },
                { label: "Source", value: "source" },
              ]}
            />
            <Tooltip title="Copy to clipboard">
              <Button
                icon={<CopyOutlined />}
                size="small"
                onClick={handleCopy}
              />
            </Tooltip>
            <Tooltip title="Helpful">
              <Button
                icon={<LikeOutlined />}
                size="small"
                type={feedback === "up" ? "primary" : "default"}
                onClick={() => handleFeedback("up")}
              />
            </Tooltip>
            <Tooltip title="Not helpful">
              <Button
                icon={<DislikeOutlined />}
                size="small"
                danger={feedback === "down"}
                type={feedback === "down" ? "primary" : "default"}
                onClick={() => handleFeedback("down")}
              />
            </Tooltip>
          </Space>
        }
      >
        {viewMode === "rendered" ? (
          <div className="overwatch-markdown">
            <Markdown remarkPlugins={[remarkGfm]}>{report || ""}</Markdown>
          </div>
        ) : (
          <pre
            style={{
              background: "#0d1117",
              color: "#c9d1d9",
              padding: 16,
              borderRadius: 6,
              overflow: "auto",
              fontSize: 13,
              fontFamily: 'Menlo, Monaco, "Courier New", monospace',
              maxHeight: "70vh",
              whiteSpace: "pre-wrap",
            }}
          >
            {report}
          </pre>
        )}
      </Card>
    </div>
  );
}
