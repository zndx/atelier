import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Segmented,
  Slider,
  Space,
  Spin,
  Tag,
  Typography,
  message,
} from "antd";
import {
  ReloadOutlined,
  RocketOutlined,
  SaveOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";

const { Title, Text, Paragraph } = Typography;

type ChoiceMeta = {
  hocon_path: string;
  label: string;
  description: string;
  type: "choice";
  choices: string[];
  default: string;
  captions: Record<string, string>;
};

type FloatMeta = {
  hocon_path: string;
  label: string;
  description: string;
  type: "float";
  min: number;
  max: number;
  step: number;
  default: number;
  caption_template: string;
};

type ParamMeta = ChoiceMeta | FloatMeta;

type SettingsResponse = {
  metadata: Record<string, ParamMeta>;
  values: Record<string, string | number>;
  overlay_keys: string[];
};

type AccelerationInfo = {
  available: boolean;
  device_count: number;
  devices: string[];
  vram_total_mib: number[];
  vram_free_mib: number[];
  driver_cuda_version: string;
  pytorch_cuda_version: string;
  summary: string;
  warnings: string[];
  methods: {
    sage: boolean;
    sage_gpu: boolean;
    shap_gpu: boolean;
    catboost_gpu: boolean;
    embedding_sharded: boolean;
  };
};

const ORDER = [
  "classify_fusion_strategy",
  "classify_llm_discount",
  "classify_discount_cosine",
  "classify_bootstrap_gap_threshold",
  "classify_bootstrap_bel_floor",
];

/** Render the caption for a float — expand {value}, {value_pct}. */
function renderCaption(meta: FloatMeta, v: number): string {
  return meta.caption_template
    .replace(/\{value_pct\}/g, Math.round(v * 100).toString())
    .replace(/\{value\}/g, v.toFixed(2));
}

/** Classify direction of a tuned value relative to the default.
 *  Used to tint the caption (green / blue / amber). */
function driftTone(meta: FloatMeta, v: number): "default" | "up" | "down" {
  const delta = v - meta.default;
  if (Math.abs(delta) < meta.step / 2) return "default";
  return delta > 0 ? "up" : "down";
}

const toneColor: Record<"default" | "up" | "down", string> = {
  default: "#389e0d",   // green  — at HOCON default
  up: "#d48806",        // amber  — looser / more ignorance
  down: "#1677ff",      // blue   — stricter / more trust
};

export default function Settings() {
  const [data, setData] = useState<SettingsResponse | null>(null);
  const [accel, setAccel] = useState<AccelerationInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<Record<string, string | number>>({});
  const [saving, setSaving] = useState(false);

  const load = () => {
    setLoading(true);
    fetch("/api/settings")
      .then((r) => r.json())
      .then((d: SettingsResponse | { error: string }) => {
        if ("error" in d) throw new Error(d.error);
        setData(d);
        setPending({});
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
    fetch("/api/acceleration")
      .then((r) => r.json())
      .then((d: AccelerationInfo | { error: string }) => {
        if (!("error" in d)) setAccel(d);
      })
      .catch(() => {});
  };

  useEffect(() => {
    load();
  }, []);

  const effective = useMemo(() => {
    if (!data) return {};
    return { ...data.values, ...pending };
  }, [data, pending]);

  const dirty = Object.keys(pending).length > 0;

  const save = async () => {
    setSaving(true);
    try {
      const r = await fetch("/api/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pending),
      });
      const body = await r.json();
      if (!r.ok || body.error) {
        message.error(body.error || `PATCH failed: ${r.status}`);
      } else {
        message.success("Settings applied — next pipeline run will use them.");
        load();
      }
    } catch (e) {
      message.error(String(e));
    } finally {
      setSaving(false);
    }
  };

  const reset = async () => {
    setSaving(true);
    try {
      const r = await fetch("/api/settings/reset", { method: "POST" });
      const body = await r.json();
      if (!r.ok || body.error) {
        message.error(body.error || `Reset failed: ${r.status}`);
      } else {
        message.success("Reverted to HOCON defaults.");
        load();
      }
    } catch (e) {
      message.error(String(e));
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div style={{ display: "flex", justifyContent: "center", padding: 60 }}>
        <Spin size="large" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <Alert
        type="error"
        message="Failed to load settings"
        description={error || "Unknown error"}
        showIcon
      />
    );
  }

  return (
    <div style={{ maxWidth: 880, margin: "0 auto" }}>
      <Space direction="vertical" size={20} style={{ width: "100%" }}>
        <div>
          <Title level={2} style={{ marginBottom: 4 }}>
            Settings
          </Title>
          <Text type="secondary">
            Session-level tuning of the DST classification pipeline.
          </Text>
        </div>

        <Alert
          type="info"
          showIcon
          message="Changes apply to the next pipeline run."
          description={
            <>
              Session overlay only — values reset when the gateway restarts.
              For permanent changes, edit <Text code>config/base.conf</Text> or
              set the corresponding environment variable.
            </>
          }
        />

        {accel && (
          <Card
            size="small"
            title={
              <Space>
                {accel.available ? (
                  <ThunderboltOutlined style={{ color: "#faad14" }} />
                ) : (
                  <RocketOutlined style={{ color: "#8c8c8c" }} />
                )}
                <Text strong>Acceleration</Text>
                <Tag color={accel.available ? "gold" : "default"}>
                  {accel.available ? "GPU" : "CPU"}
                </Tag>
              </Space>
            }
          >
            <Space direction="vertical" size={4} style={{ width: "100%" }}>
              <Text
                style={{
                  fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                  fontSize: 13,
                }}
              >
                {accel.summary}
              </Text>
              {accel.available && (
                <Space wrap size={[8, 4]}>
                  {accel.methods.sage_gpu && (
                    <Tag color="blue">SAGE: GPU kernel</Tag>
                  )}
                  {accel.methods.shap_gpu && (
                    <Tag color="blue">PermutationSHAP: GPU kernel</Tag>
                  )}
                  {accel.methods.catboost_gpu && (
                    <Tag color="blue">CatBoost: GPU training</Tag>
                  )}
                  {accel.methods.embedding_sharded && (
                    <Tag color="purple">
                      Embedding: sharded across {accel.device_count}
                    </Tag>
                  )}
                </Space>
              )}
              {!accel.available && accel.device_count > 0 && (
                <Text type="warning" style={{ fontSize: 12 }}>
                  {accel.device_count}× GPU detected but CUDA unavailable —
                  running on CPU.
                </Text>
              )}
              {accel.warnings?.length > 0 && (
                <div>
                  {accel.warnings.map((w, i) => (
                    <Text key={i} type="warning" style={{ fontSize: 12 }}>
                      ⚠ {w}
                    </Text>
                  ))}
                </div>
              )}
            </Space>
          </Card>
        )}

        {ORDER.map((key) => {
          const meta = data.metadata[key];
          if (!meta) return null;
          const current = effective[key];
          const isOverride = key in pending;
          const isOverlay = data.overlay_keys.includes(key);

          if (meta.type === "choice") {
            const v = String(current);
            const caption = meta.captions[v] || "";
            return (
              <Card
                key={key}
                size="small"
                title={
                  <Space>
                    <Text strong>{meta.label}</Text>
                    <Text code style={{ fontSize: 11 }}>
                      {meta.hocon_path}
                    </Text>
                    {isOverride && <Tag color="blue">pending</Tag>}
                    {!isOverride && isOverlay && (
                      <Tag color="geekblue">session</Tag>
                    )}
                  </Space>
                }
              >
                <Paragraph type="secondary" style={{ marginBottom: 12 }}>
                  {meta.description}
                </Paragraph>
                <Segmented
                  value={v}
                  options={meta.choices.map((c) => ({
                    label: c.charAt(0).toUpperCase() + c.slice(1),
                    value: c,
                  }))}
                  onChange={(val) =>
                    setPending((p) => ({ ...p, [key]: String(val) }))
                  }
                />
                <div style={{ marginTop: 12 }}>
                  <Text
                    style={{
                      fontFamily:
                        "ui-monospace, SFMono-Regular, Menlo, monospace",
                      fontSize: 13,
                      color: v === meta.default ? "#389e0d" : "#1677ff",
                    }}
                  >
                    {caption}
                  </Text>
                </div>
              </Card>
            );
          }

          // Float slider
          const v = Number(current);
          const tone = driftTone(meta, v);
          const caption = renderCaption(meta, v);
          return (
            <Card
              key={key}
              size="small"
              title={
                <Space>
                  <Text strong>{meta.label}</Text>
                  <Text code style={{ fontSize: 11 }}>
                    {meta.hocon_path}
                  </Text>
                  {isOverride && <Tag color="blue">pending</Tag>}
                  {!isOverride && isOverlay && (
                    <Tag color="geekblue">session</Tag>
                  )}
                </Space>
              }
            >
              <Paragraph type="secondary" style={{ marginBottom: 12 }}>
                {meta.description}
              </Paragraph>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 64px",
                  gap: 16,
                  alignItems: "center",
                }}
              >
                <Slider
                  min={meta.min}
                  max={meta.max}
                  step={meta.step}
                  value={v}
                  onChange={(val) =>
                    setPending((p) => ({ ...p, [key]: Number(val) }))
                  }
                  marks={{
                    [meta.min]: String(meta.min),
                    [meta.default]: { label: "default", style: { color: "#389e0d" } },
                    [meta.max]: String(meta.max),
                  }}
                />
                <Text
                  style={{
                    fontFamily:
                      "ui-monospace, SFMono-Regular, Menlo, monospace",
                    fontSize: 16,
                    textAlign: "right",
                  }}
                >
                  {v.toFixed(2)}
                </Text>
              </div>
              <div style={{ marginTop: 16 }}>
                <Text
                  style={{
                    fontFamily:
                      "ui-monospace, SFMono-Regular, Menlo, monospace",
                    fontSize: 13,
                    color: toneColor[tone],
                  }}
                >
                  {caption}
                </Text>
              </div>
            </Card>
          );
        })}

        <Space
          style={{
            display: "flex",
            justifyContent: "space-between",
            padding: "8px 0 32px 0",
          }}
        >
          <Button
            icon={<ReloadOutlined />}
            onClick={reset}
            disabled={saving || data.overlay_keys.length === 0}
          >
            Reset to defaults
          </Button>
          <Space>
            <Button
              onClick={() => setPending({})}
              disabled={!dirty || saving}
            >
              Discard changes
            </Button>
            <Button
              type="primary"
              icon={<SaveOutlined />}
              loading={saving}
              disabled={!dirty}
              onClick={save}
            >
              Save
            </Button>
          </Space>
        </Space>
      </Space>
    </div>
  );
}
