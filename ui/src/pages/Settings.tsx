import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  InputNumber,
  Segmented,
  Slider,
  Space,
  Spin,
  Switch,
  Tabs,
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

// ── Metadata shape ──────────────────────────────────────────────────

type BaseMeta = {
  hocon_path: string;
  label: string;
  description: string;
  group: "convergence" | "evidence" | "sampling" | "training" | "llm_system";
  default_focus?: boolean;
};

type ChoiceMeta = BaseMeta & {
  type: "choice";
  choices: string[];
  default: string;
  captions: Record<string, string>;
};

type FloatMeta = BaseMeta & {
  type: "float";
  min: number;
  max: number;
  step: number;
  default: number;
  caption_template: string;
};

type IntMeta = BaseMeta & {
  type: "int";
  min: number;
  max: number;
  step: number;
  default: number;
  caption_template: string;
};

type SwitchMeta = BaseMeta & {
  type: "switch";
  default: boolean;
  captions: { true: string; false: string } | Record<string, string>;
};

type ParamMeta = ChoiceMeta | FloatMeta | IntMeta | SwitchMeta;

type ParamValue = string | number | boolean;

type SettingsResponse = {
  metadata: Record<string, ParamMeta>;
  values: Record<string, ParamValue>;
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

// ── Tab config ──────────────────────────────────────────────────────

type TabSpec = {
  key: BaseMeta["group"];
  label: string;
};

const TABS: TabSpec[] = [
  { key: "convergence", label: "Convergence" },
  { key: "evidence", label: "Evidence & Fusion" },
  { key: "sampling", label: "Sampling" },
  { key: "training", label: "Training" },
  { key: "llm_system", label: "LLM & System" },
];

// ── Rendering helpers ───────────────────────────────────────────────

/** Expand {value}, {value_pct} in a caption template. */
function renderNumericCaption(template: string, v: number, step: number): string {
  const decimals = step < 1 ? Math.max(0, -Math.floor(Math.log10(step))) : 0;
  return template
    .replace(/\{value_pct\}/g, Math.round(v * 100).toString())
    .replace(/\{value\}/g, v.toFixed(decimals));
}

function driftTone(defaultV: number, current: number, step: number): "default" | "up" | "down" {
  const delta = current - defaultV;
  if (Math.abs(delta) < step / 2) return "default";
  return delta > 0 ? "up" : "down";
}

const toneColor: Record<"default" | "up" | "down", string> = {
  default: "#389e0d",   // green  — at HOCON default
  up: "#d48806",        // amber  — looser / higher
  down: "#1677ff",      // blue   — stricter / lower
};

// ── Control card ────────────────────────────────────────────────────

type ControlCardProps = {
  paramKey: string;
  meta: ParamMeta;
  current: ParamValue;
  pending: boolean;
  session: boolean;
  onChange: (key: string, value: ParamValue) => void;
};

function ControlCard({ paramKey, meta, current, pending, session, onChange }: ControlCardProps) {
  const header = (
    <Space>
      <Text strong>{meta.label}</Text>
      <Text code style={{ fontSize: 11 }}>
        {meta.hocon_path}
      </Text>
      {pending && <Tag color="blue">pending</Tag>}
      {!pending && session && <Tag color="geekblue">session</Tag>}
    </Space>
  );

  let body: React.ReactNode;
  let captionText: string;
  let captionColor: string;

  if (meta.type === "choice") {
    const v = String(current);
    captionText = meta.captions[v] || "";
    captionColor = v === meta.default ? toneColor.default : toneColor.down;
    body = (
      <Segmented
        value={v}
        options={meta.choices.map((c) => ({
          label: c.charAt(0).toUpperCase() + c.slice(1),
          value: c,
        }))}
        onChange={(val) => onChange(paramKey, String(val))}
      />
    );
  } else if (meta.type === "switch") {
    const v = Boolean(current);
    const vs = v ? "true" : "false";
    captionText = (meta.captions[vs as "true" | "false"] as string) || "";
    captionColor = v === meta.default ? toneColor.default : toneColor.down;
    body = (
      <Switch
        checked={v}
        onChange={(val) => onChange(paramKey, val)}
      />
    );
  } else if (meta.type === "int") {
    const v = Number(current);
    const tone = driftTone(meta.default, v, meta.step || 1);
    captionText = renderNumericCaption(meta.caption_template, v, meta.step || 1);
    captionColor = toneColor[tone];
    body = (
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 120px",
          gap: 16,
          alignItems: "center",
        }}
      >
        <Slider
          min={meta.min}
          max={meta.max}
          step={meta.step || 1}
          value={v}
          onChange={(val) => onChange(paramKey, Number(val))}
          marks={{
            [meta.min]: String(meta.min),
            [meta.default]: { label: "default", style: { color: "#389e0d" } },
            [meta.max]: String(meta.max),
          }}
        />
        <InputNumber
          min={meta.min}
          max={meta.max}
          step={meta.step || 1}
          value={v}
          onChange={(val) => {
            if (val != null) onChange(paramKey, Number(val));
          }}
          style={{ width: "100%" }}
        />
      </div>
    );
  } else {
    // float
    const v = Number(current);
    const tone = driftTone(meta.default, v, meta.step);
    captionText = renderNumericCaption(meta.caption_template, v, meta.step);
    captionColor = toneColor[tone];
    const decimals = meta.step < 1
      ? Math.max(0, -Math.floor(Math.log10(meta.step)))
      : 0;
    body = (
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
          onChange={(val) => onChange(paramKey, Number(val))}
          marks={{
            [meta.min]: String(meta.min),
            [meta.default]: { label: "default", style: { color: "#389e0d" } },
            [meta.max]: String(meta.max),
          }}
        />
        <Text
          style={{
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
            fontSize: 16,
            textAlign: "right",
          }}
        >
          {v.toFixed(decimals)}
        </Text>
      </div>
    );
  }

  return (
    <Card key={paramKey} size="small" title={header}>
      <Paragraph type="secondary" style={{ marginBottom: 12 }}>
        {meta.description}
      </Paragraph>
      {body}
      <div style={{ marginTop: 12 }}>
        <Text
          style={{
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
            fontSize: 13,
            color: captionColor,
          }}
        >
          {captionText}
        </Text>
      </div>
    </Card>
  );
}

// ── Page ────────────────────────────────────────────────────────────

export default function Settings() {
  const [data, setData] = useState<SettingsResponse | null>(null);
  const [accel, setAccel] = useState<AccelerationInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<Record<string, ParamValue>>({});
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState<string>("convergence");

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
    if (!data) return {} as Record<string, ParamValue>;
    return { ...data.values, ...pending };
  }, [data, pending]);

  /** Group parameter keys by tab assignment. */
  const byTab = useMemo(() => {
    const out: Record<string, string[]> = {
      convergence: [],
      evidence: [],
      sampling: [],
      training: [],
      llm_system: [],
    };
    if (!data) return out;
    for (const [key, meta] of Object.entries(data.metadata)) {
      const group = meta.group;
      if (group in out) out[group].push(key);
    }
    // Stable alphabetical order within each tab. Could switch to a
    // curated ordering later if we want "most-important-first" per tab.
    for (const g of Object.keys(out)) out[g].sort();
    return out;
  }, [data]);

  const dirty = Object.keys(pending).length > 0;

  const handleChange = (key: string, value: ParamValue) => {
    setPending((p) => ({ ...p, [key]: value }));
  };

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

  const renderTab = (group: string) => (
    <Space direction="vertical" size={16} style={{ width: "100%" }}>
      {byTab[group].map((key) => {
        const meta = data.metadata[key];
        if (!meta) return null;
        return (
          <ControlCard
            key={key}
            paramKey={key}
            meta={meta}
            current={effective[key]}
            pending={key in pending}
            session={!(key in pending) && data.overlay_keys.includes(key)}
            onChange={handleChange}
          />
        );
      })}
    </Space>
  );

  const tabItems = TABS.map((t) => ({
    key: t.key,
    label: (
      <Space size={4}>
        <span>{t.label}</span>
        <Tag style={{ marginRight: 0 }} color="default">
          {byTab[t.key]?.length ?? 0}
        </Tag>
      </Space>
    ),
    children: renderTab(t.key),
  }));

  return (
    <div style={{ maxWidth: 960, margin: "0 auto" }}>
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

        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={tabItems}
          tabBarStyle={{ marginBottom: 16 }}
        />

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
