import { useEffect, useMemo, useState } from "react";
import {
  Card,
  Col,
  Divider,
  Row,
  Spin,
  Table,
  Tag,
  Typography,
} from "antd";
import {
  CloudServerOutlined,
  DashboardOutlined,
  DotChartOutlined,
  ExperimentOutlined,
  FunnelPlotOutlined,
  MergeCellsOutlined,
  ToolOutlined,
} from "@ant-design/icons";
import type { AgentInfo } from "../types/canvas";
import type { ReactNode } from "react";

const { Title, Paragraph, Text } = Typography;

// Synced with KeystoneNode.tsx
interface RoleTheme {
  icon: ReactNode;
  color: string;
  bg: string;
  label: string;
}

const ROLE_THEME: Record<string, RoleTheme> = {
  classifier: {
    icon: <ExperimentOutlined />,
    color: "#1890ff",
    bg: "#e6f7ff",
    label: "Classifier",
  },
  evidence_fuser: {
    icon: <MergeCellsOutlined />,
    color: "#52c41a",
    bg: "#f6ffed",
    label: "Evidence Fuser",
  },
  visualization_director: {
    icon: <DotChartOutlined />,
    color: "#722ed1",
    bg: "#f9f0ff",
    label: "Viz Director",
  },
  sampler: {
    icon: <FunnelPlotOutlined />,
    color: "#fa8c16",
    bg: "#fff7e6",
    label: "Sampler",
  },
  synth_generator: {
    icon: <CloudServerOutlined />,
    color: "#13c2c2",
    bg: "#e6fffb",
    label: "Synth Generator",
  },
};

interface SkillInfo {
  id: string;
  title: string;
  description: string;
  content: string;
}

interface SkillRow {
  key: string;
  skill: string;
  title: string;
  agentName: string;
  agentRole: string;
  description: string;
  content: string;
}

export default function Agents() {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/api/agents")
        .then((r) => r.json())
        .then((data) => data.agents || [])
        .catch(() => []),
      fetch("/api/skills")
        .then((r) => r.json())
        .then((data) => data.skills || [])
        .catch(() => []),
    ])
      .then(([agentsData, skillsData]) => {
        setAgents(agentsData);
        setSkills(skillsData);
      })
      .finally(() => setLoading(false));
  }, []);

  const skillMap = useMemo(
    () => new Map(skills.map((s) => [s.id, s])),
    [skills],
  );

  const skillRows: SkillRow[] = agents.flatMap((agent) =>
    (agent.tool_ids || []).map((skillId) => {
      const info = skillMap.get(skillId);
      return {
        key: `${agent.id}-${skillId}`,
        skill: skillId,
        title: info?.title || skillId,
        agentName: agent.name,
        agentRole: agent.role,
        description: info?.description || skillId,
        content: info?.content || "",
      };
    }),
  );

  const skillColumns = [
    {
      title: "Skill",
      dataIndex: "skill",
      key: "skill",
      render: (_text: string, record: SkillRow) => (
        <span>
          <ToolOutlined style={{ marginRight: 6, color: "#8c8c8c" }} />
          <Text strong>{record.title}</Text>
        </span>
      ),
    },
    {
      title: "Agent",
      dataIndex: "agentName",
      key: "agent",
      render: (name: string, record: SkillRow) => {
        const theme = ROLE_THEME[record.agentRole];
        return theme ? (
          <Tag
            icon={theme.icon}
            style={{ borderColor: theme.color, color: theme.color }}
          >
            {name}
          </Tag>
        ) : (
          <Tag>{name}</Tag>
        );
      },
    },
    {
      title: "Description",
      dataIndex: "description",
      key: "description",
    },
  ];

  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          paddingTop: 64,
        }}
      >
        <Spin size="large" />
      </div>
    );
  }

  return (
    <>
      <Title level={2}>Keystone Agents</Title>
      <Paragraph type="secondary">
        Atelier's classification pipeline is driven by keystone agents,
        each specialized for a distinct phase of the Dempster-Shafer
        evidence fusion workflow. Agents are optimized using GEPA
        (Pareto-optimal prompt evolution) and Agent Lightning APO
        (automatic prompt optimization with textual gradients).
      </Paragraph>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        {agents.map((agent) => {
          const theme = ROLE_THEME[agent.role] || ROLE_THEME.classifier;
          return (
            <Col xs={24} md={8} key={agent.id}>
              <Card
                style={{
                  height: "100%",
                  borderTop: `3px solid ${theme.color}`,
                }}
                styles={{ header: { background: theme.bg } }}
                title={
                  <span style={{ color: theme.color }}>
                    {theme.icon}
                    <span style={{ marginLeft: 8 }}>{agent.name}</span>
                  </span>
                }
                extra={
                  <Tag
                    style={{
                      borderColor: theme.color,
                      color: theme.color,
                      background: "transparent",
                    }}
                  >
                    {agent.role.replace(/_/g, " ")}
                  </Tag>
                }
              >
                <Paragraph>{agent.description}</Paragraph>

                <div style={{ marginBottom: 16 }}>
                  <Text
                    strong
                    style={{ fontSize: 12, color: "#8c8c8c" }}
                  >
                    <ToolOutlined style={{ marginRight: 4 }} />
                    SKILLS
                  </Text>
                  <div style={{ marginTop: 8 }}>
                    {(agent.tool_ids || []).length > 0 ? (
                      agent.tool_ids.map((skillId) => {
                        const info = skillMap.get(skillId);
                        return (
                          <Tag key={skillId} style={{ marginBottom: 4 }}>
                            {info?.title || skillId}
                          </Tag>
                        );
                      })
                    ) : (
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        No skills configured
                      </Text>
                    )}
                  </div>
                </div>

                <div
                  style={{
                    borderTop: "1px dashed #f0f0f0",
                    paddingTop: 12,
                    marginTop: 8,
                  }}
                >
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    <DashboardOutlined style={{ marginRight: 4 }} />
                    Operational metrics available when OTel is configured
                  </Text>
                </div>
              </Card>
            </Col>
          );
        })}
      </Row>

      <Divider />

      <Title level={3}>Skills</Title>
      <Paragraph type="secondary">
        Skills are Claude Code slash commands defined as markdown files in{" "}
        <Text code>.claude/commands/</Text>. Each skill provides
        domain-specific instructions that agents use to perform classification
        tasks via the Claude Agent SDK.
      </Paragraph>
      <Table
        dataSource={skillRows}
        columns={skillColumns}
        pagination={false}
        size="small"
        style={{ marginTop: 16 }}
        expandable={{
          expandedRowRender: (record) => (
            <pre
              style={{
                margin: 0,
                padding: 16,
                background: "#fafafa",
                borderRadius: 4,
                fontSize: 13,
                lineHeight: 1.6,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                maxHeight: 400,
                overflow: "auto",
              }}
            >
              {record.content || "No content available"}
            </pre>
          ),
          rowExpandable: (record) => !!record.content,
        }}
      />
    </>
  );
}
