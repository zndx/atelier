import { Layout as AntLayout, Typography } from "antd";
import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import clouderaLogo from "../assets/Cloudera.svg";

const { Header, Content, Footer } = AntLayout;
const { Text } = Typography;

const NAV_ITEMS = [
  { path: "/agents", label: "Agents" },
  { path: "/workflows", label: "Workflows" },
  { path: "/terminal", label: "Terminal" },
  { path: "/embeddings", label: "Embeddings" },
  { path: "/status", label: "Status" },
];

interface LayoutProps {
  children: ReactNode;
  /** Lock to viewport height, hide footer — for full-bleed pages like Embeddings. */
  fullHeight?: boolean;
}

function Layout({ children, fullHeight }: LayoutProps) {
  const { pathname } = useLocation();

  return (
    <AntLayout
      style={fullHeight
        ? { height: "100vh", overflow: "hidden" }
        : { minHeight: "100vh" }
      }
    >
      <Header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          background: "#001529",
          padding: "0 clamp(12px, 2vw, 24px)",
        }}
      >
        <Link to="/" style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <img src={clouderaLogo} alt="Cloudera" style={{ height: 28 }} />
          <Text
            strong
            style={{ color: "#fff", fontSize: 18, letterSpacing: 0.5 }}
          >
            Atelier
          </Text>
        </Link>
        <nav style={{ display: "flex", gap: 4, marginLeft: 16 }}>
          {NAV_ITEMS.map(({ path, label }) => {
            const active = pathname === path || pathname.startsWith(path + "/");
            return (
              <Link
                key={path}
                to={path}
                style={{
                  color: active ? "#fff" : "rgba(255,255,255,0.65)",
                  padding: "4px 12px",
                  borderRadius: 4,
                  fontSize: 14,
                  textDecoration: "none",
                  background: active ? "rgba(255,255,255,0.1)" : "transparent",
                  transition: "all 0.2s",
                }}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </Header>
      <Content
        style={fullHeight
          ? { padding: "0 8px", display: "flex", flexDirection: "column", overflow: "hidden" }
          : { padding: "24px clamp(16px, 3vw, 48px)" }
        }
      >
        {children}
      </Content>
      {!fullHeight && (
        <Footer style={{ textAlign: "center" }}>
          <Text type="secondary">
            Atelier v0.1.0 &mdash; Agentic Classification Workbench
          </Text>
        </Footer>
      )}
    </AntLayout>
  );
}

export default Layout;
