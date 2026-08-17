import { Layout as AntLayout, Tooltip, Typography } from "antd";
import { MoonOutlined, SettingOutlined, SunOutlined } from "@ant-design/icons";
import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";
import clouderaLogo from "../assets/Cloudera.svg";
import { toggleColorMode, useColorMode } from "../theme/colorMode";
import WaffleMenu from "./WaffleMenu";

const { Header, Content, Footer } = AntLayout;
const { Text } = Typography;

const NAV_ITEMS = [
  { path: "/agents", label: "Agents" },
  { path: "/workflows", label: "Workflows" },
  { path: "/terminal", label: "Terminal" },
  { path: "/embeddings", label: "Embeddings" },
  { path: "/status", label: "Status" },
];

// Header chrome stays on the DARK ramp in both modes (see base.css note:
// raster logo asset). These are the k-ramp dark chrome colors, fixed.
const CHROME = {
  text: "#e5e8ec",
  dim: "#9fa5ac",
  fill: "#21272d",
};

interface LayoutProps {
  children: ReactNode;
  /** Lock to viewport height, hide footer — for full-bleed pages like Embeddings. */
  fullHeight?: boolean;
}

function Layout({ children, fullHeight }: LayoutProps) {
  const { pathname } = useLocation();
  const mode = useColorMode();

  const chromeLink = (active: boolean): React.CSSProperties => ({
    color: active ? CHROME.text : CHROME.dim,
    padding: "4px 12px",
    borderRadius: 4,
    fontSize: 14,
    textDecoration: "none",
    background: active ? CHROME.fill : "transparent",
    transition: "all 0.2s",
  });

  return (
    <AntLayout
      style={fullHeight
        ? { height: "100vh", overflow: "hidden" }
        : { minHeight: "100vh" }
      }
    >
      <Header
        className="atelier-chrome"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          background: "transparent", // surface comes from .atelier-chrome
          padding: "0 clamp(12px, 2vw, 24px)",
        }}
      >
        <Link to="/" style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <img src={clouderaLogo} alt="Cloudera" style={{ height: 28 }} />
          <Text
            strong
            style={{ color: CHROME.text, fontSize: 18, letterSpacing: 0.5 }}
          >
            Atelier
          </Text>
        </Link>
        <nav style={{ display: "flex", gap: 4, marginLeft: 16 }}>
          {NAV_ITEMS.map(({ path, label }) => {
            const active = pathname === path || pathname.startsWith(path + "/");
            return (
              <Link key={path} to={path} style={chromeLink(active)}>
                {label}
              </Link>
            );
          })}
        </nav>
        <div
          style={{
            marginLeft: "auto",
            display: "flex",
            alignItems: "center",
            gap: 6,
          }}
        >
          <WaffleMenu />
          <Tooltip
            title={mode === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            placement="bottomRight"
          >
            <a
              aria-label="Toggle color mode"
              onClick={() => toggleColorMode(mode)}
              style={{
                color: CHROME.dim,
                padding: "6px 10px",
                borderRadius: 4,
                fontSize: 18,
                lineHeight: 1,
                display: "inline-flex",
                alignItems: "center",
                cursor: "pointer",
                transition: "all 0.2s",
              }}
            >
              {mode === "dark" ? <SunOutlined /> : <MoonOutlined />}
            </a>
          </Tooltip>
          <Tooltip title="Settings — tune the DST pipeline" placement="bottomRight">
            <Link
              to="/settings"
              aria-label="Settings"
              style={{
                color: pathname.startsWith("/settings") ? CHROME.text : CHROME.dim,
                padding: "6px 10px",
                borderRadius: 4,
                fontSize: 18,
                lineHeight: 1,
                display: "inline-flex",
                alignItems: "center",
                background: pathname.startsWith("/settings")
                  ? CHROME.fill
                  : "transparent",
                transition: "all 0.2s",
              }}
            >
              <SettingOutlined />
            </Link>
          </Tooltip>
        </div>
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
        <Footer style={{ textAlign: "center", background: "transparent" }}>
          <Text type="secondary">
            Atelier v0.5.1 &mdash; Agentic Classification Workbench
          </Text>
        </Footer>
      )}
    </AntLayout>
  );
}

export default Layout;
