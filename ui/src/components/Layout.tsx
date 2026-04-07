import { Layout as AntLayout, Typography } from "antd";
import type { ReactNode } from "react";
import clouderaLogo from "../assets/Cloudera.svg";

const { Header, Content, Footer } = AntLayout;
const { Text } = Typography;

interface LayoutProps {
  children: ReactNode;
  /** Lock to viewport height, hide footer — for full-bleed pages like the Embeddings Viewer. */
  fullHeight?: boolean;
}

function Layout({ children, fullHeight }: LayoutProps) {
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
        <img src={clouderaLogo} alt="Cloudera" style={{ height: 28 }} />
        <Text
          strong
          style={{ color: "#fff", fontSize: 18, letterSpacing: 0.5 }}
        >
          Atelier
        </Text>
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
