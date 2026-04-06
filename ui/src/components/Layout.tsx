import { Layout as AntLayout, Typography } from "antd";
import type { ReactNode } from "react";
import clouderaLogo from "../assets/Cloudera.svg";

const { Header, Content, Footer } = AntLayout;
const { Text } = Typography;

interface LayoutProps {
  children: ReactNode;
}

function Layout({ children }: LayoutProps) {
  return (
    <AntLayout style={{ minHeight: "100vh" }}>
      <Header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          background: "#001529",
          padding: "0 24px",
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
      <Content style={{ padding: "24px 48px" }}>{children}</Content>
      <Footer style={{ textAlign: "center" }}>
        <Text type="secondary">
          Atelier v0.1.0 &mdash; Agentic Classification Workbench
        </Text>
      </Footer>
    </AntLayout>
  );
}

export default Layout;
