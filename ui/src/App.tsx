import { ConfigProvider, theme } from "antd";
import Layout from "./components/Layout";
import Landing from "./pages/Landing";

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
      <Layout>
        <Landing />
      </Layout>
    </ConfigProvider>
  );
}

export default App;
