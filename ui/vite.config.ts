import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Silently swallow proxy errors during startup (gateway starts slower than Vite)
const silenceProxyError = (err: Error, _req: unknown, _res: unknown) => {
  if ((err as NodeJS.ErrnoException).code === "ECONNREFUSED") return;
  console.error("[vite] proxy error:", err.message);
};

const gatewayPort = process.env.CDSW_APP_PORT || "8090";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ["embedding-atlas"],
  },
  server: {
    host: "0.0.0.0",
    port: 3000,
    allowedHosts: true,
    proxy: {
      "/api": {
        target: `http://localhost:${gatewayPort}`,
        changeOrigin: true,
        configure: (proxy) => { proxy.on("error", silenceProxyError); },
      },
      "/ws": {
        target: `ws://localhost:${gatewayPort}`,
        ws: true,
        configure: (proxy) => { proxy.on("error", silenceProxyError); },
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
