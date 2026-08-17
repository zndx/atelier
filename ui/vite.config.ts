import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Silently swallow proxy errors during startup (gateway starts slower than Vite)
const silenceProxyError = (err: Error, _req: unknown, _res: unknown) => {
  if ((err as NodeJS.ErrnoException).code === "ECONNREFUSED") return;
  console.error("[vite] proxy error:", err.message);
};

// Honor ${CDSW_APP_PORT:-8090} so the proxy follows whatever port CAI assigns
// at runtime instead of the hardcoded local-dev 8090.
const gatewayPort = process.env.CDSW_APP_PORT || "8090";
// :3000 is Gaius Tilt / Metaflow (kubectl port-forward). Atelier Vite owns
// :3300 and must not silently bump — waffle advertises this port.
const vitePort = Number(process.env.ATELIER_VITE_PORT || "3300");

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ["embedding-atlas"],
  },
  server: {
    host: "0.0.0.0",
    port: vitePort,
    strictPort: true,
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
