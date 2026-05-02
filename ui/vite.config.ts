// Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
//
// This file contains material proprietary to Cloudera, Inc., and is provided
// to authorized licensees solely for use in connection with the Cloudera AI
// (CAI) Application from which it was obtained.  It may not be copied,
// modified, redistributed, or used in any other manner without the express
// written consent of Cloudera, Inc.

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
