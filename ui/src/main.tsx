import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles/theme-keiretsu.css";
import "./styles/base.css";
import { initColorMode } from "./theme/colorMode";

// Keiretsu is THE Atelier site theme (decision 2026-07-23); only data-mode
// flips. Both set before first render so there is no unthemed flash.
document.documentElement.setAttribute("data-theme", "keiretsu");
initColorMode(); // restores saved dark|light; org default = dark

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
