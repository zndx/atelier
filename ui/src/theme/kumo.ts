/**
 * Kumo token mirror + AntD bridge — Atelier-born (candidate for upstream
 * feedback into cldr-design-template as the AntD-stack reference bridge).
 *
 * Values mirror styles/theme-keiretsu.css (canonical:
 * cldr-design-template@db1a423) — AntD's ConfigProvider needs concrete
 * values, not CSS vars, so the ramp is restated here. Keep the two in sync
 * when re-syncing the css from upstream.
 *
 * Keiretsu laws applied to the AntD token mapping:
 *   1. Elevation is lightness — colorBgLayout=canvas(k0) < colorBgBase(k1)
 *      < colorBgContainer/Elevated(k2); no hue shifts between surfaces.
 *   2. Accents held to accent duty — brand only on primary/link/focus.
 *   3. Contrast band — text tokens land 10-13:1 on their surfaces.
 */
import { theme as antdTheme } from "antd";
import type { ThemeConfig } from "antd";

import type { ColorMode } from "./colorMode";

export interface KumoPalette {
  canvas: string;
  base: string;
  elevated: string;
  recessed: string;
  interact: string;
  line: string;
  text: string;
  strong: string;
  subtle: string;
  inactive: string;
  brand: string;
  brandHover: string;
  danger: string;
  warning: string;
  success: string;
  info: string;
}

export const KUMO: Record<ColorMode, KumoPalette> = {
  dark: {
    canvas: "#090b0e",
    base: "#101418",
    elevated: "#191d22",
    recessed: "#21272d",
    interact: "#30363c",
    line: "#30363c",
    text: "#d4d8dd",
    strong: "#e5e8ec",
    subtle: "#9fa5ac",
    inactive: "#7b8187",
    brand: "#96a2fc",
    brandHover: "#818cf8",
    danger: "#f28881",
    warning: "#d99d54",
    success: "#4ec491",
    info: "#96a2fc",
  },
  light: {
    canvas: "#eef1f4",
    base: "#f8f9fb",
    elevated: "#ffffff",
    recessed: "#e4e8ed",
    interact: "#d9dfe5",
    line: "#d3d9df",
    text: "#25292f",
    strong: "#15191d",
    subtle: "#5d646b",
    inactive: "#8a9098",
    brand: "#4338ca",
    brandHover: "#372aa8",
    danger: "#dc2626",
    warning: "#d97706",
    success: "#059669",
    info: "#4338ca",
  },
};

/** AntD v5 theme config for the given color mode, on the kumo ramp. */
export function kumoAntdTheme(mode: ColorMode): ThemeConfig {
  const k = KUMO[mode];
  return {
    algorithm:
      mode === "dark" ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
    token: {
      colorPrimary: k.brand,
      colorLink: k.brand,
      colorLinkHover: k.brandHover,
      colorInfo: k.info,
      colorError: k.danger,
      colorWarning: k.warning,
      colorSuccess: k.success,
      colorBgLayout: k.canvas,
      colorBgBase: k.base,
      colorBgContainer: k.elevated,
      colorBgElevated: k.elevated,
      colorText: k.text,
      colorTextSecondary: k.subtle,
      colorTextTertiary: k.inactive,
      colorTextQuaternary: k.inactive,
      colorBorder: k.line,
      colorBorderSecondary: k.line,
      colorSplit: k.line,
      borderRadius: 4,
    },
  };
}
