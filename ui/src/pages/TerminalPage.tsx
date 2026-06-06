import Terminal from "../components/Terminal";

// Page height = viewport - (Header ~64px + Content vertical padding 48px
// + Footer ~70px ≈ 182px).  Rounded to 200px for a small bottom buffer
// so the terminal frame doesn't sit flush against the footer;
// ``minHeight`` keeps it usable on short viewports.  Horizontal margins
// + footer come from the standard Layout (no ``fullHeight`` prop) — same
// padding pattern as Status / Settings so terminal text doesn't run
// flush to the screen edge on wide monitors.
export default function TerminalPage() {
  return (
    <div
      style={{
        height: "calc(100vh - 200px)",
        minHeight: 400,
      }}
    >
      <Terminal style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
