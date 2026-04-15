import Terminal from "../components/Terminal";

export default function TerminalPage() {
  return (
    <div style={{ flex: 1, minHeight: 0 }}>
      <Terminal style={{ width: "100%", height: "100%" }} />
    </div>
  );
}
