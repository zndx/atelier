import { Outlet, useLocation } from "react-router-dom";
import { Header } from "./Header";

// Routes that render their own full-bleed content (no page padding,
// locked to viewport height): the dual-pane Operate home and the
// full-canvas Workflows / Embeddings detail views.
function isFullBleed(pathname: string): boolean {
  return (
    pathname === "/" ||
    pathname.startsWith("/workflows") ||
    /^\/embeddings\/.+/.test(pathname)
  );
}

export function AppShell() {
  const { pathname } = useLocation();
  const fullBleed = isFullBleed(pathname);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-surface-0">
      <Header />
      {fullBleed ? (
        <main className="min-h-0 flex-1 overflow-hidden">
          <Outlet />
        </main>
      ) : (
        <main className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
            <Outlet />
          </div>
        </main>
      )}
    </div>
  );
}
