import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { CheckCircle2, XCircle, Info } from "lucide-react";
import { cn } from "./cn";

type ToastTone = "success" | "error" | "info";
interface Toast {
  id: number;
  tone: ToastTone;
  message: string;
}

interface ToastCtx {
  push: (message: string, tone?: ToastTone) => void;
}

const Ctx = createContext<ToastCtx | null>(null);

const ICON: Record<ToastTone, ReactNode> = {
  success: <CheckCircle2 className="h-4 w-4 text-status-green" />,
  error: <XCircle className="h-4 w-4 text-status-red" />,
  info: <Info className="h-4 w-4 text-accent" />,
};

// DESIGN_TEMPLATE §5.17 — toast.
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const idRef = useRef(0);

  const push = useCallback((message: string, tone: ToastTone = "info") => {
    const id = ++idRef.current;
    setToasts((t) => [...t, { id, tone, message }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
  }, []);

  return (
    <Ctx.Provider value={{ push }}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={cn(
              "pointer-events-auto flex items-center gap-2 rounded-lg border border-surface-3 bg-surface-2 px-3 py-2 text-sm text-gray-200 shadow-lg animate-fade-slide-in",
            )}
          >
            {ICON[t.tone]}
            <span>{t.message}</span>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}

export function useToast(): ToastCtx {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useToast must be used within ToastProvider");
  return ctx;
}
