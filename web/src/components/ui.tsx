import { clsx } from "clsx";
import { Loader2, AlertTriangle, Info, CheckCircle2, XCircle } from "lucide-react";
import {
  createContext, forwardRef, useCallback, useContext, useState,
  type ButtonHTMLAttributes, type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes, type TextareaHTMLAttributes,
} from "react";

export const cx = clsx;

// ------------------------------------------------------------------ Button

type Variant = "primary" | "secondary" | "ghost" | "danger";
const variants: Record<Variant, string> = {
  primary: "bg-brand text-white hover:bg-brand-hover shadow-card dark:text-[#0e1013]",
  secondary: "bg-surface text-ink border border-line-strong hover:bg-surface-2 shadow-card",
  ghost: "text-ink-2 hover:bg-surface-3",
  danger: "bg-surface text-danger border border-line-strong hover:bg-danger-soft",
};

export const Button = forwardRef<HTMLButtonElement, ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant; size?: "sm" | "md"; loading?: boolean; icon?: ReactNode;
}>(({ variant = "secondary", size = "md", loading, icon, className, children, disabled, ...rest }, ref) => (
  <button
    ref={ref}
    disabled={disabled || loading}
    className={cx(
      "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap",
      size === "sm" ? "h-8 px-2.5 text-[13px]" : "h-9 px-3.5 text-[13.5px]",
      variants[variant], className,
    )}
    {...rest}
  >
    {loading ? <Loader2 className="size-4 animate-spin" /> : icon}
    {children}
  </button>
));
Button.displayName = "Button";

// ------------------------------------------------------------------ Layout primitives

export function Card({ className, children, padded = true }: { className?: string; children: ReactNode; padded?: boolean }) {
  return <section className={cx("bg-surface border border-line rounded-xl shadow-card min-w-0", padded && "p-5", className)}>{children}</section>;
}

export function CardHeader({ title, subtitle, actions }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex items-start justify-between gap-4 mb-4">
      <div className="min-w-0">
        <h3 className="text-[15px] font-semibold tracking-[-0.01em] text-ink">{title}</h3>
        {subtitle && <p className="text-[13px] text-muted mt-0.5">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  );
}

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between mb-6">
      <div className="max-w-3xl min-w-0">
        {eyebrow && <p className="text-[12px] font-medium uppercase tracking-[0.08em] text-brand-text mb-1.5">{eyebrow}</p>}
        <h1 className="text-[26px] leading-8 font-semibold tracking-[-0.02em] text-ink">{title}</h1>
        {description && <p className="text-[14.5px] text-muted mt-2 leading-6">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2 flex-wrap">{actions}</div>}
    </header>
  );
}

// ------------------------------------------------------------------ Form controls

const control = "w-full h-9 rounded-lg border border-line-strong bg-surface px-3 text-[13.5px] text-ink placeholder:text-faint focus:outline-none focus:border-brand focus:ring-3 focus:ring-[var(--brand-ring)] transition";

export function Field({ label, hint, children, className }: { label: ReactNode; hint?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <label className={cx("flex flex-col gap-1.5", className)}>
      <span className="text-[12.5px] font-medium text-ink-2">{label}</span>
      {children}
      {hint && <span className="text-[12px] text-faint">{hint}</span>}
    </label>
  );
}

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(({ className, ...p }, ref) => (
  <input ref={ref} className={cx(control, "num", className)} {...p} />
));
Input.displayName = "Input";

export function Select({ className, children, ...p }: SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={cx(control, "pr-8", className)} {...p}>{children}</select>;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(({ className, ...p }, ref) => (
  <textarea ref={ref} className={cx(control, "h-auto py-2 leading-5", className)} {...p} />
));
Textarea.displayName = "Textarea";

export function Slider({ label, value, min, max, step, onChange, format }: {
  label: ReactNode; value: number; min: number; max: number; step: number; onChange: (v: number) => void; format?: (v: number) => string;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="flex items-center justify-between text-[12.5px]">
        <span className="font-medium text-ink-2">{label}</span>
        <span className="num font-mono text-[12px] text-ink bg-surface-3 rounded px-1.5 py-0.5">{format ? format(value) : value}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} className="w-full" />
    </label>
  );
}

export function Segmented<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode }[];
}) {
  return (
    <div className="inline-flex rounded-lg bg-surface-3 p-0.5 border border-line" role="tablist">
      {options.map((o) => (
        <button
          key={o.value} role="tab" aria-selected={value === o.value} onClick={() => onChange(o.value)}
          className={cx("px-3 h-8 rounded-md text-[13px] font-medium transition-colors",
            value === o.value ? "bg-surface text-ink shadow-card" : "text-muted hover:text-ink")}
        >{o.label}</button>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ Display

export function Badge({ tone = "neutral", children, className }: { tone?: "neutral" | "brand" | "ok" | "warn" | "danger" | "accent"; children: ReactNode; className?: string }) {
  const t = {
    neutral: "bg-surface-3 text-ink-2 border-line",
    brand: "bg-brand-soft text-brand-text border-transparent",
    ok: "bg-ok-soft text-ok border-transparent",
    warn: "bg-warn-soft text-warn border-transparent",
    danger: "bg-danger-soft text-danger border-transparent",
    accent: "bg-accent-soft text-accent border-transparent",
  }[tone];
  return <span className={cx("inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11.5px] font-medium whitespace-nowrap", t, className)}>{children}</span>;
}

export function Stat({ label, value, unit, tone, hint }: { label: string; value: ReactNode; unit?: string; tone?: "ok" | "warn" | "danger"; hint?: ReactNode }) {
  const color = tone === "ok" ? "text-ok" : tone === "warn" ? "text-warn" : tone === "danger" ? "text-danger" : "text-ink";
  return (
    <div className="bg-surface border border-line rounded-xl px-4 py-3.5 shadow-card min-w-0">
      <div className="text-[12px] font-medium text-muted truncate">{label}</div>
      <div className={cx("num mt-1 text-[22px] leading-7 font-semibold tracking-[-0.02em]", color)}>
        {value}{unit && <span className="text-[13px] font-medium text-muted ml-1">{unit}</span>}
      </div>
      {hint && <div className="text-[12px] text-faint mt-0.5">{hint}</div>}
    </div>
  );
}

export function Callout({ tone = "info", title, children }: { tone?: "info" | "warn" | "danger" | "ok"; title?: ReactNode; children: ReactNode }) {
  const map = {
    info: { cls: "bg-surface-2 border-line text-ink-2", Icon: Info, ic: "text-brand-text" },
    warn: { cls: "bg-warn-soft border-transparent text-ink-2", Icon: AlertTriangle, ic: "text-warn" },
    danger: { cls: "bg-danger-soft border-transparent text-ink-2", Icon: XCircle, ic: "text-danger" },
    ok: { cls: "bg-ok-soft border-transparent text-ink-2", Icon: CheckCircle2, ic: "text-ok" },
  }[tone];
  return (
    <div className={cx("flex gap-2.5 rounded-lg border px-3.5 py-3 text-[13px] leading-5", map.cls)}>
      <map.Icon className={cx("size-4 mt-0.5 shrink-0", map.ic)} />
      <div className="min-w-0">{title && <div className="font-semibold text-ink mb-0.5">{title}</div>}{children}</div>
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return <div className="flex items-center gap-2 text-muted text-[13px]"><Loader2 className="size-4 animate-spin" />{label}</div>;
}

export function Empty({ icon, title, children, action }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center text-center py-12 px-6">
      {icon && <div className="size-10 rounded-xl bg-surface-3 grid place-items-center text-muted mb-3">{icon}</div>}
      <div className="font-semibold text-ink">{title}</div>
      {children && <div className="text-[13px] text-muted mt-1 max-w-md">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Table({ head, children, className }: { head: ReactNode[]; children: ReactNode; className?: string }) {
  return (
    <div className={cx("overflow-x-auto", className)}>
      <table className="w-full text-[13px]">
        <thead>
          <tr className="border-b border-line">
            {head.map((h, i) => <th key={i} className="text-left font-medium text-muted px-3 py-2 whitespace-nowrap">{h}</th>)}
          </tr>
        </thead>
        <tbody className="divide-y divide-line">{children}</tbody>
      </table>
    </div>
  );
}

// ------------------------------------------------------------------ Toasts

type Toast = { id: number; msg: string; tone: "ok" | "error" };
const ToastCtx = createContext<(msg: string, tone?: "ok" | "error") => void>(() => {});
export const useToast = () => useContext(ToastCtx);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const push = useCallback((msg: string, tone: "ok" | "error" = "ok") => {
    const id = Date.now() + Math.random();
    setItems((x) => [...x, { id, msg, tone }]);
    setTimeout(() => setItems((x) => x.filter((t) => t.id !== id)), tone === "error" ? 6500 : 3200);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 max-w-sm" role="status" aria-live="polite">
        {items.map((t) => (
          <div key={t.id} className={cx("rounded-lg border px-4 py-3 text-[13px] shadow-pop bg-surface flex gap-2",
            t.tone === "error" ? "border-danger/40 text-danger" : "border-line text-ink")}>
            {t.tone === "error" ? <XCircle className="size-4 mt-0.5 shrink-0" /> : <CheckCircle2 className="size-4 mt-0.5 shrink-0 text-ok" />}
            <span>{t.msg}</span>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}
