import { Download, Share, X } from "lucide-react";
import { useState } from "react";
import { useI18n } from "@/lib/i18n";
import { useInstall } from "@/lib/pwa";

/** "Install app" control for the sidebar: a real prompt where the browser supports it, instructions on iOS. */
export function InstallApp() {
  const { installed, canPrompt, ios, install } = useInstall();
  const [help, setHelp] = useState(false);
  const { t } = useI18n();
  if (installed || (!canPrompt && !ios)) return null;
  return (
    <div className="relative">
      <button onClick={() => (canPrompt ? install() : setHelp((h) => !h))}
        className="w-full flex items-center justify-center gap-1.5 rounded-lg border border-line-strong h-8 text-[12.5px] font-medium text-ink-2 hover:bg-surface-3">
        <Download className="size-3.5" /> {t("layout.install")}
      </button>
      {help && (
        <div role="dialog" aria-label="Install on iPhone or iPad" className="absolute bottom-10 left-0 right-0 z-10 rounded-lg border border-line bg-surface p-3 shadow-pop text-[12.5px] text-ink-2 leading-5">
          <button onClick={() => setHelp(false)} className="absolute right-1.5 top-1.5 size-6 grid place-items-center text-muted" aria-label="Close"><X className="size-3.5" /></button>
          {t("layout.installIos").split("{share}").map((part, i) => (i === 0 ? <span key={i}>{part}</span> : <span key={i}><Share className="inline size-3.5 -mt-0.5" />{part}</span>))}
        </div>
      )}
    </div>
  );
}
