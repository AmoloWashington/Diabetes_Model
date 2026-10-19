import { useMutation } from "@tanstack/react-query";
import { CheckCircle2, Send } from "lucide-react";
import { useState } from "react";
import { Button, Callout, Card, Field, PageHeader, Textarea, cx } from "@/components/ui";
import { api } from "@/lib/api";
import { useI18n, type Key } from "@/lib/i18n";

const ROLES = ["clinician", "diabetes_educator", "researcher", "lecturer", "student", "person_with_diabetes", "family_carer", "other"] as const;
type Role = (typeof ROLES)[number];
// Page names are sent in English so reviews can be compared whatever the interface language.
const PAGES: { key: Key; en: string }[] = [
  { key: "nav.cgm", en: "Glucose data (CGM)" }, { key: "nav.monogenic", en: "Monogenic diabetes" },
  { key: "nav.meal", en: "Meal simulation" }, { key: "nav.betacell", en: "β-cell dynamics" },
  { key: "nav.minimal", en: "Insulin sensitivity" }, { key: "nav.clinical", en: "Clinical indices" },
  { key: "nav.biophysics", en: "Membrane biophysics" }, { key: "nav.math", en: "Math & physics" },
  { key: "nav.explorer", en: "3D cell explorer" }, { key: "nav.dna", en: "DNA lab" },
  { key: "nav.predict", en: "RNA 3D prediction" }, { key: "nav.risk", en: "Diabetes risk model" },
  { key: "nav.assistant", en: "Research assistant" },
];

export default function Feedback() {
  const { t, lang } = useI18n();
  const [role, setRole] = useState<Role | null>(null);
  const [pages, setPages] = useState<string[]>([]);
  const [rating, setRating] = useState<number | null>(null);
  const [use, setUse] = useState<"yes" | "maybe" | "no" | null>(null);
  const [concerns, setConcerns] = useState("");
  const [suggestions, setSuggestions] = useState("");
  const [website, setWebsite] = useState("");  // honeypot
  const [problem, setProblem] = useState<string | null>(null);

  const send = useMutation({
    mutationFn: () => api.post<{ ok: boolean }>("/api/feedback", {
      role, rating, would_use: use, pages, accuracy_concerns: concerns.trim(), suggestions: suggestions.trim(), language: lang, website,
    }),
  });

  function submit() {
    if (!role) return setProblem(t("fb.chooseRole"));
    if (!rating) return setProblem(t("fb.chooseRating"));
    if (!use) return setProblem(t("fb.chooseUse"));
    setProblem(null);
    send.mutate();
  }

  if (send.isSuccess) {
    return (
      <div>
        <PageHeader eyebrow={t("fb.eyebrow")} title={t("fb.title")} />
        <Card><div className="flex items-center gap-3 py-6"><CheckCircle2 className="size-6 text-ok" /><p className="text-[15px]">{t("fb.thanks")}</p></div></Card>
      </div>
    );
  }

  const chip = (active: boolean) => cx("rounded-lg border px-3 py-2 text-[13px] text-left transition-colors",
    active ? "bg-brand-soft border-brand text-brand-text font-medium" : "border-line-strong text-ink-2 hover:bg-surface-2");

  return (
    <div className="max-w-3xl">
      <PageHeader eyebrow={t("fb.eyebrow")} title={t("fb.title")} description={t("fb.description")} />
      <Card>
        <form className="flex flex-col gap-6" onSubmit={(e) => { e.preventDefault(); submit(); }}>
          <fieldset>
            <legend className="text-[13.5px] font-semibold mb-2">{t("fb.role")}</legend>
            <div className="grid sm:grid-cols-2 gap-2">
              {ROLES.map((r) => (
                <button type="button" key={r} aria-pressed={role === r} onClick={() => setRole(r)} className={chip(role === r)}>{t(`fb.role.${r}` as Key)}</button>
              ))}
            </div>
          </fieldset>

          <fieldset>
            <legend className="text-[13.5px] font-semibold mb-2">{t("fb.pages")}</legend>
            <div className="flex flex-wrap gap-2">
              {PAGES.map((p) => {
                const on = pages.includes(p.en);
                return <button type="button" key={p.en} aria-pressed={on} className={chip(on)}
                  onClick={() => setPages((cur) => (on ? cur.filter((x) => x !== p.en) : [...cur, p.en]))}>{t(p.key)}</button>;
              })}
            </div>
          </fieldset>

          <fieldset>
            <legend className="text-[13.5px] font-semibold mb-2">{t("fb.rating")}</legend>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-[12px] text-muted">{t("fb.rating1")}</span>
              {[1, 2, 3, 4, 5].map((n) => (
                <button type="button" key={n} aria-pressed={rating === n} aria-label={`${n} / 5`} onClick={() => setRating(n)}
                  className={cx("size-10 rounded-lg border text-[14px] font-semibold num", rating === n ? "bg-brand text-white border-brand" : "border-line-strong text-ink-2 hover:bg-surface-2")}>{n}</button>
              ))}
              <span className="text-[12px] text-muted">{t("fb.rating5")}</span>
            </div>
          </fieldset>

          <fieldset>
            <legend className="text-[13.5px] font-semibold mb-2">{t("fb.wouldUse")}</legend>
            <div className="flex gap-2">
              {(["yes", "maybe", "no"] as const).map((v) => (
                <button type="button" key={v} aria-pressed={use === v} onClick={() => setUse(v)} className={chip(use === v)}>{t(`fb.${v}` as Key)}</button>
              ))}
            </div>
          </fieldset>

          <Field label={t("fb.concerns")} hint={t("fb.concernsHint")}>
            <Textarea rows={4} maxLength={4000} value={concerns} onChange={(e) => setConcerns(e.target.value)} />
          </Field>
          <Field label={t("fb.suggestions")}>
            <Textarea rows={4} maxLength={4000} value={suggestions} onChange={(e) => setSuggestions(e.target.value)} />
          </Field>

          {/* Honeypot: hidden from people and screen readers; bots that fill every field are discarded. */}
          <div aria-hidden="true" className="absolute -left-[9999px] h-0 w-0 overflow-hidden">
            <label>Website<input tabIndex={-1} autoComplete="off" value={website} onChange={(e) => setWebsite(e.target.value)} /></label>
          </div>

          <p className="text-[12.5px] text-muted">{t("fb.noPersonal")}</p>
          {problem && <Callout tone="warn">{problem}</Callout>}
          {send.isError && <Callout tone="danger">{(send.error as Error).message}</Callout>}
          <div><Button type="submit" variant="primary" icon={<Send className="size-4" />} loading={send.isPending}>{t("fb.submit")}</Button></div>
        </form>
      </Card>
    </div>
  );
}
