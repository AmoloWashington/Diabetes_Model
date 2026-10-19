import { BrainCircuit } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { setPendingQuestion } from "@/lib/chatStore";
import { useI18n } from "@/lib/i18n";
import { Button } from "./ui";

/** Opens the research assistant with this page's current results as context. */
export function AskAI({ page, summary, question, label }: { page: string; summary: string; question: string; label?: string }) {
  const nav = useNavigate();
  const { t } = useI18n();
  return (
    <Button icon={<BrainCircuit className="size-4" />} disabled={!summary}
      onClick={() => { setPendingQuestion({ question, context: { page, summary: summary.slice(0, 6000) } }); nav("/assistant"); }}>
      {label ?? t("common.askAi")}
    </Button>
  );
}
