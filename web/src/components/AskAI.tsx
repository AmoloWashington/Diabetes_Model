import { BrainCircuit } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { setPendingQuestion } from "@/lib/chatStore";
import { Button } from "./ui";

/** Opens the research assistant with this page's current results as context. */
export function AskAI({ page, summary, question }: { page: string; summary: string; question: string }) {
  const nav = useNavigate();
  return (
    <Button icon={<BrainCircuit className="size-4" />} disabled={!summary}
      onClick={() => { setPendingQuestion({ question, context: { page, summary: summary.slice(0, 6000) } }); nav("/assistant"); }}>
      Ask AI about this
    </Button>
  );
}
