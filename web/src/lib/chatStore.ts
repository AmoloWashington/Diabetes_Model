// Conversations persisted in this browser (localStorage), plus a hand-off slot
// used by "Ask AI about this" buttons to open the assistant with page context.

export interface ToolEvent { name: string; input: unknown; status: "running" | "ok" | "error"; output?: string }
export interface ChatMsg {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  tools?: ToolEvent[];
  model?: string;
  tokens?: number;
  error?: string;
  context?: { page: string; summary: string };
}
export interface Conversation { id: string; title: string; updated: number; messages: ChatMsg[] }

const KEY = "glucolab-conversations-v1";
const MAX_CONVERSATIONS = 30;

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(KEY);
    const list = raw ? (JSON.parse(raw) as Conversation[]) : [];
    return Array.isArray(list) ? list.sort((a, b) => b.updated - a.updated) : [];
  } catch {
    return [];
  }
}

export function saveConversations(list: Conversation[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(list.slice(0, MAX_CONVERSATIONS)));
  } catch {
    /* storage full or unavailable: conversations stay in memory only */
  }
}

export const newId = () => Math.random().toString(36).slice(2) + Date.now().toString(36);

let pending: { question: string; context: { page: string; summary: string } } | null = null;
export function setPendingQuestion(q: typeof pending) { pending = q; }
export function takePendingQuestion() { const p = pending; pending = null; return p; }
