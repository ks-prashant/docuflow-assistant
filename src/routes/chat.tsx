import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { supabase } from "@/integrations/supabase/client";
import { AppShell } from "@/components/AppShell";
import { ArrowUp, FileText, Eraser, ChevronDown, FileQuestion } from "lucide-react";

export const Route = createFileRoute("/chat")({
  head: () => ({
    meta: [
      { title: "Chat — DocuFlow" },
      { name: "description", content: "Ask questions across your uploaded documents." },
    ],
  }),
  component: ChatPage,
});

type Citation = { document: string; page?: number; snippet?: string };
type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
};
type IndexedDoc = { id: string; filename: string };

const ALL_DOCS = "__all__";

function newThreadId() {
  return crypto.randomUUID();
}

// Mirrors supabase/functions/rag/index.ts's NOT_FOUND_MESSAGE + isRefusal() so the
// UI can detect a refusal client-side and render a warmer empty-state instead of a
// plain bubble. Keep these two in sync if the backend wording ever changes.
const NOT_FOUND_MESSAGE = "That's not in the provided documents";
function isRefusalText(text: string): boolean {
  const normalize = (s: string) =>
    s.trim().toLowerCase().replace(/['’]/g, "'").replace(/[.!\s]+$/, "");
  return normalize(text) === normalize(NOT_FOUND_MESSAGE);
}

const SUGGESTED_QUESTIONS = [
  "What does this cover?",
  "What are the exclusions?",
  "What is the sum insured?",
];

// The backend cites passages with bracket markers like "[4]" / "[7]" whose numbers
// are the passage's rank among ALL retrieved chunks (up to 8) — not sequential
// among just the CITED ones, so raw text can jump "[4] ... [7]" confusingly. The
// `citations` array the backend sends is always in ascending original-number order
// (supabase/functions/rag/index.ts: `chunks.filter` preserves index order), so we
// can safely renumber to a clean 1..N and rewrite each marker as a markdown link
// (`#cite-N`) that a custom renderer turns into a clickable badge.
function prepareAnswerMarkdown(content: string, citationCount: number): string {
  if (citationCount === 0) return content;
  const seen: number[] = [];
  for (const m of content.matchAll(/\[(\d+)\]/g)) {
    const n = Number(m[1]);
    if (!seen.includes(n)) seen.push(n);
  }
  seen.sort((a, b) => a - b);
  const displayNumber = new Map(seen.map((orig, i) => [orig, i + 1]));
  return content.replace(/\[(\d+)\]/g, (full, numStr) => {
    const disp = displayNumber.get(Number(numStr));
    return disp ? `[${disp}](#cite-${disp})` : full;
  });
}

// The backend's citation snippet is a hard 240-char slice of the source chunk, so
// it often starts/ends mid-sentence. Purely cosmetic client-side cleanup: trim to
// the nearest full sentence and mark a mid-passage start with a leading "…" rather
// than silently showing a fragment.
function cleanSnippet(snippet: string): string {
  let s = snippet.trim();
  const wasTruncated = s.endsWith("…");
  if (wasTruncated) s = s.slice(0, -1).trim();
  const lastEnd = Math.max(s.lastIndexOf(". "), s.lastIndexOf("? "), s.lastIndexOf("! "));
  if (lastEnd > s.length * 0.4) {
    s = s.slice(0, lastEnd + 1);
  } else if (wasTruncated) {
    s = s + "…";
  }
  if (/^[a-z]/.test(s)) s = "…" + s;
  return s;
}

function ChatPage() {
  // Fresh thread per visit. History is deliberately NOT loaded from the DB —
  // new messages are still persisted (tagged with this thread id) but old
  // conversations don't reappear when you open the page.
  const [threadId, setThreadId] = useState<string>(() => newThreadId());
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [streamingText, setStreamingText] = useState<string | null>(null);
  // A soft, non-technical notice shown when the backend degraded gracefully (e.g.
  // reranker fell back). Full detail always goes to the console for debugging.
  const [warning, setWarning] = useState<string | null>(null);

  // Document scope. `ALL_DOCS` means "search across every indexed document"
  // (unchanged retrieval); any other value is a specific document.id that
  // gets forwarded to the edge function → SQL retrieval filter.
  const [indexedDocs, setIndexedDocs] = useState<IndexedDoc[]>([]);
  const [scopeId, setScopeId] = useState<string>(ALL_DOCS);

  const scrollerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Load the list of indexed documents for the selector. Poll modestly so a
  // newly-indexed doc appears without a manual refresh.
  useEffect(() => {
    let cancelled = false;
    async function loadDocs() {
      const { data, error } = await supabase
        .from("documents")
        .select("id, filename, status")
        .eq("status", "indexed")
        .order("created_at", { ascending: false });
      if (cancelled) return;
      if (error) {
        setWarning(`Couldn't load your document list: ${error.message}`);
        return;
      }
      setIndexedDocs((data ?? []).map((d) => ({ id: d.id, filename: d.filename })));
    }
    loadDocs();
    const t = setInterval(loadDocs, 8000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streamingText]);

  useEffect(() => {
    textareaRef.current?.focus();
  }, [threadId, sending]);

  function resetThread() {
    setMessages([]);
    setStreamingText(null);
    setWarning(null);
    setThreadId(newThreadId());
  }

  function onScopeChange(next: string) {
    if (next === scopeId) return;
    setScopeId(next);
    // Switching documents starts a new conversation — otherwise the model
    // would carry context about one doc into questions about another.
    resetThread();
  }

  const scopedDoc = scopeId === ALL_DOCS ? null : indexedDocs.find((d) => d.id === scopeId);
  const scopeLabel = scopedDoc ? scopedDoc.filename : "your documents";

  // Core send path, parameterized so both the input box and the refusal's
  // suggested-question chips can trigger it.
  async function sendText(text: string) {
    if (!text || sending) return;
    setSending(true);
    setWarning(null);

    // Optimistic user bubble so the UI updates immediately.
    const localUser: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
      citations: [],
    };
    setMessages((m) => [...m, localUser]);

    // Persist the user message (tagged with the current thread).
    const { error: userInsertErr } = await supabase.from("chat_messages").insert({
      role: "user",
      content: text,
      citations: [],
      thread_id: threadId,
    });
    if (userInsertErr) {
      setWarning(`Couldn't save your message: ${userInsertErr.message}`);
    }

    setStreamingText("");
    let full = "";
    let citations: Citation[] = [];
    let streamError: string | null = null;

    try {
      const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
      const KEY = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;
      const res = await fetch(`${SUPABASE_URL}/functions/v1/rag`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          apikey: KEY,
          Authorization: `Bearer ${KEY}`,
        },
        body: JSON.stringify({
          action: "ask",
          question: text,
          // Only send documentId when a specific doc is selected. Omitting it
          // preserves the exact current behavior (search across all docs).
          ...(scopeId !== ALL_DOCS ? { documentId: scopeId } : {}),
        }),
      });
      if (!res.ok || !res.body) throw new Error(`Answering service error (${res.status})`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      const handle = (line: string) => {
        const t = line.trim();
        if (!t) return;
        let ev: { type?: string; text?: string; citations?: Citation[]; warning?: string; error?: string };
        try {
          ev = JSON.parse(t);
        } catch {
          return;
        }
        if (ev.type === "token") {
          full += ev.text ?? "";
          setStreamingText(full);
        } else if (ev.type === "done") {
          citations = ev.citations ?? [];
          if (ev.warning) {
            // Keep full technical detail in the console; show only a soft,
            // non-technical notice in the chat itself.
            console.warn("[rag]", ev.warning);
            setWarning("This answer used a backup method — it should still be accurate.");
          }
        } else if (ev.type === "error") {
          streamError = ev.error ?? "Unknown error";
        }
      };

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) handle(line);
      }
      handle(buf);

      if (streamError) throw new Error(streamError);

      const assistant: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: full || "Something went wrong answering that.",
        citations,
      };
      setMessages((m) => [...m, assistant]);

      const { error: asstInsertErr } = await supabase.from("chat_messages").insert({
        role: "assistant",
        content: assistant.content,
        citations: assistant.citations,
        thread_id: threadId,
      });
      if (asstInsertErr) {
        setWarning(`Couldn't save the reply: ${asstInsertErr.message}`);
      }
    } catch (e) {
      const errText =
        "I couldn't reach the answering service. " +
        (e instanceof Error ? e.message : "Please try again.");
      const assistant: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: errText,
        citations: [],
      };
      setMessages((m) => [...m, assistant]);
      setWarning(errText);
      await supabase.from("chat_messages").insert({
        role: "assistant",
        content: errText,
        citations: [],
        thread_id: threadId,
      });
    }

    setStreamingText(null);
    setSending(false);
  }

  async function send() {
    const text = input.trim();
    if (!text || sending) return;
    setInput("");
    await sendText(text);
  }

  return (
    <AppShell>
      <div className="mx-auto max-w-3xl px-6 h-[calc(100vh-4rem)] flex flex-col">
        {/* Scope + Clear controls */}
        <div className="pt-6 pb-2 flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-xs uppercase tracking-wider text-muted-foreground shrink-0">
              Answering from:
            </span>
            <div className="relative">
              <select
                value={scopeId}
                onChange={(e) => onScopeChange(e.target.value)}
                className="appearance-none rounded-lg border border-border bg-card pl-3 pr-8 py-1.5 text-sm font-medium hover:border-primary/50 focus:outline-none focus:ring-2 focus:ring-primary/20 max-w-[280px] truncate"
                aria-label="Choose which document to chat with"
              >
                <option value={ALL_DOCS}>All documents</option>
                {indexedDocs.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.filename}
                  </option>
                ))}
              </select>
              <ChevronDown className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            </div>
          </div>
          {messages.length > 0 && (
            <button
              onClick={resetThread}
              className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-2.5 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:border-primary/50 transition-colors"
              aria-label="Clear chat"
            >
              <Eraser className="h-3.5 w-3.5" />
              Clear chat
            </button>
          )}
        </div>

        <div ref={scrollerRef} className="flex-1 overflow-y-auto py-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <h1 className="font-display text-5xl leading-tight">
                Ask your documents anything
              </h1>
              <p className="mt-4 text-muted-foreground max-w-md">
                Answers appear with citations to the exact passage. Currently answering
                from <span className="text-foreground font-medium">{scopeLabel === "your documents" ? "all documents" : scopeLabel}</span>.
              </p>
            </div>
          ) : (
            <div className="space-y-8">
              {messages.map((m) => (
                <MessageBubble key={m.id} message={m} scopeLabel={scopeLabel} onSuggest={sendText} />
              ))}
              {streamingText !== null && streamingText.length > 0 ? (
                <div className="space-y-3">
                  <div className="text-[15px] leading-relaxed">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents()}>
                      {streamingText}
                    </ReactMarkdown>
                    <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse rounded-sm bg-primary/70" />
                  </div>
                </div>
              ) : (
                sending && <ThinkingIndicator />
              )}
            </div>
          )}
        </div>

        <div className="pb-8 pt-2">
          {warning && (
            <div className="mb-3 flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
              <span className="font-medium">Heads up:</span>
              <span className="flex-1">{warning}</span>
              <button
                onClick={() => setWarning(null)}
                className="text-amber-700/60 hover:text-amber-700 dark:text-amber-400/60"
                aria-label="Dismiss"
              >
                ✕
              </button>
            </div>
          )}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              send();
            }}
            className="relative rounded-2xl border border-border bg-card shadow-sm focus-within:border-primary/60 focus-within:ring-4 focus-within:ring-primary/10 transition"
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={2}
              placeholder={
                scopedDoc
                  ? `Ask about ${scopedDoc.filename}…`
                  : "Ask a question about your documents…"
              }
              className="w-full resize-none bg-transparent px-5 py-4 pr-14 text-[15px] outline-none placeholder:text-muted-foreground"
            />
            <button
              type="submit"
              disabled={!input.trim() || sending}
              className="absolute right-3 bottom-3 h-9 w-9 rounded-lg bg-primary text-primary-foreground flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition"
              aria-label="Send"
            >
              <ArrowUp className="h-4 w-4" />
            </button>
          </form>
          <p className="mt-2 text-xs text-muted-foreground text-center">
            Press Enter to send · Shift + Enter for a new line
          </p>
        </div>
      </div>
    </AppShell>
  );
}

// Three gently bouncing dots — a more natural "the assistant is working" signal
// than a plain "Thinking…" label, and reads consistently with the streaming
// cursor once tokens start arriving.
function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-1 py-1" aria-label="Thinking">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 rounded-full bg-muted-foreground/50 animate-bounce"
          style={{ animationDelay: `${i * 120}ms` }}
        />
      ))}
    </div>
  );
}

// Shared markdown rendering rules for assistant text — used for both the final
// message and the live streaming preview. `onCiteClick` is optional; when
// omitted (the streaming preview, which has no stable citation list yet),
// citation markers just render inert since `prepareAnswerMarkdown` hasn't run.
function markdownComponents(onCiteClick?: (n: number) => void) {
  return {
    p: ({ children }: { children?: ReactNode }) => <p className="mb-2 last:mb-0">{children}</p>,
    strong: ({ children }: { children?: ReactNode }) => (
      <strong className="font-semibold text-foreground">{children}</strong>
    ),
    ul: ({ children }: { children?: ReactNode }) => (
      <ul className="my-2 ml-5 list-disc space-y-1">{children}</ul>
    ),
    ol: ({ children }: { children?: ReactNode }) => (
      <ol className="my-2 ml-5 list-decimal space-y-1">{children}</ol>
    ),
    li: ({ children }: { children?: ReactNode }) => <li className="pl-1">{children}</li>,
    h1: ({ children }: { children?: ReactNode }) => (
      <h3 className="mt-3 mb-1 text-base font-semibold">{children}</h3>
    ),
    h2: ({ children }: { children?: ReactNode }) => (
      <h3 className="mt-3 mb-1 text-base font-semibold">{children}</h3>
    ),
    h3: ({ children }: { children?: ReactNode }) => (
      <h4 className="mt-2 mb-1 text-[15px] font-semibold">{children}</h4>
    ),
    a: ({ href, children }: { href?: string; children?: ReactNode }) => {
      if (href?.startsWith("#cite-")) {
        const n = Number(href.replace("#cite-", ""));
        return (
          <button
            type="button"
            onClick={() => onCiteClick?.(n)}
            className="mx-0.5 inline-flex h-4 min-w-4 translate-y-[-3px] items-center justify-center rounded-full bg-primary/10 px-1 text-[10px] font-semibold text-primary hover:bg-primary/20 transition-colors"
          >
            {n}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer" className="text-primary underline underline-offset-2">
          {children}
        </a>
      );
    },
  };
}

function MessageBubble({
  message,
  scopeLabel,
  onSuggest,
}: {
  message: Message;
  scopeLabel: string;
  onSuggest: (q: string) => void;
}) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [highlightNum, setHighlightNum] = useState<number | null>(null);

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl bg-primary px-4 py-3 text-[15px] text-primary-foreground">
          {message.content}
        </div>
      </div>
    );
  }

  // Refusal: a warm empty-state with suggested next questions. There is nothing
  // to cite, so no Source panel is rendered at all.
  if (isRefusalText(message.content)) {
    return <RefusalCard scopeLabel={scopeLabel} onSuggest={onSuggest} />;
  }

  function handleCiteClick(n: number) {
    setSourcesOpen(true);
    setHighlightNum(n);
    // Let the panel expand before scrolling to its now-rendered card.
    requestAnimationFrame(() => {
      document.getElementById(`cite-${n}-${message.id}`)?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    });
    window.setTimeout(() => setHighlightNum(null), 1600);
  }

  const displayText = prepareAnswerMarkdown(message.content, message.citations.length);

  return (
    <div className="space-y-3">
      <div className="text-[15px] leading-relaxed">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents(handleCiteClick)}>
          {displayText}
        </ReactMarkdown>
      </div>
      <SourcePanel
        citations={message.citations}
        messageId={message.id}
        expanded={sourcesOpen}
        onToggle={() => setSourcesOpen((v) => !v)}
        highlightNum={highlightNum}
      />
    </div>
  );
}

function RefusalCard({
  scopeLabel,
  onSuggest,
}: {
  scopeLabel: string;
  onSuggest: (q: string) => void;
}) {
  return (
    <div className="space-y-3 rounded-xl border border-border bg-card/60 px-4 py-3.5">
      <div className="flex items-start gap-2.5">
        <div className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-muted">
          <FileQuestion className="h-3.5 w-3.5 text-muted-foreground" />
        </div>
        <p className="text-[15px] leading-relaxed text-muted-foreground">
          I couldn't find anything about that in{" "}
          <span className="font-medium text-foreground">{scopeLabel}</span>. Try one of
          these instead, or rephrase your question:
        </p>
      </div>
      <div className="flex flex-wrap gap-2 pl-8">
        {SUGGESTED_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            onClick={() => onSuggest(q)}
            className="rounded-full border border-border bg-background px-3 py-1.5 text-xs font-medium hover:border-primary/50 hover:text-primary transition-colors"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}

function SourcePanel({
  citations,
  messageId,
  expanded,
  onToggle,
  highlightNum,
}: {
  citations: Citation[];
  messageId: string;
  expanded: boolean;
  onToggle: () => void;
  highlightNum: number | null;
}) {
  if (citations.length === 0) return null;

  return (
    <div>
      <button
        type="button"
        onClick={onToggle}
        className="flex items-center gap-1.5 rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-muted-foreground hover:text-foreground hover:border-primary/40 transition-colors"
        aria-expanded={expanded}
      >
        <FileText className="h-3 w-3" />
        Sources ({citations.length})
        <ChevronDown className={`h-3 w-3 transition-transform ${expanded ? "rotate-180" : ""}`} />
      </button>
      {expanded && (
        <ul className="mt-2 space-y-2">
          {citations.map((c, i) => {
            const n = i + 1;
            return (
              <li
                key={i}
                id={`cite-${n}-${messageId}`}
                className={
                  "rounded-lg border px-3 py-2 text-sm transition-colors " +
                  (highlightNum === n
                    ? "border-primary/60 bg-primary/5"
                    : "border-border bg-card/60")
                }
              >
                <div className="flex items-center gap-2">
                  <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-primary/10 text-[10px] font-semibold text-primary">
                    {n}
                  </span>
                  <span className="font-medium truncate">{c.document}</span>
                  {c.page ? (
                    <span className="shrink-0 text-xs text-muted-foreground">p.{c.page}</span>
                  ) : null}
                </div>
                {c.snippet ? (
                  <p className="mt-1 pl-6 text-muted-foreground italic">
                    "{cleanSnippet(c.snippet)}"
                  </p>
                ) : null}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
