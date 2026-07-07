import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { AppShell } from "@/components/AppShell";
import { ArrowUp, FileText, Eraser, ChevronDown } from "lucide-react";

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

function ChatPage() {
  // Fresh thread per visit. History is deliberately NOT loaded from the DB —
  // new messages are still persisted (tagged with this thread id) but old
  // conversations don't reappear when you open the page.
  const [threadId, setThreadId] = useState<string>(() => newThreadId());
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [streamingText, setStreamingText] = useState<string | null>(null);
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
  const scopeLabel = scopedDoc ? scopedDoc.filename : "All documents";

  async function send() {
    const text = input.trim();
    if (!text || sending) return;
    setSending(true);
    setInput("");
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
          if (ev.warning) setWarning(ev.warning);
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
                from <span className="text-foreground font-medium">{scopeLabel}</span>.
              </p>
            </div>
          ) : (
            <div className="space-y-8">
              {messages.map((m) => (
                <MessageBubble key={m.id} message={m} />
              ))}
              {streamingText !== null && streamingText.length > 0 ? (
                <div className="space-y-3">
                  <div className="text-[15px] leading-relaxed whitespace-pre-wrap">
                    {streamingText}
                    <span className="ml-0.5 inline-block h-4 w-1.5 translate-y-0.5 animate-pulse rounded-sm bg-primary/70" />
                  </div>
                </div>
              ) : (
                sending && (
                  <div className="text-sm text-muted-foreground animate-pulse">Thinking…</div>
                )
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

function MessageBubble({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl bg-primary px-4 py-3 text-[15px] text-primary-foreground">
          {message.content}
        </div>
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div className="text-[15px] leading-relaxed whitespace-pre-wrap">{message.content}</div>
      <SourceBlock citations={message.citations} />
    </div>
  );
}

function SourceBlock({ citations }: { citations: Citation[] }) {
  return (
    <div className="rounded-xl border border-border bg-card/60 px-4 py-3">
      <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        <FileText className="h-3.5 w-3.5" />
        Source
      </div>
      {citations.length === 0 ? (
        <div className="mt-2 text-sm text-muted-foreground italic">
          No source cited for this reply.
        </div>
      ) : (
        <ul className="mt-2 space-y-2">
          {citations.map((c, i) => (
            <li key={i} className="text-sm">
              <span className="font-medium">{c.document}</span>
              {c.page ? <span className="text-muted-foreground"> · p.{c.page}</span> : null}
              {c.snippet ? (
                <p className="mt-1 text-muted-foreground border-l-2 border-border pl-3">
                  {c.snippet}
                </p>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
