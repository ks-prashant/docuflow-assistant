import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { AppShell } from "@/components/AppShell";
import { ArrowUp, FileText } from "lucide-react";

export const Route = createFileRoute("/chat")({
  head: () => ({
    meta: [
      { title: "Chat — Paperline" },
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
  created_at: string;
};

function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  // The answer as it streams in, token by token. null = not currently streaming.
  const [streamingText, setStreamingText] = useState<string | null>(null);
  // Non-fatal notices from the backend (e.g. reranker fell back, PII redaction
  // degraded). Surfaced so nothing fails silently, but not stored in chat history.
  const [warning, setWarning] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);

  async function load() {
    const { data } = await supabase
      .from("chat_messages")
      .select("*")
      .order("created_at", { ascending: true });
    if (data) setMessages(data as Message[]);
  }

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    scrollerRef.current?.scrollTo({ top: scrollerRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streamingText]);

  async function send() {
    const text = input.trim();
    if (!text || sending) return;
    setSending(true);
    setInput("");
    setWarning(null);

    await supabase.from("chat_messages").insert({
      role: "user",
      content: text,
      citations: [],
    });
    await load();

    // Ask the RAG edge function. It STREAMS newline-delimited JSON events:
    //   {type:"token",text}  — answer text, already PII-masked, as it's written
    //   {type:"done",citations,warning} — final metadata once the answer completes
    //   {type:"error",error} — a failure to surface
    // We render tokens live, then persist the finished message to the DB.
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
        body: JSON.stringify({ action: "ask", question: text }),
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
          return; // ignore any partial/non-JSON line
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

      // Read the stream, dispatching each complete newline-delimited event.
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) handle(line);
      }
      handle(buf); // trailing event, if any

      if (streamError) throw new Error(streamError);

      await supabase.from("chat_messages").insert({
        role: "assistant",
        content: full || "Something went wrong answering that.",
        citations,
      });
    } catch (e) {
      await supabase.from("chat_messages").insert({
        role: "assistant",
        content:
          "I couldn't reach the answering service. " +
          (e instanceof Error ? e.message : "Please try again."),
        citations: [],
      });
    }

    await load();          // pull the persisted message into the list…
    setStreamingText(null); // …then drop the live bubble (avoids a flash-gap)
    setSending(false);
  }

  return (
    <AppShell>
      <div className="mx-auto max-w-3xl px-6 h-[calc(100vh-4rem)] flex flex-col">
        <div ref={scrollerRef} className="flex-1 overflow-y-auto py-10">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center">
              <h1 className="font-display text-5xl leading-tight">
                Ask your documents anything
              </h1>
              <p className="mt-4 text-muted-foreground max-w-md">
                Answers appear with citations to the exact passage in your PDFs.
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
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={2}
              placeholder="Ask a question about your documents…"
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
          No source yet — citations will appear here once AI answering is enabled.
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
