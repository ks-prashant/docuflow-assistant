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
  }, [messages]);

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

    // Ask the RAG edge function: it embeds the question, retrieves the closest
    // chunks, and answers strictly from them — returning { answer, citations }.
    try {
      const { data, error } = await supabase.functions.invoke("rag", {
        body: { action: "ask", question: text },
      });
      if (error) throw error;
      // Backend returns a structured error instead of throwing on some failures.
      if (data?.error) throw new Error(data.error);
      if (data?.warning) setWarning(data.warning);

      await supabase.from("chat_messages").insert({
        role: "assistant",
        content: data?.answer ?? "Something went wrong answering that.",
        citations: data?.citations ?? [],
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

    await load();
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
              {sending && (
                <div className="text-sm text-muted-foreground animate-pulse">Thinking…</div>
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
