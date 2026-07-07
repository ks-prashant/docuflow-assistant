import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { AppShell } from "@/components/AppShell";
import { FileText, Upload, Trash2, Loader2, RefreshCw } from "lucide-react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Your documents — DocuFlow" },
      {
        name: "description",
        content:
          "Upload PDFs to build a private, cited knowledge base. DocuFlow only answers from what you upload.",
      },
    ],
  }),
  component: DocumentsPage,
});

type Doc = {
  id: string;
  filename: string;
  file_path: string;
  file_size: number | null;
  status: string;
  created_at: string;
};

function formatSize(bytes: number | null) {
  if (!bytes) return "—";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function DocumentsPage() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Tracks which documents are currently being (re)indexed, so we can show a
  // spinner and disable the button per-row.
  const [reindexing, setReindexing] = useState<Record<string, boolean>>({});
  const inputRef = useRef<HTMLInputElement>(null);

  async function load() {
    const { data, error } = await supabase
      .from("documents")
      .select("*")
      .order("created_at", { ascending: false });
    if (error) setError(error.message);
    else setDocs((data as Doc[]) ?? []);
  }

  useEffect(() => {
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, []);

  async function handleFiles(files: FileList | null) {
    console.log("[upload] handleFiles", files?.length);
    if (!files || files.length === 0) return;
    setError(null);
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        console.log("[upload] file", file.name, file.size, file.type);
        if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
          setError("Only PDF files are supported.");
          continue;
        }
        const path = `${crypto.randomUUID()}-${file.name}`;
        const { error: upErr } = await supabase.storage
          .from("documents")
          .upload(path, file, { contentType: "application/pdf" });
        console.log("[upload] storage result", upErr);
        if (upErr) {
          setError(`Storage upload failed: ${upErr.message}`);
          throw upErr;
        }
        const { data: inserted, error: insErr } = await supabase
          .from("documents")
          .insert({
            filename: file.name,
            file_path: path,
            file_size: file.size,
            status: "uploaded",
          })
          .select()
          .single();
        console.log("[upload] insert result", inserted, insErr);
        if (insErr) {
          setError(`DB insert failed: ${insErr.message}`);
          throw insErr;
        }

        if (inserted) {
          await load();
          supabase.functions
            .invoke("rag", { body: { action: "ingest", documentId: inserted.id } })
            .then((r) => {
              console.log("[upload] ingest result", r);
              load();
            })
            .catch((e) => {
              console.error("[upload] Ingest failed:", e);
              setError(`Ingest failed: ${e?.message ?? e}`);
            });
        }
      }
      await load();
    } catch (e) {
      console.error("[upload] error", e);
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function remove(doc: Doc) {
    await supabase.storage.from("documents").remove([doc.file_path]);
    await supabase.from("documents").delete().eq("id", doc.id);
    await load();
  }

  // Retry processing for a document that never got indexed (e.g. uploaded before
  // ingestion was wired up, or a run that failed). The edge function's ingest is
  // idempotent — it clears old chunks first — so this is safe to run repeatedly.
  async function reindex(doc: Doc) {
    setReindexing((r) => ({ ...r, [doc.id]: true }));
    try {
      const { error } = await supabase.functions.invoke("rag", {
        body: { action: "ingest", documentId: doc.id },
      });
      if (error) throw error;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Re-index failed");
    } finally {
      setReindexing((r) => ({ ...r, [doc.id]: false }));
      await load();
    }
  }

  return (
    <AppShell>
      <div className="mx-auto max-w-3xl px-6 py-16">
        <div className="mb-12">
          <h1 className="font-display text-5xl leading-[1.1]">
            Answers you can trust, sourced from your documents.
          </h1>
          <p className="mt-5 text-muted-foreground max-w-xl text-[15px] leading-relaxed">
            DocuFlow only answers from what you upload — every response is cited,
            and if it's not in your documents, it says so.
          </p>
        </div>

        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
          className={
            "block cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-colors " +
            (dragging
              ? "border-primary bg-accent/40"
              : "border-border hover:border-primary/60 hover:bg-accent/20")
          }
        >
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf"
            multiple
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
          <div className="mx-auto h-12 w-12 rounded-xl bg-accent flex items-center justify-center">
            {uploading ? (
              <Loader2 className="h-5 w-5 text-primary animate-spin" />
            ) : (
              <Upload className="h-5 w-5 text-primary" />
            )}
          </div>
          <div className="mt-4 text-base font-medium">
            {uploading ? "Uploading..." : "Drop PDFs here, or click to browse"}
          </div>
          <div className="mt-1 text-sm text-muted-foreground">
            PDF only · up to 20MB per file
          </div>
        </label>

        {error && (
          <div className="mt-4 rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}

        <div className="mt-12">
          <div className="flex items-baseline justify-between mb-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Library
            </h2>
            <span className="text-sm text-muted-foreground">{docs.length} files</span>
          </div>

          {docs.length === 0 ? (
            <div className="rounded-xl border border-border bg-card py-12 text-center text-sm text-muted-foreground">
              No documents yet.
            </div>
          ) : (
            <ul className="divide-y divide-border rounded-xl border border-border bg-card overflow-hidden">
              {docs.map((d) => (
                <li key={d.id} className="flex items-center gap-4 px-5 py-4">
                  <div className="h-10 w-10 rounded-lg bg-accent flex items-center justify-center shrink-0">
                    <FileText className="h-5 w-5 text-primary" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium truncate">{d.filename}</span>
                      <StatusBadge status={d.status} />
                    </div>
                    <div className="text-xs text-muted-foreground mt-0.5">
                      {formatSize(d.file_size)} ·{" "}
                      {new Date(d.created_at).toLocaleDateString(undefined, {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      })}
                    </div>
                  </div>
                  {/* Offer a retry for anything that isn't already searchable.
                      The status pill next to the filename (StatusBadge) shows the
                      live lifecycle state; this button lets a stuck doc retry. */}
                  {d.status !== "indexed" && d.status !== "processing" && (
                    <button
                      onClick={() => reindex(d)}
                      disabled={reindexing[d.id]}
                      className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-muted-foreground hover:text-primary hover:bg-accent transition-colors disabled:opacity-50"
                      aria-label="Re-index"
                    >
                      <RefreshCw
                        className={"h-3.5 w-3.5 " + (reindexing[d.id] ? "animate-spin" : "")}
                      />
                      {reindexing[d.id] ? "Indexing…" : "Re-index"}
                    </button>
                  )}
                  <button
                    onClick={() => remove(d)}
                    className="p-2 rounded-lg text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                    aria-label="Delete"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </AppShell>
  );
}

// Small coloured pill showing where a document is in the ingestion lifecycle:
// uploaded → processing → indexed, or the failure states failed / no_text.
function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    indexed: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
    processing: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
    uploaded: "bg-muted text-muted-foreground",
    failed: "bg-destructive/10 text-destructive",
    no_text: "bg-destructive/10 text-destructive",
  };
  const labels: Record<string, string> = {
    indexed: "Indexed",
    processing: "Processing",
    uploaded: "Not indexed",
    failed: "Failed",
    no_text: "No text found",
  };
  const style = styles[status] ?? "bg-muted text-muted-foreground";
  const label = labels[status] ?? status;
  return (
    <span
      className={
        "shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide " +
        style
      }
    >
      {label}
    </span>
  );
}
