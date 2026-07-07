import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { AppShell } from "@/components/AppShell";
import { FileText, Upload, Trash2, Loader2 } from "lucide-react";

export const Route = createFileRoute("/")({
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
  }, []);

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    setError(null);
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
          setError("Only PDF files are supported.");
          continue;
        }
        const path = `${crypto.randomUUID()}-${file.name}`;
        const { error: upErr } = await supabase.storage
          .from("documents")
          .upload(path, file, { contentType: "application/pdf" });
        if (upErr) throw upErr;
        const { error: insErr } = await supabase.from("documents").insert({
          filename: file.name,
          file_path: path,
          file_size: file.size,
          status: "uploaded",
        });
        if (insErr) throw insErr;
      }
      await load();
    } catch (e) {
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

  return (
    <AppShell>
      <div className="mx-auto max-w-3xl px-6 py-16">
        <div className="mb-10">
          <h1 className="font-display text-5xl leading-tight">Your documents</h1>
          <p className="mt-3 text-muted-foreground max-w-lg">
            Upload PDFs to build a private knowledge base. Then head to Chat to ask
            questions and get answers with citations.
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
                    <div className="font-medium truncate">{d.filename}</div>
                    <div className="text-xs text-muted-foreground mt-0.5">
                      {formatSize(d.file_size)} ·{" "}
                      {new Date(d.created_at).toLocaleDateString(undefined, {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      })}
                    </div>
                  </div>
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
