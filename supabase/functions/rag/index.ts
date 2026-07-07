// =============================================================================
// RAG core — a single, self-contained edge function.
//
// This one module is the entire "AI brain" of the app. It exposes two actions:
//
//   • action: "ingest"  → take an uploaded PDF, split it into chunks, embed each
//                          chunk with OpenAI, and store the vectors in Postgres.
//   • action: "ask"     → embed the user's question, retrieve ONLY the closest
//                          chunks, and let the LLM answer strictly from those.
//
// Design promises (the things you'd defend to a client):
//   1. The model NEVER sees your whole library — only the handful of chunks the
//      vector search returns for this specific question. That is the core of
//      Retrieval-Augmented Generation: ground the answer in retrieved evidence.
//   2. The system prompt forbids outside knowledge and requires a citation, and
//      the model is told to say "That's not in the provided documents" when the
//      retrieved chunks don't contain the answer.
//   3. temperature = 0.1 → near-deterministic, faithful answers, not creative ones.
//   4. The OpenAI key is read from a secret (Deno.env), never hardcoded.
//
// SECRET REQUIRED:  OPENAI_API_KEY   (add it in Lovable Cloud → project secrets)
// =============================================================================

import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
// unpdf is a serverless-friendly build of Mozilla's pdf.js — it extracts text
// from a PDF without native binaries, which is exactly what a Deno edge runtime
// needs. We ask it for text PER PAGE so every chunk can carry a page number.
import { extractText, getDocumentProxy } from "https://esm.sh/unpdf@0.12.1";
// Retrieval quality + safety are split into their own small, swappable modules.
import { rerank } from "./rerank.ts";     // wide-recall candidates → precise top-N
import { maskOutputs } from "./mask.ts";  // OUTPUT-ONLY PII masking (answer + snippets)

// ---------------------------------------------------------------------------
// Configuration — all the knobs in one place so they're easy to explain/tune.
// ---------------------------------------------------------------------------
const EMBEDDING_MODEL = "text-embedding-3-small"; // 1536 dims → matches our vector column
const CHAT_MODEL = "gpt-4o-mini";                 // cheap, strong instruction-following
const CHAT_TEMPERATURE = 0.1;                      // low = faithful, repeatable answers

const CHUNK_SIZE = 1000;      // ~target characters per chunk (a few paragraphs)
const CHUNK_OVERLAP = 150;    // characters shared between neighbours so we don't
                              // slice a sentence/idea cleanly in half at a boundary
const EMBED_BATCH_SIZE = 96;  // how many chunks we embed per OpenAI request

// Retrieval is now WIDE then TIGHT: cast a wide net for recall, rerank for precision.
const RETRIEVE_CANDIDATES = 25; // how many to pull from EACH search arm (semantic + keyword)
const RERANK_TOP_N = 8;         // how many survive reranking and reach the answer model
const SEMANTIC_FLOOR = 0.05;    // a very low garbage floor ONLY. We no longer refuse on a
                                // similarity threshold — we always pass candidates to the
                                // LLM and let its prompt rules decide (Change 2).
const RRF_K = 60;               // Reciprocal Rank Fusion constant (the standard default)

// The refusal string is defined once so the prompt and any fallback stay in sync.
const NOT_FOUND_MESSAGE = "That's not in the provided documents";

/**
 * Detect whether the model refused, tolerant of the small variations an LLM adds
 * (a trailing period, different casing, curly vs straight apostrophe). We compare
 * on a normalized form so a refusal is never mistaken for a real answer — that
 * mismatch was what previously let a refusal carry a bogus citation.
 */
function isRefusal(text: string): boolean {
  const normalize = (s: string) =>
    s.trim().toLowerCase().replace(/['’]/g, "'").replace(/[.!\s]+$/, "");
  return normalize(text) === normalize(NOT_FOUND_MESSAGE);
}

// Standard CORS headers so the browser app can call this function directly.
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

// ---------------------------------------------------------------------------
// Clients / secrets.
//
// Lovable Cloud auto-injects SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY into
// every edge function. We use the SERVICE ROLE key here (not the public key)
// because this trusted server code needs to read storage and write embeddings
// on the user's behalf. OPENAI_API_KEY is the one secret YOU must add.
// ---------------------------------------------------------------------------
const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY");
const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY);

// ===========================================================================
// OpenAI helpers
// ===========================================================================

/**
 * Embed one or more texts with text-embedding-3-small.
 * Returns one 1536-number vector per input, in the same order.
 */
async function embed(inputs: string[]): Promise<number[][]> {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model: EMBEDDING_MODEL, input: inputs }),
  });
  if (!res.ok) {
    throw new Error(`OpenAI embeddings failed: ${res.status} ${await res.text()}`);
  }
  const json = await res.json();
  // The API preserves input order but also returns an index; sort to be safe.
  return json.data
    .sort((a: { index: number }, b: { index: number }) => a.index - b.index)
    .map((d: { embedding: number[] }) => d.embedding);
}

// ===========================================================================
// Text chunking
//
// LLMs and embeddings work best on bite-sized passages, and small chunks make
// citations precise ("this paragraph on page 4" beats "somewhere in this file").
// We split each PAGE independently so a chunk never straddles two pages and its
// page_number stays truthful. Overlap keeps ideas that sit on a boundary intact.
// ===========================================================================
function chunkText(text: string): string[] {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return [];

  const chunks: string[] = [];
  let start = 0;
  while (start < clean.length) {
    let end = Math.min(start + CHUNK_SIZE, clean.length);

    // Prefer to end a chunk at a sentence/space boundary near the target size,
    // rather than mid-word, so chunks read naturally.
    if (end < clean.length) {
      const window = clean.slice(start, end);
      const lastBreak = Math.max(
        window.lastIndexOf(". "),
        window.lastIndexOf("? "),
        window.lastIndexOf("! "),
        window.lastIndexOf("\n"),
      );
      if (lastBreak > CHUNK_SIZE * 0.5) end = start + lastBreak + 1;
    }

    const piece = clean.slice(start, end).trim();
    if (piece) chunks.push(piece);

    if (end >= clean.length) break;
    start = end - CHUNK_OVERLAP; // step back so the next chunk overlaps this one
  }
  return chunks;
}

// ===========================================================================
// ACTION 1 — INGEST: PDF  →  chunks  →  embeddings  →  Postgres
// ===========================================================================
async function ingest(documentId: string) {
  // 1. Look up the document row the frontend created on upload.
  const { data: doc, error: docErr } = await supabase
    .from("documents")
    .select("id, filename, file_path")
    .eq("id", documentId)
    .single();
  if (docErr || !doc) throw new Error(`Document not found: ${documentId}`);

  // Mark it "processing" so the UI can show progress and we can tell if a run
  // died halfway. We also clear any old chunks so re-ingesting is idempotent.
  await supabase.from("documents").update({ status: "processing" }).eq("id", documentId);
  await supabase.from("document_chunks").delete().eq("document_id", documentId);

  try {
    // 2. Download the raw PDF bytes from the 'documents' storage bucket.
    const { data: file, error: dlErr } = await supabase
      .storage.from("documents").download(doc.file_path);
    if (dlErr || !file) throw new Error(`Could not download PDF: ${dlErr?.message}`);
    const bytes = new Uint8Array(await file.arrayBuffer());

    // 3. Extract text, one string per page (mergePages: false keeps pages split).
    const pdf = await getDocumentProxy(bytes);
    const { text: pages } = await extractText(pdf, { mergePages: false });
    const pageTexts: string[] = Array.isArray(pages) ? pages : [pages];

    // 4. Chunk every page, remembering which page each chunk came from.
    const records: { content: string; page_number: number }[] = [];
    pageTexts.forEach((pageText, i) => {
      for (const content of chunkText(pageText ?? "")) {
        records.push({ content, page_number: i + 1 }); // pages are 1-indexed for humans
      }
    });

    if (records.length === 0) {
      // A scanned/image-only PDF yields no extractable text — flag it clearly
      // instead of silently indexing nothing.
      await supabase.from("documents").update({ status: "no_text" }).eq("id", documentId);
      return { documentId, chunks: 0, note: "No extractable text (is this a scanned PDF?)" };
    }

    // 5. Embed the chunks in batches, then store each chunk + its vector.
    let chunkIndex = 0;
    for (let i = 0; i < records.length; i += EMBED_BATCH_SIZE) {
      const batch = records.slice(i, i + EMBED_BATCH_SIZE);
      const vectors = await embed(batch.map((r) => r.content));

      const rows = batch.map((r, j) => ({
        document_id: documentId,
        chunk_index: chunkIndex++,
        content: r.content,
        page_number: r.page_number,
        // pgvector accepts its text form "[0.1,0.2,...]", which is exactly what
        // JSON.stringify(array) produces — so we store the vector as a string.
        embedding: JSON.stringify(vectors[j]),
      }));

      const { error: insErr } = await supabase.from("document_chunks").insert(rows);
      if (insErr) throw new Error(`Failed to store chunks: ${insErr.message}`);
    }

    // 6. Done — the document is now searchable.
    await supabase.from("documents").update({ status: "indexed" }).eq("id", documentId);
    return { documentId, chunks: records.length };
  } catch (err) {
    // Never leave a document stuck on "processing"; record the failure.
    await supabase.from("documents").update({ status: "failed" }).eq("id", documentId);
    throw err;
  }
}

// A retrieved chunk, in the shape both search RPCs return (semantic adds
// `similarity`, keyword adds `rank`; we only rely on the shared fields + `id`).
interface Chunk {
  id: string;
  document_id: string;
  content: string;
  page_number: number | null;
  filename: string;
}

/**
 * Reciprocal Rank Fusion — merge several ranked lists into one.
 * Each list votes for a chunk by its POSITION (rank 1, 2, 3…); a chunk's fused
 * score is the sum of 1/(k + rank) across the lists it appears in. This rewards
 * chunks that rank high in EITHER search without needing the two scores (cosine
 * vs ts_rank, which aren't comparable) to be on the same scale. Standard k=60.
 */
function rrf(lists: Chunk[][], k: number): Chunk[] {
  const score = new Map<string, number>();
  const byId = new Map<string, Chunk>();
  for (const list of lists) {
    list.forEach((item, i) => {
      const rank = i + 1; // 1-based
      score.set(item.id, (score.get(item.id) ?? 0) + 1 / (k + rank));
      if (!byId.has(item.id)) byId.set(item.id, item);
    });
  }
  return [...byId.values()].sort((a, b) => score.get(b.id)! - score.get(a.id)!);
}

// ===========================================================================
// ACTION 2 — ASK: question → hybrid retrieve → rerank → grounded answer → mask
// ===========================================================================
async function ask(question: string) {
  const warnings: string[] = []; // soft, non-fatal issues to surface in the UI

  // 1. Turn the question into the same kind of vector as our chunks.
  const [questionEmbedding] = await embed([question]);

  // 2. HYBRID RETRIEVAL — run both searches in parallel:
  //    • semantic (meaning) via pgvector, with only a very low garbage floor;
  //    • keyword (exact terms) via Postgres full-text search.
  //    Either arm failing is a real error we surface — never a silent empty result.
  const [sem, fts] = await Promise.all([
    supabase.rpc("match_document_chunks", {
      query_embedding: JSON.stringify(questionEmbedding),
      match_count: RETRIEVE_CANDIDATES,
      min_similarity: SEMANTIC_FLOOR,
    }),
    supabase.rpc("match_document_chunks_fts", {
      query_text: question,
      match_count: RETRIEVE_CANDIDATES,
    }),
  ]);
  if (sem.error) throw new Error(`Semantic retrieval failed: ${sem.error.message}`);
  if (fts.error) throw new Error(`Keyword retrieval failed: ${fts.error.message}`);

  // 3. Fuse the two ranked lists into one candidate list (Reciprocal Rank Fusion).
  const fused = rrf([(sem.data ?? []) as Chunk[], (fts.data ?? []) as Chunk[]], RRF_K);

  // Only refuse up front if there is genuinely nothing to work with (e.g. no
  // documents indexed). Otherwise we ALWAYS let the model judge relevance.
  if (fused.length === 0) {
    return { answer: NOT_FOUND_MESSAGE, citations: [] };
  }

  // 4. RERANK for precision: score the wide candidate set against the question and
  //    keep only the best few. Degrades gracefully (Cohere → LLM → retrieval order).
  const reranked = await rerank(question, fused.slice(0, RETRIEVE_CANDIDATES), RERANK_TOP_N);
  if (reranked.warning) warnings.push(reranked.warning);
  const chunks = reranked.items;

  // 5. Build a numbered context block. Numbering lets the model cite "[1]", and
  //    lets us map that citation back to a real file + page for the UI.
  const context = chunks
    .map((c, i) => `[${i + 1}] (${c.filename}, page ${c.page_number ?? "?"})\n${c.content}`)
    .join("\n\n---\n\n");

  // 6. The system prompt — this is what keeps the model honest. It is grounded
  //    (only the passages, no outside knowledge) but NOT extractive-only: the
  //    model may summarize and synthesize ACROSS the passages, so broad questions
  //    like "what is this about?" get a real answer instead of a refusal.
  const systemPrompt = [
    "You are a document assistant. Answer the user's question using ONLY the",
    "numbered context passages provided below. Follow these rules exactly:",
    "1. Use only information stated in or directly supported by the passages.",
    "   Never use outside knowledge and never invent facts.",
    "2. You MAY combine and summarize across multiple passages to answer broad",
    "   questions (e.g. what a document is about, or an overview) — as long as",
    "   every claim is grounded in the passages.",
    `3. Only if the passages genuinely do not address the question, reply with`,
    `   exactly: "${NOT_FOUND_MESSAGE}" and nothing else. Do not refuse just`,
    "   because the answer is spread across passages or not phrased as a summary.",
    "4. When you answer, cite the passage(s) you used with their bracket numbers,",
    "   e.g. [1] or [2]. Quote source wording where helpful. Be concise.",
    "",
    "Context passages:",
    context,
  ].join("\n");

  // 7. Call the chat model with temperature 0.1 for faithful, stable answers.
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: CHAT_MODEL,
      temperature: CHAT_TEMPERATURE,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: question },
      ],
    }),
  });
  if (!res.ok) {
    throw new Error(`OpenAI chat failed: ${res.status} ${await res.text()}`);
  }
  const json = await res.json();
  const answer: string = json.choices?.[0]?.message?.content?.trim() ?? NOT_FOUND_MESSAGE;

  // 8. If the model refused, return no citations. isRefusal() is tolerant of a
  //    trailing period / casing so a refusal can never slip through and pick up
  //    a bogus source (the bug behind "refusal WITH a citation").
  if (isRefusal(answer)) {
    return { answer: NOT_FOUND_MESSAGE, citations: [] };
  }

  // 9. Attach citations. We surface the passages the model actually cited (by
  //    bracket number); if it answered without brackets, we fall back to the
  //    top 3 retrieved chunks (already ordered best-first) so the source box
  //    honestly reflects what grounded the answer.
  const citedNumbers = new Set(
    [...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])),
  );
  const chosen = citedNumbers.size > 0
    ? chunks.filter((_, i) => citedNumbers.has(i + 1))
    : chunks.slice(0, 3);

  const rawCitations = chosen.map((c) => ({
    document: c.filename,
    page: c.page_number ?? undefined,
    snippet: c.content.length > 240 ? c.content.slice(0, 240) + "…" : c.content,
  }));

  // 10. PII MASKING — the very last step, on OUTPUT ONLY. We mask the answer and
  //     every citation snippet together in one pass. Everything above this line
  //     (retrieval, reranking, the model) worked on the REAL text; only what we
  //     hand back to the browser is redacted. Fails closed (never leaks).
  const toMask = [answer, ...rawCitations.map((c) => c.snippet)];
  const masked = await maskOutputs(toMask);
  if (masked.warning) warnings.push(masked.warning);

  const citations = rawCitations.map((c, i) => ({ ...c, snippet: masked.texts[i + 1] }));

  return {
    answer: masked.texts[0],
    citations,
    warning: warnings.length ? warnings.join(" | ") : undefined,
  };
}

// ===========================================================================
// HTTP entrypoint — routes a single POST to the right action.
// ===========================================================================
Deno.serve(async (req) => {
  // Browser preflight.
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    // Fail fast and loud if the required secret is missing.
    if (!OPENAI_API_KEY) {
      throw new Error("Missing secret OPENAI_API_KEY. Add it in Lovable Cloud project secrets.");
    }

    const body = await req.json();
    const action = body.action as "ingest" | "ask";

    let result: unknown;
    if (action === "ingest") {
      if (!body.documentId) throw new Error("ingest requires a documentId");
      result = await ingest(body.documentId);
    } else if (action === "ask") {
      const question = (body.question ?? "").toString().trim();
      if (!question) throw new Error("ask requires a non-empty question");
      result = await ask(question);
    } else {
      throw new Error(`Unknown action: ${action}. Use "ingest" or "ask".`);
    }

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("[rag] error:", err);
    return new Response(
      JSON.stringify({ error: err instanceof Error ? err.message : String(err) }),
      { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }
});
