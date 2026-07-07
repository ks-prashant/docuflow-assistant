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
import { rerank } from "./rerank.ts";               // wide-recall candidates → precise top-N
import { maskStructured, maskSpans, findNameAddressSpans } from "./mask.ts"; // OUTPUT-ONLY PII masking
import { transformQuery } from "./querytransform.ts"; // expand question → search variants (Change 5)
import { chunkPage } from "./chunk.ts";             // structure-aware chunking (Change 7)
import { summarizeDocument, generateContexts } from "./context.ts"; // per-chunk context (Change 6)

// ---------------------------------------------------------------------------
// Configuration — all the knobs in one place so they're easy to explain/tune.
// ---------------------------------------------------------------------------
const EMBEDDING_MODEL = "text-embedding-3-small"; // 1536 dims → matches our vector column
// Two model tiers (Change 7): a stronger model writes the final answer; a cheap
// model does the many helper steps (query transform, context generation, LLM
// rerank fallback, PII redaction). Switch the answer model here in one place.
const ANSWER_MODEL = "gpt-4o-mini";  // final answer — fast + cheap; a detailed prompt
                                     // carries answer quality. Switch here in one place.
                                     // (Helper steps — query transform, context gen,
                                     // rerank fallback, snippet redaction — also use
                                     // gpt-4o-mini, set inside their own modules.)
const CHAT_TEMPERATURE = 0.1;        // low = faithful, repeatable answers

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

// Text chunking now lives in the structure-aware module ./chunk.ts (chunkPage),
// which preserves headings and keeps tables/lists intact (Change 7).

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

    // 4. STRUCTURE-AWARE CHUNKING (Change 7): split each page while preserving
    //    headings and keeping tables/lists intact. Each chunk remembers its page
    //    and the nearest heading it sits under.
    const records: { content: string; page_number: number; heading: string | null }[] = [];
    pageTexts.forEach((pageText, i) => {
      for (const c of chunkPage(pageText ?? "", CHUNK_SIZE, CHUNK_OVERLAP)) {
        records.push({ content: c.content, page_number: i + 1, heading: c.heading });
      }
    });

    if (records.length === 0) {
      // A scanned/image-only PDF yields no extractable text — flag it clearly
      // instead of silently indexing nothing.
      await supabase.from("documents").update({ status: "no_text" }).eq("id", documentId);
      return { documentId, chunks: 0, note: "No extractable text (is this a scanned PDF?)" };
    }

    // 5. CONTEXTUAL RETRIEVAL (Change 6): summarize the doc once, then generate a
    //    one-line context per chunk. We embed context+chunk, but STORE the raw
    //    chunk. If context generation degrades, we embed the raw text instead.
    const summary = await summarizeDocument(pageTexts, records.map((r) => r.heading ?? ""));
    const { contexts, warning: ctxWarning } = await generateContexts(
      summary,
      records.map((r) => ({ content: r.content, heading: r.heading })),
    );

    // 6. Embed in batches (context-prepended text) and store each chunk + vector.
    let chunkIndex = 0;
    for (let i = 0; i < records.length; i += EMBED_BATCH_SIZE) {
      const batch = records.slice(i, i + EMBED_BATCH_SIZE);
      const batchContexts = contexts.slice(i, i + EMBED_BATCH_SIZE);
      // The text we EMBED = context line + raw chunk (or just the chunk if the
      // context degraded). The text we STORE in `content` is always the raw chunk.
      const embedInputs = batch.map((r, j) =>
        batchContexts[j] ? `${batchContexts[j]}\n\n${r.content}` : r.content
      );
      const vectors = await embed(embedInputs);

      const rows = batch.map((r, j) => ({
        document_id: documentId,
        chunk_index: chunkIndex++,
        content: r.content,       // raw chunk — used for citations + keyword search
        page_number: r.page_number,
        heading: r.heading,
        context_line: batchContexts[j],
        // pgvector accepts its text form "[0.1,0.2,...]", which is exactly what
        // JSON.stringify(array) produces — so we store the vector as a string.
        embedding: JSON.stringify(vectors[j]),
      }));

      const { error: insErr } = await supabase.from("document_chunks").insert(rows);
      if (insErr) throw new Error(`Failed to store chunks: ${insErr.message}`);
    }

    // 7. Done — the document is now searchable.
    await supabase.from("documents").update({ status: "indexed" }).eq("id", documentId);
    return { documentId, chunks: records.length, warning: ctxWarning };
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

// Decide whether a question needs the (latency-adding) query-transform step.
// Long, specific questions are usually self-sufficient, so we skip the rewrite to
// save ~0.5–1.5s. Short or vaguely-worded ones ("whose policy is this?") benefit
// from jargon expansion, so we keep it for them.
function needsRephrase(question: string): boolean {
  const words = question.trim().split(/\s+/).filter(Boolean);
  if (words.length <= 6) return true;                              // short → expand
  if (words.length < 10 && /\b(this|that|these|those|it|whose|they|there)\b/i.test(question)) {
    return true;                                                   // vague reference → expand
  }
  return false;                                                    // long & specific → skip
}

// The answer instructions. Detailed on HOW to write a good grounded answer, and —
// because answers now STREAM token-by-token — it masks personal data INLINE as it
// writes (so nothing unmasked ever reaches the browser). A deterministic regex net
// on the outgoing stream still guarantees the structured PII types.
function buildSystemPrompt(context: string): string {
  return [
    "You are a meticulous insurance-document assistant. Answer the user's question",
    "using ONLY the numbered context passages below.",
    "",
    "HOW TO ANSWER WELL:",
    "- Ground every statement in the passages. Never use outside knowledge or guess.",
    "- Start with a direct answer to the exact question, then the key supporting detail.",
    "- You MAY combine and summarize across passages for broad questions.",
    "- Preserve specifics exactly: numbers, percentages, monetary limits, time periods,",
    "  and section names (e.g. \"1% of Sum Insured per day\", \"Section B.3\").",
    "- When values come from a table/sub-limit list, pair each value with what it applies to.",
    "- Format with brief markdown: a lead sentence, then bullet points for multiple values.",
    "  Be concise — no filler, no repeating the question.",
    "- Cite the passages you used with bracket numbers like [1], [2].",
    `- ONLY if the passages genuinely do not address the question, reply with EXACTLY`,
    `  "${NOT_FOUND_MESSAGE}" and nothing else. Do not refuse just because the answer is`,
    "  spread across passages or not phrased as a summary.",
    "",
    "PRIVACY — mask personal data IN YOUR ANSWER as you write it:",
    "- Replace person names, full postal addresses, phone numbers, email addresses,",
    "  policy/certificate/ID numbers, PAN and Aadhaar numbers with asterisks, leaving",
    "  only the LAST 3 characters visible. Example: \"Prashant Kumar Singh\" ->",
    "  \"*****************ngh\"; policy \"2856000112902\" -> \"**********902\".",
    "- Do NOT mask insurer/company names, product names, or ordinary values like",
    "  coverage amounts, percentages, dates, or section numbers.",
    "",
    "Context passages:",
    context,
  ].join("\n");
}

// ===========================================================================
// ACTION 2 — ASK (STREAMING): question → hybrid retrieve → rerank → stream a
// grounded, PII-masked answer token-by-token → send citations at the end.
//
// Returns a ReadableStream of newline-delimited JSON events:
//   {"type":"token","text":"..."}                          (answer text, masked)
//   {"type":"done","citations":[...],"warning":"...","rerankBackend":"..."}
//   {"type":"error","error":"..."}
// ===========================================================================
function askStream(question: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (obj: unknown) => controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"));
      const warnings: string[] = [];
      try {
        // 1. QUERY TRANSFORM — only for short/vague questions (Change: skip when clear).
        let variants = [question];
        if (needsRephrase(question)) {
          const t = await transformQuery(question);
          if (t.warning) warnings.push(t.warning);
          variants = t.variants;
        }

        // 2. Embed all variants in one call.
        const variantEmbeddings = await embed(variants);

        // 3. HYBRID RETRIEVAL per variant (semantic + keyword), all in parallel.
        const searches = await Promise.all(
          variants.flatMap((variant, i) => [
            supabase.rpc("match_document_chunks", {
              query_embedding: JSON.stringify(variantEmbeddings[i]),
              match_count: RETRIEVE_CANDIDATES,
              min_similarity: SEMANTIC_FLOOR,
            }),
            supabase.rpc("match_document_chunks_fts", {
              query_text: variant,
              match_count: RETRIEVE_CANDIDATES,
            }),
          ]),
        );
        for (const r of searches) {
          if (r.error) throw new Error(`Retrieval failed: ${r.error.message}`);
        }

        // 4. Fuse all ranked lists with Reciprocal Rank Fusion.
        const fused = rrf(searches.map((r) => (r.data ?? []) as Chunk[]), RRF_K);
        if (fused.length === 0) {
          send({ type: "token", text: NOT_FOUND_MESSAGE });
          send({ type: "done", citations: [], warning: joinWarnings(warnings) });
          controller.close();
          return;
        }

        // 5. Rerank against the ORIGINAL question; keep the best few.
        const reranked = await rerank(question, fused.slice(0, RETRIEVE_CANDIDATES), RERANK_TOP_N);
        if (reranked.warning) warnings.push(reranked.warning);
        const chunks = reranked.items;

        // 6. Build numbered context + the detailed, mask-aware system prompt.
        const context = chunks
          .map((c, i) => `[${i + 1}] (${c.filename}, page ${c.page_number ?? "?"})\n${c.content}`)
          .join("\n\n---\n\n");
        const systemPrompt = buildSystemPrompt(context);

        // 7. STREAM the answer from OpenAI. Two masking layers protect the stream:
        //    (a) regex net for structured PII (phone/email/IDs/PAN/Aadhaar), and
        //    (b) deterministic NAME/ADDRESS masking using spans we extract from the
        //        retrieved chunks up front (the answer's names come from there) —
        //        because the model alone does NOT reliably mask names inline.
        //    We start the span extraction and the answer generation in parallel so
        //    the extra call barely adds latency, and flush at SENTENCE boundaries so
        //    a whole name is always within the segment we mask.
        const spansPromise = findNameAddressSpans(chunks.map((c) => c.content))
          .catch(() => null); // null → couldn't extract; we degrade + warn below

        const res = await fetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${OPENAI_API_KEY}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: ANSWER_MODEL,
            temperature: CHAT_TEMPERATURE,
            stream: true,
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: question },
            ],
          }),
        });
        if (!res.ok || !res.body) {
          throw new Error(`OpenAI chat failed: ${res.status} ${res.body ? await res.text() : "(no body)"}`);
        }

        const piiSpans = await spansPromise;
        if (piiSpans === null) {
          warnings.push("Name/address masking degraded (span extraction failed); structured PII still masked");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let sseBuf = "";     // raw SSE lines not yet parsed
        let outBuf = "";     // model text not yet safe to flush
        let fullAnswer = ""; // the masked answer we accumulate for refusal/citation logic

        // Flush emits masked text up to the last SENTENCE boundary (so a full name
        // is never split across segments). Structured regex + known name/address
        // spans are applied to each segment before it leaves the server.
        const flush = (final: boolean) => {
          if (!outBuf) return;
          let upto: number;
          if (final) {
            upto = outBuf.length;
          } else {
            const matches = [...outBuf.matchAll(/[.!?\n]["')\]]?\s/g)];
            if (matches.length > 0) {
              const last = matches[matches.length - 1];
              upto = (last.index ?? 0) + last[0].length;
            } else if (outBuf.length > 400) {
              const ws = outBuf.lastIndexOf(" "); // fallback for a very long run
              upto = ws >= 0 ? ws + 1 : 0;
            } else {
              upto = 0; // hold until we have a complete sentence
            }
          }
          if (upto === 0) return;
          let seg = maskStructured(outBuf.slice(0, upto));
          if (piiSpans) seg = maskSpans(seg, piiSpans);
          outBuf = outBuf.slice(upto);
          fullAnswer += seg;
          send({ type: "token", text: seg });
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          sseBuf += decoder.decode(value, { stream: true });
          const lines = sseBuf.split("\n");
          sseBuf = lines.pop() ?? "";
          for (const line of lines) {
            const t = line.trim();
            if (!t.startsWith("data:")) continue;
            const data = t.slice(5).trim();
            if (data === "[DONE]") continue;
            try {
              const delta = JSON.parse(data).choices?.[0]?.delta?.content;
              if (typeof delta === "string" && delta) {
                outBuf += delta;
                flush(false);
              }
            } catch { /* ignore keep-alive / non-JSON lines */ }
          }
        }
        flush(true);
        const answer = fullAnswer.trim();

        // 8. Refusal → no citations, and we SKIP the snippet scrub (no PII to mask).
        if (isRefusal(answer)) {
          send({ type: "done", citations: [], warning: joinWarnings(warnings), rerankBackend: reranked.backend });
          controller.close();
          return;
        }

        // 9. Choose citations from the bracket numbers the model cited (or top-3),
        //    and mask each snippet with the SAME spans we already extracted (the
        //    snippets come from these very chunks, so coverage is exact — no extra
        //    LLM call needed). Skipped entirely on refusals above (no PII to mask).
        const cited = new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])));
        const chosen = cited.size > 0 ? chunks.filter((_, i) => cited.has(i + 1)) : chunks.slice(0, 3);
        const citations = chosen.map((c) => {
          const raw = c.content.length > 240 ? c.content.slice(0, 240) + "…" : c.content;
          let snippet = maskStructured(raw);
          if (piiSpans) snippet = maskSpans(snippet, piiSpans);
          return { document: c.filename, page: c.page_number ?? undefined, snippet };
        });

        send({ type: "done", citations, warning: joinWarnings(warnings), rerankBackend: reranked.backend });
        controller.close();
      } catch (err) {
        // Surface any failure to the UI as a stream error event (never silent).
        send({ type: "error", error: err instanceof Error ? err.message : String(err) });
        controller.close();
      }
    },
  });
}

function joinWarnings(warnings: string[]): string | undefined {
  return warnings.length ? warnings.join(" | ") : undefined;
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

    // "ask" STREAMS its response (newline-delimited JSON events); "ingest" returns
    // a single JSON object once processing finishes.
    if (action === "ask") {
      const question = (body.question ?? "").toString().trim();
      if (!question) throw new Error("ask requires a non-empty question");
      return new Response(askStream(question), {
        headers: {
          ...corsHeaders,
          "Content-Type": "application/x-ndjson; charset=utf-8",
          "Cache-Control": "no-cache",
        },
      });
    }

    let result: unknown;
    if (action === "ingest") {
      if (!body.documentId) throw new Error("ingest requires a documentId");
      result = await ingest(body.documentId);
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
