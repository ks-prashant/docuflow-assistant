// =============================================================================
// Reranker — a small, swappable module.
//
// Why it exists: retrieval (hybrid search) is tuned for RECALL — cast a wide net
// (~25 candidates) so we don't miss the right passage. But feeding 25 passages to
// the answer model dilutes it. A reranker is a second, sharper judge that reads
// the question together with each candidate and scores true relevance, so we can
// keep only the best 8. Wide net for recall, tight rerank for precision.
//
// Two interchangeable backends, chosen at runtime:
//   • Cohere Rerank  — a purpose-built cross-encoder (used when COHERE_API_KEY is set)
//   • LLM fallback   — gpt-4o-mini scores each candidate (used if no key, or if
//                      Cohere errors) so the pipeline NEVER breaks.
//
// The module is generic: it reorders whatever candidate objects you pass in (they
// only need a `content` string) and returns the top N, plus an optional `warning`
// the caller can surface in the UI when it had to degrade to the fallback.
// =============================================================================

const COHERE_API_KEY = Deno.env.get("COHERE_API_KEY");
const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY");

// Any candidate with text to judge. We keep it generic so retrieval can attach
// whatever fields it likes (page_number, filename, similarity…) and get them back.
export interface Rerankable {
  content: string;
}

export interface RerankResult<T extends Rerankable> {
  items: T[];           // the top-N candidates, most-relevant first
  backend: "cohere" | "llm";
  warning?: string;     // set when we fell back or partially degraded
}

/**
 * Rerank `candidates` against `query` and return the best `topN`.
 * Never throws for ranking reasons — on any backend failure it degrades to the
 * LLM, and if even that fails it returns the original order (truncated) with a
 * warning, so the answer pipeline keeps working.
 */
export async function rerank<T extends Rerankable>(
  query: string,
  candidates: T[],
  topN: number,
): Promise<RerankResult<T>> {
  if (candidates.length === 0) return { items: [], backend: COHERE_API_KEY ? "cohere" : "llm" };

  // Prefer Cohere when its key is present; fall back to the LLM on any error.
  if (COHERE_API_KEY) {
    try {
      const items = await cohereRerank(query, candidates, topN);
      return { items, backend: "cohere" };
    } catch (err) {
      const warning = `Cohere rerank failed, used LLM fallback: ${msg(err)}`;
      return { ...(await llmRerank(query, candidates, topN)), warning };
    }
  }
  return llmRerank(query, candidates, topN);
}

// --- Backend 1: Cohere Rerank (cross-encoder) --------------------------------
async function cohereRerank<T extends Rerankable>(
  query: string,
  candidates: T[],
  topN: number,
): Promise<T[]> {
  const res = await fetch("https://api.cohere.com/v2/rerank", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${COHERE_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: "rerank-english-v3.0",
      query,
      documents: candidates.map((c) => c.content),
      top_n: Math.min(topN, candidates.length),
    }),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  const json = await res.json();
  // Cohere returns [{ index, relevance_score }] already sorted best-first.
  return (json.results as { index: number }[]).map((r) => candidates[r.index]);
}

// --- Backend 2: LLM fallback (gpt-4o-mini scores every candidate) ------------
async function llmRerank<T extends Rerankable>(
  query: string,
  candidates: T[],
  topN: number,
): Promise<RerankResult<T>> {
  // One batched call: score all candidates 0..1 and return JSON. Cheaper and more
  // consistent than one request per candidate.
  const numbered = candidates
    .map((c, i) => `[${i}] ${c.content.slice(0, 500)}`)
    .join("\n\n");

  const prompt =
    "Score how well each passage helps answer the QUESTION, from 0 (irrelevant) " +
    "to 1 (directly answers it). Return ONLY JSON: " +
    '{"scores":[{"i":<index>,"s":<0..1>}, ...]} with one entry per passage.\n\n' +
    `QUESTION: ${query}\n\nPASSAGES:\n${numbered}`;

  try {
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        temperature: 0,
        response_format: { type: "json_object" },
        messages: [{ role: "user", content: prompt }],
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const json = await res.json();
    const parsed = JSON.parse(json.choices?.[0]?.message?.content ?? "{}");
    const scores: number[] = candidates.map((_, i) => {
      const hit = (parsed.scores as { i: number; s: number }[] | undefined)?.find((x) => x.i === i);
      return typeof hit?.s === "number" ? hit.s : 0;
    });
    const ranked = candidates
      .map((c, i) => ({ c, s: scores[i] }))
      .sort((a, b) => b.s - a.s)
      .slice(0, topN)
      .map((x) => x.c);
    return { items: ranked, backend: "llm" };
  } catch (err) {
    // Last-resort safety net: keep the retrieval order (already decent) so the
    // answer step still runs, and tell the caller we couldn't rerank.
    return {
      items: candidates.slice(0, topN),
      backend: "llm",
      warning: `Reranking unavailable, used retrieval order: ${msg(err)}`,
    };
  }
}

function msg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
