// =============================================================================
// Contextual retrieval (Change 6) — give each chunk its own context before we
// embed it. A chunk like "1% of Sum Insured per day" is meaningless on its own;
// prefixed with "This is the Room Rent sub-limit in the hospitalization section
// of the HDFC ERGO health policy," its embedding lands near questions about bed
// limits. We generate that one-liner with a cheap model and PREPEND it to the
// embedded text (the raw chunk is still stored + searched separately).
//
// Cost control: we summarize the whole document ONCE and pass that short summary
// (plus each chunk's heading) as reference — instead of re-reading the full doc
// for every chunk — and we generate contexts in batches.
// =============================================================================

const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY");
const HELPER_MODEL = "gpt-4o-mini"; // cheap; context generation is a helper step
const CONTEXT_BATCH_SIZE = 12;      // chunks described per LLM call

async function chat(messages: unknown, jsonMode: boolean): Promise<string> {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: HELPER_MODEL,
      temperature: 0,
      ...(jsonMode ? { response_format: { type: "json_object" } } : {}),
      messages,
    }),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  const json = await res.json();
  return json.choices?.[0]?.message?.content ?? "";
}

/**
 * One short whole-document summary, used as cheap reference for per-chunk context.
 * Never throws — an empty summary just means contexts rely on headings alone.
 */
export async function summarizeDocument(
  pageTexts: string[],
  headings: string[],
): Promise<string> {
  try {
    const sample = pageTexts.join("\n").replace(/\s+/g, " ").slice(0, 6000);
    const headingList = [...new Set(headings.filter(Boolean))].slice(0, 40).join("; ");
    const content =
      `In 2-3 sentences, summarize this document: what product/document it is, ` +
      `who it is for, and the main topics/sections it covers. Be specific ` +
      `(e.g. insurer and product name if present).\n\n` +
      `SECTION HEADINGS: ${headingList}\n\nSAMPLE TEXT: ${sample}`;
    return (await chat([{ role: "user", content }], false)).trim();
  } catch {
    return ""; // non-fatal: proceed without a summary
  }
}

/**
 * Generate a one-line context per chunk, in batches. Returns an array aligned to
 * `chunks` where each entry is the context string, or null if that batch failed
 * (the caller then embeds that chunk's raw text — degrade, never break). Sets a
 * warning if any batch degraded.
 */
export async function generateContexts(
  summary: string,
  chunks: { content: string; heading: string | null }[],
): Promise<{ contexts: (string | null)[]; warning?: string }> {
  const contexts: (string | null)[] = new Array(chunks.length).fill(null);
  let degraded = 0;

  for (let start = 0; start < chunks.length; start += CONTEXT_BATCH_SIZE) {
    const batch = chunks.slice(start, start + CONTEXT_BATCH_SIZE);
    const listed = batch
      .map((c, i) =>
        `[${i}] heading: ${c.heading ?? "(none)"}\n${c.content.slice(0, 400)}`
      )
      .join("\n\n");
    const content =
      `DOCUMENT SUMMARY: ${summary || "(not available)"}\n\n` +
      `For each passage below, write ONE short sentence (max ~20 words) saying ` +
      `what it is and where it sits in the document, so it can be understood on ` +
      `its own. Use the heading. Do not answer questions or add extra text.\n` +
      `Return ONLY JSON: {"contexts":[{"i":<index>,"c":"<sentence>"}, ...]} with ` +
      `one entry per passage.\n\n${listed}`;

    try {
      const raw = await chat([{ role: "user", content }], true);
      const parsed = JSON.parse(raw || "{}");
      const arr = (parsed.contexts ?? []) as { i: number; c: string }[];
      batch.forEach((_, i) => {
        const hit = arr.find((x) => x.i === i);
        contexts[start + i] = typeof hit?.c === "string" ? hit.c.trim() : null;
        if (contexts[start + i] === null) degraded++;
      });
    } catch {
      // Whole batch failed → leave as null (caller embeds raw text for these).
      degraded += batch.length;
    }
  }

  return {
    contexts,
    warning: degraded > 0
      ? `Context generation degraded for ${degraded}/${chunks.length} chunks (embedded raw text instead)`
      : undefined,
  };
}
