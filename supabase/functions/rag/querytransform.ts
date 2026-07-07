// =============================================================================
// Query transformation — a small, swappable module that runs BEFORE retrieval.
//
// Why it exists: users and insurance documents use different words for the same
// thing. A user asks "co-pay?"; the policy says "cost sharing" or "deductible".
// A user asks "whose policy is this?"; the document just lists a "Policyholder
// Name". Searching the user's literal words misses these. So we ask a cheap model
// to rewrite the question into a few better SEARCH queries first:
//   (a) expand insurance jargon / synonyms / acronyms, and
//   (b) turn short/vague questions into concrete search terms.
//
// It returns 2–3 variants (multi-query). The caller runs retrieval for EACH and
// fuses the results, so a hit under ANY phrasing surfaces. It never breaks the
// pipeline: on any failure it falls back to just the original question.
// =============================================================================

const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY");
const HELPER_MODEL = "gpt-4o-mini"; // cheap; this is a helper step, not the answer

export interface QueryTransform {
  variants: string[];   // includes the original question, plus rewrites
  warning?: string;     // set if we had to fall back (surfaced in the UI)
}

/**
 * Turn a raw question into 2–3 retrieval queries. The original question is always
 * kept as one variant so we never do worse than before.
 */
export async function transformQuery(question: string): Promise<QueryTransform> {
  const prompt =
    `You rewrite a user's question into search queries for an insurance-policy ` +
    `search engine. Do BOTH:\n` +
    `- Expand insurance jargon, synonyms and acronyms (e.g. "co-pay" <-> ` +
    `"co-payment" / "cost sharing"; "room rent" <-> "bed limit"; "sum insured" ` +
    `<-> "coverage amount"; "policyholder" <-> "insured / name on policy").\n` +
    `- Rewrite short or vague questions into concrete search terms (e.g. ` +
    `"whose policy is this?" -> "policyholder name, insured person name, name on ` +
    `policy document").\n` +
    `Return ONLY JSON: {"variants": ["...", "..."]} with 2 to 3 concise queries. ` +
    `Do not answer the question.\n\n` +
    `QUESTION: ${question}`;

  try {
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: HELPER_MODEL,
        temperature: 0,
        response_format: { type: "json_object" },
        messages: [{ role: "user", content: prompt }],
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    const json = await res.json();
    const parsed = JSON.parse(json.choices?.[0]?.message?.content ?? "{}");
    const raw = Array.isArray(parsed.variants) ? parsed.variants : [];
    const cleaned = raw
      .filter((v: unknown): v is string => typeof v === "string" && v.trim().length > 0)
      .map((v: string) => v.trim())
      .slice(0, 3);

    // Always include the original question, de-duplicated, first.
    const variants = dedupe([question, ...cleaned]);
    return { variants };
  } catch (err) {
    // Fail safe: search the original question only.
    return {
      variants: [question],
      warning: `Query expansion unavailable, used the original question: ${
        err instanceof Error ? err.message : String(err)
      }`,
    };
  }
}

function dedupe(items: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const s of items) {
    const key = s.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      out.push(s);
    }
  }
  return out;
}
