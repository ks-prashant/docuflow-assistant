// =============================================================================
// PII masking — OUTPUT ONLY.
//
// This module masks personal data in what we SEND BACK to the browser: the final
// answer text and every citation snippet. It is deliberately the very last step.
//
// Non-negotiable rule: it NEVER touches stored chunks, embeddings, or the text we
// search on. Retrieval and the answer model still see the real values (so "whose
// policy is this?" can be answered) — we only redact the words the user sees.
//
// Two layers, by design:
//   1. Regex layer (deterministic, high precision) for STRUCTURED data — phone,
//      email, long/grouped numeric IDs (policy/certificate), PAN, Aadhaar.
//   2. LLM layer (gpt-4o-mini, temperature 0) for names + full street addresses,
//      which regex handles poorly. It returns the text unchanged except PII.
//
// Fail-closed: if the LLM layer errors or misbehaves, we still return the
// regex-masked text (structured PII already hidden) plus a warning — we never
// return fully-unmasked output.
//
// Masking style: replace every character except the last `VISIBLE_TAIL` with "*".
//   policy "2856000112902" -> "**********902"
//   phone  "7512345676"    -> "*******676"
// =============================================================================

const OPENAI_API_KEY = Deno.env.get("OPENAI_API_KEY");

// The single knob for "how many trailing characters stay visible". Change here only.
export const VISIBLE_TAIL = 3;

/** Replace all but the last VISIBLE_TAIL characters with "*". Short values are fully masked. */
export function maskValue(s: string): string {
  if (s.length <= VISIBLE_TAIL) return "*".repeat(s.length);
  return "*".repeat(s.length - VISIBLE_TAIL) + s.slice(-VISIBLE_TAIL);
}

// --- Layer 1: deterministic regex masking ------------------------------------
// Applied in order. Emails go first so their embedded dots/digits aren't caught
// by the numeric rules. Each match is replaced by maskValue() of the match.
const PII_PATTERNS: RegExp[] = [
  // Email addresses.
  /[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}/g,
  // Aadhaar — 12 digits, optionally grouped 4-4-4 by spaces/hyphens.
  /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/g,
  // PAN — 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F).
  /\b[A-Z]{5}[0-9]{4}[A-Z]\b/g,
  // Long or grouped numeric identifiers: policy numbers, certificate numbers,
  // and phone numbers. Matches a run that starts and ends with a digit and spans
  // at least ~9 digit/space/hyphen characters — covers "7512345676" and
  // "2856 2076 3772 4200 000" while ignoring short amounts like "12,345".
  /\b\d[\d\s-]{7,}\d\b/g,
];

/** Mask all structured PII. Pure and deterministic — safe to unit-test offline. */
export function maskStructured(text: string): string {
  let out = text;
  for (const re of PII_PATTERNS) {
    out = out.replace(re, (match) => maskValue(match));
  }
  return out;
}

// --- Layer 2: LLM redaction for names + addresses ----------------------------
// One batched call redacts an array of texts at once (answer + each snippet).
async function llmRedactBatch(texts: string[]): Promise<string[]> {
  const prompt =
    `You are a redaction filter. For each input string, replace ONLY personal ` +
    `names and full street/postal addresses with asterisks, keeping the LAST ` +
    `${VISIBLE_TAIL} characters of each masked value visible ` +
    `(e.g. "Prashant Singh" -> "*********ngh"). Do NOT change anything else — ` +
    `keep all other characters, numbers, punctuation, and existing asterisks ` +
    `byte-for-byte identical. Do not add commentary.\n` +
    `Return ONLY JSON: {"out": ["<redacted 0>", "<redacted 1>", ...]} with the ` +
    `same number of items, in the same order.\n\n` +
    `INPUTS:\n${JSON.stringify(texts)}`;

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
  const out = parsed.out;
  // Guard: the model must return exactly one redacted string per input, else we
  // can't trust the mapping — treat as failure so the caller falls back safely.
  if (!Array.isArray(out) || out.length !== texts.length || out.some((s) => typeof s !== "string")) {
    throw new Error("LLM redaction returned an unexpected shape");
  }
  return out as string[];
}

/**
 * Mask an array of outgoing texts (answer + citation snippets).
 * Order matters: regex first (always applied), then the LLM name/address pass on
 * the regex-masked text. Fail-closed — on any LLM problem we return the
 * regex-masked text and a warning, never the raw text.
 */
export async function maskOutputs(
  texts: string[],
): Promise<{ texts: string[]; warning?: string }> {
  const structured = texts.map(maskStructured);
  try {
    const redacted = await llmRedactBatch(structured);
    return { texts: redacted };
  } catch (err) {
    return {
      texts: structured, // structured PII is still masked — we never leak those
      warning: `Name/address redaction unavailable; structured PII still masked: ${
        err instanceof Error ? err.message : String(err)
      }`,
    };
  }
}
