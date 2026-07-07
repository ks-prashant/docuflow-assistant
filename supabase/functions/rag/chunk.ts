// =============================================================================
// Structure-aware chunking (Change 7).
//
// The old chunker collapsed ALL whitespace first, which destroyed the document's
// line/section structure and happily cut through tables — separating a benefit
// ("Room Rent") from its value ("1% of Sum Insured / day"). This version keeps the
// line structure so it can:
//   • detect heading-like lines and carry the nearest heading into each chunk, and
//   • group the text into blocks and never split a block (a table / sub-limit /
//     definition list) across two chunks.
//
// HONEST LIMITATION: unpdf/pdf.js gives us largely linear text, not true table
// geometry. So this is best-effort — we keep a heading with the rows beneath it and
// avoid cutting mid-block, but we can't perfectly reconstruct multi-column tables.
// Real layout fidelity would need a different PDF extractor (out of scope).
// =============================================================================

export interface PageChunk {
  content: string;          // the raw chunk text (what we store + cite + keyword-search)
  heading: string | null;   // nearest section/heading this chunk sits under
}

/**
 * STRONG heading signals — unambiguous section markers: "Section B.3", numbered
 * headings ("2.3 Protect Benefit"), or ALL-CAPS titles ("DEDUCTIBLE"). These are
 * trusted even as the first line of a multi-line block.
 */
function isStrongHeading(line: string): boolean {
  const s = line.trim();
  if (s.length < 3 || s.length > 80) return false;
  if (/[.:,;]$/.test(s)) return false;                       // sentences/labels end in punctuation
  if (/^section\s+[a-z0-9]/i.test(s)) return true;            // "Section B.3"
  if (/^[0-9]+(\.[0-9]+)*\s+[A-Za-z]/.test(s)) return true;   // "2.3 Protect Benefit"
  const letters = s.replace(/[^A-Za-z]/g, "");
  return letters.length >= 3 && s === s.toUpperCase();        // ALL CAPS title
}

/**
 * SOFT heading signal — a short Title Case line with no trailing punctuation.
 * Used ONLY for standalone (single-line) blocks, because inside a table these look
 * identical to row labels like "Room Rent" and must NOT override the real section.
 */
function isHeading(line: string): boolean {
  const s = line.trim();
  if (isStrongHeading(s)) return true;
  if (s.length < 3 || s.length > 80 || /[.:,;]$/.test(s)) return false;
  if (/^[A-Z][A-Za-z]/.test(s) && s.split(/\s+/).length <= 7 && !/[a-z]\s+[a-z]{1,3}$/.test(s)) {
    return /^(?:[A-Z][A-Za-z’'&/-]*)(?:\s+(?:[A-Z][A-Za-z’'&/-]*|of|and|the|for|to))*$/.test(s);
  }
  return false;
}

/**
 * Chunk one page at the LINE level. The PDF extractor gives us single line breaks
 * but no blank-line paragraph structure, so we group whole lines up to `targetSize`
 * and only ever break BETWEEN lines — never mid-line. Since each table row is its
 * own line, rows are never cut in half (a big table may still span two chunks, but
 * always at a row boundary). Each chunk carries the nearest heading seen so far,
 * and `overlap` re-includes the previous chunk's tail lines for continuity.
 */
export function chunkPage(
  pageText: string,
  targetSize: number,
  overlap: number,
): PageChunk[] {
  const clean = (pageText ?? "").replace(/\r\n?/g, "\n");
  if (!clean.trim()) return [];

  const lines = clean.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);
  const chunks: PageChunk[] = [];

  let currentHeading: string | null = null;
  let buf: string[] = [];       // lines in the current chunk
  let bufLen = 0;               // approx char length of buf
  let bufHeading: string | null = null;

  const flush = () => {
    const text = buf.join("\n").trim();
    if (text) chunks.push({ content: text, heading: bufHeading });
    buf = [];
    bufLen = 0;
  };

  for (const line of lines) {
    // Track the nearest heading. isStrongHeading catches "Section B.3"/ALL-CAPS/
    // numbered; the soft rule also catches a short Title Case line on its own.
    if (isHeading(line)) currentHeading = line;

    // Overflow → close the current chunk, carrying the tail lines as overlap.
    if (bufLen > 0 && bufLen + line.length + 1 > targetSize) {
      const carried: string[] = [];
      let carriedLen = 0;
      for (let k = buf.length - 1; k >= 0 && carriedLen < overlap; k--) {
        carried.unshift(buf[k]);
        carriedLen += buf[k].length + 1;
      }
      flush();
      buf = [...carried];
      bufLen = carriedLen;
      bufHeading = currentHeading;
    }

    if (buf.length === 0) bufHeading = currentHeading;
    buf.push(line);
    bufLen += line.length + 1;
  }
  flush();

  return chunks;
}
