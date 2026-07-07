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
 * A "block" is a run of consecutive non-empty lines (a paragraph, a table, or a
 * list). We split on blank lines. Blocks are the atomic unit we never break across
 * chunks — that is what keeps a table's rows with their header.
 */
function splitIntoBlocks(pageText: string): string[] {
  return pageText
    .replace(/\r\n?/g, "\n")
    .split(/\n{2,}|\n(?=\s*$)/)   // blank-line boundaries
    .map((b) => b.replace(/[ \t]+\n/g, "\n").trim())
    .filter((b) => b.length > 0);
}

/**
 * Chunk one page. `targetSize` is the soft prose target; `overlap` re-includes the
 * tail of the previous chunk for prose continuity. Blocks are never split; a block
 * larger than the target becomes its own (over-sized) chunk rather than being cut.
 */
export function chunkPage(
  pageText: string,
  targetSize: number,
  overlap: number,
): PageChunk[] {
  const clean = (pageText ?? "").replace(/\r\n?/g, "\n");
  if (!clean.trim()) return [];

  const blocks = splitIntoBlocks(clean);
  const chunks: PageChunk[] = [];

  let currentHeading: string | null = null;
  let buf = "";
  let bufHeading: string | null = null;

  const flush = () => {
    const text = buf.trim();
    if (text) chunks.push({ content: text, heading: bufHeading });
    buf = "";
  };

  for (const block of blocks) {
    // A block that is ONLY a heading updates context but isn't a chunk on its own.
    const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
    if (lines.length === 1 && isHeading(lines[0])) {
      flush();
      currentHeading = lines[0];
      continue;
    }
    // If a multi-line block starts with a STRONG heading, adopt it (but ignore
    // soft title-case first lines, which are usually table row labels).
    if (lines.length > 1 && isStrongHeading(lines[0])) {
      currentHeading = lines[0];
    }

    // Start a fresh chunk if adding this block would overflow the prose target
    // (but never split the block itself).
    if (buf && (buf.length + block.length + 1) > targetSize) {
      const tail = overlap > 0 ? buf.slice(-overlap) : "";
      flush();
      buf = tail ? tail + "\n" : "";
    }
    if (!buf) bufHeading = currentHeading; // record the heading this chunk opened under
    buf += (buf ? "\n" : "") + block;

    // A single block bigger than the target is its own chunk (don't cut a table).
    if (buf.length >= targetSize) flush();
  }
  flush();

  return chunks;
}
