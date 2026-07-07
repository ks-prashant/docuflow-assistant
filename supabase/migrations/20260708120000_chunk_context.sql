-- =============================================================================
-- Contextual retrieval + structure-aware chunking support.
--
-- Two new per-chunk metadata columns (both nullable, so existing rows stay valid
-- until re-indexed):
--   • heading      — the nearest section/heading the chunk sits under, detected
--                    during structure-aware chunking (Change 7).
--   • context_line — a one-line, LLM-generated description of what the chunk is
--                    and where it sits in the document (Change 6). We also PREPEND
--                    this line to the text we embed, so each chunk's vector carries
--                    its own context instead of floating free.
--
-- Note: `content` still stores the RAW chunk text, so citations and the keyword
-- (content_tsv) search keep working on the real words. Only the EMBEDDED text is
-- context-prepended. No RPC changes are needed — richer embeddings simply improve
-- match_document_chunks ranking automatically.
-- =============================================================================

ALTER TABLE public.document_chunks
  ADD COLUMN IF NOT EXISTS heading      text,
  ADD COLUMN IF NOT EXISTS context_line text;
