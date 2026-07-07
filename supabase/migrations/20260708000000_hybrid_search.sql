-- =============================================================================
-- Hybrid retrieval: add keyword (full-text) search alongside the existing
-- pgvector semantic search, so questions with exact terms — "co-payment",
-- a policyholder's name, "Section B.3" — are found even when their *meaning*
-- vector isn't the closest. We keep semantic search untouched and ADD a
-- parallel keyword search; the edge function fuses the two result lists.
-- =============================================================================

-- 1. Full-text search column.
--    `content_tsv` is a GENERATED column: Postgres derives it automatically from
--    `content` for every existing row (on creation) and every future insert/update.
--    That means no trigger and no manual backfill — existing chunks are covered
--    the moment this runs. 'english' applies stemming/stop-words so "paying" and
--    "pay" match.
ALTER TABLE public.document_chunks
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

-- 2. GIN index makes keyword search fast (the standard index type for tsvector).
CREATE INDEX IF NOT EXISTS document_chunks_content_tsv_idx
  ON public.document_chunks USING gin (content_tsv);

-- 3. The vector (HNSW) index that an earlier migration file defined but was never
--    applied to the live DB. Speeds up semantic search; cosine ops must match the
--    `<=>` operator used by match_document_chunks.
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
  ON public.document_chunks USING hnsw (embedding vector_cosine_ops);

-- 4. Keyword-search RPC, mirroring match_document_chunks' shape so the edge
--    function can treat both result lists the same way. Ranked by ts_rank
--    (higher = better).
--    Note on OR semantics: websearch_to_tsquery gives us stemming + safe parsing
--    of natural, user-typed queries, but it ANDs every term — too strict for a
--    full question (it would require ALL words in one chunk). We swap '&' for '|'
--    on its text form to get OR semantics: a chunk matches if it contains ANY
--    significant term. Phrase operators (<->) from quoted input are preserved.
CREATE OR REPLACE FUNCTION public.match_document_chunks_fts(
  query_text  text,
  match_count int DEFAULT 25
)
RETURNS TABLE (
  id          uuid,
  document_id uuid,
  content     text,
  page_number int,
  filename    text,
  rank        float
)
LANGUAGE sql
STABLE
SET search_path = public
AS $$
  WITH q AS (
    SELECT NULLIF(
      replace(websearch_to_tsquery('english', query_text)::text, '&', '|'), ''
    )::tsquery AS tsq
  )
  SELECT
    c.id,
    c.document_id,
    c.content,
    c.page_number,
    d.filename,
    ts_rank(c.content_tsv, q.tsq) AS rank
  FROM public.document_chunks AS c
  JOIN public.documents       AS d ON d.id = c.document_id
  CROSS JOIN q
  -- q.tsq IS NULL for an empty/garbage query → no rows (semantic search still runs).
  WHERE q.tsq IS NOT NULL AND c.content_tsv @@ q.tsq
  ORDER BY rank DESC
  LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION public.match_document_chunks_fts(text, int)
  TO anon, authenticated, service_role;
