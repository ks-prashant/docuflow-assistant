-- =============================================================================
-- Retrieval primitive for the RAG pipeline.
--
-- Semantic search is "find the chunks whose meaning is closest to the question".
-- With pgvector, "closest in meaning" == "closest vector". We store a 1536-dim
-- embedding per chunk (produced by OpenAI text-embedding-3-small) and compare it
-- to the question's embedding using COSINE DISTANCE (the `<=>` operator).
--
-- We do this comparison INSIDE Postgres — not in the edge function — for two
-- reasons a client would care about:
--   1. Correctness/scale: the DB can use an index and never has to ship every
--      chunk's 1536 numbers over the network to be ranked in JavaScript.
--   2. Simplicity: the edge function asks one question ("give me the top N")
--      and gets back exactly the rows it needs, already joined to the filename.
-- =============================================================================

-- Approximate-nearest-neighbour index so top-k search stays fast as the corpus
-- grows. `vector_cosine_ops` must match the distance operator used below (`<=>`).
-- HNSW gives good recall/latency without needing to be tuned per-dataset.
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
  ON public.document_chunks
  USING hnsw (embedding vector_cosine_ops);

-- match_document_chunks: given a question embedding, return the most similar
-- chunks. `similarity` is returned as 1 - cosine_distance so that 1.0 == identical
-- and higher == more relevant (easier for the caller to threshold and display).
CREATE OR REPLACE FUNCTION public.match_document_chunks(
  query_embedding vector(1536),
  match_count     int   DEFAULT 5,
  min_similarity  float DEFAULT 0.0
)
RETURNS TABLE (
  id          uuid,
  document_id uuid,
  content     text,
  page_number int,
  filename    text,
  similarity  float
)
LANGUAGE sql
STABLE
AS $$
  SELECT
    c.id,
    c.document_id,
    c.content,
    c.page_number,
    d.filename,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM public.document_chunks AS c
  JOIN public.documents       AS d ON d.id = c.document_id
  WHERE c.embedding IS NOT NULL
    -- Drop weak matches early so a totally unrelated question returns nothing,
    -- which is what lets the LLM confidently say "not in the documents".
    AND (1 - (c.embedding <=> query_embedding)) >= min_similarity
  -- Order by raw distance (ascending = closest first) so the index can be used.
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- The edge function calls this via PostgREST, so the API roles need EXECUTE.
GRANT EXECUTE ON FUNCTION public.match_document_chunks(vector, int, float)
  TO anon, authenticated, service_role;
