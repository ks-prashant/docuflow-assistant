
ALTER TABLE public.chat_messages ADD COLUMN IF NOT EXISTS thread_id uuid;
CREATE INDEX IF NOT EXISTS chat_messages_thread_id_idx ON public.chat_messages (thread_id);

CREATE OR REPLACE FUNCTION public.match_document_chunks(
  query_embedding vector,
  match_count integer DEFAULT 5,
  min_similarity double precision DEFAULT 0.0,
  filter_document_id uuid DEFAULT NULL
)
RETURNS TABLE(id uuid, document_id uuid, content text, page_number integer, filename text, similarity double precision)
LANGUAGE sql
STABLE
SET search_path TO 'public'
AS $function$
  SELECT
    c.id,
    c.document_id,
    c.content,
    c.page_number,
    d.filename,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM public.document_chunks c
  JOIN public.documents d ON d.id = c.document_id
  WHERE 1 - (c.embedding <=> query_embedding) >= min_similarity
    AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
$function$;

CREATE OR REPLACE FUNCTION public.match_document_chunks_fts(
  query_text text,
  match_count integer DEFAULT 25,
  filter_document_id uuid DEFAULT NULL
)
RETURNS TABLE(id uuid, document_id uuid, content text, page_number integer, filename text, rank double precision)
LANGUAGE sql
STABLE
SET search_path TO 'public'
AS $function$
  WITH q AS (
    SELECT NULLIF(
      replace(websearch_to_tsquery('english', query_text)::text, '&', '|'), ''
    )::tsquery AS tsq
  )
  SELECT c.id, c.document_id, c.content, c.page_number, d.filename,
         ts_rank(c.content_tsv, q.tsq) AS rank
  FROM public.document_chunks AS c
  JOIN public.documents AS d ON d.id = c.document_id
  CROSS JOIN q
  WHERE q.tsq IS NOT NULL AND c.content_tsv @@ q.tsq
    AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
  ORDER BY rank DESC
  LIMIT match_count;
$function$;
