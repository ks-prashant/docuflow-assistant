## Overview

Four scoped changes: rebrand, Documents page copy, fresh-thread-per-visit chat with Clear Chat, and per-document chat scoping (retrieval-level). Part 4 requires SQL + edge function changes and a redeploy.

## Part 1 — Rebrand to DocuFlow

Edit `src/components/AppShell.tsx`: change wordmark "Paperline" → "DocuFlow"; keep the existing round icon + serif style. Add a small muted byline "by Prashant" next to the wordmark (smaller font, `text-muted-foreground`, tight leading, no visual competition).

`src/routes/__root.tsx` already says "DocuFlow" in `<title>`/OG tags — no change. `src/routes/chat.tsx`'s title "Chat — Paperline" → "Chat — DocuFlow".

## Part 2 — Documents page header/subheader

Edit `src/routes/index.tsx`. Replace the "Your documents" block with:

- H1: "Answers you can trust, sourced from your documents."
- Sub: "DocuFlow only answers from what you upload — every response is cited, and if it's not in your documents, it says so."

Keep the same layout (header → dropzone → error → library). Widen the subheader max-width slightly (`max-w-xl`) so the longer line isn't cramped, and add a bit of vertical breathing room.

Also add a matching `head()` for `/` with a real per-page title/description (currently missing).

## Part 3 — Fresh chat thread per visit + Clear Chat

Edit `src/routes/chat.tsx`:

- Remove the initial `load()` (don't fetch `chat_messages` on mount). Start `messages` as `[]`.
- Generate a `threadId` (crypto.randomUUID) in a `useState` initializer for this visit.
- Persist new user/assistant messages tagged with this `threadId` (see below).
- Render only messages sent during this visit (kept in local state, no DB fetch).
- Add a "Clear chat" button in the chat header area (visible when messages exist). Clicking it: clears local `messages`, resets streaming state, and mints a new `threadId`. No DB delete.

Storage: to tag messages with a thread, add a nullable `thread_id uuid` column to `chat_messages` via migration (backfill left null; no data loss). Inserts include the current `threadId`. This is additive — nothing else in the app reads it today.

## Part 4 — Per-document chat scoping (retrieval-level)

Goal: user picks "All documents" (default) or a specific indexed doc; when a doc is picked, retrieval is filtered to that doc's chunks before RRF + rerank.

### UI (`src/routes/chat.tsx`)

- Add a selector at the top of the chat page, populated from `documents` where `status='indexed'`, plus "All documents" as the first/default option.
- Always show current scope prominently, e.g. "Answering from: hdfc policy doc.PDF" or "Answering from: All documents".
- Changing the selection triggers the same reset as Clear Chat (clear messages, new threadId) so contexts don't blend.
- Pass `documentId` (or omit for "All") in the `ask` request body.

### Edge function (`supabase/functions/rag/index.ts`)

- Accept optional `documentId` from the request body; validate it's a uuid string if present.
- `askStream(question, documentId?)`: pass `documentId` into both RPC calls (new optional arg). Rest of pipeline (RRF, rerank, prompt, streaming, PII, citations) is unchanged — filtering happens at the SQL layer, so downstream stages naturally operate on the filtered set.

### Database (new migration)

Extend both retrieval RPCs to accept an optional `filter_document_id uuid default null`. When non-null, add `AND c.document_id = filter_document_id` to the WHERE clause. When null, behavior is identical to today.

```sql
-- match_document_chunks: add filter_document_id uuid default null
-- WHERE 1 - (c.embedding <=> query_embedding) >= min_similarity
--   AND (filter_document_id IS NULL OR c.document_id = filter_document_id)

-- match_document_chunks_fts: add filter_document_id uuid default null
-- WHERE q.tsq IS NOT NULL AND c.content_tsv @@ q.tsq
--   AND (filter_document_id IS NULL OR c.document_id = filter_document_id)
```

`CREATE OR REPLACE FUNCTION` with the same return shape → no other callers break. `src/integrations/supabase/types.ts` is auto-regenerated and picks up the new optional arg; the app doesn't call these RPCs directly (only the edge function does).

### Deploy

After the migration lands, redeploy the `rag` edge function so it uses the new arg.

## Verification

- Build passes, upload/index/chat still work end-to-end (verify via `curl_edge_functions` with and without `documentId`).
- With a specific doc selected, citations only reference that filename.
- Clear Chat and switching selector both reset the visible thread and mint a new id; DB rows remain.
- Errors from ingest/ask still surface into the existing error banner / stream error event.

## Answers I'll give you at the end

(a) **Yes, edge function changed** — new optional `documentId` handling in `askStream` and the request router. Requires redeploying `rag` (via `supabase--deploy_edge_functions`) after the DB migration adds the new RPC args.

(b) **Plain-English filter logic**: Retrieval runs two searches in parallel — a vector/semantic search and a keyword/full-text search — both against the `document_chunks` table. When the user picks a specific document, we pass that document's id into both searches, and each search adds a single WHERE condition: "only consider chunks whose `document_id` equals the selected one." Everything downstream (fusing the two lists with RRF, reranking with the cross-encoder, building the prompt, citations) then operates on a set that's already restricted to that document, so the model literally cannot see or cite anything from other files. When "All documents" is selected, no filter is applied and retrieval behaves exactly as before.
