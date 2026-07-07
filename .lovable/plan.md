## Problem

After picking/dropping a PDF on the Documents page, nothing appears in the list. Confirmed:

- `documents` table is empty (0 rows).
- `rag` edge function has no invocation logs.
- The initial `GET /documents` request works (anon reads are fine).
- `OPENAI_API_KEY` secret is present.

So the failure is happening **before** the DB insert — during `supabase.storage.from("documents").upload(...)` or earlier — and the error is being swallowed into the small red banner (or not shown at all).

## Plan

1. **Instrument the upload path** in `src/routes/index.tsx` so we can see exactly where it fails:
   - `console.log` at start of `handleFiles`, after storage upload, after DB insert, after `functions.invoke("rag", ...)`.
   - Surface any storage/insert error into the on-page error banner instead of only `throw`ing.
   - Add a visible "processing" row immediately after insert so the user sees progress even if ingest is slow.

2. **Verify the `documents` storage bucket policies** allow anon `INSERT` (uploads from the browser use the anon key). If missing, add a policy so anon can insert/select/delete objects in the `documents` bucket. Public read is not required — signed URLs / edge function download via service role handle that.

3. **Reproduce in the sandbox** with Playwright: open `/`, upload a small test PDF, capture console + network, and confirm the storage POST reaches Supabase and returns 200. If it fails with status 0 in the preview iframe (known Lovable preview fetch-proxy issue with some Supabase endpoints), test on the published URL to confirm.

4. **Wire ingest feedback**: once upload works, poll `documents` row status (`uploaded → processing → indexed`) so the UI reflects the RAG pipeline progress instead of looking frozen.

5. **Report back** with the exact failing step and the applied fix. No changes to the RAG edge function itself unless step 3 shows an issue there.

## Technical notes

- Storage bucket is private; browser upload with anon key requires an explicit `storage.objects` INSERT policy scoped to `bucket_id = 'documents'`.
- The edge function is already configured with `verify_jwt = false`, so `supabase.functions.invoke("rag", ...)` from the browser will reach it once a document row exists.
- Not touching `client.ts`, `.env`, or `supabase/config.toml`.
