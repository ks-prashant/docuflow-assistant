-- =============================================================================
-- Store each document's personal-data spans (names + full addresses) once, at
-- ingestion. Answer-time masking then reads this known list and masks it
-- deterministically in the streamed answer — reliable and with no per-question
-- LLM guess. (Structured PII — phone/email/IDs/PAN/Aadhaar — is still handled by
-- regex at output time and needs no stored list.)
-- =============================================================================

ALTER TABLE public.documents
  ADD COLUMN IF NOT EXISTS pii_spans jsonb NOT NULL DEFAULT '[]'::jsonb;
