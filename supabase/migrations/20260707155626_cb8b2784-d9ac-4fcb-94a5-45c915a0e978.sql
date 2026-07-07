
CREATE POLICY "Public upload documents bucket" ON storage.objects FOR INSERT WITH CHECK (bucket_id = 'documents');
CREATE POLICY "Public read documents bucket" ON storage.objects FOR SELECT USING (bucket_id = 'documents');
CREATE POLICY "Public delete documents bucket" ON storage.objects FOR DELETE USING (bucket_id = 'documents');
