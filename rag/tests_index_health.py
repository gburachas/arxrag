from django.test import TestCase
from django.core.management import call_command
from rag.models import Chunk, Document
from rag.ingest import DIM, INDEX_PATH
import os, faiss, numpy as np

class IndexHealthTest(TestCase):
    def setUp(self):
        # Create one doc with two good chunks
        d = Document.objects.create(arxiv_id='x1', title='T', authors='', pdf_path='p')
        import numpy as np
        v1 = np.ones(DIM, dtype='float32'); v1 /= np.linalg.norm(v1)
        v2 = np.zeros(DIM, dtype='float32'); v2[0]=1
        Chunk.objects.create(doc=d, kind='text', content='alpha beta gamma', ord=0, vector=v1.tobytes())
        Chunk.objects.create(doc=d, kind='text', content='delta epsilon zeta', ord=1, vector=v2.tobytes())
        # Build mismatched index containing only first vector
        idx = faiss.IndexFlatIP(DIM)
        idx.add(v1.reshape(1,-1))
        faiss.write_index(idx, INDEX_PATH)

    def test_rebuild_fixes_mismatch(self):
        idx = faiss.read_index(INDEX_PATH)
        self.assertEqual(idx.ntotal, 1)
        # Rebuild
        call_command('rebuild_faiss')
        idx2 = faiss.read_index(INDEX_PATH)
        self.assertEqual(idx2.ntotal, 2)
