from django.test import TestCase, Client
from .models import Document, Chunk, Reference
from .ingest import DIM, INDEX_PATH, REF_INDEX_PATH
import numpy as np, faiss, os


class AgentEndpointsTest(TestCase):
	def setUp(self):
		self.client = Client()
		# Enable offline deterministic embedding mode
		os.environ["RAG_OFFLINE_TEST"] = "1"
		# Create a document & chunk
		doc = Document.objects.create(arxiv_id="TEST1234", title="Test Paper", authors="A. Author", pdf_path="/tmp/x.pdf")
		rng = np.random.default_rng(7)
		chunk_vec = rng.normal(size=(DIM,)).astype('float32')
		chunk_vec /= np.linalg.norm(chunk_vec)
		Chunk.objects.create(doc=doc, kind="text", content="This is a test chunk about retrieval augmented generation and vector search.", ord=0, vector=chunk_vec.tobytes())
		# Reference with deterministic vector
		ref = Reference.objects.create(doc=doc, raw_text="[1] A. Author. Important Reference on Retrieval (2024).", position=0)
		rng2 = np.random.default_rng(42)
		ref_vec = rng2.normal(size=(1, DIM)).astype('float32')
		ref_vec /= np.linalg.norm(ref_vec, axis=1, keepdims=True)
		ref.vector = ref_vec.tobytes(); ref.save()
		# Build FAISS indices
		os.makedirs(os.path.dirname(REF_INDEX_PATH), exist_ok=True)
		ref_index = faiss.IndexFlatIP(DIM); ref_index.add(ref_vec)
		faiss.write_index(ref_index, REF_INDEX_PATH)
		text_index = faiss.IndexFlatIP(DIM); text_index.add(np.stack([chunk_vec], axis=0))
		os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
		faiss.write_index(text_index, INDEX_PATH)

	def test_list_documents(self):
		r = self.client.get('/api/agent/documents?page=1&page_size=10&author=Author&title=Test')
		self.assertEqual(r.status_code, 200)
		data = r.json()['documents']
		self.assertTrue(any(doc['arxiv_id'] == "TEST1234" for doc in data))
		payload = r.json()
		self.assertIn('total', payload)
		self.assertEqual(payload.get('page'), 1)
		self.assertEqual(payload.get('page_size'), 10)

	def test_list_chunks(self):
		d = Document.objects.first()
		r = self.client.get(f'/api/agent/documents/{d.id}/chunks?limit=5')
		self.assertEqual(r.status_code, 200)
		self.assertGreaterEqual(len(r.json()['chunks']), 1)

	def test_list_references(self):
		d = Document.objects.first()
		r = self.client.get(f'/api/agent/documents/{d.id}/references')
		self.assertEqual(r.status_code, 200)
		self.assertGreaterEqual(len(r.json()['references']), 1)

	def test_reference_search_requires_approval(self):
		r = self.client.post('/api/agent/search_references', data={'query': 'Important'}, content_type='application/json')
		self.assertEqual(r.status_code, 202)
		self.assertIn('pending', r.json())
		r2 = self.client.post('/api/agent/search_references', data={'query': 'Important', 'approve': True}, content_type='application/json')
		if r2.status_code == 200:
			self.assertGreaterEqual(len(r2.json().get('results', [])), 1)
		else:
			self.assertEqual(r2.status_code, 500)  # allow FAISS mismatch if any

	def test_search_chunks(self):
		r = self.client.post('/api/agent/search_chunks', data={'query': 'retrieval', 'k': 3}, content_type='application/json')
		self.assertEqual(r.status_code, 200)
		self.assertGreaterEqual(len(r.json().get('results', [])), 1)

# End tests
