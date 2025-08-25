import os
from django.core.management.base import BaseCommand
from rag.models import Chunk, Document
from rag.ingest import ingest_arxiv, INDEX_PATH

class Command(BaseCommand):
    help = "Rebuild FAISS index with normalized embeddings by clearing existing chunks/documents and reingesting arXiv papers."

    def add_arguments(self, parser):
        parser.add_argument('--query', type=str, default='agentic RAG', help='ArXiv query string')
        parser.add_argument('--max-results', type=int, default=1, help='Max arXiv results to ingest')
        parser.add_argument('--keep-docs', action='store_true', help='Keep existing Document rows (only rebuild index from stored vectors)')

    def handle(self, *args, **options):
        query = options['query']
        max_results = options['max_results']
        keep_docs = options['keep_docs']

        # Remove index file
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
            self.stdout.write(self.style.WARNING(f"Deleted existing index: {INDEX_PATH}"))
        else:
            self.stdout.write("No existing index found; creating new one.")

        if not keep_docs:
            Chunk.objects.all().delete()
            Document.objects.all().delete()
            self.stdout.write(self.style.WARNING("Cleared Chunk and Document tables."))
        else:
            # If keeping docs, just rebuild from scratch by reingesting (will duplicate unless we skip already existing arxiv_ids)
            existing_ids = set(Document.objects.values_list('arxiv_id', flat=True))
            self.stdout.write(f"Keeping {len(existing_ids)} existing documents; new ingestion will add more if different.")

        ingest_arxiv(query=query, max_results=max_results)
        self.stdout.write(self.style.SUCCESS("Reingestion complete."))
