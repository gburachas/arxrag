from django.core.management.base import BaseCommand
from rag.models import Chunk
from rag.ingest import DIM, INDEX_PATH, save_index
import faiss, numpy as np, os

class Command(BaseCommand):
    help = "Rebuild the FAISS chunk index from all Chunk vectors in the database (skips malformed dimensions)."

    def handle(self, *args, **options):
        qs = Chunk.objects.order_by('id')
        good = []
        bad = 0
        for c in qs:
            dim = len(c.vector) // 4
            if dim != DIM:
                bad += 1
                continue
            good.append(c)
        self.stdout.write(f"Good chunks: {len(good)}  (skipped malformed: {bad})")
        if not good:
            self.stdout.write(self.style.WARNING("No good chunks to index."))
            return
        arr = np.vstack([np.frombuffer(c.vector, dtype='float32') for c in good])
        idx = faiss.IndexFlatIP(DIM)
        idx.add(arr)
        tmp_path = INDEX_PATH + '.tmp'
        faiss.write_index(idx, tmp_path)
        os.replace(tmp_path, INDEX_PATH)
        self.stdout.write(self.style.SUCCESS(f"Rebuilt index with {idx.ntotal} vectors -> {INDEX_PATH}"))
