from django.apps import AppConfig
import os, faiss, numpy as np


class RagConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rag'
    index_mismatch = False
    index_expected = 0
    index_actual = 0

    def ready(self):  # noqa: D401
        """On startup, compare number of good-dimension chunk vectors in DB vs FAISS index size.

        Sets class attributes (index_mismatch, index_expected, index_actual) used for a UI banner.
        Avoids expensive rebuild; only logs warning when mismatch > 5% or abs diff >= 10.
        """
        from .models import Chunk  # local import
        from .ingest import DIM, INDEX_PATH
        try:
            qs = Chunk.objects.all().only('id','vector')
            good = 0
            for c in qs:
                if len(c.vector)//4 == DIM:
                    good += 1
            expected = good
            actual = 0
            if os.path.exists(INDEX_PATH):
                try:
                    idx = faiss.read_index(INDEX_PATH)
                    actual = idx.ntotal
                except Exception:
                    actual = -1
            RagConfig.index_expected = expected
            RagConfig.index_actual = actual
            RagConfig.index_mismatch = (actual != expected) and (actual < 0 or abs(actual-expected) >= 10 or (expected and abs(actual-expected)/expected > 0.05))
            if RagConfig.index_mismatch:
                print(f"[RAG] WARNING: FAISS index count {actual} != DB good-chunk count {expected}. Run `python manage.py rebuild_faiss`." )
        except Exception as e:  # noqa: BLE001
            print("[RAG] Health check failed:", e)
