from django.db import migrations

def purge_and_rebuild(apps, schema_editor):
    Document = apps.get_model('rag','Document')
    Chunk = apps.get_model('rag','Chunk')
    try:
        bad = Document.objects.get(id=26)
    except Document.DoesNotExist:
        return
    # Delete doc cascades chunks
    bad.delete()
    # Rebuild FAISS index from remaining good chunks
    from rag.ingest import DIM, INDEX_PATH
    import faiss, numpy as np, os
    qs = Chunk.objects.order_by('id')
    good = []
    for c in qs:
        dim = len(c.vector)//4
        if dim == DIM:
            good.append(c)
    if not good:
        # remove index if exists
        if os.path.exists(INDEX_PATH):
            try: os.remove(INDEX_PATH)
            except OSError: pass
        return
    arr = np.vstack([np.frombuffer(c.vector, dtype='float32') for c in good])
    idx = faiss.IndexFlatIP(DIM)
    idx.add(arr)
    tmp = INDEX_PATH + '.tmp'
    faiss.write_index(idx, tmp)
    import os
    os.replace(tmp, INDEX_PATH)

class Migration(migrations.Migration):
    dependencies = [
        ('rag', '0002_reference'),
    ]

    operations = [
        migrations.RunPython(purge_and_rebuild, reverse_code=migrations.RunPython.noop)
    ]
