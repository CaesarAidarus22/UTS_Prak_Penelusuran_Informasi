from whoosh.fields import Schema, TEXT, ID
from whoosh import index
import os, shutil

def create_index(index_dir, documents):
    """
    Membuat index Whoosh dari daftar dokumen (string).
    Folder index lama akan dihapus agar tidak bentrok.
    """
    schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))

    # hapus index lama biar bersih
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.mkdir(index_dir)

    # buat index baru
    idx = index.create_in(index_dir, schema)
    writer = idx.writer()

    for i, doc in enumerate(documents):
        writer.add_document(title=f"Doc_{i}", content=doc)

    writer.commit()
    print(f"Index Whoosh berhasil dibuat di folder '{index_dir}'.")
    return idx
