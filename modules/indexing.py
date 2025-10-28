from whoosh.fields import Schema, TEXT, ID
from whoosh import index
import os

def create_index(index_dir, documents):
    schema = Schema(title=TEXT(stored=True), content=TEXT)
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    idx = index.create_in(index_dir, schema)
    writer = idx.writer()
    for i, doc in enumerate(documents):
        writer.add_document(title=f"Doc_{i}", content=doc)
    writer.commit()
    return idx
