# modules/search.py
from sklearn.metrics.pairwise import cosine_similarity

def search_query(query, docs, vectorizer):
    qv = vectorizer.transform([query])
    dv = vectorizer.transform(docs)
    scores = cosine_similarity(qv, dv).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked  # JANGAN dipotong di sini
