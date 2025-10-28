from sklearn.metrics.pairwise import cosine_similarity

def search_query(query, docs, vectorizer):
    query_vec = vectorizer.transform([query])
    doc_vectors = vectorizer.transform(docs)
    scores = cosine_similarity(query_vec, doc_vectors).flatten()
    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
    return ranked[:5]  # top 5 hasil
