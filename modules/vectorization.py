from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def vectorize_documents(docs):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs)
    return X, vectorizer
