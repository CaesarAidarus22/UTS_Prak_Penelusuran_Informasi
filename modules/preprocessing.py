import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# === Inisialisasi Stemmer & Stopwords ===
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Stopwords bahasa Indonesia + Inggris
stop_words = set(stopwords.words('indonesian')).union(ENGLISH_STOP_WORDS)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)
  