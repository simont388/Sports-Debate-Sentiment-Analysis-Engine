import json
import os
import pickle
import re
from collections import Counter, defaultdict
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD

# Build LSI (Latent Semantic Indexing) models from corpus using TF-IDF + Truncated SVD
# Produces two models: paragraph-level (k=150) for search, document-level (k=50) for debate filtering

CORPUS_FILE = "data/corpus/corpus.json"
MODELS_DIR = "models"
PARAGRAPH_MODEL_OUT = os.path.join(MODELS_DIR, "lsi_paragraphs.pkl")
DOCUMENT_MODEL_OUT = os.path.join(MODELS_DIR, "lsi_documents.pkl")
STOPWORDS_FILE = "config/common_words"
K_PARAGRAPHS = 150
K_DOCUMENTS = 50

os.makedirs(MODELS_DIR, exist_ok=True)

stemmer = SnowballStemmer('english')

def read_stopwords(filepath):
    if not os.path.exists(filepath):
        print(F"WARNING: stopwords file not found at {filepath}, defaulting to minimal list")
        return {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "of", "in", "on", "at", "to", "for", "with", "by", "from",
            "and", "or", "but", "not", "this", "that", "it", "he", "she",
            "they", "we", "you", "his", "her", "their", "its", "our", "my",
            "as", "if", "so", "up", "out", "about", "into", "then", "than",
            "also", "just", "more", "no", "all", "one", "who", "which"
        }
    with open(filepath) as f:
        return set(x.strip() for x in f.readlines())
    
def tokenize_and_clean(text, stopwords):
    # Lowercase, remove stopwords/punctuation, stem remaining tokens
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if re.match(r'^[a-z]+$', t)]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def compute_doc_freqs(tokenized_docs):
    # Count how many documents each term appears in (for IDF calculation)
    freq = Counter()
    for tokens in tokenized_docs:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            freq[token] += 1
    return freq

def compute_tfidf(tokens, doc_freqs, N):
    # Compute TF-IDF weights for a single document's tokens
    tf_vec = Counter(tokens)
    tfidf_vec = {}
    for word, tf in tf_vec.items():
        df = doc_freqs.get(word, 1)
        idf = np.log(N/df)
        tfidf_vec[word] = tf*idf
    return tfidf_vec

def build_svd(tfidf_vectors, k):
    # Build term-document matrix and run TruncatedSVD to get latent space
    all_words = set()
    for vec in tfidf_vectors:
        all_words.update(vec.keys())
    unique_words = sorted(list(all_words))
    word_map = {word:i for i, word in enumerate(unique_words)}
    num_terms = len(unique_words)
    num_docs = len(tfidf_vectors)
    print(f"Building term-document matrix: {num_terms} terms x {num_docs} documents")

    A = np.zeros((num_terms, num_docs))
    for j, vec in enumerate(tfidf_vectors):
        for word, weight in vec.items():
            if word in word_map:
                A[word_map[word], j] = weight
    print(f"Running truncated SVD with k={k}")
    svd = TruncatedSVD(n_components=k, random_state=42)
    term_reps = svd.fit_transform(A)
    Vtk = svd.components_
    Vtk_scaled = Vtk * svd.singular_values_[:, np.newaxis]
    Sk = np.diag(svd.singular_values_)
    Sk_inv = np.linalg.inv(Sk)
    Uk = term_reps @ Sk_inv

    return Uk, Sk_inv, Vtk_scaled, word_map

def query_projection(query_vec, word_map, Uk, Sk_inv):
    # Project a query TF-IDF vector into the same latent space as documents
    q_vec = np.zeros(len(word_map))
    for word, weight in query_vec.items():
        if word in word_map:
            q_vec[word_map[word]] = weight
    
    q_projected = q_vec @ Uk @ Sk_inv
    return q_projected

def search_lsi(q_projected, Vtk):
    # Compute cosine similarity between projected query and all documents
    dots = q_projected @ Vtk
    q_norm = np.linalg.norm(q_projected)
    doc_norms = np.linalg.norm(Vtk, axis=0)
    scores = dots / (doc_norms * q_norm + 1e-10)

    ranked = sorted(
        enumerate(scores.tolist()),
        key = lambda x: x[1],
        reverse=True
    )
    return ranked

def group_chunks_to_docs(paragaphs):
    # Group paragraph chunks into logical documents by debate + source
    doc_groups = defaultdict(list)
    for chunk in paragaphs:
        debates = chunk.get("debates", ["unknown"])
        source = chunk.get("source", "unknown")
        for debate in debates:
            key = f"{debate}::{source}"
            doc_groups[key].append(chunk["text"])
    
    grouped_docs = []
    group_keys = []
    for key, texts in doc_groups.items():
        grouped_docs.append(" ".join(texts))
        group_keys.append(key)
    return grouped_docs, group_keys

def build_lsi():
    print("=== Building LSI Matrices ===\n")

    print(f"Loading corpus...")
    with open(CORPUS_FILE, "r", encoding='utf-8') as f:
        paragraphs = json.load(f)
    print(f"Loaded {len(paragraphs)} paragraph chunks\n")

    stopwords = read_stopwords(STOPWORDS_FILE)

    print(f"--- Building paragraph level matrix ---")
    print(f"Tokenizing paragraphs...")

    tokenized_paragraphs = [tokenize_and_clean(p["text"], stopwords) for p in paragraphs]
    valid_indices = [i for i, t in enumerate(tokenized_paragraphs) if len(t) > 0]
    valid_paragraphs = [paragraphs[i] for i in valid_indices]
    valid_tokens = [tokenized_paragraphs[i] for i in valid_indices]
    print(f"Valid paragraphs after tokenization: {len(valid_paragraphs)}, "
          f"{len(paragraphs) - len(valid_paragraphs)} filtered out")
    
    para_doc_freqs = compute_doc_freqs(valid_tokens)
    N_para = len(valid_tokens)

    print(f"Computing TF-IDF vectors...")
    para_tfidf_vecs = [
        compute_tfidf(tokens, para_doc_freqs, N_para) for tokens in valid_tokens
    ]

    print(f"Building SVD (k = {K_PARAGRAPHS})...")
    para_Uk, para_Sk_inv, para_Vtk, para_word_map = build_svd(para_tfidf_vecs, K_PARAGRAPHS)
    para_model = {
        "Uk" : para_Uk,
        "Sk_inv" : para_Sk_inv,
        "Vtk" : para_Vtk,
        "word_map" : para_word_map,
        "doc_freqs" : para_doc_freqs,
        "N" : N_para,
        "paragraphs" : valid_paragraphs,
        "k" : K_PARAGRAPHS
    }

    with open(PARAGRAPH_MODEL_OUT, "wb") as f:
        pickle.dump(para_model, f)
    print(f"Saved paragraph level model to: {PARAGRAPH_MODEL_OUT}\n")

    print("--- Building document level matrix ---")

    grouped_texts, group_keys = group_chunks_to_docs(paragraphs)
    print(f"Created {len(grouped_texts)} logical documents from {len(paragraphs)} chunks")

    tokenized_docs = [
        tokenize_and_clean(text, stopwords) for text in grouped_texts
    ]

    valid_doc_indices = [i for i, t in enumerate(tokenized_docs) if len(t) > 0]
    valid_doc_texts = [grouped_texts[i] for i in valid_doc_indices]
    valid_doc_keys = [group_keys[i] for i in valid_doc_indices]
    valid_doc_tokens = [tokenized_docs[i] for i in valid_doc_indices]

    doc_doc_freqs = compute_doc_freqs(valid_doc_tokens)
    N_docs = len(valid_doc_tokens)

    print("Computing TF-IDF vectors for documents...")
    doc_tfidf_vecs = [
        compute_tfidf(tokens, doc_doc_freqs, N_docs) for tokens in valid_doc_tokens
    ]

    print(f"Building SVD (k = {K_DOCUMENTS})")
    doc_Uk, doc_Sk_inv, doc_Vtk, doc_word_map = build_svd(doc_tfidf_vecs, K_DOCUMENTS)
    doc_model = {
        "Uk" : doc_Uk,
        "Sk_inv" : doc_Sk_inv,
        "Vtk" : doc_Vtk,
        "word_map" : doc_word_map,
        "doc_freqs" : doc_doc_freqs,
        "N" : N_docs,
        "doc_keys" : valid_doc_keys,
        "k" : K_DOCUMENTS
    }

    with open(DOCUMENT_MODEL_OUT, "wb") as f:
        pickle.dump(doc_model, f)
    print(f"Saved document level model to: {DOCUMENT_MODEL_OUT}\n")

if __name__ == "__main__":
    build_lsi()