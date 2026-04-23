"""BM25 and TF-IDF retrieval indexes."""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BM25Index:
    """BM25 index using CountVectorizer + sparse matrices."""

    def __init__(self, chunk_texts, chunk_ids, k1=1.5, b=0.75):
        self.chunk_ids = chunk_ids
        self.chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        self.k1 = k1
        self.b = b

        self.vectorizer = CountVectorizer()
        tf_raw = self.vectorizer.fit_transform(chunk_texts)
        n_docs = tf_raw.shape[0]

        doc_lens = tf_raw.sum(axis=1).A1
        self.avgdl = doc_lens.mean()

        df = (tf_raw > 0).sum(axis=0).A1
        self.idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

        #BM25-adjusted TF
        tf_float = tf_raw.astype(np.float64)
        len_norm = k1 * (1.0 - b + b * doc_lens / self.avgdl)
        self.adjusted_tf = tf_float.copy().tocsr()
        for i in range(n_docs):
            start, end = self.adjusted_tf.indptr[i], self.adjusted_tf.indptr[i + 1]
            data = self.adjusted_tf.data[start:end]
            data[:] = data * (k1 + 1.0) / (data + len_norm[i])

        print(f"BM25 index built: {n_docs} docs, {tf_raw.shape[1]} terms (sparse matrix)")

    def score_all(self, query):
        """Score all chunks against a query. Returns score array."""
        q_vec = self.vectorizer.transform([query])
        q_idf = q_vec.multiply(self.idf)
        scores = q_idf.dot(self.adjusted_tf.T).toarray().flatten()
        return scores

    def score(self, query):
        """Score all chunks. Returns {chunk_id: score} dict."""
        scores = self.score_all(query)
        return dict(zip(self.chunk_ids, scores.tolist()))

    def top_k(self, query, k=10):
        """Return top-k (chunk_id, score) pairs by BM25."""
        scores = self.score_all(query)
        top_idxs = np.argsort(scores)[-k:][::-1]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idxs]


class TfidfIndex:
    """TF-IDF retrieval index using sklearn."""

    def __init__(self, max_features=50000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            sublinear_tf=True,
        )
        self.chunk_ids = None
        self.chunk_id_to_idx = None
        self.tfidf_matrix = None

    def build_index(self, chunk_texts, chunk_ids):
        """Fit and transform TF-IDF on all chunks."""
        self.chunk_ids = chunk_ids
        self.chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        print(f"Building TF-IDF index over {len(chunk_texts)} chunks...")
        self.tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
        print(f"TF-IDF index built: {self.tfidf_matrix.shape[0]} docs, {self.tfidf_matrix.shape[1]} features")

    def encode_query(self, query):
        """Transform a query into TF-IDF vector."""
        return self.vectorizer.transform([query])

    def score(self, query):
        """Score all chunks by cosine similarity. Returns {chunk_id: score}."""
        query_vec = self.encode_query(query)
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return dict(zip(self.chunk_ids, scores.tolist()))

    def score_all(self, query):
        """Score all chunks by cosine similarity. Returns score array."""
        query_vec = self.encode_query(query)
        return cosine_similarity(query_vec, self.tfidf_matrix).flatten()

    def top_k(self, query, k=10):
        """Return top-k (chunk_id, score) pairs by TF-IDF cosine similarity."""
        scores = self.score_all(query)
        top_idxs = np.argsort(scores)[-k:][::-1]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idxs]
