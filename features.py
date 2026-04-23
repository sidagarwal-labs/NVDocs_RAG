"""Feature computation for (question, chunk) pairs."""

import gc
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_features(triples_df, bm25_index, tfidf_index):
    """Compute full feature matrix for all (question, chunk) pairs in triples_df."""
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize

    n_pairs = len(triples_df)
    unique_questions = triples_df['question'].unique()
    print(f"Computing features for {n_pairs:,} pairs ({len(unique_questions):,} unique questions)...", flush=True)

    q_to_idx = {q: i for i, q in enumerate(unique_questions)}
    q_indices = triples_df['question'].map(q_to_idx).values
    c_indices = triples_df['chunk_id'].map(bm25_index.chunk_id_to_idx).values

    #BM25 scoring: element-wise sparse multiply per batch of pairs
    print("BM25 scoring (sparse pair-wise, batched)...", flush=True)
    q_vecs_bm25 = bm25_index.vectorizer.transform(unique_questions)
    q_idf = q_vecs_bm25.multiply(bm25_index.idf).tocsr()
    del q_vecs_bm25; gc.collect()
    adj_tf = sp.csr_matrix(bm25_index.adjusted_tf)
    BATCH = 100000
    bm25_scores = np.empty(n_pairs, dtype=np.float32)
    for start in range(0, n_pairs, BATCH):
        end = min(start + BATCH, n_pairs)
        q_batch = q_idf[q_indices[start:end]]
        c_batch = adj_tf[c_indices[start:end]]
        bm25_scores[start:end] = np.array(q_batch.multiply(c_batch).sum(axis=1)).ravel().astype(np.float32)
        del q_batch, c_batch
        if start % 500000 == 0:
            print(f"  BM25 {start:,}/{n_pairs:,}...", flush=True)
    del q_idf, adj_tf; gc.collect()
    print(f"  BM25 done: {bm25_scores.shape}", flush=True)

    #TF-IDF cosine similarity: element-wise sparse multiply per batch
    print("TF-IDF scoring (sparse pair-wise, batched)...", flush=True)
    q_vecs_tfidf = tfidf_index.vectorizer.transform(unique_questions)
    q_norm = normalize(q_vecs_tfidf, norm='l2').tocsr()
    del q_vecs_tfidf; gc.collect()
    c_norm = normalize(sp.csr_matrix(tfidf_index.tfidf_matrix), norm='l2').tocsr()
    cosine_sims = np.empty(n_pairs, dtype=np.float32)
    for start in range(0, n_pairs, BATCH):
        end = min(start + BATCH, n_pairs)
        q_batch = q_norm[q_indices[start:end]]
        c_batch = c_norm[c_indices[start:end]]
        cosine_sims[start:end] = np.array(q_batch.multiply(c_batch).sum(axis=1)).ravel().astype(np.float32)
        del q_batch, c_batch
        if start % 500000 == 0:
            print(f"  TF-IDF {start:,}/{n_pairs:,}...", flush=True)
    del q_norm, c_norm; gc.collect()
    print(f"  TF-IDF done: {cosine_sims.shape}", flush=True)

    #text overlap (cached question tokens, vectorized chunk tokens)
    print("Computing text overlap features...", flush=True)
    overlap_ratios = np.zeros(n_pairs, dtype=np.float32)
    overlap_counts = np.zeros(n_pairs, dtype=np.int32)
    q_lengths = np.zeros(n_pairs, dtype=np.int32)

    _q_token_cache = {}
    for q in unique_questions:
        words = q.lower().split()
        _q_token_cache[q] = (set(words), len(words))

    _questions = triples_df['question'].values
    _chunk_texts = triples_df['chunk_text'].values
    for i in range(n_pairs):
        q_set, q_len = _q_token_cache[_questions[i]]
        c_tokens = set(_chunk_texts[i].lower().split())
        if q_set:
            n_overlap = len(q_set & c_tokens)
            overlap_ratios[i] = n_overlap / len(q_set)
            overlap_counts[i] = n_overlap
        q_lengths[i] = q_len
        if i % 200000 == 0:
            print(f"  overlap {i:,}/{n_pairs:,}...", flush=True)
    del _q_token_cache; gc.collect()
    print("  overlap done", flush=True)

    features_df = pd.DataFrame({
        'bm25_score': bm25_scores,
        'cosine_similarity': cosine_sims,
        'token_overlap_ratio': overlap_ratios,
        'token_overlap_count': overlap_counts,
        'question_length': q_lengths,
    })
    del bm25_scores, cosine_sims, overlap_ratios, overlap_counts, q_lengths; gc.collect()

    #per-question rank features (vectorized groupby)
    print("Computing rank features...", flush=True)
    _df = pd.DataFrame({
        'question': triples_df['question'].values,
        'bm25': features_df['bm25_score'].values,
        'tfidf': features_df['cosine_similarity'].values,
    })
    features_df['bm25_rank'] = _df.groupby('question', sort=False)['bm25'].rank(ascending=False, method='first').astype(np.int32).values
    features_df['tfidf_rank'] = _df.groupby('question', sort=False)['tfidf'].rank(ascending=False, method='first').astype(np.int32).values
    features_df['bm25_reciprocal_rank'] = (1.0 / features_df['bm25_rank']).astype(np.float32)
    features_df['rank_diff'] = (features_df['bm25_rank'] - features_df['tfidf_rank']).astype(np.int32)
    del _df; gc.collect()
    print("  rank features done", flush=True)

    #one-hot encode categorical columns
    query_type_dummies = pd.get_dummies(triples_df['query_type'], prefix='qtype')
    reasoning_type_dummies = pd.get_dummies(triples_df['reasoning_type'], prefix='rtype')

    result = pd.concat([
        triples_df.reset_index(drop=True),
        features_df.reset_index(drop=True),
        query_type_dummies.reset_index(drop=True),
        reasoning_type_dummies.reset_index(drop=True),
    ], axis=1)

    return result


def get_feature_columns(df):
    """Return feature column names (excludes metadata/target columns)."""
    exclude = {
        'question', 'chunk_id', 'chunk_text', 'relevance',
        'query_type', 'reasoning_type', 'doc_id', 'chunk_ids', 'original_chunk_id'
    }
    return [c for c in df.columns if c not in exclude]
