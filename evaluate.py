"""Ranking evaluation metrics and candidate preparation."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def dcg_at_k(relevances, k):
    """Compute DCG@k."""
    relevances = np.array(relevances[:k])
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(len(relevances)) + 2)
    return np.sum(relevances / discounts)


def ndcg_at_k(relevances, k):
    """Compute NDCG@k."""
    dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(relevances, k, total_relevant):
    """Compute Recall@k."""
    if total_relevant == 0:
        return 0.0
    return sum(relevances[:k]) / total_relevant


def reciprocal_rank(relevances):
    """Compute Reciprocal Rank."""
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(relevances):
    """Compute Average Precision."""
    if sum(relevances) == 0:
        return 0.0
    precisions = []
    relevant_count = 0
    for i, rel in enumerate(relevances):
        if rel > 0:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    return sum(precisions) / sum(relevances)


def evaluate_ranking(df, score_col, k_values=[5, 10, 20]):
    """Evaluate ranking model, returns dict of metric_name -> value."""
    results = {f'NDCG@{k}': [] for k in k_values}
    results.update({f'Recall@{k}': [] for k in k_values})
    results['MRR'] = []
    results['MAP'] = []

    for question, group in df.groupby('question'):
        sorted_group = group.sort_values(score_col, ascending=False)
        relevances = sorted_group['relevance'].tolist()
        total_relevant = sum(relevances)

        for k in k_values:
            results[f'NDCG@{k}'].append(ndcg_at_k(relevances, k))
            results[f'Recall@{k}'].append(recall_at_k(relevances, k, total_relevant))

        results['MRR'].append(reciprocal_rank(relevances))
        results['MAP'].append(average_precision(relevances))

    return {metric: np.mean(values) for metric, values in results.items()}


def prepare_candidates(qa_list, bm25_index, tfidf_index, chunk_lookup,
                       feature_cols, candidate_k=100, batch_size=500,
                       inject_golden=False):
    """Retrieve BM25∪TF-IDF top-k candidates and compute features per question.

    Returns list of dicts with candidate_ids, features, golden_ids, metadata.
    Uses dense×sparse batched matmul to avoid scipy OOM and per-question overhead.

    If inject_golden=True, golden chunks missing from the retrieval pool are
    appended to the candidate list (with their actual BM25/TF-IDF scores).
    This ensures every training question has at least one positive label.
    Only use inject_golden=True for training data — never for evaluation.
    """
    from tqdm import tqdm
    import gc

    questions = [qa['question'] for qa in qa_list]
    unique_questions = list(dict.fromkeys(questions))
    q_to_idx = {q: i for i, q in enumerate(unique_questions)}
    n_unique = len(unique_questions)

    # pre-collect golden chunk_ids per unique question for injection
    golden_by_qi = {}
    if inject_golden:
        for qa in qa_list:
            qi = q_to_idx[qa['question']]
            if qi not in golden_by_qi:
                golden_by_qi[qi] = set()
            golden_by_qi[qi].update(qa['chunk_ids'])

    print(f"  Encoding {n_unique} unique questions...")
    bm25_q_idf = bm25_index.vectorizer.transform(unique_questions).multiply(bm25_index.idf).tocsr()
    tfidf_q_vecs = tfidf_index.vectorizer.transform(unique_questions)

    bm25_id_to_idx = bm25_index.chunk_id_to_idx
    tfidf_id_to_idx = tfidf_index.chunk_id_to_idx

    #pre-transpose once
    bm25_adj_T = bm25_index.adjusted_tf.T.tocsc()
    tfidf_mat_T = tfidf_index.tfidf_matrix.T.tocsc()

    #batched scoring: convert query batch to dense, then dense×sparse → dense result
    #avoids scipy sparse×sparse nnz over-estimation (which caused 72 GiB OOM)
    print(f"  Scoring {n_unique} questions in batches of {batch_size}...")
    q_data: dict[int, tuple] = {}

    for batch_start in range(0, n_unique, batch_size):
        batch_end = min(batch_start + batch_size, n_unique)

        #dense query batch × sparse doc matrix → dense scores
        bm25_batch = np.asarray(bm25_q_idf[batch_start:batch_end].toarray() @ bm25_adj_T)
        tfidf_batch = np.asarray(tfidf_q_vecs[batch_start:batch_end].toarray() @ tfidf_mat_T)

        for local_i in range(batch_end - batch_start):
            qi = batch_start + local_i
            bm25_scores = bm25_batch[local_i].ravel()
            tfidf_scores = tfidf_batch[local_i].ravel()

            bm25_topk_idx = np.argpartition(bm25_scores, -candidate_k)[-candidate_k:]
            bm25_topk = bm25_topk_idx[np.argsort(bm25_scores[bm25_topk_idx])[::-1]]
            tfidf_topk_idx = np.argpartition(tfidf_scores, -candidate_k)[-candidate_k:]
            tfidf_topk = tfidf_topk_idx[np.argsort(tfidf_scores[tfidf_topk_idx])[::-1]]

            seen = set()
            candidates = []
            for idx in list(bm25_topk) + list(tfidf_topk):
                cid = bm25_index.chunk_ids[idx]
                if cid not in seen:
                    seen.add(cid)
                    candidates.append(cid)

            # inject missing golden chunks so the model always has positives
            if inject_golden and qi in golden_by_qi:
                for gid in golden_by_qi[qi]:
                    if gid not in seen and gid in bm25_id_to_idx:
                        seen.add(gid)
                        candidates.append(gid)

            bm25_dict = {cid: float(bm25_scores[bm25_id_to_idx[cid]]) for cid in candidates}
            tfidf_dict = {cid: float(tfidf_scores[tfidf_id_to_idx[cid]]) for cid in candidates}
            q_data[qi] = (candidates, bm25_dict, tfidf_dict)

        del bm25_batch, tfidf_batch
        gc.collect()

        if (batch_start // batch_size) % 10 == 0:
            print(f"    {batch_end}/{n_unique} questions scored")

    #build feature vectors per QA pair
    precomputed = []
    for qa in tqdm(qa_list, desc="Building features"):
        question = qa['question']
        golden_ids = set(qa['chunk_ids'])
        qi = q_to_idx[question]

        candidates, bm25_dict, tfidf_dict = q_data[qi]

        # compute rank features (1-based, lower = better)
        bm25_sorted = sorted(candidates, key=lambda cid: bm25_dict[cid], reverse=True)
        tfidf_sorted = sorted(candidates, key=lambda cid: tfidf_dict[cid], reverse=True)
        bm25_rank_map = {cid: rank + 1 for rank, cid in enumerate(bm25_sorted)}
        tfidf_rank_map = {cid: rank + 1 for rank, cid in enumerate(tfidf_sorted)}

        q_tokens = set(question.lower().split())
        q_len = len(question.split())
        rows = []
        for chunk_id in candidates:
            chunk = chunk_lookup[chunk_id]
            c_tokens = set(chunk['chunk_text'].lower().split())
            overlap_count = len(q_tokens & c_tokens)
            b_rank = bm25_rank_map[chunk_id]
            t_rank = tfidf_rank_map[chunk_id]

            rows.append({
                'bm25_score': bm25_dict[chunk_id],
                'cosine_similarity': tfidf_dict[chunk_id],
                'token_overlap_ratio': overlap_count / len(q_tokens) if q_tokens else 0.0,
                'token_overlap_count': overlap_count,
                'question_length': q_len,
                'chunk_word_count': chunk['word_count'],
                'chunk_sentence_count': chunk['sentence_count'],
                'question_complexity': qa['question_complexity'],
                'hop_count': qa['hop_count'],
                'bm25_rank': b_rank,
                'tfidf_rank': t_rank,
                'bm25_reciprocal_rank': 1.0 / b_rank,
                'rank_diff': b_rank - t_rank,
            })

        feat_df = pd.DataFrame(rows)
        for col in feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0

        bm25_arr = np.array([bm25_dict[cid] for cid in candidates])
        tfidf_arr = np.array([tfidf_dict[cid] for cid in candidates])

        precomputed.append({
            'candidate_ids': candidates,
            'features': feat_df[feature_cols].values,
            'bm25_scores': bm25_arr,
            'tfidf_scores': tfidf_arr,
            'golden_ids': golden_ids,
            'metadata': {
                'question': question,
                'query_type': qa['query_type'],
                'reasoning_type': qa['reasoning_type'],
                'n_candidates': len(candidates),
                'n_golden': len(golden_ids),
                'golden_in_candidates': len(golden_ids & set(candidates)),
            },
        })

    return precomputed


def evaluate_from_candidates(precomputed, model, k_values=[5, 10, 20], show_progress=True):
    """Score candidates with a model and compute ranking metrics.

    model=None -> BM25 baseline, model='tfidf' -> TF-IDF baseline.
    """
    from tqdm.auto import tqdm

    results = {f'NDCG@{k}': [] for k in k_values}
    results.update({f'Recall@{k}': [] for k in k_values})
    results['MRR'] = []
    results['MAP'] = []
    meta_rows = []

    iterator = tqdm(precomputed, desc="Evaluating", disable=not show_progress)

    #batch predictions for speed
    batched_scores = None
    if model is not None and model != 'tfidf':
        sizes = [len(e['candidate_ids']) for e in precomputed]
        all_features = np.vstack([e['features'] for e in precomputed])
        if hasattr(model, 'predict_proba'):
            all_scores = model.predict_proba(all_features)[:, 1]
        else:
            all_scores = model.predict(all_features)
        offsets = np.cumsum([0] + sizes)
        batched_scores = [all_scores[offsets[i]:offsets[i + 1]] for i in range(len(sizes))]

    for i, entry in enumerate(iterator):
        candidates = entry['candidate_ids']
        golden_ids = entry['golden_ids']

        if model is None:
            scores = entry['bm25_scores']
        elif model == 'tfidf':
            scores = entry['tfidf_scores']
        else:
            assert batched_scores is not None
            scores = batched_scores[i]

        ranked_idx = np.argsort(-scores)
        relevances = [1 if candidates[j] in golden_ids else 0 for j in ranked_idx]
        total_relevant = len(golden_ids)

        for k in k_values:
            results[f'NDCG@{k}'].append(ndcg_at_k(relevances, k))
            results[f'Recall@{k}'].append(recall_at_k(relevances, k, total_relevant))
        results['MRR'].append(reciprocal_rank(relevances))
        results['MAP'].append(average_precision(relevances))
        meta_rows.append(entry['metadata'])

    metrics = {m: np.mean(v) for m, v in results.items()}
    return metrics, pd.DataFrame(meta_rows)


def candidates_to_training_data(precomputed, feature_cols):
    """Convert precomputed candidates into X, y, groups arrays for ranker training."""
    X_rows = []
    y_rows = []
    groups = []

    for entry in precomputed:
        candidates = entry['candidate_ids']
        golden_ids = entry['golden_ids']
        features = entry['features']

        labels = np.array([1 if cid in golden_ids else 0 for cid in candidates])

        X_rows.append(features)
        y_rows.append(labels)
        groups.append(len(candidates))

    X = np.vstack(X_rows)
    y = np.concatenate(y_rows)
    groups = np.array(groups)

    return X, y, groups


def evaluate_full_retrieval(test_qa, bm25_index, tfidf_index, chunk_lookup,
                            model, feature_cols, candidate_k=100,
                            k_values=[5, 10, 20]):
    """Convenience wrapper: prepare candidates then evaluate."""
    precomputed = prepare_candidates(test_qa, bm25_index, tfidf_index,
                                     chunk_lookup, feature_cols, candidate_k)
    return evaluate_from_candidates(precomputed, model, k_values)


def print_evaluation(metrics, model_name="Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation Results")
    print(f"{'='*50}")
    for metric, value in sorted(metrics.items()):
        print(f"  {metric:15s}: {value:.4f}")
    print(f"{'='*50}")
