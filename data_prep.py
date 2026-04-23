"""Data preparation: flatten dataset into (question, chunk, relevance) triples."""

import random
import pandas as pd
from tqdm import tqdm


def extract_all_chunks(dataset_split, chunk_offset=0, doc_offset=0):
    """Flatten all chunks from a dataset split.

    Each chunk gets a globally unique integer chunk_id (starting from chunk_offset)
    and a doc_id (starting from doc_offset). Chunks from the same document share a doc_id.

    Returns: (chunks_list, next_chunk_offset, next_doc_offset)
    """
    chunks = []
    cid = chunk_offset
    for doc_idx, record in enumerate(tqdm(dataset_split, desc="Extracting chunks")):
        did = doc_offset + doc_idx
        for chunk in record['chunks']:
            chunks.append({
                'chunk_id': cid,
                'doc_id': did,
                'original_chunk_id': chunk['chunk_id'],
                'chunk_text': chunk['text'],
                'word_count': chunk['word_count'],
                'sentence_count': chunk['sentence_count'],
            })
            cid += 1
    next_doc_offset = doc_offset + len(dataset_split)
    return chunks, cid, next_doc_offset


def extract_qa_pairs(dataset_split, chunks_by_doc_and_original_id, doc_offset=0):
    """Flatten all QA pairs, mapping segment_ids to global chunk_ids.

    chunks_by_doc_and_original_id: dict of (doc_id, original_chunk_id) -> chunk_id
    """
    qa_pairs = []
    for doc_idx, record in enumerate(tqdm(dataset_split, desc="Extracting QA pairs")):
        did = doc_offset + doc_idx
        for qa in record['deduplicated_qa_pairs']:
            mapped_ids = []
            for sid in qa['segment_ids']:
                key = (did, sid)
                if key in chunks_by_doc_and_original_id:
                    mapped_ids.append(chunks_by_doc_and_original_id[key])
            qa_pairs.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'query_type': qa['query_type'],
                'reasoning_type': qa['reasoning_type'],
                'question_complexity': qa['question_complexity'],
                'hop_count': qa['hop_count'],
                'chunk_ids': mapped_ids,
                'doc_id': did,
            })
    return qa_pairs


def build_chunk_lookup(all_chunks):
    """Build lookup dicts from a flat list of chunks.

    Returns:
        chunk_by_id: {chunk_id: chunk_dict}
        chunk_by_doc_original: {(doc_id, original_chunk_id): chunk_id}
    """
    chunk_by_id = {c['chunk_id']: c for c in all_chunks}
    chunk_by_doc_original = {
        (c['doc_id'], c['original_chunk_id']): c['chunk_id']
        for c in all_chunks
    }
    return chunk_by_id, chunk_by_doc_original


def create_relevance_triples(qa_pairs, all_chunks, neg_per_positive=4, seed=42):
    """Create positive and random-negative triples for EDA/feature analysis."""
    random.seed(seed)

    chunk_by_id = {c['chunk_id']: c for c in all_chunks}
    all_chunk_ids = list(chunk_by_id.keys())

    triples = []
    for qa in tqdm(qa_pairs, desc="Creating triples"):
        relevant_ids = set(qa['chunk_ids'])

        #positives
        for cid in relevant_ids:
            if cid not in chunk_by_id:
                continue
            chunk = chunk_by_id[cid]
            triples.append({
                'question': qa['question'],
                'chunk_id': cid,
                'chunk_text': chunk['chunk_text'],
                'relevance': 1,
                'query_type': qa['query_type'],
                'reasoning_type': qa['reasoning_type'],
                'question_complexity': qa['question_complexity'],
                'hop_count': qa['hop_count'],
                'chunk_word_count': chunk['word_count'],
                'chunk_sentence_count': chunk['sentence_count'],
                'doc_id': qa['doc_id'],
            })

        #negatives (resample on collision)
        n_neg = len(relevant_ids) * neg_per_positive
        neg_samples = set()
        while len(neg_samples) < n_neg:
            c = random.choice(all_chunk_ids)
            if c not in relevant_ids:
                neg_samples.add(c)

        for cid in neg_samples:
            chunk = chunk_by_id[cid]
            triples.append({
                'question': qa['question'],
                'chunk_id': cid,
                'chunk_text': chunk['chunk_text'],
                'relevance': 0,
                'query_type': qa['query_type'],
                'reasoning_type': qa['reasoning_type'],
                'question_complexity': qa['question_complexity'],
                'hop_count': qa['hop_count'],
                'chunk_word_count': chunk['word_count'],
                'chunk_sentence_count': chunk['sentence_count'],
                'doc_id': qa['doc_id'],
            })

    return pd.DataFrame(triples)
