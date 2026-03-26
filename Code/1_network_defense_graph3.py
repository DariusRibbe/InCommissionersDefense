#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


INPUT_CSV = "/fs/dss/work/role6027/defence_bertopic_network_out/df_sentences_with_topics.csv"

OUT_DIR = "/fs/dss/work/role6027/defence_bertopic_network_out/bertopic_sentence_vs_context_out"
MODEL_DIR = os.path.join(OUT_DIR, "model")
TABLE_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

OUT_CSV_ENRICHED = os.path.join(TABLE_DIR, "df_with_bertopic_sentence_and_context.csv")
OUT_TOPICS_INFO = os.path.join(TABLE_DIR, "topics_info.csv")
OUT_UNIQUE_WORDS = os.path.join(TABLE_DIR, "topics_unique_words.csv")


SENTENCE_COL = "satz_text"
CONTEXT_COL = "context_sentence_pm1"

# Optional row id column (if absent we create one)
ROW_ID_COL = "row_id"

# Output columns
TOPIC_SENT_OUT = "topic_sentence_bertopic"
TOPIC_CTX_OUT  = "topic_context_pm1_bertopic"
PROB_SENT_OUT  = "topic_sentence_prob"
PROB_CTX_OUT   = "topic_context_prob"
TOP3_SENT_OUT  = "topic_sentence_top3"
TOP3_CTX_OUT   = "topic_context_top3"



LANGUAGE = "multilingual"  # or "english" / "german" depending on your texts
NR_TOPICS = "auto"         # reduction target; "auto" uses hierarchical reduction
MIN_TOPIC_SIZE = 30        # HDBSCAN parameter in BERTopic (controls granularity)
TOP_N_WORDS = 15           # top words for topic representation
NGRAM_RANGE = (1, 2)       # vectorizer ngram range for c-TF-IDF


EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


UNIQUE_WORDS_PER_TOPIC = 15
UNIQUE_CANDIDATE_WORDS = 60  


def _clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nAvailable columns:\n{list(df.columns)}")

def _safe_make_row_id(df: pd.DataFrame) -> pd.DataFrame:
    if ROW_ID_COL not in df.columns:
        df = df.copy()
        df[ROW_ID_COL] = np.arange(len(df), dtype=int)
    return df

def _topk_from_probs(probs: Optional[np.ndarray], k: int = 3) -> List[Tuple[int, float]]:

    if probs is None:
        return []
    probs = np.asarray(probs)
    if probs.ndim != 1 or probs.size == 0:
        return []
    idx = np.argsort(-probs)[:k]
    return [(int(i), float(probs[i])) for i in idx]

def _serialize_topk(topk: List[Tuple[int, float]]) -> str:
    return json.dumps([{"topic_index": t, "prob": p} for t, p in topk], ensure_ascii=False)


def build_bertopic_model():
    # BERTopic + dependencies
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        ngram_range=NGRAM_RANGE,
        stop_words=None,  # let c-TF-IDF handle; set to "english"/custom list if desired
    )

    topic_model = BERTopic(
        language=LANGUAGE,
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=TOP_N_WORDS,
        calculate_probabilities=True,
        verbose=True,
    )

    return topic_model

def reduce_topics_if_requested(topic_model, docs: List[str], topics: List[int]):

    if NR_TOPICS is None:
        return topic_model, topics

    try:
        reduced_model = topic_model.reduce_topics(docs, topics, nr_topics=NR_TOPICS)
        # After reduce_topics, topics are updated in the model; we should refetch doc topics
        # by transforming again so both sentence/context use the reduced topic space.
        return reduced_model, None
    except Exception as e:
        print(f"WARNING: reduce_topics failed ({e}). Continuing without reduction.", file=sys.stderr)
        return topic_model, topics

def export_model(topic_model, out_dir: str):

    model_path = os.path.join(out_dir, "bertopic_model")
    os.makedirs(out_dir, exist_ok=True)

    try:
        topic_model.save(model_path, serialization="safetensors")
        return model_path
    except TypeError:
        # older versions
        topic_model.save(model_path)
        return model_path
    except Exception as e:
        raise RuntimeError(f"Failed to save BERTopic model: {e}")

def compute_unique_words(topic_model, unique_words_per_topic: int, candidate_top_n: int) -> pd.DataFrame:

    topics = sorted([t for t in topic_model.get_topics().keys() if t != -1])


    ranks: Dict[int, Dict[str, int]] = {}
    for t in topics:
        words = topic_model.get_topic(t) or []
        words = words[:candidate_top_n]
        ranks[t] = {w: i for i, (w, _score) in enumerate(words)}

    rows = []
    for t in topics:
        for w, r in ranks[t].items():
            other_best = None
            for t2 in topics:
                if t2 == t:
                    continue
                r2 = ranks[t2].get(w, None)
                if r2 is None:
                    continue
                if other_best is None or r2 < other_best:
                    other_best = r2

            # If absent elsewhere, treat as very unique
            if other_best is None:
                uniq_score = 10_000  # huge
            else:
                uniq_score = (other_best - r)

            rows.append({"topic": t, "word": w, "rank_in_topic": r, "best_rank_other_topics": other_best, "uniqueness": uniq_score})

    dfu = pd.DataFrame(rows)

    # pick top unique words per topic
    out = (
        dfu.sort_values(["topic", "uniqueness"], ascending=[True, False])
           .groupby("topic", as_index=False)
           .head(unique_words_per_topic)
           .reset_index(drop=True)
    )
    return out

def topics_info_table(topic_model) -> pd.DataFrame:
    info = topic_model.get_topic_info()
    return info

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing INPUT_CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = _safe_make_row_id(df)

    _ensure_columns(df, [ROW_ID_COL, SENTENCE_COL, CONTEXT_COL])

    # Clean texts
    df[SENTENCE_COL] = df[SENTENCE_COL].map(_clean_text)
    df[CONTEXT_COL] = df[CONTEXT_COL].map(_clean_text)

    # Filter out empty rows for training (but keep them in output)
    train_mask = (df[CONTEXT_COL].str.len() > 0)
    train_docs = df.loc[train_mask, CONTEXT_COL].tolist()

    if len(train_docs) < 10:
        raise ValueError("Too few non-empty context documents to train BERTopic (need >= 10).")

    # 1) Fit ONE shared topic model on CONTEXT docs (shared topic categories for both)
    topic_model = build_bertopic_model()
    topics_ctx_train, probs_ctx_train = topic_model.fit_transform(train_docs)

    # 2) Topic reduction (auto) to improve interpretability / merge similar topics
    topic_model, maybe_topics = reduce_topics_if_requested(topic_model, train_docs, topics_ctx_train)

    # ( ensures topic ids match reduced model)
    topics_ctx_all, probs_ctx_all = topic_model.transform(df[CONTEXT_COL].tolist())

    # 3) Transform sentence-only texts using the SAME model -> same topic IDs/categories
    topics_sent_all, probs_sent_all = topic_model.transform(df[SENTENCE_COL].tolist())

    # 4) Store per-row most relevant topic + probability
    df[TOPIC_CTX_OUT] = topics_ctx_all
    df[TOPIC_SENT_OUT] = topics_sent_all

    topic_ids_sorted = list(topic_model.get_topics().keys())
    topic_to_index = {tid: i for i, tid in enumerate(topic_ids_sorted)}

    def assigned_prob(assigned_topic: int, prob_vec: Optional[np.ndarray]) -> float:
        if prob_vec is None:
            return np.nan
        idx = topic_to_index.get(int(assigned_topic), None)
        if idx is None:
            return np.nan
        if idx >= len(prob_vec):
            return np.nan
        return float(prob_vec[idx])

    ctx_probs_assigned = []
    sent_probs_assigned = []
    ctx_top3 = []
    sent_top3 = []

    for t_ctx, p_ctx, t_sent, p_sent in zip(topics_ctx_all, probs_ctx_all, topics_sent_all, probs_sent_all):
        ctx_probs_assigned.append(assigned_prob(t_ctx, p_ctx))
        sent_probs_assigned.append(assigned_prob(t_sent, p_sent))

        # top-3 topics by prob (index -> topic_id)
        if p_ctx is not None and len(p_ctx) == len(topic_ids_sorted):
            topk_idx = np.argsort(-p_ctx)[:3]
            topk = [(int(topic_ids_sorted[i]), float(p_ctx[i])) for i in topk_idx]
            ctx_top3.append(_serialize_topk(topk))
        else:
            ctx_top3.append("[]")

        if p_sent is not None and len(p_sent) == len(topic_ids_sorted):
            topk_idx = np.argsort(-p_sent)[:3]
            topk = [(int(topic_ids_sorted[i]), float(p_sent[i])) for i in topk_idx]
            sent_top3.append(_serialize_topk(topk))
        else:
            sent_top3.append("[]")

    df[PROB_CTX_OUT] = ctx_probs_assigned
    df[PROB_SENT_OUT] = sent_probs_assigned
    df[TOP3_CTX_OUT] = ctx_top3
    df[TOP3_SENT_OUT] = sent_top3

    # 5) Export enriched dataset
    df.to_csv(OUT_CSV_ENRICHED, index=False)
    print(f"Saved enriched CSV: {OUT_CSV_ENRICHED}")

    # 6) Export topic info
    info = topics_info_table(topic_model)
    info.to_csv(OUT_TOPICS_INFO, index=False)
    print(f"Saved topic info: {OUT_TOPICS_INFO}")

    # 7) Export "most unique words" per topic
    unique_df = compute_unique_words(
        topic_model,
        unique_words_per_topic=UNIQUE_WORDS_PER_TOPIC,
        candidate_top_n=UNIQUE_CANDIDATE_WORDS,
    )
    unique_df.to_csv(OUT_UNIQUE_WORDS, index=False)
    print(f"Saved unique words per topic: {OUT_UNIQUE_WORDS}")

    # 8) Save model for reuse
    model_path = export_model(topic_model, MODEL_DIR)
    print(f"Saved BERTopic model: {model_path}")

    print("Done. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
