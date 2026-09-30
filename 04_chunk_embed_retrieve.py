# Stage 4: Chunking, Vector Storage (TF-IDF), and Retrieval Evaluation

import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CLEAN_PATH = "D:\M.Sc Course Materials\CSC 796 - Data Engineering\Term Paper\hana_clean.csv"
CHUNK_WORDS = 120
CHUNK_OVERLAP = 20

def chunk_text(text, doc_id, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = str(text).split()
    chunks, start, idx = [], 0, 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({"chunk_id": f"{doc_id}-C{idx}", "doc_id": doc_id, "text": chunk})
        if end == len(words):
            break
        start, idx = end - overlap, idx + 1
    return chunks

def build_chunks(df):
    all_chunks = []
    for _, row in df.iterrows():
        all_chunks.extend(chunk_text(row["main"], row["doc_id"]))
    return pd.DataFrame(all_chunks)

def build_vector_store(chunks_df):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(chunks_df["text"])
    return vectorizer, matrix

def retrieve(query, vectorizer, matrix, chunks_df, meta_df, top_k=5):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = []
    for i in top_idx:
        row = chunks_df.iloc[i]
        cat = meta_df.loc[meta_df["doc_id"] == row["doc_id"], "category"].values[0]
        results.append({
            "chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "category": cat,
            "similarity": round(float(sims[i]), 4), "snippet": row["text"][:160] + "..."
        })
    return results

if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    chunks_df = build_chunks(df)
    vectorizer, matrix = build_vector_store(chunks_df)

    summary = {
        "n_documents": len(df),
        "n_chunks": len(chunks_df),
        "avg_chunks_per_doc": round(len(chunks_df) / len(df), 2),
        "chunk_size_words": CHUNK_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP,
        "vocabulary_size": len(vectorizer.vocabulary_),
    }

    test_queries = [
        ("What new legislation or government policy was announced?", "politics_government"),
        ("Which team won the championship game?", "sports"),
        ("What breakthrough in artificial intelligence or computing was reported?", "science_technology"),
        ("What religious ceremony or historical anniversary took place?", "history_religion"),
        ("Which celebrity made a public appearance or announcement?", "celebrity"),
        ("What civil rights protest or social movement was covered?", "social_culture_civil_rights"),
    ]

    eval_rows = []
    for q, expected_cat in test_queries:
        results = retrieve(q, vectorizer, matrix, chunks_df, df, top_k=5)
        top1_correct = results[0]["category"] == expected_cat
        precision_at_5 = sum(1 for r in results if r["category"] == expected_cat) / 5
        eval_rows.append({
            "query": q, "expected_category": expected_cat,
            "top1_category": results[0]["category"], "top1_correct": top1_correct,
            "precision_at_5": precision_at_5, "top1_similarity": results[0]["similarity"],
        })

    summary["retrieval_evaluation"] = eval_rows
    summary["top1_accuracy"] = sum(r["top1_correct"] for r in eval_rows) / len(eval_rows)
    summary["mean_precision_at_5"] = round(sum(r["precision_at_5"] for r in eval_rows) / len(eval_rows), 3)

    print(json.dumps(summary, indent=2, default=str))
    chunks_df.to_csv("D:\M.Sc Course Materials\CSC 796 - Data Engineering\Term Paper\hana_chunks.csv", index=False)
