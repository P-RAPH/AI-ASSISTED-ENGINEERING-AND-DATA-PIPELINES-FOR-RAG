# Stage 3: Data Provenance and Content-Quality Analysis

import pandas as pd
import re
import json

RAW_PATH = "D:\M.Sc Course Materials\CSC 796 - Data Engineering\Term Paper\hana_chatgpt.csv"

def word_count(t):
    return len(str(t).split())

def sentence_count(t):
    sents = re.split(r"[.!?]+", str(t))
    return max(1, len([s for s in sents if s.strip()]))

def type_token_ratio(t):
    words = re.findall(r"[a-zA-Z']+", str(t).lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def run():
    df = pd.read_csv(RAW_PATH)
    paired = df.dropna(subset=["rephrase"]).copy()  # rows with a successful AI rephrase
    n_pairs = len(paired)

    for col, label in [("main", "human"), ("rephrase", "ai")]:
        paired[f"{label}_words"] = paired[col].apply(word_count)
        paired[f"{label}_sentences"] = paired[col].apply(sentence_count)
        paired[f"{label}_avg_sent_len"] = paired[f"{label}_words"] / paired[f"{label}_sentences"]
        paired[f"{label}_ttr"] = paired[col].apply(type_token_ratio)

    summary = {
        "n_paired_articles": n_pairs,
        "human_mean_words": round(paired["human_words"].mean(), 1),
        "ai_mean_words": round(paired["ai_words"].mean(), 1),
        "human_mean_sentences": round(paired["human_sentences"].mean(), 1),
        "ai_mean_sentences": round(paired["ai_sentences"].mean(), 1),
        "human_mean_avg_sentence_length": round(paired["human_avg_sent_len"].mean(), 2),
        "ai_mean_avg_sentence_length": round(paired["ai_avg_sent_len"].mean(), 2),
        "human_mean_type_token_ratio": round(paired["human_ttr"].mean(), 4),
        "ai_mean_type_token_ratio": round(paired["ai_ttr"].mean(), 4),
    }

    # Simple, transparent heuristic provenance flag: articles with a longer
    # average sentence length AND higher lexical diversity than a fitted
    # midpoint are flagged as "possibly AI-generated". Evaluated on a
    # balanced blind sample of human + AI texts pulled from the same pairs.
    sample_n = min(200, n_pairs)
    sample = paired.sample(n=sample_n, random_state=42)
    blind = []
    for _, row in sample.iterrows():
        blind.append({"text": row["main"], "true_label": "human"})
        blind.append({"text": row["rephrase"], "true_label": "ai"})
    blind_df = pd.DataFrame(blind)
    blind_df["avg_sent_len"] = blind_df["text"].apply(lambda t: word_count(t) / sentence_count(t))
    blind_df["ttr"] = blind_df["text"].apply(type_token_ratio)

    sent_len_threshold = blind_df["avg_sent_len"].median()
    ttr_threshold = blind_df["ttr"].median()
    blind_df["predicted_label"] = blind_df.apply(
        lambda r: "ai" if (r["avg_sent_len"] > sent_len_threshold and r["ttr"] > ttr_threshold) else "human",
        axis=1
    )
    accuracy = (blind_df["predicted_label"] == blind_df["true_label"]).mean()
    tp = ((blind_df["predicted_label"] == "ai") & (blind_df["true_label"] == "ai")).sum()
    fp = ((blind_df["predicted_label"] == "ai") & (blind_df["true_label"] == "human")).sum()
    fn = ((blind_df["predicted_label"] == "human") & (blind_df["true_label"] == "ai")).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    summary["heuristic_eval_sample_size"] = len(blind_df)
    summary["heuristic_sent_len_threshold"] = round(sent_len_threshold, 2)
    summary["heuristic_ttr_threshold"] = round(ttr_threshold, 4)
    summary["heuristic_accuracy"] = round(accuracy, 3)
    summary["heuristic_precision_ai_class"] = round(precision, 3)
    summary["heuristic_recall_ai_class"] = round(recall, 3)

    return summary

if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
