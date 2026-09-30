# Stage 1: Data Ingestion and Profiling

import pandas as pd
import json

RAW_PATH = "/D:\M.Sc Course Materials\CSC 796 - Data Engineering\Term Paper\hana_chatgpt.csv"

def ingest(path):
    return pd.read_csv(path)

def profile(df):
    report = {}
    report["n_rows"] = len(df)
    report["n_columns"] = df.shape[1]
    report["columns"] = list(df.columns)
    report["agency_counts"] = df["agency"].value_counts().to_dict()
    report["topic_counts_raw"] = df["topic"].value_counts().to_dict()
    report["missing_values"] = df.isnull().sum().to_dict()
    report["duplicate_main_rows"] = int(df.duplicated(subset=["main"]).sum())
    report["duplicate_title_rows"] = int(df.duplicated(subset=["title"]).sum())
    report["word_count_stats"] = df["words"].describe().to_dict()
    # cross-check reported word count against actual token count of 'main'
    df["_actual_words"] = df["main"].str.split().apply(len)
    report["word_count_mismatch_rows"] = int((df["_actual_words"] != df["words"]).sum())
    return report

if __name__ == "__main__":
    df = ingest(RAW_PATH)
    rep = profile(df)
    print(json.dumps(rep, indent=2, default=str))
