# validate_master_dataset.py
# -------------------------------------------------------------
# 🎯 Purpose: Perform detailed sanity checks on movies_master.parquet
# and ensure embeddings_corpus.jsonl integrity.

import pandas as pd
import numpy as np
import json
import os

BASE = "data"
master_path = os.path.join(BASE, "movies_master.parquet")
corpus_path = os.path.join(BASE, "embeddings_corpus.jsonl")

# === Load master dataset ===
df = pd.read_parquet(master_path)
print(f"✅ Loaded master dataset with {len(df)} rows and {len(df.columns)} columns.\n")

# === 1️⃣ Check uniqueness ===
duplicate_ids = df["id"].duplicated().sum()
print(f"Duplicate movie IDs: {duplicate_ids}")

duplicate_titles = df["title"].duplicated().sum()
print(f"Duplicate movie titles: {duplicate_titles}\n")

# === 2️⃣ Missing data stats ===
missing_summary = df.isna().mean().sort_values(ascending=False)
print("🔍 Missing value ratios (top 10):")
print(missing_summary.head(10), "\n")

# === 3️⃣ Descriptive stats for ratings ===
rating_cols = ["vote_average", "vote_count", "weighted_rating", "user_rating_mean"]
desc = df[rating_cols].describe()
print("🎯 Rating distribution summary:")
print(desc, "\n")

# === 4️⃣ Genre & Director coverage ===
def safe_len(x):
    if isinstance(x, (list, np.ndarray)):
        return len(x)
    return 0

genre_coverage = df["genres"].apply(safe_len).mean()

director_coverage = df["director"].notna().mean()
print(f"Average #genres per movie: {genre_coverage:.2f}")
print(f"Director coverage: {director_coverage*100:.1f}%\n")

# === 5️⃣ Verify corpus alignment ===
corpus_lines = sum(1 for _ in open(corpus_path, "r", encoding="utf-8"))
if corpus_lines == len(df):
    print(f"✅ Embedding corpus alignment OK — {corpus_lines} records match master dataset.")
else:
    print(f"⚠️ Corpus size mismatch: {corpus_lines} vs {len(df)} rows in master dataset.")
