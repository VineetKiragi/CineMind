# build_master_dataset.py
# ---------------------------------------------------------
# 🎯 Purpose: Create a clean, deduplicated, enriched movie-level dataset
# ready for embeddings and retrieval.

import pandas as pd
import numpy as np
import json
import ast
import os

# === Utility functions ===
def parse_genres(genre_str):
    """
    Handles double-encoded JSON-like genre strings from movies_metadata.csv.
    Works for strings like:
        "[{'id': 28, 'name': 'Action'}, {'id': 35, 'name': 'Comedy'}]"
    """
    if pd.isna(genre_str) or genre_str in ("[]", "", None):
        return []
    try:
        # Step 1: Remove outer quotes if double-encoded
        if isinstance(genre_str, str) and genre_str.startswith('"[') and genre_str.endswith(']"'):
            genre_str = ast.literal_eval(genre_str)
        # Step 2: Evaluate inner Python-style list
        parsed = ast.literal_eval(genre_str)
        if isinstance(parsed, list):
            return [g["name"] for g in parsed if isinstance(g, dict) and "name" in g]
        return []
    except Exception:
        return []


def parse_keywords(kw_str):
    """Safely parse keywords field."""
    if pd.isna(kw_str) or kw_str in ("[]", "", None):
        return []
    try:
        parsed = ast.literal_eval(kw_str)
        if isinstance(parsed, list):
            return [k["name"] for k in parsed if isinstance(k, dict) and "name" in k]
        return []
    except Exception:
        return []

def extract_cast(x):
    """Extracts top 5 cast names from 'cast' field in credits.csv."""
    if pd.isna(x) or not isinstance(x, str):
        return []
    try:
        parsed = ast.literal_eval(x)
        return [c["name"] for c in parsed[:5] if isinstance(c, dict) and "name" in c]
    except Exception:
        return []

def extract_director(x):
    """Extracts first director from 'crew' field in credits.csv."""
    if pd.isna(x) or not isinstance(x, str):
        return None
    try:
        parsed = ast.literal_eval(x)
        directors = [c["name"] for c in parsed if isinstance(c, dict) and c.get("job") == "Director"]
        return directors[0] if directors else None
    except Exception:
        return None

def compute_weighted_rating(v, R, C, m):
    """IMDb-style weighted rating."""
    return (v / (v + m)) * R + (m / (v + m)) * C if (v + m) > 0 else R


# === Load raw data ===
BASE = "data"
movies = pd.read_csv(os.path.join(BASE, "movies_metadata.csv"), low_memory=False)
ratings = pd.read_csv(os.path.join(BASE, "ratings_small.csv"))
credits = pd.read_csv(os.path.join(BASE, "credits.csv"))
keywords = pd.read_csv(os.path.join(BASE, "keywords.csv"))

print(f"Movies: {len(movies)}, Ratings: {len(ratings)}, Credits: {len(credits)}, Keywords: {len(keywords)}")

# === Clean & select movie fields ===
movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
movies.dropna(subset=["id"], inplace=True)
movies["id"] = movies["id"].astype(int)

movies = movies[[
    "id", "title", "overview", "release_date", "vote_average", "vote_count", "genres"
]]
movies["genres"] = movies["genres"].apply(parse_genres)
movies["year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year

# === Aggregate ratings ===
ratings_grouped = ratings.groupby("movieId").agg(
    user_rating_mean=("rating", "mean"),
    user_rating_median=("rating", "median"),
    user_rating_count=("rating", "count"),
    user_rating_std=("rating", "std")
).reset_index()

# === Join movies with ratings ===
df = movies.merge(ratings_grouped, left_on="id", right_on="movieId", how="left")

# === Add credits (cast/director) ===
credits["cast_top"] = credits["cast"].apply(extract_cast)
credits["director"] = credits["crew"].apply(extract_director)
credits = credits[["id", "cast_top", "director"]]
df = df.merge(credits, on="id", how="left")

# === Add keywords ===
keywords["keywords"] = keywords["keywords"].apply(parse_keywords)
df = df.merge(keywords, on="id", how="left")

# === Compute Weighted Rating (IMDb formula) ===
C = df["vote_average"].mean()
m = df["vote_count"].quantile(0.8)
df["weighted_rating"] = df.apply(
    lambda x: compute_weighted_rating(x["vote_count"], x["vote_average"], C, m), axis=1
)

# === Drop duplicates & invalid rows ===
df.drop_duplicates(subset=["id"], inplace=True)
df.dropna(subset=["title", "overview"], inplace=True)

print(f"✅ Cleaned dataset size: {len(df)} unique movies")

# === Save master file ===
master_path = os.path.join(BASE, "movies_master.parquet")
df.to_parquet(master_path, index=False)
print(f"✅ Saved master dataset to {master_path}")

# === Build embedding corpus (rich text) ===
def build_corpus_row(row):
    genres = ", ".join(row["genres"]) if isinstance(row["genres"], list) else ""
    cast = ", ".join(row["cast_top"]) if isinstance(row["cast_top"], list) else ""
    keywords = ", ".join(row["keywords"]) if isinstance(row["keywords"], list) else ""
    text = (
        f"Title: {row['title']} ({int(row['year']) if not np.isnan(row['year']) else 'Unknown'})\n"
        f"Genres: {genres}\n"
        f"Director: {row['director']}\n"
        f"Cast: {cast}\n"
        f"Keywords: {keywords}\n"
        f"Rating: {round(row['weighted_rating'], 2)}\n"
        f"Overview: {row['overview']}"
    )
    meta = {
        "title": row["title"],
        "year": row["year"],
        "genres": row["genres"],
        "director": row["director"],
        "rating": row["weighted_rating"],
        "vote_count": row["vote_count"],
    }
    return {"page_content": text, "metadata": meta}

corpus = df.apply(build_corpus_row, axis=1).to_list()
corpus_path = os.path.join(BASE, "embeddings_corpus.jsonl")
with open(corpus_path, "w", encoding="utf-8") as f:
    for rec in corpus:
        f.write(json.dumps(rec) + "\n")

print(f"✅ Embedding corpus saved to {corpus_path}")
print("🎉 Data build completed successfully — master dataset and corpus ready!")
