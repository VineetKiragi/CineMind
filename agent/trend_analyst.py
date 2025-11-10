# agent/trend_analyst.py
# ---------------------------------------------------------
# 🎯 Purpose: Retrieve semantically relevant movies from FAISS index
# using precomputed embeddings (downloaded once if missing).

import os
import json
import zipfile
import urllib.request
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# === Load API key ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# === Constants ===
FAISS_DIR = "data/faiss_index"
FAISS_URL = "https://github.com/VineetKiragi/CineMind/releases/download/v0.1.0/faiss_index.zip"
FAISS_ZIP = os.path.join(FAISS_DIR, "faiss_index.zip")
os.makedirs(FAISS_DIR, exist_ok=True)

# === Ensure FAISS index is present ===
if not os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
    print("⚠️ FAISS index missing — downloading from GitHub release...")
    try:
        urllib.request.urlretrieve(FAISS_URL, FAISS_ZIP)
        with zipfile.ZipFile(FAISS_ZIP, "r") as zip_ref:
            zip_ref.extractall(FAISS_DIR)
        print("✅ FAISS index downloaded and extracted successfully.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download FAISS index: {e}")

# === Load FAISS index ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
vectorstore = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)


# === Utility ===
def build_search_prompt(profile_json: str):
    """Converts structured user preferences into a natural query string."""
    try:
        profile = json.loads(profile_json)
    except json.JSONDecodeError:
        return "Movies similar to the user's tastes"

    parts = []
    if profile.get("genres"):
        parts.append(f"genres: {', '.join(profile['genres'])}")
    if profile.get("tone"):
        parts.append(f"tone: {', '.join(profile['tone'])}")
    if profile.get("decade"):
        parts.append(f"from the {', '.join(profile['decade'])}")
    if profile.get("people"):
        parts.append(f"involving {', '.join(profile['people'])}")
    if profile.get("other_preferences"):
        parts.append(f"themes: {', '.join(profile['other_preferences'])}")

    return "Recommend movies that match " + ", ".join(parts)


# === Core Retrieval ===
def analyze_trends(profile_json: str, k=5):
    """Queries the FAISS vectorstore and returns ranked recommendations."""
    query = build_search_prompt(profile_json)
    print(f"\n🔍 Trend Analyst Query: {query}\n")

    results = vectorstore.similarity_search_with_score(query, k=k)
    recommendations = []
    seen_titles = set()

    for doc, score in results:
        title = doc.metadata.get("title", "Unknown")
        if title not in seen_titles:
            seen_titles.add(title)
            recommendations.append({
                "title": title,
                "year": doc.metadata.get("year"),
                "genres": doc.metadata.get("genres"),
                "director": doc.metadata.get("director"),
                "rating": float(round(doc.metadata.get("rating", 0), 2)),
                "score": float(round(score, 3))
            })

    print("🎞 Top Retrieved Candidates:")
    for r in recommendations:
        print(f"• {r['title']} ({r['year']}) | {r['genres']} | Score: {r['score']}")

    return recommendations


if __name__ == "__main__":
    # Test mode
    sample_profile = json.dumps({
        "genres": ["romance", "comedy"],
        "tone": ["light-hearted"],
        "decade": ["2000s"],
        "people": [],
        "other_preferences": ["feel-good", "happy ending"]
    })
    analyze_trends(sample_profile)
