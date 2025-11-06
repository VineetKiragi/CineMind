# test_faiss_retrieval.py
# ------------------------------------------------------
# 🎯 Purpose: Verify FAISS index retrieval works correctly.

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# === Load environment & embeddings ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

FAISS_PATH = "data/faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def search_movies(query, k=5):
    results = vectorstore.similarity_search_with_score(query, k=k)
    print(f"\n🔍 Query: {query}\n")
    for i, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        print(f"{i}. 🎬 {meta.get('title')} ({meta.get('year')}) — Score: {round(score, 3)}")
        print(f"   Genres: {meta.get('genres')}")
        print(f"   Director: {meta.get('director')}")
        print(f"   Rating: {meta.get('rating')}")
        print(f"   Overview: {doc.page_content[:250]}...\n")

if __name__ == "__main__":
    # Try some sample queries
    search_movies("space exploration and artificial intelligence")
    search_movies("goofy buddy cop action comedy")
    search_movies("romantic movies directed by Christopher Nolan")
