# create_embeddings_faiss.py
# -------------------------------------------------------------
# 🎯 Purpose: Convert movie corpus into OpenAI embeddings and
# store them in a local FAISS index for semantic retrieval.

import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from tqdm import tqdm

# === Load environment variables ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env")

# === File paths ===
CORPUS_PATH = "data/embeddings_corpus.jsonl"
INDEX_DIR = "data/faiss_index"

# === Read the corpus ===
print("📖 Loading corpus from JSONL...")
documents = []
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        doc = Document(page_content=rec["page_content"], metadata=rec["metadata"])
        documents.append(doc)

print(f"✅ Loaded {len(documents)} documents for embedding.")

# === Initialize embeddings ===
print("⚙️ Creating OpenAI embeddings (text-embedding-3-large)...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=openai_api_key
)

# === Build FAISS index ===
print("🧠 Generating and indexing embeddings (this may take a few minutes)...")
vectorstore = FAISS.from_documents(tqdm(documents), embeddings)

# === Save index ===
vectorstore.save_local(INDEX_DIR)
print(f"✅ FAISS index saved at: {INDEX_DIR}")

print("🎉 Embedding generation complete — CineMind vector store is ready!")
