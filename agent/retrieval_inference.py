# retrieval_inference.py
# ---------------------------------------------------------
# 🎯 Purpose: Combine FAISS retrieval with GPT reasoning to
# produce context-aware movie recommendations.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# === Load API key ===
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file.")

# === Load FAISS index ===
FAISS_PATH = "data/faiss_index"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# === Create retriever ===
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === Configure GPT model ===
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7, api_key=openai_api_key)

# === Create prompt template ===
template = """You are CineMind, an AI movie recommendation assistant.
Use the following retrieved movie information to provide thoughtful recommendations.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# === Create RAG chain ===
def format_docs(docs):
    return "\n\n".join([f"Title: {doc.metadata.get('title', 'Unknown')}\nGenres: {doc.metadata.get('genres', 'N/A')}\nContent: {doc.page_content}" for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# === Inference function ===
def cine_recommend(query: str):
    """Run a query through FAISS + GPT reasoning."""
    print(f"\n🎬 User query: {query}\n")

    # Get retrieved documents
    docs = retriever.invoke(query)

    # Get AI response
    response = rag_chain.invoke(query)

    print("🧠 CineMind reasoning output:\n")
    print(response)
    print("\n--- Top retrieved movies ---")
    for doc in docs:
        meta = doc.metadata
        print(f"• {meta.get('title')} ({meta.get('year')}) — Genres: {meta.get('genres')}")

# === Example interactive usage ===
if __name__ == "__main__":
    queries = [
        "I loved Interstellar and Arrival — recommend similar thought-provoking sci-fi.",
        "Suggest light-hearted romantic comedies from the 2000s.",
        "Movies with artificial intelligence themes but emotionally deep.",
    ]
    for q in queries:
        cine_recommend(q)
