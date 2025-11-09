# app.py
# ---------------------------------------------------------
# 🎬 CineMind — Interactive Movie Recommendation Chat
# Frontend built with Streamlit

import streamlit as st
import os, sys, json

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.coordinator import run_cinemind_pipeline

st.set_page_config(page_title="🎬 CineMind", page_icon="🎥", layout="wide")

# === Header ===
st.title("🎬 CineMind — Your AI Movie Curator")
st.markdown(
    "Ask CineMind for recommendations — it uses multi-agent reasoning and deep movie knowledge "
    "to tailor suggestions based on your taste."
)

# === Sidebar ===
st.sidebar.header("Settings")
st.sidebar.markdown("💡 *Powered by OpenAI GPT-4 + FAISS + LangChain Agents*")
if "history" not in st.session_state:
    st.session_state.history = []

# === Chat Input ===
user_query = st.text_input("🎤 Ask CineMind something like:",
                           placeholder="I loved Interstellar and Arrival, suggest similar movies...")

if st.button("✨ Recommend") or user_query:
    with st.spinner("CineMind is thinking..."):
        try:
            result = run_cinemind_pipeline(user_query)
            st.session_state.history.append({"query": user_query, "result": result})
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# === Display Chat History ===
st.markdown("---")
for chat in reversed(st.session_state.history):
    st.markdown(f"**🧑‍🎓 You:** {chat['query']}")
    st.markdown(f"**🎬 CineMind:** {chat['result']}")
    st.markdown("---")
