# user_profiler.py
# ---------------------------------------------------------
# 🎯 Purpose: Extract structured user preferences from natural-language input.
# Works as the first agent in the CineMind multi-agent pipeline.

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3, api_key=openai_api_key)

# === Prompt template ===
template = """
You are CineMind's User Profiler Agent.
Analyze the following user query and extract their movie preferences.

User Query: {query}

Return a JSON object with keys:
- "genres": list of genres or themes
- "tone": list of tone or mood descriptors
- "decade": list of decade or period clues
- "people": list of directors or actors mentioned
- "other_preferences": any extra info (e.g., story elements, settings, pacing)
"""

prompt = ChatPromptTemplate.from_template(template)

# === Chain definition ===
profile_chain = prompt | llm | StrOutputParser()


def extract_user_profile(query: str):
    """
    Run the profiler agent to get structured preferences.
    """
    print(f"\n🎬 Profiling user query: {query}\n")
    response = profile_chain.invoke({"query": query})
    print("🧠 Extracted profile:\n", response)
    return response


if __name__ == "__main__":
    test_queries = [
        "I loved Interstellar and Inception but want something lighter and more romantic.",
        "Recommend funny adventure movies from the 90s with Robin Williams.",
        "I like realistic dramas with strong female leads from the 2010s."
    ]

    for q in test_queries:
        extract_user_profile(q)
