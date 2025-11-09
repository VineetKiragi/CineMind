# content_curator.py
# ---------------------------------------------------------
# 🎯 Purpose: Transform retrieved movie candidates + user profile
# into a conversational, ranked recommendation message.

import os, json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# === Initialize LLM ===
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.7, api_key=openai_api_key)

# === Prompt template ===
template = """
You are CineMind's Content Curator Agent.
You receive:
1. A JSON object describing the user's preferences.
2. A list of candidate movies with metadata (title, year, genres, rating, etc.).

Your job:
- Select the 3–5 best recommendations that best fit the user's taste.
- Explain *why* each movie suits the user, referencing tone, genre, era, or theme.
- Write naturally, like a friendly movie expert.
- Keep the tone engaging and concise (about 1–2 sentences per movie).

User Profile:
{profile}

Candidate Movies:
{candidates}

Return your final recommendations as a conversational paragraph list.
"""

prompt = ChatPromptTemplate.from_template(template)
curation_chain = prompt | llm | StrOutputParser()

def curate_recommendations(profile_json: str, candidate_list: list):
    """
    Given a user profile (JSON str) and candidate list (list of dicts),
    return CineMind's curated recommendation message.
    """
    # Pretty-format candidates for prompt readability
    candidates_str = json.dumps(candidate_list, indent=2)
    print("\n🎨 Curating final recommendations...\n")
    result = curation_chain.invoke({
        "profile": profile_json,
        "candidates": candidates_str
    })
    print("🧠 CineMind Curator Output:\n")
    print(result)
    return result

# === Example usage ===
if __name__ == "__main__":
    sample_profile = json.dumps({
        "genres": ["romance", "comedy"],
        "tone": ["light-hearted"],
        "decade": ["2000s"],
        "people": [],
        "other_preferences": ["feel-good", "happy ending"]
    })

    sample_candidates = [
        {"title": "Serendipity", "year": 2001, "genres": ["Comedy","Romance"], "rating": 7.3},
        {"title": "Just Like Heaven", "year": 2005, "genres": ["Comedy","Romance","Fantasy"], "rating": 7.1},
        {"title": "Amélie", "year": 2001, "genres": ["Comedy","Romance"], "rating": 8.1},
        {"title": "(500) Days of Summer", "year": 2009, "genres": ["Comedy","Drama","Romance"], "rating": 7.7}
    ]

    curate_recommendations(sample_profile, sample_candidates)
