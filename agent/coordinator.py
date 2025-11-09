# coordinator.py
# ---------------------------------------------------------
# 🎯 Purpose: Orchestrate CineMind's multi-agent workflow
# (Profiler → Trend Analyst → Content Curator)

import json
import sys
import os

# Add parent directory to path for flexible imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Try relative imports first (for package usage)
    from .user_profiler import extract_user_profile
    from .trend_analyst import analyze_trends
    from .content_curator import curate_recommendations
except ImportError:
    # Fall back to direct imports (for direct script execution)
    from user_profiler import extract_user_profile
    from trend_analyst import analyze_trends
    from content_curator import curate_recommendations


def run_cinemind_pipeline(user_query: str):
    print(f"\n🎬 User Query: {user_query}\n{'-'*70}")

    # --- Step 1: Profile user intent ---
    profile_output = extract_user_profile(user_query)

    # Some LLMs wrap JSON in backticks or markdown → clean it
    cleaned = (
        profile_output.replace("```json", "")
        .replace("```", "")
        .strip()
    )

    # --- Step 2: Retrieve candidate movies ---
    candidates = analyze_trends(cleaned, k=8)

    # --- Step 3: Curate final recommendations ---
    final_output = curate_recommendations(cleaned, candidates)

    print("\n✅ CineMind Final Recommendation:\n")
    print(final_output)
    return final_output


if __name__ == "__main__":
    test_queries = [
        "I loved Interstellar and Inception but want something more emotional and romantic.",
        "Recommend light-hearted comedies from the 90s with Robin Williams.",
        "Suggest deep sci-fi dramas about human emotion and space exploration."
    ]

    for q in test_queries:
        run_cinemind_pipeline(q)
