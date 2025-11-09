# api.py
# ---------------------------------------------------------
# 🎯 Purpose: Expose CineMind's multi-agent reasoning pipeline as REST API endpoints

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os, sys, json

# Ensure project root is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.coordinator import run_cinemind_pipeline
from agent.user_profiler import extract_user_profile

# ---------------------------------------------------------
# 🚀 Initialize app
app = FastAPI(
    title="CineMind API",
    description="Multi-agent AI movie recommendation API powered by GPT-4 and FAISS.",
    version="1.0"
)

# Enable CORS (so React / Next.js / Streamlit can access it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 📥 Request model
class QueryRequest(BaseModel):
    query: str

# ---------------------------------------------------------
# 🧠 Endpoints

@app.post("/recommend")
def recommend_movies(request: QueryRequest):
    """Main endpoint for CineMind recommendations."""
    try:
        result = run_cinemind_pipeline(request.query)
        return {"status": "success", "query": request.query, "recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/profile")
def extract_profile(request: QueryRequest):
    """Endpoint for debugging user profiling only."""
    try:
        profile = extract_user_profile(request.query)
        cleaned = profile.replace("```json", "").replace("```", "").strip()
        return {"status": "success", "profile": json.loads(cleaned)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "🎬 CineMind API is running! Use /recommend or /profile."}
