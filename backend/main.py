from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager

from backend.ml_engine import analyze_review
from backend.database import init_db, add_review, setup_vote, get_recent_reviews

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database on startup
    init_db()
    yield

app = FastAPI(title="Review Credibility Analyzer API", lifespan=lifespan)

class ReviewRequest(BaseModel):
    review: str

class TopWord(BaseModel):
    word: str
    contribution: float
    tfidf: float

class AnalysisResponse(BaseModel):
    review_id: int
    score: float
    status: str
    confidence: str
    behavior_adjustment: float
    top_words: List[TopWord]
    reasoning: List[str]

class VoteRequest(BaseModel):
    review_id: int
    vote: str

class ReviewItem(BaseModel):
    id: int
    text: str
    ml_score: float
    upvotes: int
    downvotes: int
    created_at: str
    final_score: float

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(req: ReviewRequest):
    if not req.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    
    try:
        result = analyze_review(req.review)
        
        # Save to database and retrieve id
        review_id = add_review(req.review, result['score'])
        
        # Append id directly returning identical schema matching AnalysisResponse
        result['review_id'] = review_id
        
        return result
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/vote")
async def vote_endpoint(req: VoteRequest):
    if req.vote not in ["up", "down"]:
        raise HTTPException(status_code=400, detail="Vote must be 'up' or 'down'.")
        
    try:
        updated_counts = setup_vote(req.review_id, req.vote)
        return {"status": "success", "counts": updated_counts}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/reviews", response_model=List[ReviewItem])
async def get_reviews_endpoint():
    try:
        reviews = get_recent_reviews(limit=20)
        return reviews
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "ML Review Analyzer Backend is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
