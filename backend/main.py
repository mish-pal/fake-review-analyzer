from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager

from backend.ml_engine import analyze_review, analyze_batch
from backend.database import init_db, add_review, get_recent_reviews
import requests
from bs4 import BeautifulSoup

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


class ReviewItem(BaseModel):
    id: int
    text: str
    ml_score: float

    created_at: str
    final_score: float

class BatchRequest(BaseModel):
    reviews: List[str]

class BatchItemResponse(BaseModel):
    text: str
    score: float
    status: str

class TopWordBatch(BaseModel):
    word: str
    contribution: float
    tfidf_sum: float

class BatchResponse(BaseModel):
    results: List[BatchItemResponse]
    metrics: Dict[str, Any]
    common_reasons: List[str] = []
    top_batch_words: List[TopWordBatch] = []

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

@app.post("/analyze_batch", response_model=BatchResponse)
async def analyze_batch_endpoint(req: BatchRequest):
    if not req.reviews:
        raise HTTPException(status_code=400, detail="Reviews list cannot be empty.")
    
    try:
        result = analyze_batch(req.reviews)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/scrape")
async def scrape_endpoint(url: str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US, en;q=0.5',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.content, 'html.parser')
        
        review_elements = soup.find_all('span', {'data-hook': 'review-body'})
        reviews = []
        for el in review_elements:
            text = el.get_text(separator=' ', strip=True)
            if text and len(text) > 20:
                reviews.append(text)
                
        # Fallback to paragraph extraction if standard review blocks not found
        if not reviews:
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text.split()) > 15:
                    reviews.append(text)
                    
        return {"url": url, "reviews": reviews[:20]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
