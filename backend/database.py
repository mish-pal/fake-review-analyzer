import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), 'reviews.db')

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                ml_score REAL NOT NULL,
                upvotes INTEGER DEFAULT 0,
                downvotes INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def add_review(text: str, ml_score: float) -> int:
    with get_db() as conn:
        cursor = conn.execute('''
            INSERT INTO reviews (text, ml_score)
            VALUES (?, ?)
        ''', (text, ml_score))
        conn.commit()
        return cursor.lastrowid

def setup_vote(review_id: int, vote: str) -> dict:
    with get_db() as conn:
        if vote == "up":
            conn.execute('UPDATE reviews SET upvotes = upvotes + 1 WHERE id = ?', (review_id,))
        elif vote == "down":
            conn.execute('UPDATE reviews SET downvotes = downvotes + 1 WHERE id = ?', (review_id,))
        else:
            raise ValueError("Vote must be 'up' or 'down'")
            
        cursor = conn.execute('SELECT upvotes, downvotes FROM reviews WHERE id = ?', (review_id,))
        row = cursor.fetchone()
        conn.commit()
        
        if row:
            return dict(row)
        else:
            raise ValueError("Review not found")

def get_recent_reviews(limit: int = 20):
    with get_db() as conn:
        cursor = conn.execute('''
            SELECT id, text, ml_score, upvotes, downvotes, created_at
            FROM reviews
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            
            # Calculate final score with vote adjustments
            ml_score = row_dict['ml_score']
            upvotes = row_dict['upvotes']
            downvotes = row_dict['downvotes']
            
            total_votes = upvotes + downvotes
            vote_score = (upvotes - downvotes) / (total_votes + 1)
            adjustment = vote_score * 10
            
            final_score = ml_score + adjustment
            final_score = max(0.0, min(100.0, final_score))
            
            row_dict['final_score'] = round(final_score, 2)
            results.append(row_dict)
            
        return results
