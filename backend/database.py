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



def get_recent_reviews(limit: int = 20):
    with get_db() as conn:
        cursor = conn.execute('''
            SELECT id, text, ml_score, created_at
            FROM reviews
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        results = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            
            # Use ml_score as final_score directly since voting is removed
            ml_score = row_dict['ml_score']
            
            final_score = ml_score
            final_score = max(0.0, min(100.0, final_score))
            
            row_dict['final_score'] = round(final_score, 2)
            results.append(row_dict)
            
        return results
