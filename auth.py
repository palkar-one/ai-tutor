import sqlite3
from datetime import datetime

DB_NAME = "users.db"

# In auth.py
import json

def init_roadmap_table():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS roadmap_progress (
            username TEXT,
            domain TEXT,
            roadmap_json TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(username, domain)
        )
    """)
    conn.commit()
    conn.close()

def init_user_roadmaps_table():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS roadmaps (
            username TEXT,
            domain TEXT,
            roadmap TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_roadmap(username, domain, roadmap_data):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    roadmap_json = json.dumps(roadmap_data)
    c.execute("""
        INSERT OR REPLACE INTO roadmap_progress (username, domain, roadmap_json) 
        VALUES (?, ?, ?)
    """, (username, domain, roadmap_json))
    conn.commit()
    conn.close()


def load_roadmap(username, domain):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        SELECT roadmap_json FROM roadmap_progress
        WHERE username = ? AND domain = ?
    """, (username, domain))
    result = c.fetchone()
    conn.close()
    if result:
        return json.loads(result[0])
    return None



def save_user_roadmap(username, domain, roadmap):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            INSERT INTO roadmaps (username, domain, roadmap, timestamp)
            VALUES (?, ?, ?, ?)
        """, (username, domain, roadmap, timestamp))
        conn.commit()

def get_user_roadmaps(username):
    with sqlite3.connect(DB_NAME) as conn:
        rows = conn.execute("""
            SELECT domain, roadmap, timestamp FROM roadmaps 
            WHERE username = ? ORDER BY timestamp DESC
        """, (username,)).fetchall()
    return rows
