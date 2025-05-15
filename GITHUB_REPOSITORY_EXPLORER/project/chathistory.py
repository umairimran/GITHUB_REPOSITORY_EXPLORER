import sqlite3
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def init_db():
    """Initialize the SQLite database and create necessary tables."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         index_name TEXT NOT NULL,
         query TEXT NOT NULL,
         response TEXT NOT NULL,
         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def get_conversation_history(index_name, limit=5):
    """Get the last 'limit' conversations for a given index."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        SELECT query, response FROM conversations 
        WHERE index_name = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (index_name, limit))
    history = c.fetchall()
    conn.close()
    return history

def store_conversation(index_name, query, response):
    """Store a new conversation in the database."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO conversations (index_name, query, response)
        VALUES (?, ?, ?)
    ''', (index_name, query, response))
    conn.commit()
    conn.close()

def delete_conversation_history(index_name):
    """Delete all conversation history for a given index."""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('DELETE FROM conversations WHERE index_name = ?', (index_name,))
    conn.commit()
    conn.close()
    logger.info(f"Deleted conversation history for index: {index_name}") 