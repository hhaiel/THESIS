import os
import sqlite3
import pandas as pd
from pathlib import Path
import streamlit as st
from typing import Optional, List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDatabase:
    def __init__(self, db_path: str = "data/sentiment_corrections.db"):
        """Initialize the database connection."""
        self.db_path = db_path
        self._ensure_db_directory()
        self._init_db()

    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _init_db(self):
        """Initialize the database and create tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                # Create sentiment corrections table
                c.execute('''
                    CREATE TABLE IF NOT EXISTS sentiment_corrections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        comment TEXT NOT NULL,
                        corrected_sentiment TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        language TEXT,
                        confidence FLOAT DEFAULT 1.0
                    )
                ''')
                # Create index for faster queries
                c.execute('''
                    CREATE INDEX IF NOT EXISTS idx_comment 
                    ON sentiment_corrections(comment)
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def save_correction(self, comment: str, corrected_sentiment: str, 
                       language: Optional[str] = None, confidence: float = 1.0) -> bool:
        """
        Save a sentiment correction to the database.
        
        Args:
            comment: The original comment text
            corrected_sentiment: The corrected sentiment value
            language: Optional language of the comment
            confidence: Confidence score for the correction (default: 1.0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO sentiment_corrections 
                    (comment, corrected_sentiment, language, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (comment, corrected_sentiment, language, confidence))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving correction: {e}")
            return False

    def get_corrections(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve sentiment corrections from the database.
        
        Args:
            limit: Optional limit on number of corrections to retrieve
            
        Returns:
            DataFrame containing the corrections
        """
        try:
            query = "SELECT * FROM sentiment_corrections ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error retrieving corrections: {e}")
            return pd.DataFrame()

    def get_correction_count(self) -> int:
        """Get the total number of corrections in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM sentiment_corrections")
                return c.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting correction count: {e}")
            return 0

    def export_to_csv(self, output_path: str) -> bool:
        """
        Export all corrections to a CSV file.
        
        Args:
            output_path: Path where the CSV file should be saved
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df = self.get_corrections()
            df.to_csv(output_path, index=False)
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def backup_database(self, backup_path: str) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path where the backup should be saved
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as source:
                with sqlite3.connect(backup_path) as dest:
                    source.backup(dest)
            return True
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False

    def clear_database(self) -> bool:
        """Clear all records from the sentiment_corrections table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("DELETE FROM sentiment_corrections")
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

# Initialize global database instance
db = SentimentDatabase()

# Streamlit session state initialization
def init_session_state():
    """Initialize Streamlit session state for sentiment corrections."""
    if 'sentiment_corrections' not in st.session_state:
        st.session_state.sentiment_corrections = db.get_corrections() 