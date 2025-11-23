import sqlite3
import time
import logging
from src.config import DB_NAME

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Professional database manager for trading data"""

    def __init__(self, db_name: str = DB_NAME):
        self.db_name = db_name
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Trading results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_results (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    symbol TEXT,
                    signal_type TEXT,
                    confidence REAL,
                    entry_price REAL,
                    exit_price REAL,
                    result INTEGER,
                    pnl_pips REAL,
                    strategy TEXT
                )
            ''')

            # Pattern learning table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_learning (
                    pattern_id TEXT PRIMARY KEY,
                    win_count INTEGER DEFAULT 0,
                    trade_count INTEGER DEFAULT 0,
                    last_updated REAL
                )
            ''')

            conn.commit()
            conn.close()
            logger.info("✅ Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def log_trade(self, trade_id, symbol, signal_type, confidence, entry, exit_p, won, pips, strategy):
        """Helper to log a completed trade"""
        try:
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trading_results (
                    id, timestamp, symbol, signal_type, confidence,
                    entry_price, exit_price, result, pnl_pips, strategy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id, time.time(), symbol, signal_type,
                confidence, entry, exit_p, 1 if won else 0,
                pips, strategy
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
