"""
Database manager for OHLC historical data storage.

Uses SQLite for local storage of cryptocurrency price data.
"""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manage SQLite database for OHLC price data.

    Schema:
        ohlc_history: symbol, date, open, high, low, close, volume, source, created_at
    """

    def __init__(self, db_path: str = "crypto_prices.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.engine = create_engine(f"sqlite:///{db_path}")
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema if not exists."""
        with self.engine.connect() as conn:
            # Main OHLC history table
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS ohlc_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL NOT NULL,
                    volume REAL,
                    source VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            """
                )
            )

            # Index for efficient queries
            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_date
                ON ohlc_history(symbol, date DESC)
            """
                )
            )

            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_ohlc_date
                ON ohlc_history(date)
            """
                )
            )

            # Metadata table for tracking data freshness
            conn.execute(
                text(
                    """
                CREATE TABLE IF NOT EXISTS data_metadata (
                    symbol VARCHAR(20) PRIMARY KEY,
                    last_updated TIMESTAMP,
                    earliest_date DATE,
                    latest_date DATE,
                    total_records INTEGER,
                    primary_source VARCHAR(20)
                )
            """
                )
            )

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def get_ohlc(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Retrieve OHLC data from database.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            start_date: Start date (None = earliest available)
            end_date: End date (None = latest available)

        Returns:
            DataFrame with OHLC data, DatetimeIndex
        """
        conditions = ["symbol = :symbol"]
        params = {"symbol": symbol}

        if start_date:
            conditions.append("date >= :start")
            params["start"] = start_date.isoformat()

        if end_date:
            conditions.append("date <= :end")
            params["end"] = end_date.isoformat()

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT date, open, high, low, close, volume, source
            FROM ohlc_history
            WHERE {where_clause}
            ORDER BY date
        """

        df = pd.read_sql(text(query), self.engine, params=params)

        if df.empty:
            return df

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        return df

    def upsert_ohlc(self, symbol: str, df: pd.DataFrame, source: str):
        """
        Insert or update OHLC data.

        Args:
            symbol: Crypto symbol
            df: DataFrame with OHLC data (DatetimeIndex)
            source: Data source name (e.g., 'yfinance', 'coingecko')
        """
        if df.empty:
            return

        # Prepare data for insertion
        data = df.copy()
        data["symbol"] = symbol
        data["source"] = source
        data["date"] = data.index.date

        # Rename columns to match schema
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        data.rename(columns=column_mapping, inplace=True)

        # Select only needed columns
        columns = ["symbol", "date", "open", "high", "low", "close", "volume", "source"]
        data = data[[c for c in columns if c in data.columns]]

        # Use INSERT OR REPLACE for upsert
        with self.engine.connect() as conn:
            for _, row in data.iterrows():
                conn.execute(
                    text(
                        """
                    INSERT OR REPLACE INTO ohlc_history
                    (symbol, date, open, high, low, close, volume, source)
                    VALUES (:symbol, :date, :open, :high, :low, :close, :volume, :source)
                """
                    ),
                    {
                        "symbol": row["symbol"],
                        "date": str(row["date"]),
                        "open": row.get("open"),
                        "high": row.get("high"),
                        "low": row.get("low"),
                        "close": row["close"],
                        "volume": row.get("volume", 0),
                        "source": row["source"],
                    },
                )
            conn.commit()

        # Update metadata
        self._update_metadata(symbol, source)
        logger.info(f"Stored {len(data)} rows for {symbol} from {source}")

    def _update_metadata(self, symbol: str, source: str):
        """Update metadata table for a symbol."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                SELECT MIN(date), MAX(date), COUNT(*)
                FROM ohlc_history
                WHERE symbol = :symbol
            """
                ),
                {"symbol": symbol},
            ).fetchone()

            if result and result[2] > 0:
                conn.execute(
                    text(
                        """
                    INSERT OR REPLACE INTO data_metadata
                    (symbol, last_updated, earliest_date, latest_date, total_records, primary_source)
                    VALUES (:symbol, :updated, :earliest, :latest, :total, :source)
                """
                    ),
                    {
                        "symbol": symbol,
                        "updated": datetime.now().isoformat(),
                        "earliest": result[0],
                        "latest": result[1],
                        "total": result[2],
                        "source": source,
                    },
                )
                conn.commit()

    def get_latest_date(self, symbol: str) -> Optional[date]:
        """Get the most recent date for a symbol."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT MAX(date) FROM ohlc_history WHERE symbol = :symbol"),
                {"symbol": symbol},
            ).fetchone()

            if result and result[0]:
                return datetime.strptime(result[0], "%Y-%m-%d").date()
            return None

    def get_earliest_date(self, symbol: str) -> Optional[date]:
        """Get the earliest date for a symbol."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT MIN(date) FROM ohlc_history WHERE symbol = :symbol"),
                {"symbol": symbol},
            ).fetchone()

            if result and result[0]:
                return datetime.strptime(result[0], "%Y-%m-%d").date()
            return None

    def get_symbols(self) -> list[str]:
        """Get list of all symbols in database."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT DISTINCT symbol FROM ohlc_history ORDER BY symbol")
            ).fetchall()
            return [row[0] for row in result]

    def get_metadata(self, symbol: str) -> Optional[dict]:
        """Get metadata for a symbol."""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT * FROM data_metadata WHERE symbol = :symbol"),
                {"symbol": symbol},
            ).fetchone()

            if result:
                return {
                    "symbol": result[0],
                    "last_updated": result[1],
                    "earliest_date": result[2],
                    "latest_date": result[3],
                    "total_records": result[4],
                    "primary_source": result[5],
                }
            return None

    def delete_symbol(self, symbol: str):
        """Delete all data for a symbol."""
        with self.engine.connect() as conn:
            conn.execute(
                text("DELETE FROM ohlc_history WHERE symbol = :symbol"),
                {"symbol": symbol},
            )
            conn.execute(
                text("DELETE FROM data_metadata WHERE symbol = :symbol"),
                {"symbol": symbol},
            )
            conn.commit()
            logger.info(f"Deleted all data for {symbol}")

    def vacuum(self):
        """Optimize database by running VACUUM."""
        with self.engine.connect() as conn:
            conn.execute(text("VACUUM"))
            logger.info("Database vacuumed")
