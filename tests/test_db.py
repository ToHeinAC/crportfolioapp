"""Tests for DatabaseManager."""

import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from crportfolio.data.db import DatabaseManager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = DatabaseManager(db_path)
    yield db
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_ohlc_df():
    """Create sample OHLC DataFrame."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "Open": [100 + i for i in range(30)],
            "High": [105 + i for i in range(30)],
            "Low": [95 + i for i in range(30)],
            "Close": [102 + i for i in range(30)],
            "Volume": [1000000 + i * 10000 for i in range(30)],
        },
        index=dates,
    )


class TestDatabaseManager:
    """Test suite for DatabaseManager."""

    def test_init_creates_tables(self, temp_db):
        """Test that initialization creates required tables."""
        with temp_db.engine.connect() as conn:
            from sqlalchemy import text

            # Check ohlc_history table exists
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='ohlc_history'")
            ).fetchone()
            assert result is not None

            # Check data_metadata table exists
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='data_metadata'")
            ).fetchone()
            assert result is not None

    def test_upsert_and_get_ohlc(self, temp_db, sample_ohlc_df):
        """Test storing and retrieving OHLC data."""
        temp_db.upsert_ohlc("BTC", sample_ohlc_df, "test_source")

        # Retrieve data
        result = temp_db.get_ohlc("BTC")

        assert not result.empty
        assert len(result) == 30
        assert "close" in result.columns
        assert result["close"].iloc[0] == 102

    def test_get_ohlc_with_date_range(self, temp_db, sample_ohlc_df):
        """Test retrieving OHLC data with date filters."""
        temp_db.upsert_ohlc("ETH", sample_ohlc_df, "test_source")

        start = date(2024, 1, 10)
        end = date(2024, 1, 20)
        result = temp_db.get_ohlc("ETH", start, end)

        assert not result.empty
        assert len(result) == 11  # 10th to 20th inclusive

    def test_get_ohlc_empty_for_unknown_symbol(self, temp_db):
        """Test that unknown symbol returns empty DataFrame."""
        result = temp_db.get_ohlc("UNKNOWN")
        assert result.empty

    def test_get_latest_date(self, temp_db, sample_ohlc_df):
        """Test getting latest date for a symbol."""
        temp_db.upsert_ohlc("SOL", sample_ohlc_df, "test_source")

        latest = temp_db.get_latest_date("SOL")
        assert latest == date(2024, 1, 30)

    def test_get_earliest_date(self, temp_db, sample_ohlc_df):
        """Test getting earliest date for a symbol."""
        temp_db.upsert_ohlc("SOL", sample_ohlc_df, "test_source")

        earliest = temp_db.get_earliest_date("SOL")
        assert earliest == date(2024, 1, 1)

    def test_get_symbols(self, temp_db, sample_ohlc_df):
        """Test getting list of all symbols."""
        temp_db.upsert_ohlc("BTC", sample_ohlc_df, "test_source")
        temp_db.upsert_ohlc("ETH", sample_ohlc_df, "test_source")

        symbols = temp_db.get_symbols()
        assert "BTC" in symbols
        assert "ETH" in symbols
        assert len(symbols) == 2

    def test_get_metadata(self, temp_db, sample_ohlc_df):
        """Test getting metadata for a symbol."""
        temp_db.upsert_ohlc("BTC", sample_ohlc_df, "yfinance")

        metadata = temp_db.get_metadata("BTC")
        assert metadata is not None
        assert metadata["symbol"] == "BTC"
        assert metadata["total_records"] == 30
        assert metadata["primary_source"] == "yfinance"

    def test_delete_symbol(self, temp_db, sample_ohlc_df):
        """Test deleting all data for a symbol."""
        temp_db.upsert_ohlc("BTC", sample_ohlc_df, "test_source")
        assert not temp_db.get_ohlc("BTC").empty

        temp_db.delete_symbol("BTC")
        assert temp_db.get_ohlc("BTC").empty
        assert temp_db.get_metadata("BTC") is None

    def test_upsert_updates_existing(self, temp_db, sample_ohlc_df):
        """Test that upsert updates existing records."""
        temp_db.upsert_ohlc("BTC", sample_ohlc_df, "source1")

        # Create updated data with different values
        updated_df = sample_ohlc_df.copy()
        updated_df["Close"] = [200 + i for i in range(30)]

        temp_db.upsert_ohlc("BTC", updated_df, "source2")

        result = temp_db.get_ohlc("BTC")
        assert result["close"].iloc[0] == 200  # Should be updated value
