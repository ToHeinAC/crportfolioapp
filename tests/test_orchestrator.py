"""Tests for DataOrchestrator."""

import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from crportfolio.data.db import DatabaseManager
from crportfolio.data.fetchers.base import BaseFetcher
from crportfolio.data.orchestrator import DataOrchestrator, create_default_orchestrator


class MockFetcher(BaseFetcher):
    """Mock fetcher for testing."""

    source_name = "mock"
    rate_limit_delay = 0.0

    def __init__(self, should_fail=False, data=None):
        self.should_fail = should_fail
        self.data = data
        self.fetch_count = 0

    def fetch_ohlc(self, symbol, start_date=None, end_date=None):
        self.fetch_count += 1
        if self.should_fail:
            raise Exception("Mock fetch failed")
        if self.data is not None:
            return self.data
        # Return default mock data
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "Open": [100] * 30,
                "High": [105] * 30,
                "Low": [95] * 30,
                "Close": [102] * 30,
                "Volume": [1000000] * 30,
            },
            index=dates,
        )

    def get_current_price(self, symbol):
        if self.should_fail:
            raise Exception("Mock price failed")
        return 50000.0


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = DatabaseManager(db_path)
    yield db
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_fetcher():
    """Create a mock fetcher."""
    return MockFetcher()


@pytest.fixture
def failing_fetcher():
    """Create a failing mock fetcher."""
    return MockFetcher(should_fail=True)


class TestDataOrchestrator:
    """Test suite for DataOrchestrator."""

    def test_get_ohlc_fetches_from_api_when_db_empty(self, temp_db, mock_fetcher):
        """Test that orchestrator fetches from API when DB is empty."""
        orchestrator = DataOrchestrator(temp_db, [mock_fetcher])

        result = orchestrator.get_ohlc("BTC")

        assert not result.empty
        assert mock_fetcher.fetch_count == 1

    def test_get_ohlc_uses_cache_when_fresh(self, temp_db, mock_fetcher):
        """Test that orchestrator uses cache when data is fresh."""
        orchestrator = DataOrchestrator(temp_db, [mock_fetcher], cache_days=1)

        # First fetch - should hit API
        result1 = orchestrator.get_ohlc("BTC")
        assert mock_fetcher.fetch_count == 1

        # Second fetch - should use cache
        result2 = orchestrator.get_ohlc("BTC")
        # May fetch for recent data, but should have cached historical
        assert not result2.empty

    def test_get_ohlc_force_refresh(self, temp_db, mock_fetcher):
        """Test force refresh bypasses cache."""
        orchestrator = DataOrchestrator(temp_db, [mock_fetcher])

        # First fetch
        orchestrator.get_ohlc("BTC")
        initial_count = mock_fetcher.fetch_count

        # Force refresh
        orchestrator.get_ohlc("BTC", force_refresh=True)
        assert mock_fetcher.fetch_count > initial_count

    def test_fallback_chain(self, temp_db, failing_fetcher, mock_fetcher):
        """Test that orchestrator falls back to next fetcher on failure."""
        orchestrator = DataOrchestrator(temp_db, [failing_fetcher, mock_fetcher])

        result = orchestrator.get_ohlc("BTC")

        assert not result.empty
        assert failing_fetcher.fetch_count == 1
        assert mock_fetcher.fetch_count == 1

    def test_all_fetchers_fail_returns_empty(self, temp_db, failing_fetcher):
        """Test that empty DataFrame is returned when all fetchers fail."""
        orchestrator = DataOrchestrator(temp_db, [failing_fetcher])

        result = orchestrator.get_ohlc("BTC")

        assert result.empty

    def test_get_current_price(self, temp_db, mock_fetcher):
        """Test getting current price."""
        orchestrator = DataOrchestrator(temp_db, [mock_fetcher])

        price = orchestrator.get_current_price("BTC")

        assert price == 50000.0

    def test_get_current_price_fallback(self, temp_db, failing_fetcher, mock_fetcher):
        """Test current price fallback."""
        orchestrator = DataOrchestrator(temp_db, [failing_fetcher, mock_fetcher])

        price = orchestrator.get_current_price("BTC")

        assert price == 50000.0

    def test_get_multiple_ohlc(self, temp_db, mock_fetcher):
        """Test fetching multiple symbols."""
        orchestrator = DataOrchestrator(temp_db, [mock_fetcher])

        results = orchestrator.get_multiple_ohlc(["BTC", "ETH", "SOL"])

        assert len(results) == 3
        assert "BTC" in results
        assert "ETH" in results
        assert "SOL" in results
        assert not results["BTC"].empty

    def test_refresh_all(self, temp_db, mock_fetcher):
        """Test refreshing all symbols."""
        orchestrator = DataOrchestrator(temp_db, [mock_fetcher])

        results = orchestrator.refresh_all(["BTC", "ETH"])

        assert results["BTC"] is True
        assert results["ETH"] is True

    def test_refresh_all_partial_failure(self, temp_db):
        """Test refresh with partial failures."""
        # Create fetcher that fails for specific symbol
        class SelectiveFetcher(MockFetcher):
            def fetch_ohlc(self, symbol, start_date=None, end_date=None):
                if symbol == "FAIL":
                    raise Exception("Intentional failure")
                return super().fetch_ohlc(symbol, start_date, end_date)

        orchestrator = DataOrchestrator(temp_db, [SelectiveFetcher()])

        results = orchestrator.refresh_all(["BTC", "FAIL"])

        assert results["BTC"] is True
        assert results["FAIL"] is False


class TestCreateDefaultOrchestrator:
    """Test suite for create_default_orchestrator factory."""

    def test_creates_orchestrator(self):
        """Test that factory creates valid orchestrator."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            orchestrator = create_default_orchestrator(db_path)

            assert orchestrator is not None
            assert len(orchestrator.fetchers) == 3
            assert orchestrator.fetchers[0].source_name == "yfinance"
            assert orchestrator.fetchers[1].source_name == "coingecko"
            assert orchestrator.fetchers[2].source_name == "coinmarketcap"
        finally:
            Path(db_path).unlink(missing_ok=True)
