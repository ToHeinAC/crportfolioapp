"""Tests for data fetchers."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from crportfolio.data.fetchers import (
    BaseFetcher,
    CoinGeckoFetcher,
    CoinMarketCapScraper,
    YFinanceFetcher,
)
from crportfolio.data.fetchers.coingecko_fetcher import COINGECKO_ID_MAPPING
from crportfolio.data.fetchers.yfinance_fetcher import YFINANCE_TICKER_MAPPING


class TestYFinanceFetcher:
    """Test suite for YFinanceFetcher."""

    def test_get_ticker_default(self):
        """Test default ticker format."""
        fetcher = YFinanceFetcher()
        assert fetcher._get_ticker("BTC") == "BTC-USD"
        assert fetcher._get_ticker("ETH") == "ETH-USD"

    def test_get_ticker_special_mapping(self):
        """Test special ticker mappings."""
        fetcher = YFinanceFetcher()
        for symbol, expected in YFINANCE_TICKER_MAPPING.items():
            assert fetcher._get_ticker(symbol) == expected

    def test_source_name(self):
        """Test source name property."""
        fetcher = YFinanceFetcher()
        assert fetcher.source_name == "yfinance"

    def test_rate_limit_delay(self):
        """Test rate limit delay property."""
        fetcher = YFinanceFetcher()
        assert fetcher.rate_limit_delay == 1.5

    @patch("crportfolio.data.fetchers.yfinance_fetcher.yf.download")
    def test_fetch_ohlc_success(self, mock_download):
        """Test successful OHLC fetch."""
        # Create mock data
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        mock_df = pd.DataFrame(
            {
                "Open": [100] * 10,
                "High": [105] * 10,
                "Low": [95] * 10,
                "Close": [102] * 10,
                "Volume": [1000000] * 10,
            },
            index=dates,
        )
        mock_download.return_value = mock_df

        fetcher = YFinanceFetcher()
        result = fetcher.fetch_ohlc("BTC")

        assert not result.empty
        assert "Open" in result.columns
        assert "Close" in result.columns
        mock_download.assert_called_once()

    @patch("crportfolio.data.fetchers.yfinance_fetcher.yf.download")
    def test_fetch_ohlc_empty_raises(self, mock_download):
        """Test that empty data raises ValueError."""
        mock_download.return_value = pd.DataFrame()

        fetcher = YFinanceFetcher()
        with pytest.raises(ValueError, match="No data returned"):
            fetcher.fetch_ohlc("INVALID")


class TestCoinGeckoFetcher:
    """Test suite for CoinGeckoFetcher."""

    def test_get_coin_id(self):
        """Test symbol to CoinGecko ID mapping."""
        fetcher = CoinGeckoFetcher()
        assert fetcher._get_coin_id("BTC") == "bitcoin"
        assert fetcher._get_coin_id("ETH") == "ethereum"
        assert fetcher._get_coin_id("SOL") == "solana"

    def test_get_coin_id_cleans_numbers(self):
        """Test that numbers are stripped from symbols."""
        fetcher = CoinGeckoFetcher()
        # SUPER8290 should map to SUPER -> superfarm
        assert fetcher._get_coin_id("SUPER8290") == "superfarm"

    def test_get_coin_id_unknown(self):
        """Test unknown symbol returns None."""
        fetcher = CoinGeckoFetcher()
        assert fetcher._get_coin_id("UNKNOWN123") is None

    def test_source_name(self):
        """Test source name property."""
        fetcher = CoinGeckoFetcher()
        assert fetcher.source_name == "coingecko"

    def test_rate_limit_delay(self):
        """Test rate limit delay property."""
        fetcher = CoinGeckoFetcher()
        assert fetcher.rate_limit_delay == 2.0


class TestCoinMarketCapScraper:
    """Test suite for CoinMarketCapScraper."""

    def test_get_slug(self):
        """Test symbol to CMC slug mapping."""
        scraper = CoinMarketCapScraper()
        assert scraper._get_slug("BTC") == "bitcoin"
        assert scraper._get_slug("ETH") == "ethereum"

    def test_source_name(self):
        """Test source name property."""
        scraper = CoinMarketCapScraper()
        assert scraper.source_name == "coinmarketcap"

    def test_rate_limit_delay(self):
        """Test rate limit delay is conservative."""
        scraper = CoinMarketCapScraper()
        assert scraper.rate_limit_delay >= 5.0

    def test_create_price_dataframe(self):
        """Test price DataFrame creation."""
        scraper = CoinMarketCapScraper()
        df = scraper._create_price_dataframe(50000.0)

        assert not df.empty
        assert len(df) == 31  # 30 days + today
        assert df["Close"].iloc[-1] == 50000.0
        assert "Open" in df.columns
        assert "High" in df.columns
        assert "Low" in df.columns


class TestBaseFetcher:
    """Test suite for BaseFetcher normalization."""

    def test_normalize_dataframe_adds_missing_columns(self):
        """Test that normalize adds missing columns."""
        # Create a concrete implementation for testing
        fetcher = YFinanceFetcher()

        # DataFrame with only Close
        df = pd.DataFrame(
            {"Close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3),
        )

        result = fetcher.normalize_dataframe(df)

        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Volume" in result.columns

    def test_normalize_dataframe_preserves_existing(self):
        """Test that normalize preserves existing columns."""
        fetcher = YFinanceFetcher()

        df = pd.DataFrame(
            {
                "Open": [99, 100, 101],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [100, 101, 102],
                "Volume": [1000, 2000, 3000],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        result = fetcher.normalize_dataframe(df)

        assert result["Open"].tolist() == [99, 100, 101]
        assert result["Volume"].tolist() == [1000, 2000, 3000]
