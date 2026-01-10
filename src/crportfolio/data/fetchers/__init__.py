"""
Data fetchers for various crypto price sources.

Each fetcher implements the BaseFetcher interface and provides:
- fetch_ohlc(): Fetch OHLC historical data
- get_current_price(): Fetch current price
- Rate limiting and error handling
"""

from .base import BaseFetcher
from .yfinance_fetcher import YFinanceFetcher
from .coingecko_fetcher import CoinGeckoFetcher
from .cmc_scraper import CoinMarketCapScraper

__all__ = [
    "BaseFetcher",
    "YFinanceFetcher",
    "CoinGeckoFetcher",
    "CoinMarketCapScraper",
]
