"""
Data orchestrator for smart data fetching.

Implements a DB-first approach:
1. Check database for existing data
2. Determine missing date ranges
3. Fetch only missing data from APIs
4. Store new data in database
5. Return complete dataset
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from .db import DatabaseManager
from .fetchers.base import BaseFetcher

logger = logging.getLogger(__name__)


class DataOrchestrator:
    """
    Smart data fetching orchestrator.

    Uses database as cache and only fetches missing data from APIs.
    Falls back through multiple data sources if primary fails.
    """

    def __init__(
        self,
        db: DatabaseManager,
        fetchers: list[BaseFetcher],
        cache_days: int = 1,
    ):
        """
        Initialize orchestrator.

        Args:
            db: DatabaseManager instance
            fetchers: List of fetchers in priority order (first = highest priority)
            cache_days: Consider data stale after this many days
        """
        self.db = db
        self.fetchers = fetchers
        self.cache_days = cache_days

    def get_ohlc(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get OHLC data for a symbol.

        Smart fetching:
        - Uses DB for historical data
        - Only fetches new data from APIs
        - Falls back through fetcher chain on failure

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            start_date: Start date (None = earliest available)
            end_date: End date (None = today)
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            DataFrame with OHLC data
        """
        if end_date is None:
            end_date = date.today()

        # Get existing data from database
        db_data = self.db.get_ohlc(symbol, start_date, end_date)

        # Determine what data we need to fetch
        if force_refresh or db_data.empty:
            # Fetch all data
            new_data = self._fetch_with_fallback(symbol, start_date, end_date)
            if not new_data.empty:
                return new_data
            return db_data  # Return empty or whatever we have

        # Check if we need to fetch new data
        latest_in_db = db_data.index.max().date() if not db_data.empty else None
        cache_threshold = date.today() - timedelta(days=self.cache_days)

        if latest_in_db and latest_in_db >= cache_threshold:
            # Data is fresh enough
            logger.info(f"Using cached data for {symbol} (latest: {latest_in_db})")
            return db_data

        # Fetch missing recent data
        fetch_start = latest_in_db + timedelta(days=1) if latest_in_db else start_date
        new_data = self._fetch_with_fallback(symbol, fetch_start, end_date)

        if new_data.empty:
            return db_data

        # Combine existing and new data
        combined = pd.concat([db_data, new_data])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)

        # Filter to requested date range
        if start_date:
            combined = combined[combined.index >= pd.Timestamp(start_date)]
        if end_date:
            combined = combined[combined.index <= pd.Timestamp(end_date)]

        return combined

    def _fetch_with_fallback(
        self,
        symbol: str,
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> pd.DataFrame:
        """
        Fetch data using fallback chain.

        Tries each fetcher in order until one succeeds.

        Args:
            symbol: Crypto symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLC data (empty if all fail)
        """
        for fetcher in self.fetchers:
            try:
                logger.info(f"Trying {fetcher.source_name} for {symbol}")
                data = fetcher.fetch_ohlc(symbol, start_date, end_date)

                if data.empty:
                    logger.warning(f"{fetcher.source_name} returned empty data for {symbol}")
                    continue

                # Store in database
                self.db.upsert_ohlc(symbol, data, fetcher.source_name)

                logger.info(
                    f"Successfully fetched {len(data)} rows for {symbol} "
                    f"from {fetcher.source_name}"
                )
                return data

            except Exception as e:
                logger.warning(f"{fetcher.source_name} failed for {symbol}: {e}")
                continue

        logger.error(f"All fetchers failed for {symbol}")
        return pd.DataFrame()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Crypto symbol

        Returns:
            Current price in USD, or None if all sources fail
        """
        for fetcher in self.fetchers:
            try:
                price = fetcher.get_current_price(symbol)
                logger.info(f"Got price ${price:.2f} for {symbol} from {fetcher.source_name}")
                return price
            except Exception as e:
                logger.warning(f"{fetcher.source_name} price failed for {symbol}: {e}")
                continue

        logger.error(f"Could not get current price for {symbol}")
        return None

    def get_multiple_ohlc(
        self,
        symbols: list[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        progress_callback: Optional[callable] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Get OHLC data for multiple symbols.

        Args:
            symbols: List of crypto symbols
            start_date: Start date
            end_date: End date
            progress_callback: Optional callback(symbol, index, total) for progress updates

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        results = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(symbol, i, total)

            try:
                data = self.get_ohlc(symbol, start_date, end_date)
                results[symbol] = data
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        return results

    def refresh_all(self, symbols: list[str]) -> dict[str, bool]:
        """
        Force refresh all data for given symbols.

        Args:
            symbols: List of symbols to refresh

        Returns:
            Dictionary mapping symbol -> success status
        """
        results = {}
        for symbol in symbols:
            try:
                data = self.get_ohlc(symbol, force_refresh=True)
                results[symbol] = not data.empty
            except Exception as e:
                logger.error(f"Failed to refresh {symbol}: {e}")
                results[symbol] = False
        return results


def create_default_orchestrator(db_path: str = "crypto_prices.db") -> DataOrchestrator:
    """
    Create an orchestrator with default configuration.

    Uses: YFinance -> CoinGecko -> CoinMarketCap scraper

    Args:
        db_path: Path to SQLite database

    Returns:
        Configured DataOrchestrator
    """
    from .fetchers import CoinGeckoFetcher, CoinMarketCapScraper, YFinanceFetcher

    db = DatabaseManager(db_path)
    fetchers = [
        YFinanceFetcher(),
        CoinGeckoFetcher(),
        CoinMarketCapScraper(),
    ]

    return DataOrchestrator(db, fetchers)
