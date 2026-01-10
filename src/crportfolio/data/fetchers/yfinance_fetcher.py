"""
Yahoo Finance data fetcher implementation.
"""

import logging
import random
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from .base import BaseFetcher

logger = logging.getLogger(__name__)


# Special ticker mappings for Yahoo Finance
YFINANCE_TICKER_MAPPING = {
    "SUPER": "SUPER8290-USD",
    "PRIME": "PRIME23711-USD",
    "TIA": "TIA22861-USD",
    "FORT": "FORT20622-USD",
    "JUP": "JUP29210-USD",
    "SUI": "SUI20947-USD",
    "APT": "APT21794-USD",
    "BANANA": "BANANA28066-USD",
}


class YFinanceFetcher(BaseFetcher):
    """Fetch cryptocurrency data from Yahoo Finance."""

    source_name = "yfinance"
    rate_limit_delay = 1.5  # 1.5 seconds between requests

    def __init__(self, max_retries: int = 3, base_retry_delay: float = 2.0):
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self._last_request_time = 0

    def _get_ticker(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance ticker format."""
        # Check for special mappings first
        if symbol in YFINANCE_TICKER_MAPPING:
            return YFINANCE_TICKER_MAPPING[symbol]
        # Default format: SYMBOL-USD
        return f"{symbol}-USD"

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed + random.uniform(0, 0.5)
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def fetch_ohlc(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLC data from Yahoo Finance.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            start_date: Start date (None = max history)
            end_date: End date (None = today)

        Returns:
            DataFrame with OHLC data
        """
        ticker = self._get_ticker(symbol)

        for retry in range(self.max_retries):
            try:
                self._rate_limit()

                # Determine period or date range
                if start_date is None:
                    # Fetch maximum history
                    df = yf.download(ticker, period="max", progress=False)
                else:
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = (
                        end_date.strftime("%Y-%m-%d")
                        if end_date
                        else datetime.now().strftime("%Y-%m-%d")
                    )
                    df = yf.download(ticker, start=start_str, end=end_str, progress=False)

                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Validate data
                if df.empty or len(df) < 1:
                    raise ValueError(f"No data returned for {ticker}")

                if "Close" not in df.columns or df["Close"].isnull().all():
                    raise ValueError(f"Invalid Close data for {ticker}")

                logger.info(f"YFinance: Fetched {len(df)} rows for {symbol}")
                return self.normalize_dataframe(df)

            except Exception as e:
                error_msg = str(e)
                is_rate_limit = "rate limit" in error_msg.lower()

                if retry < self.max_retries - 1:
                    wait_time = self.base_retry_delay * (2**retry) + random.uniform(0, 1)
                    if is_rate_limit:
                        logger.warning(
                            f"YFinance rate limit for {symbol}, retry {retry+1}/{self.max_retries} "
                            f"in {wait_time:.1f}s"
                        )
                    else:
                        logger.warning(
                            f"YFinance error for {symbol}: {error_msg}, "
                            f"retry {retry+1}/{self.max_retries}"
                        )
                    time.sleep(wait_time)
                else:
                    logger.error(f"YFinance failed for {symbol} after {self.max_retries} retries: {e}")
                    raise

        return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get current price from Yahoo Finance."""
        ticker = self._get_ticker(symbol)
        self._rate_limit()

        try:
            # Fetch last 5 days to ensure we get data
            df = yf.download(ticker, period="5d", progress=False)

            # Handle multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or "Close" not in df.columns:
                raise ValueError(f"No price data for {ticker}")

            return float(df["Close"].iloc[-1])
        except Exception as e:
            logger.error(f"YFinance current price failed for {symbol}: {e}")
            raise
