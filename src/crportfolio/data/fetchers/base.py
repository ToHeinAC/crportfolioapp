"""
Abstract base class for all data fetchers.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional
import pandas as pd


class BaseFetcher(ABC):
    """Abstract base class for cryptocurrency data fetchers."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this data source."""
        pass

    @property
    @abstractmethod
    def rate_limit_delay(self) -> float:
        """Minimum delay between requests in seconds."""
        pass

    @abstractmethod
    def fetch_ohlc(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLC data for a symbol within date range.

        Args:
            symbol: The crypto symbol (e.g., 'BTC', 'ETH')
            start_date: Start date for historical data (None = earliest available)
            end_date: End date for historical data (None = today)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index should be DatetimeIndex
        """
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Fetch current price for a symbol.

        Args:
            symbol: The crypto symbol (e.g., 'BTC', 'ETH')

        Returns:
            Current price in USD
        """
        pass

    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame to standard OHLC format.

        Ensures columns: Open, High, Low, Close, Volume
        Ensures DatetimeIndex
        """
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        # Ensure all required columns exist
        for col in required_columns:
            if col not in df.columns:
                if col == "Volume":
                    df[col] = 0
                elif col == "Open":
                    df[col] = df["Close"].shift(1).fillna(df["Close"])
                elif col == "High":
                    df[col] = df["Close"] * 1.005
                elif col == "Low":
                    df[col] = df["Close"] * 0.995

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Select only required columns in order
        return df[required_columns]
