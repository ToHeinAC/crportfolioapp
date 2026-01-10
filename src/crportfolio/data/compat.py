"""
Compatibility layer for integrating new data layer with existing Streamlit app.

Provides drop-in compatible functions that use the new orchestrator
while maintaining the same interface as the original get_data2() function.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

from .db import DatabaseManager
from .fetchers import CoinGeckoFetcher, CoinMarketCapScraper, YFinanceFetcher
from .orchestrator import DataOrchestrator

logger = logging.getLogger(__name__)

# Global orchestrator instance (lazy initialized)
_orchestrator: Optional[DataOrchestrator] = None
_db: Optional[DatabaseManager] = None


def get_orchestrator(db_path: str = "crypto_prices.db") -> DataOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator, _db

    if _orchestrator is None:
        _db = DatabaseManager(db_path)
        fetchers = [
            YFinanceFetcher(),
            CoinGeckoFetcher(),
            CoinMarketCapScraper(),
        ]
        _orchestrator = DataOrchestrator(_db, fetchers)
        logger.info(f"Initialized data orchestrator with DB at {db_path}")

    return _orchestrator


def get_database() -> Optional[DatabaseManager]:
    """Get the database manager instance."""
    return _db


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_crypto_data(
    pairs: list[str],
    period: str = "max",
    show_progress: bool = True,
    debug_mode: bool = False,
) -> list[pd.DataFrame]:
    """
    Fetch cryptocurrency OHLC data for multiple pairs.

    Drop-in replacement for the original get_data2() function.
    Uses the new orchestrator with DB caching.

    Args:
        pairs: List of trading pairs (e.g., ['BTC-USD', 'ETH-USD'])
        period: Time period ('max', '1y', '6m', '3m', '1m', '7d')
        show_progress: Whether to show Streamlit progress bar
        debug_mode: Whether to show debug information

    Returns:
        List of DataFrames with OHLC data, one per pair
    """
    orchestrator = get_orchestrator()
    data = []
    debug_info = []

    # Parse period to date range
    end_date = date.today()
    if period == "max":
        start_date = None  # Fetch all available
    elif period == "1y":
        start_date = end_date - timedelta(days=365)
    elif period == "6m":
        start_date = end_date - timedelta(days=180)
    elif period == "3m":
        start_date = end_date - timedelta(days=90)
    elif period == "1m":
        start_date = end_date - timedelta(days=30)
    elif period == "7d":
        start_date = end_date - timedelta(days=7)
    else:
        start_date = None

    # Progress bar
    if show_progress:
        progress_text = "Fetching data..."
        my_bar = st.progress(0.0, text=progress_text)

    total = len(pairs)

    for i, pair in enumerate(pairs):
        # Extract symbol from pair (e.g., 'BTC-USD' -> 'BTC')
        symbol = pair.split("-")[0]
        # Remove any trailing numbers (e.g., 'SUPER8290' -> 'SUPER')
        clean_symbol = "".join([c for c in symbol if not c.isdigit()])

        item_debug = {"pair": pair, "symbol": clean_symbol, "success": False}

        try:
            df = orchestrator.get_ohlc(clean_symbol, start_date, end_date)

            if not df.empty:
                # Normalize column names to match legacy format (uppercase)
                column_mapping = {
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
                df.rename(columns=column_mapping, inplace=True)
                
                # Ensure we have standard column names
                if "Adj Close" not in df.columns and "Close" in df.columns:
                    df["Adj Close"] = df["Close"]

                item_debug["success"] = True
                item_debug["data_points"] = len(df)
                item_debug["date_range"] = f"{df.index.min()} to {df.index.max()}"

                if debug_mode:
                    metadata = _db.get_metadata(clean_symbol) if _db else None
                    if metadata:
                        item_debug["source"] = metadata.get("primary_source", "unknown")

                if show_progress:
                    st.success(f"Fetched {len(df)} rows for {pair}")
            else:
                item_debug["error"] = "Empty data returned"
                if show_progress:
                    st.warning(f"No data available for {pair}")

            data.append(df)

        except Exception as e:
            logger.error(f"Failed to fetch {pair}: {e}")
            item_debug["error"] = str(e)
            if show_progress:
                st.error(f"Error fetching {pair}: {e}")
            # Append empty DataFrame to maintain index alignment
            data.append(pd.DataFrame())

        debug_info.append(item_debug)

        # Update progress
        if show_progress:
            percent = (i + 1) / total
            my_bar.progress(percent, text=f"{progress_text} {int(percent * 100)}%")

    if show_progress:
        my_bar.progress(1.0, text="Finished fetching data")

    # Show debug info if enabled
    if debug_mode:
        with st.expander("Data Retrieval Debug Info"):
            st.json(debug_info)

    return data


def get_current_prices(symbols: list[str]) -> dict[str, float]:
    """
    Get current prices for multiple symbols.

    Args:
        symbols: List of symbols (e.g., ['BTC', 'ETH'])

    Returns:
        Dictionary mapping symbol -> price
    """
    orchestrator = get_orchestrator()
    prices = {}

    for symbol in symbols:
        try:
            price = orchestrator.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
        except Exception as e:
            logger.warning(f"Could not get price for {symbol}: {e}")

    return prices


def refresh_data(symbols: list[str]) -> dict[str, bool]:
    """
    Force refresh data for symbols.

    Args:
        symbols: List of symbols to refresh

    Returns:
        Dictionary mapping symbol -> success status
    """
    orchestrator = get_orchestrator()
    return orchestrator.refresh_all(symbols)


def get_db_stats() -> dict:
    """Get database statistics."""
    if _db is None:
        return {"initialized": False}

    symbols = _db.get_symbols()
    stats = {
        "initialized": True,
        "db_path": str(_db.db_path),
        "total_symbols": len(symbols),
        "symbols": symbols,
    }

    return stats
