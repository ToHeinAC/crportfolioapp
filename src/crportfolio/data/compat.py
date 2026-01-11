"""
Compatibility layer for integrating new data layer with existing Streamlit app.

Provides drop-in compatible functions that use the new orchestrator
while maintaining the same interface as the original get_data2() function.

Supports both local (SQLite caching) and cloud (in-memory caching) deployments.
"""

import logging
import os
import time
import random
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

from .db import DatabaseManager
from .fetchers import CoinGeckoFetcher, CoinMarketCapScraper, YFinanceFetcher
from .orchestrator import DataOrchestrator

logger = logging.getLogger(__name__)

# Constants for rate limiting
MAX_RETRIES = 3
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 30.0  # seconds


def is_cloud_deployment() -> bool:
    """
    Detect if running on Streamlit Cloud.
    
    Streamlit Cloud sets specific environment variables that we can check.
    """
    indicators = [
        os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit.io",
        os.environ.get("IS_STREAMLIT_CLOUD") == "true",
        # Streamlit Cloud runs in a specific environment
        os.environ.get("HOME", "").startswith("/home/appuser"),
    ]
    return any(indicators)


def is_sqlite_available() -> bool:
    """Check if SQLite persistent storage is available and writable."""
    if is_cloud_deployment():
        return False
    
    try:
        test_path = "crypto_prices.db"
        if os.path.exists(test_path):
            return os.access(test_path, os.W_OK)
        else:
            try:
                with open(test_path, 'a'):
                    pass
                return True
            except (IOError, OSError):
                return False
    except Exception:
        return False


def get_default_period() -> str:
    """
    Get default data fetch period based on deployment environment.
    
    Cloud: Shorter period to reduce API calls and memory usage.
    Local: Full history for comprehensive analysis.
    """
    if is_cloud_deployment():
        return "1y"  # 1 year for cloud to reduce load
    return "max"  # Full history for local


@st.cache_resource
def _get_database(db_path: str = "crypto_prices.db") -> Optional[DatabaseManager]:
    """
    Get cached database connection.
    
    Uses @st.cache_resource to ensure single connection across reruns.
    Returns None if SQLite is not available (e.g., on Streamlit Cloud).
    """
    if not is_sqlite_available():
        logger.info("SQLite not available (cloud deployment), using in-memory caching only")
        return None
    
    try:
        db = DatabaseManager(db_path)
        logger.info(f"Database connection established at {db_path}")
        return db
    except Exception as e:
        logger.warning(f"Failed to initialize database: {e}")
        return None


@st.cache_resource
def _get_fetchers() -> list:
    """Get cached fetcher instances."""
    return [
        YFinanceFetcher(),
        CoinGeckoFetcher(),
        CoinMarketCapScraper(),
    ]


@st.cache_resource
def _get_orchestrator(db_path: str = "crypto_prices.db") -> Optional[DataOrchestrator]:
    """
    Get or create the cached orchestrator instance.
    
    Uses @st.cache_resource to ensure single instance across reruns.
    Returns None if database is not available (graceful degradation).
    """
    db = _get_database(db_path)
    fetchers = _get_fetchers()
    
    if db is None:
        logger.info("Running in cloud mode without persistent database")
        return None
    
    orchestrator = DataOrchestrator(db, fetchers)
    logger.info(f"Initialized data orchestrator with DB at {db_path}")
    return orchestrator


def get_orchestrator(db_path: str = "crypto_prices.db") -> Optional[DataOrchestrator]:
    """Public accessor for orchestrator."""
    return _get_orchestrator(db_path)


def get_database() -> Optional[DatabaseManager]:
    """Public accessor for database."""
    return _get_database()


def _fetch_with_retry(
    fetcher,
    symbol: str,
    start_date: Optional[date],
    end_date: Optional[date],
) -> pd.DataFrame:
    """
    Fetch data with exponential backoff retry logic.
    
    Handles rate limiting gracefully.
    """
    for attempt in range(MAX_RETRIES):
        try:
            df = fetcher.fetch_ohlc(symbol, start_date, end_date)
            return df
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(x in error_str for x in ['rate', 'limit', '429', 'too many'])
            
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                logger.warning(f"Rate limit hit for {symbol}, retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                raise
    
    return pd.DataFrame()


def _fetch_single_symbol_cloud(
    symbol: str,
    start_date: Optional[date],
    end_date: Optional[date],
) -> pd.DataFrame:
    """
    Fetch data for a single symbol in cloud mode (no DB caching).
    
    Uses fetcher chain with fallback.
    """
    fetchers = _get_fetchers()
    
    for fetcher in fetchers:
        try:
            df = _fetch_with_retry(fetcher, symbol, start_date, end_date)
            if not df.empty:
                logger.info(f"Fetched {len(df)} rows for {symbol} from {fetcher.source_name}")
                return df
        except Exception as e:
            logger.warning(f"{fetcher.source_name} failed for {symbol}: {e}")
            continue
    
    logger.error(f"All fetchers failed for {symbol}")
    return pd.DataFrame()


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to match legacy format (uppercase)."""
    if df.empty:
        return df
    
    column_mapping = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=column_mapping)
    
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    
    return df


@st.cache_data(show_spinner=False, ttl=3600, max_entries=100)
def _fetch_single_cached(
    pair: str,
    start_date_str: Optional[str],
    end_date_str: Optional[str],
) -> pd.DataFrame:
    """
    Cached fetch for a single pair (cloud mode).
    
    Uses string dates for cache key compatibility.
    """
    symbol = pair.split("-")[0]
    clean_symbol = "".join([c for c in symbol if not c.isdigit()])
    
    start_date = date.fromisoformat(start_date_str) if start_date_str else None
    end_date = date.fromisoformat(end_date_str) if end_date_str else date.today()
    
    df = _fetch_single_symbol_cloud(clean_symbol, start_date, end_date)
    return _normalize_dataframe(df)


def fetch_crypto_data(
    pairs: list[str],
    period: str = "max",
    show_progress: bool = True,
    debug_mode: bool = False,
) -> list[pd.DataFrame]:
    """
    Fetch cryptocurrency OHLC data for multiple pairs.

    Drop-in replacement for the original get_data2() function.
    
    Behavior:
    - Local: Uses SQLite DB caching via orchestrator
    - Cloud: Uses st.cache_data for in-memory caching

    Args:
        pairs: List of trading pairs (e.g., ['BTC-USD', 'ETH-USD'])
        period: Time period ('max', '1y', '6m', '3m', '1m', '7d')
        show_progress: Whether to show Streamlit progress bar
        debug_mode: Whether to show debug information

    Returns:
        List of DataFrames with OHLC data, one per pair
    """
    # Adjust period for cloud deployment
    if is_cloud_deployment() and period == "max":
        period = get_default_period()
        logger.info(f"Cloud deployment: adjusted period from 'max' to '{period}'")
    
    # Parse period to date range
    end_date = date.today()
    period_days = {
        "max": None,
        "1y": 365,
        "6m": 180,
        "3m": 90,
        "1m": 30,
        "7d": 7,
    }
    days = period_days.get(period)
    start_date = end_date - timedelta(days=days) if days else None
    
    # Convert to strings for cache key
    start_date_str = start_date.isoformat() if start_date else None
    end_date_str = end_date.isoformat()
    
    orchestrator = _get_orchestrator()
    data = []
    debug_info = []
    
    # Progress bar
    if show_progress:
        progress_text = "Fetching data..."
        my_bar = st.progress(0.0, text=progress_text)
    
    total = len(pairs)
    cloud_mode = orchestrator is None
    
    if cloud_mode and debug_mode:
        st.info("🌐 Running in cloud mode (no persistent database)")
    
    for i, pair in enumerate(pairs):
        symbol = pair.split("-")[0]
        clean_symbol = "".join([c for c in symbol if not c.isdigit()])
        
        item_debug = {"pair": pair, "symbol": clean_symbol, "success": False, "mode": "cloud" if cloud_mode else "local"}
        
        try:
            if cloud_mode:
                # Cloud mode: Use cached fetch per symbol
                df = _fetch_single_cached(pair, start_date_str, end_date_str)
            else:
                # Local mode: Use orchestrator with DB
                df = orchestrator.get_ohlc(clean_symbol, start_date, end_date)
                df = _normalize_dataframe(df)
            
            if not df.empty:
                item_debug["success"] = True
                item_debug["data_points"] = len(df)
                item_debug["date_range"] = f"{df.index.min()} to {df.index.max()}"
                
                if debug_mode and not cloud_mode:
                    db = _get_database()
                    if db:
                        metadata = db.get_metadata(clean_symbol)
                        if metadata:
                            item_debug["source"] = metadata.get("primary_source", "unknown")
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
            data.append(pd.DataFrame())
        
        debug_info.append(item_debug)
        
        # Update progress
        if show_progress:
            percent = (i + 1) / total
            my_bar.progress(percent, text=f"{progress_text} {int(percent * 100)}%")
        
        # Add small delay between requests in cloud mode to avoid rate limits
        if cloud_mode and i < total - 1:
            time.sleep(0.5)
    
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
    orchestrator = _get_orchestrator()
    prices = {}
    
    if orchestrator is None:
        # Cloud mode: fetch directly
        fetchers = _get_fetchers()
        for symbol in symbols:
            for fetcher in fetchers:
                try:
                    price = fetcher.get_current_price(symbol)
                    if price is not None:
                        prices[symbol] = price
                        break
                except Exception as e:
                    logger.warning(f"{fetcher.source_name} price failed for {symbol}: {e}")
                    continue
    else:
        # Local mode: use orchestrator
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
    orchestrator = _get_orchestrator()
    
    if orchestrator is None:
        # Cloud mode: clear cache and return
        st.cache_data.clear()
        return {s: True for s in symbols}
    
    return orchestrator.refresh_all(symbols)


def get_db_stats() -> dict:
    """Get database statistics."""
    db = _get_database()
    
    if db is None:
        return {
            "initialized": False,
            "mode": "cloud" if is_cloud_deployment() else "local_no_db",
            "message": "Using in-memory caching (no persistent database)"
        }
    
    symbols = db.get_symbols()
    return {
        "initialized": True,
        "mode": "local",
        "db_path": str(db.db_path),
        "total_symbols": len(symbols),
        "symbols": symbols,
    }


def get_deployment_info() -> dict:
    """Get information about the current deployment environment."""
    return {
        "is_cloud": is_cloud_deployment(),
        "sqlite_available": is_sqlite_available(),
        "default_period": get_default_period(),
        "cache_ttl_seconds": 3600,
        "max_retries": MAX_RETRIES,
    }
