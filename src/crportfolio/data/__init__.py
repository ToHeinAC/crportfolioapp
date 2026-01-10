"""
Data acquisition and storage module for Crypto Portfolio App.

This module provides:
- DatabaseManager: SQLite storage for OHLC historical data
- DataOrchestrator: Smart data fetching with DB-first approach
- Fetchers: Multiple data source implementations (yfinance, CoinGecko, web scraping)
"""

from .db import DatabaseManager
from .orchestrator import DataOrchestrator

__all__ = ["DatabaseManager", "DataOrchestrator"]
