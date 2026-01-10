"""
CoinGecko API data fetcher implementation.
"""

import logging
import random
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from pycoingecko import CoinGeckoAPI

from .base import BaseFetcher

logger = logging.getLogger(__name__)


# Mapping of crypto symbols to CoinGecko IDs
COINGECKO_ID_MAPPING = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ATOM": "cosmos",
    "LINK": "chainlink",
    "ONT": "ontology",
    "AAVE": "aave",
    "ICP": "internet-computer",
    "RAY": "raydium",
    "VOXEL": "voxies",
    "BOME": "book-of-meme",
    "VANRY": "vanar-chain",
    "AGLD": "adventure-gold",
    "SUPER": "superfarm",
    "PHB": "phoenix-global",
    "PRIME": "echelon-prime",
    "TIA": "celestia",
    "INJ": "injective-protocol",
    "MDT": "measurable-data-token",
    "MPL": "maple",
    "AKT": "akash-network",
    "SUI": "sui",
    "APT": "aptos",
    "JTO": "jito",
    "TURBO": "turbos-finance",
    "FET": "fetch-ai",
    "ONDO": "ondo-finance",
    "USDT": "tether",
    "USDC": "usd-coin",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "MATIC": "matic-network",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "SHIB": "shiba-inu",
    "LTC": "litecoin",
    "UNI": "uniswap",
    "XLM": "stellar",
    "XMR": "monero",
    "BCH": "bitcoin-cash",
    "ALGO": "algorand",
    "NEAR": "near",
    "RENDER": "render-token",
    "CRO": "cronos",
    "TAO": "bittensor",
}


class CoinGeckoFetcher(BaseFetcher):
    """Fetch cryptocurrency data from CoinGecko API."""

    source_name = "coingecko"
    rate_limit_delay = 2.0  # CoinGecko free tier is rate-limited

    def __init__(self, max_retries: int = 3):
        self.cg = CoinGeckoAPI()
        self.max_retries = max_retries
        self._last_request_time = 0

    def _get_coin_id(self, symbol: str) -> Optional[str]:
        """Convert symbol to CoinGecko coin ID."""
        # Clean symbol (remove numbers)
        clean_symbol = "".join([c for c in symbol if not c.isdigit()])
        return COINGECKO_ID_MAPPING.get(clean_symbol) or COINGECKO_ID_MAPPING.get(symbol)

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
        Fetch OHLC data from CoinGecko.

        Note: CoinGecko free API only provides prices (not full OHLC).
        Open, High, Low are approximated from Close prices.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            start_date: Start date (None = 365 days ago, max for free tier)
            end_date: End date (None = today)

        Returns:
            DataFrame with OHLC data (approximated)
        """
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            raise ValueError(f"No CoinGecko mapping for symbol: {symbol}")

        # CoinGecko free tier limits to 365 days
        days = 365
        if start_date and end_date:
            delta = (end_date - start_date).days
            days = min(delta, 365)

        for retry in range(self.max_retries):
            try:
                self._rate_limit()

                market_data = self.cg.get_coin_market_chart_by_id(
                    id=coin_id, vs_currency="usd", days=str(days)
                )

                prices = market_data.get("prices", [])
                if not prices:
                    raise ValueError(f"No price data returned for {coin_id}")

                # Create DataFrame
                df = pd.DataFrame(prices, columns=["timestamp", "Close"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)

                # Approximate OHLC from Close prices
                df["Open"] = df["Close"].shift(1)
                df["High"] = df["Close"] * 1.005
                df["Low"] = df["Close"] * 0.995
                df["Volume"] = 0

                # Fill first row Open
                if pd.isna(df["Open"].iloc[0]):
                    df.loc[df.index[0], "Open"] = df["Close"].iloc[0]

                logger.info(f"CoinGecko: Fetched {len(df)} rows for {symbol}")
                return self.normalize_dataframe(df)

            except Exception as e:
                if retry < self.max_retries - 1:
                    wait_time = 2 * (retry + 1) + random.uniform(0, 1)
                    logger.warning(
                        f"CoinGecko error for {symbol}: {e}, retry {retry+1}/{self.max_retries}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"CoinGecko failed for {symbol} after {self.max_retries} retries: {e}")
                    raise

        return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get current price from CoinGecko."""
        coin_id = self._get_coin_id(symbol)
        if not coin_id:
            raise ValueError(f"No CoinGecko mapping for symbol: {symbol}")

        self._rate_limit()

        try:
            price_data = self.cg.get_price(ids=coin_id, vs_currencies="usd")
            price = price_data.get(coin_id, {}).get("usd")
            if price is None:
                raise ValueError(f"No price data for {coin_id}")
            return float(price)
        except Exception as e:
            logger.error(f"CoinGecko current price failed for {symbol}: {e}")
            raise
