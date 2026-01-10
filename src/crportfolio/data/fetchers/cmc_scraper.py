"""
CoinMarketCap web scraper for cryptocurrency data.

This scraper respects robots.txt and implements rate limiting.
Use as a last resort when API-based sources fail.
"""

import logging
import random
import time
from datetime import date, datetime, timedelta
from typing import Optional
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .base import BaseFetcher

logger = logging.getLogger(__name__)


# Mapping of crypto symbols to CoinMarketCap slugs
CMC_SLUG_MAPPING = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ATOM": "cosmos",
    "LINK": "chainlink",
    "AAVE": "aave",
    "ICP": "internet-computer",
    "RAY": "raydium",
    "SUPER": "superfarm",
    "PRIME": "echelon-prime",
    "TIA": "celestia",
    "INJ": "injective",
    "SUI": "sui",
    "APT": "aptos",
    "FET": "artificial-superintelligence-alliance",
    "ONDO": "ondo",
    "USDT": "tether",
    "USDC": "usd-coin",
    "BNB": "bnb",
    "XRP": "xrp",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "DOT": "polkadot-new",
    "AVAX": "avalanche",
    "SHIB": "shiba-inu",
    "LTC": "litecoin",
    "UNI": "uniswap",
    "NEAR": "near-protocol",
    "RENDER": "render",
    "TAO": "bittensor",
}


class CoinMarketCapScraper(BaseFetcher):
    """
    Scrape cryptocurrency data from CoinMarketCap.

    Ethical scraping practices:
    - Respects robots.txt
    - Rate limiting (5 seconds between requests)
    - Proper User-Agent header
    - Only scrapes publicly available data
    """

    source_name = "coinmarketcap"
    rate_limit_delay = 5.0  # Conservative rate limit

    BASE_URL = "https://coinmarketcap.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "CryptoPortfolioApp/1.0 (Educational/Personal Use)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self._last_request_time = 0
        self._robot_parser = self._init_robots_txt()

    def _init_robots_txt(self) -> RobotFileParser:
        """Parse robots.txt for compliance checking."""
        rp = RobotFileParser()
        rp.set_url(f"{self.BASE_URL}/robots.txt")
        try:
            rp.read()
        except Exception as e:
            logger.warning(f"Could not read robots.txt: {e}")
        return rp

    def _can_fetch(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        try:
            return self._robot_parser.can_fetch("*", url)
        except Exception:
            return True  # Allow if we can't check

    def _get_slug(self, symbol: str) -> Optional[str]:
        """Convert symbol to CoinMarketCap slug."""
        clean_symbol = "".join([c for c in symbol if not c.isdigit()])
        return CMC_SLUG_MAPPING.get(clean_symbol) or CMC_SLUG_MAPPING.get(symbol)

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed + random.uniform(0, 1)
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def fetch_ohlc(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch price data from CoinMarketCap.

        Note: CoinMarketCap's historical data page uses JavaScript rendering,
        so this method fetches the current price page and creates a simple
        price point. For full historical data, consider using their API.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            start_date: Start date (ignored for scraping)
            end_date: End date (ignored for scraping)

        Returns:
            DataFrame with approximated OHLC data from current price
        """
        slug = self._get_slug(symbol)
        if not slug:
            raise ValueError(f"No CoinMarketCap mapping for symbol: {symbol}")

        url = f"{self.BASE_URL}/currencies/{slug}/"

        if not self._can_fetch(url):
            raise PermissionError(f"robots.txt disallows scraping {url}")

        self._rate_limit()

        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # Try to extract current price from the page
            price = self._extract_price(soup, symbol)

            if price is None:
                raise ValueError(f"Could not extract price for {symbol}")

            # Create a DataFrame with the current price
            # Since we can only reliably get current price, create minimal data
            df = self._create_price_dataframe(price)

            logger.info(f"CMC Scraper: Got current price ${price:.2f} for {symbol}")
            return self.normalize_dataframe(df)

        except requests.RequestException as e:
            logger.error(f"CMC request failed for {symbol}: {e}")
            raise ConnectionError(f"Failed to fetch {url}: {e}")

    def _extract_price(self, soup: BeautifulSoup, symbol: str) -> Optional[float]:
        """Extract current price from CoinMarketCap page."""
        try:
            # Try multiple selectors as CMC changes their HTML structure
            price_selectors = [
                'span[data-test="text-cdp-price-display"]',
                'div.priceValue span',
                'div[class*="priceValue"]',
                'span[class*="price"]',
            ]

            for selector in price_selectors:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(strip=True)
                    # Clean price text: remove $, commas, etc.
                    price_text = text.replace("$", "").replace(",", "").strip()
                    try:
                        return float(price_text)
                    except ValueError:
                        continue

            # Fallback: search for price pattern in page
            import re

            price_pattern = r'\$[\d,]+\.?\d*'
            matches = re.findall(price_pattern, soup.get_text())
            if matches:
                price_text = matches[0].replace("$", "").replace(",", "")
                return float(price_text)

            return None

        except Exception as e:
            logger.warning(f"Price extraction failed for {symbol}: {e}")
            return None

    def _create_price_dataframe(self, price: float) -> pd.DataFrame:
        """Create a DataFrame with the current price."""
        # Create date range for last 30 days (approximation)
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=30)
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        df = pd.DataFrame(index=date_range)
        df["Open"] = price
        df["High"] = price * 1.005
        df["Low"] = price * 0.995
        df["Close"] = price
        df["Volume"] = 0

        return df

    def get_current_price(self, symbol: str) -> float:
        """Get current price from CoinMarketCap."""
        slug = self._get_slug(symbol)
        if not slug:
            raise ValueError(f"No CoinMarketCap mapping for symbol: {symbol}")

        url = f"{self.BASE_URL}/currencies/{slug}/"

        if not self._can_fetch(url):
            raise PermissionError(f"robots.txt disallows scraping {url}")

        self._rate_limit()

        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            price = self._extract_price(soup, symbol)

            if price is None:
                raise ValueError(f"Could not extract price for {symbol}")

            return price

        except Exception as e:
            logger.error(f"CMC current price failed for {symbol}: {e}")
            raise
