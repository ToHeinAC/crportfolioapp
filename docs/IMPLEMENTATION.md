# Implementation Guide

## Architecture Overview

```
crportfolioapp/
├── crportfolioapp.py          # Main Streamlit application
├── src/crportfolio/           # Data layer package
│   ├── data/
│   │   ├── db.py              # DatabaseManager - SQLite operations
│   │   ├── orchestrator.py    # DataOrchestrator - smart fetching
│   │   ├── compat.py          # Compatibility layer for main app
│   │   └── fetchers/          # Data source implementations
│   │       ├── base.py        # Abstract BaseFetcher
│   │       ├── yfinance_fetcher.py
│   │       ├── coingecko_fetcher.py
│   │       └── cmc_scraper.py
├── pyproject.toml             # UV package definition (source of truth)
├── requirements.txt           # Auto-generated for Streamlit Cloud
└── scripts/
    └── sync-requirements.sh   # Sync pyproject.toml -> requirements.txt
```

## Data Flow

```
User uploads Excel → Parse portfolio
                          ↓
                    Generate symbol list
                          ↓
              DataOrchestrator.get_ohlc()
                    /           \
            Check DB            Fetch missing
          (historical)          (APIs)
                    \           /
                     Combine data
                          ↓
                    Store new data in DB
                          ↓
                    Return to app
                          ↓
                    Visualizations
```

## Setup Instructions

### Local Development

```bash
# Clone repository
git clone <repo-url>
cd crportfolioapp

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the app
uv run streamlit run crportfolioapp.py
```

### DevContainer (VS Code / GitHub Codespaces)

1. Open project in VS Code
2. When prompted, click "Reopen in Container"
3. Wait for container to build (installs uv and dependencies)
4. App starts automatically at http://localhost:8501

### Streamlit Cloud Deployment

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Set main file: `crportfolioapp.py`
4. Deploy (uses `requirements.txt` automatically)

**Note**: SQLite database will be created in the app directory. Data persists within the session but may reset on app restart in cloud deployments.

## Data Layer Components

### DatabaseManager (`src/crportfolio/data/db.py`)

Handles SQLite operations for OHLC price storage.

```python
from crportfolio.data.db import DatabaseManager

db = DatabaseManager("crypto_prices.db")

# Store data
db.upsert_ohlc("BTC", df, source="yfinance")

# Retrieve data
df = db.get_ohlc("BTC", start_date, end_date)

# Check latest date
latest = db.get_latest_date("BTC")

# Get all cached symbols
symbols = db.get_symbols()
```

### DataOrchestrator (`src/crportfolio/data/orchestrator.py`)

Smart data fetching with DB-first approach and fallback chain.

```python
from crportfolio.data.orchestrator import create_default_orchestrator

orch = create_default_orchestrator("crypto_prices.db")

# Get OHLC data (checks DB first, fetches missing from APIs)
df = orch.get_ohlc("BTC", start_date, end_date)

# Force refresh from APIs
df = orch.get_ohlc("BTC", force_refresh=True)

# Get data for multiple symbols
results = orch.get_multiple_ohlc(["BTC", "ETH", "SOL"])
```

### Fetchers (`src/crportfolio/data/fetchers/`)

Each fetcher implements the `BaseFetcher` interface:

```python
class BaseFetcher(ABC):
    @property
    def source_name(self) -> str: ...
    @property
    def rate_limit_delay(self) -> float: ...
    def fetch_ohlc(self, symbol, start_date, end_date) -> pd.DataFrame: ...
    def get_current_price(self, symbol) -> float: ...
```

**YFinanceFetcher**: Primary source, uses Yahoo Finance API
- Rate limit: 1.5s between requests
- Supports special ticker mappings (SUPER -> SUPER8290-USD)
- Automatic retry with exponential backoff

**CoinGeckoFetcher**: First fallback
- Rate limit: 2s between requests
- Free tier limited to 365 days history
- Approximates OHLC from price data

**CoinMarketCapScraper**: Last resort
- Rate limit: 5s between requests
- Respects robots.txt
- Only gets current price (creates 30-day approximation)

## Adding New Data Sources

1. Create new fetcher in `src/crportfolio/data/fetchers/`:

```python
from .base import BaseFetcher

class NewSourceFetcher(BaseFetcher):
    source_name = "newsource"
    rate_limit_delay = 2.0

    def fetch_ohlc(self, symbol, start_date, end_date):
        # Implement fetching logic
        # Return DataFrame with: Open, High, Low, Close, Volume
        # Index should be DatetimeIndex
        pass

    def get_current_price(self, symbol):
        # Return float price
        pass
```

2. Add to `__init__.py`:

```python
from .newsource_fetcher import NewSourceFetcher
__all__.append("NewSourceFetcher")
```

3. Add to orchestrator in `compat.py`:

```python
fetchers = [
    YFinanceFetcher(),
    CoinGeckoFetcher(),
    NewSourceFetcher(),  # Add in priority order
    CoinMarketCapScraper(),
]
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `UV_HTTP_TIMEOUT` | HTTP timeout for uv | 30s |
| `PORT` | Server port (Streamlit Cloud) | 8501 |

### Symbol Mappings

Special ticker mappings are defined in each fetcher:

- **YFinance**: `src/crportfolio/data/fetchers/yfinance_fetcher.py`
- **CoinGecko**: `src/crportfolio/data/fetchers/coingecko_fetcher.py`
- **CoinMarketCap**: `src/crportfolio/data/fetchers/cmc_scraper.py`

To add a new symbol mapping:

```python
# In yfinance_fetcher.py
YFINANCE_TICKER_MAPPING = {
    "NEWSYM": "NEWSYM12345-USD",  # Add mapping
}

# In coingecko_fetcher.py
COINGECKO_ID_MAPPING = {
    "NEWSYM": "new-symbol-coingecko-id",
}
```

## Troubleshooting

### Rate Limiting

**Symptoms**: "Rate limit" errors, slow data fetching

**Solutions**:
1. Increase `rate_limit_delay` in fetchers
2. Use database cache (reduces API calls)
3. Enable debug mode to see which source is being used

### Missing Data

**Symptoms**: Empty charts, "No data available" messages

**Solutions**:
1. Check symbol mapping exists for the asset
2. Verify asset is available on data sources
3. Check database for cached data: `db.get_symbols()`
4. Force refresh: Clear cache in Options menu

### Database Issues

**Symptoms**: Data not persisting, slow queries

**Solutions**:
1. Check database file exists: `crypto_prices.db`
2. Run vacuum: `db.vacuum()`
3. Check disk space
4. Delete and recreate database if corrupted

### Import Errors

**Symptoms**: "New data layer not available" message

**Solutions**:
1. Ensure `src/` directory exists
2. Check Python path includes `src/`
3. Run `uv sync` to reinstall dependencies

## Testing

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_fetchers.py

# Run with coverage
uv run pytest --cov=src/crportfolio
```

### Manual Testing

```python
# Test database
from crportfolio.data.db import DatabaseManager
db = DatabaseManager("test.db")
# ... test operations

# Test fetchers
from crportfolio.data.fetchers import YFinanceFetcher
f = YFinanceFetcher()
df = f.fetch_ohlc("BTC")
print(df.head())

# Test orchestrator
from crportfolio.data.orchestrator import create_default_orchestrator
orch = create_default_orchestrator("test.db")
df = orch.get_ohlc("ETH")
```

## Performance Optimization

1. **Database indexing**: Indexes on `(symbol, date)` for fast queries
2. **Caching**: `@st.cache_data` with 1-hour TTL
3. **Smart fetching**: Only fetch missing data ranges
4. **Rate limiting**: Prevent API blocks and retries
5. **Parallel fetching**: Multiple fetchers can run in sequence within orchestrator
