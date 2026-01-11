# CLAUDE.md - Development Notes for AI Assistance

## Project Overview

Streamlit-based cryptocurrency portfolio tracking application with multi-source data fetching and SQLite caching.

## Key Files

| File | Purpose |
|------|---------|
| `crportfolioapp.py` | Main Streamlit app (~1500 lines) |
| `src/crportfolio/data/db.py` | SQLite database manager |
| `src/crportfolio/data/orchestrator.py` | Smart data fetching |
| `src/crportfolio/data/compat.py` | Compatibility layer |
| `src/crportfolio/data/fetchers/*.py` | Data source implementations |
| `pyproject.toml` | Dependencies (source of truth) |
| `requirements.txt` | Auto-generated for Streamlit Cloud |

## Development Commands

```bash
# Install dependencies
uv sync

# Run the app
uv run streamlit run crportfolioapp.py

# Sync requirements.txt from pyproject.toml
./scripts/sync-requirements.sh

# Run tests
uv run pytest

# Add a new dependency
uv add <package>
# Then run: ./scripts/sync-requirements.sh
```

## Architecture Decisions

### Why SQLite for caching?
- No external database setup required
- Works locally and in cloud deployments
- Simple backup (single file)
- Sufficient performance for this use case

### Why multiple data fetchers?
- Yahoo Finance: Best historical data but rate limited
- CoinGecko: Good fallback, free API
- CoinMarketCap: Last resort when others fail
- Chain ensures reliability

### Why separate data layer package?
- Clean separation of concerns
- Testable in isolation
- Can be reused in other projects
- Easier to maintain and extend

## Common Tasks

### Adding a new cryptocurrency

1. Add symbol mapping to relevant fetchers:
   - `src/crportfolio/data/fetchers/yfinance_fetcher.py` - `YFINANCE_TICKER_MAPPING`
   - `src/crportfolio/data/fetchers/coingecko_fetcher.py` - `COINGECKO_ID_MAPPING`
   - `src/crportfolio/data/fetchers/cmc_scraper.py` - `CMC_SLUG_MAPPING`

### Adding a new data source

1. Create fetcher class implementing `BaseFetcher`
2. Add to `src/crportfolio/data/fetchers/__init__.py`
3. Add to fetcher list in `compat.py`

### Modifying the database schema

1. Update `DatabaseManager._init_schema()` in `db.py`
2. Consider migration strategy for existing data
3. Test with fresh database

### Adding a new visualization

1. Create plot function in `crportfolioapp.py`
2. Add to appropriate section (Portfolio, Asset Cats, OHLC)
3. Use `st.plotly_chart()` for interactive charts

## Known Issues

### CoinMarketCap scraping fragility
CoinMarketCap frequently changes their HTML structure. The scraper may break and require selector updates.

### yfinance rate limits
Yahoo Finance has undocumented rate limits. Exponential backoff helps but occasional failures occur.

### Streamlit Cloud SQLite
SQLite database may reset on app restart in cloud deployments. Consider external database for production.

## Code Style

- Use type hints where practical
- Follow PEP 8 (enforced by ruff)
- Keep functions focused and testable
- Document public APIs with docstrings

## Testing Strategy

- Unit tests for data layer components
- Integration tests for orchestrator
- Manual testing for Streamlit UI
- Test with both fresh and cached data

## Debugging Tips

1. Enable debug mode in Options menu
2. Check logs in terminal for API errors
3. Inspect database: `sqlite3 crypto_prices.db ".tables"`
4. Test fetchers individually in Python REPL

## Performance Notes

- First load fetches all data (slow, ~30-60s)
- Subsequent loads use cache (fast, <5s)
- Database caches indefinitely until cleared
- Streamlit cache (1 hour TTL) for session

## Streamlit Cloud Deployment

### Requirements Management

- **Source of truth**: `pyproject.toml` (managed by `uv`)
- **Generated file**: `requirements.txt` (for Streamlit Cloud)
- **Sync command**: `./scripts/sync-requirements.sh`

**Important**: Always run `./scripts/sync-requirements.sh` after changing dependencies in `pyproject.toml`. Streamlit Cloud reads `requirements.txt`, not `pyproject.toml`.

### Key Files for Deployment

| File | Purpose |
|------|---------|
| `requirements.txt` | Dependencies (auto-generated, DO NOT edit manually) |
| `runtime.txt` | Python version (currently 3.11) |
| `.streamlit/config.toml` | UI theme and server config |
| `.streamlit/secrets.toml` | Local secrets (NOT committed, use Streamlit Cloud UI for production) |

### yfinance Version Constraint

yfinance is pinned to `<0.2.58` because versions 0.2.58+ require `curl-cffi`, which has compilation issues on Streamlit Cloud. The fallback chain (CoinGecko, CoinMarketCap) handles any rate limiting on older yfinance versions.

### Deploy Checklist

1. Run `./scripts/sync-requirements.sh` to update requirements.txt
2. Commit changes including updated requirements.txt
3. Push to GitHub
4. In Streamlit Cloud: configure secrets via the UI (not secrets.toml)
5. Note: SQLite data is ephemeral on Streamlit Cloud (resets on restart)

## Security Considerations

- No secrets stored in code
- `.streamlit/secrets.toml` is gitignored
- Excel files processed locally
- No user authentication (single-user app)
- Web scraper respects robots.txt
