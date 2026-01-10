# Product Requirements Document: Crypto Portfolio Dashboard

## Overview

The Crypto Portfolio Dashboard is a Streamlit-based web application for tracking and analyzing cryptocurrency portfolios. Users can upload their portfolio holdings via Excel files and view comprehensive analytics including current values, gains/losses, category breakdowns, and technical charts.

## Target Users

- **Crypto Investors**: Individual investors tracking personal crypto portfolios
- **Analysts**: Users needing technical analysis tools for individual assets
- **Portfolio Managers**: Users managing diversified crypto holdings across categories

## User Stories

### Portfolio Management

1. **As a crypto investor**, I want to upload my portfolio positions from an Excel file so that I can quickly see my holdings without manual data entry.

2. **As a crypto investor**, I want to see my total portfolio value vs. invested amount so that I can understand my overall performance at a glance.

3. **As a crypto investor**, I want to view gains/losses per asset so that I can identify my best and worst performing investments.

4. **As a crypto investor**, I want to see weekly and monthly comparisons so that I can track recent performance trends.

### Category Analysis

5. **As a portfolio manager**, I want to group my assets by category (Layer 1, DeFi, etc.) so that I can understand my exposure to different sectors.

6. **As a portfolio manager**, I want a treemap visualization so that I can quickly identify my largest positions and their performance.

7. **As an analyst**, I want to compare category performance over time so that I can adjust my allocation strategy.

### Technical Analysis

8. **As an analyst**, I want OHLC candlestick charts so that I can perform technical analysis on individual assets.

9. **As an analyst**, I want to see moving averages (20, 50, 200 day) so that I can identify trends.

10. **As an analyst**, I want Bollinger Bands and RSI indicators so that I can identify overbought/oversold conditions.

### Data & Performance

11. **As a user**, I want data to load quickly without excessive API calls so that I can use the app efficiently.

12. **As a user**, I want the app to work even when some data sources are unavailable so that I have a reliable experience.

13. **As a developer**, I want historical data cached in a database so that repeated requests are fast.

## Functional Requirements

### FR1: Portfolio Upload and Parsing
- Accept Excel (.xlsx) files with portfolio data
- Required columns: Asset symbol (index), Anzahl (quantity), Kaufpreis $ (purchase price), Kategorie (category)
- Calculate derived metrics (current value, gain/loss)

### FR2: Real-time and Historical Price Data
- Primary source: Yahoo Finance via yfinance
- Fallback 1: CoinGecko API
- Fallback 2: CoinMarketCap web scraping
- Store historical data in SQLite database
- Fetch only missing data from APIs

### FR3: Portfolio Visualization
- Total investment vs. current value comparison
- Asset-level bar charts (Top 10, Top 20, All)
- Treemap by category with gain/loss coloring
- Sparkline charts for recent performance

### FR4: OHLC Technical Charts
- Interactive candlestick charts
- Moving averages: 20-day, 50-day, 200-day
- Bollinger Bands (20-day, 2 std dev)
- RSI indicator (14-period)
- Volume with EMA overlay

### FR5: Category Analysis
- Group assets by category
- Calculate category totals and performance
- Time evolution comparison chart
- Single category deep-dive view

## Non-Functional Requirements

### NFR1: Performance
- Initial load with cached data: < 5 seconds
- Fresh data fetch for 20 assets: < 60 seconds
- Database query response: < 100ms

### NFR2: Reliability
- Graceful degradation when data sources fail
- Fallback to cached/historical data
- Clear error messages for failed requests

### NFR3: Usability
- Mobile-responsive layout (Streamlit wide mode)
- Interactive charts with hover details
- Progress indicators during data fetching

### NFR4: Maintainability
- Modular data layer architecture
- Clear separation of concerns
- Comprehensive logging for debugging

## Data Model

### Portfolio Excel Schema
```
| Column      | Type   | Description              |
|-------------|--------|--------------------------|
| Name (index)| String | Asset symbol (BTC, ETH)  |
| Anzahl      | Float  | Quantity held            |
| Kaufpreis $ | Float  | Average purchase price   |
| Kategorie   | String | Asset category           |
```

### Database Schema (OHLC History)
```sql
CREATE TABLE ohlc_history (
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL NOT NULL,
    volume REAL,
    source VARCHAR(20) NOT NULL,
    UNIQUE(symbol, date)
);
```

## Success Metrics

1. **Data Freshness**: 95% of displayed prices < 1 hour old
2. **Load Performance**: 90% of page loads < 5 seconds with cache
3. **API Efficiency**: 50% reduction in API calls via caching
4. **Reliability**: < 1% of sessions experience data fetch failures

## Future Considerations

- Real-time price updates via WebSocket
- Portfolio history tracking and time-travel view
- Multi-currency support (EUR, GBP)
- Import from exchange APIs (Binance, Coinbase)
- Alerts and notifications
- PDF report generation
