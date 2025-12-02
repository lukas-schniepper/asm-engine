# Portfolio Performance Tracking System - Implementation Plan

**Version:** 1.0
**Date:** 2025-12-02
**Status:** Draft

---

## Executive Summary

This document outlines the implementation plan for a Portfolio Performance Tracking System that will:
- Track daily performance of alpha portfolios
- Show portfolios WITH and WITHOUT risk overlays (Conservative Model, TrendRegimeV2)
- Mirror real trading performance
- Use the styling from the asm-models comprehensive email dashboards

---

## Table of Contents

1. [Architecture Decision](#1-architecture-decision)
2. [Storage Recommendation](#2-storage-recommendation)
3. [Comprehensive Dashboard Analysis](#3-comprehensive-dashboard-analysis)
4. [Database Schema](#4-database-schema)
5. [File Structure](#5-file-structure)
6. [Implementation Phases](#6-implementation-phases)
7. [Data Flow](#7-data-flow)
8. [UI Components](#8-ui-components)
9. [Error Handling & Resilience](#9-error-handling--resilience)
10. [Future Extensibility](#10-future-extensibility)

---

## 1. Architecture Decision

### Recommendation: Option A - Extend Existing asm-engine Streamlit App

**Rationale:**

| Factor | Assessment |
|--------|------------|
| **Data Proximity** | Portfolio tracking needs direct access to `price_data`, `ticker_period` - all in asm-engine's Supabase |
| **Styling Reuse** | Dashboard styling is pure Python + HTML/CSS - easily portable to new page |
| **User Experience** | Users already use Backtester/Optimizer/Data Mgmt - adding "Performance Tracker" as 4th page is natural |
| **Deployment** | Single Streamlit Cloud deployment, no additional infrastructure |
| **Code Reuse** | Can directly import existing performance calculation utilities |
| **Overlay Integration** | Import allocation functions from asm-models via S3 data |

### Critical Requirement: App Resilience

The existing Streamlit app must remain operational even if the new Performance Tracker page has errors:

```python
# In streamlit_app.py - wrap new page in try/except
try:
    if page == "Performance Tracker":
        from app.pages.performance_tracker import show_performance_tracker_ui
        show_performance_tracker_ui()
except Exception as e:
    st.error(f"Performance Tracker is temporarily unavailable: {e}")
    st.info("Please use other pages while we fix this issue.")
    import traceback
    st.code(traceback.format_exc())
```

---

## 2. Storage Recommendation

### Question: S3 vs Supabase for tracking data?

### Recommendation: **Supabase** for tracking data

**Reasoning:**

| Criteria | Supabase | S3 |
|----------|----------|-----|
| **Existing Infrastructure** | Already configured in asm-engine | Used by asm-models/asm-data |
| **Query Patterns** | Relational queries (filter by date, portfolio, variant) | File-based, requires full file loads |
| **Real-time Updates** | Native SQL transactions | Eventually consistent |
| **Schema Enforcement** | Strong typing, constraints, indexes | Schema-less |
| **Streamlit Integration** | Direct via SQLModel/SQLAlchemy | Requires boto3 + pandas |
| **Cost** | Already paying for it | Additional S3 costs |

### Data Flow Strategy

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   asm-data      │     │   asm-models    │     │   asm-engine    │
│   (S3)          │────▶│   (S3)          │────▶│   (Supabase)    │
│                 │     │                 │     │                 │
│ features_       │     │ allocation_     │     │ portfolio_      │
│ latest.parquet  │     │ history.csv     │     │ daily_nav       │
│ spy.csv         │     │ today_alloc.json│     │ overlay_signals │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       ▲
        └───────────────────────┴───────────────────────┘
                    Nightly sync via GitHub Actions
```

**Key Principle:** Always load data from S3 when we need asm-models data, since asm-data workflows update S3 nightly.

### S3 Integration Pattern

```python
# AlphaMachine_core/tracking/s3_adapter.py

import boto3
import pandas as pd
from io import BytesIO

class S3DataLoader:
    """Load data from asm-models S3 bucket."""

    def __init__(self, bucket_name: str = "asm-models-data"):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name

    def load_features_latest(self) -> pd.DataFrame:
        """Load features_latest.parquet from S3."""
        obj = self.s3.get_object(Bucket=self.bucket, Key="features_latest.parquet")
        return pd.read_parquet(BytesIO(obj['Body'].read()))

    def load_allocation_history(self, model: str) -> pd.DataFrame:
        """Load allocation history for a model."""
        key = f"models/{model}/allocation_history.csv"
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_csv(BytesIO(obj['Body'].read()))

    def load_spy_prices(self) -> pd.DataFrame:
        """Load SPY price data."""
        obj = self.s3.get_object(Bucket=self.bucket, Key="spy.csv")
        return pd.read_csv(BytesIO(obj['Body'].read()))
```

---

## 3. Comprehensive Dashboard Analysis

### Target: asm-models Comprehensive Dashboard

The comprehensive dashboard attached to allocation emails is the TARGET styling. Key file:
- `asm-models/Optimizers/conservative_oct16/dashboards/oct16_DASHBOARD.html` (1.5MB)

### Dashboard Sections (20 total)

| # | Section | Description | Priority |
|---|---------|-------------|----------|
| 1 | Header | Gradient header with title, subtitle, config badge | Critical |
| 2 | Performance Summary | 6 KPI cards (CAGR, Sharpe, Sortino, Max DD, Vol, Calmar) | Critical |
| 3 | IS vs OOS Performance | Comparison table (Strategy vs SPY, IS vs OOS vs Total) | High |
| 4 | Annual Performance | Year-by-year returns with IS/OOS badges | Critical |
| 5 | Monthly Performance | Scrollable table with monthly returns | Critical |
| 6 | Crisis Period Analysis | GFC, COVID, 2022 Bear, 2025 Tariffs performance | High |
| 7 | Parameter Grid | Fixed vs Optimized parameters with rationale | Medium |
| 8 | Parameter Stability | Frequency analysis across periods | Medium |
| 9 | Methodology | Walk-forward approach documentation | Low |
| 10 | Period-by-Period Results | Detailed train/test metrics per period | Medium |
| 11 | Cumulative Returns Chart | Plotly line chart with IS/OOS shading | Critical |
| 12 | Drawdown Chart | Plotly area chart showing drawdowns | Critical |
| 13 | Allocation Dynamics Chart | Plotly showing equity % over time | High |

### CSS/Styling Patterns

```css
/* Color Palette */
--header-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
--success: #10b981;
--success-bg: #dcfce7;
--danger: #ef4444;
--danger-bg: #fee2e2;
--warning: #f59e0b;
--warning-bg: #fef3c7;
--info: #0284c7;
--info-bg: #dbeafe;
--muted: #64748b;
--border: #e2e8f0;
--bg-light: #f8fafc;

/* Typography */
--font-family: system-ui, -apple-system, 'Segoe UI', Arial, sans-serif;
--h1: 700 28px;
--h2: 600 18px;
--h3: 600 14px;
--kpi-value: 700 20px;
--label: 600 12px uppercase;
--body: 400 12-13px;

/* Layout */
--container-max: 1600px;
--card-radius: 16px;
--section-padding: 24px 28px;
--grid-gap: 16px 20px;
--kpi-grid: repeat(auto-fit, minmax(180px, 1fr));
```

### Badge Classes

```css
.badge-is { background: #dbeafe; color: #1e40af; }  /* In-Sample */
.badge-oos { background: #dcfce7; color: #166534; } /* Out-of-Sample */
.positive { color: #10b981; }
.negative { color: #ef4444; }
```

### Chart Configuration (Plotly)

```javascript
// Cumulative Returns Chart
{
    type: 'scatter',
    mode: 'lines',
    line: { color: '#0284c7', width: 2 },  // Strategy
    // SPY: #94a3b8, width: 1.5, dash: 'dot'
}

// Drawdown Chart
{
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    fillcolor: 'rgba(239, 68, 68, 0.1)',
    line: { color: '#ef4444' }
}

// Allocation Chart
{
    type: 'scatter',
    mode: 'lines',
    fill: 'tozeroy',
    fillcolor: 'rgba(59, 130, 246, 0.2)',
    line: { color: '#3b82f6' }
}

// Layout buttons for IS/OOS range switching
updatemenus: [{
    type: 'buttons',
    buttons: [
        {label: 'Full History', method: 'relayout', args: ['xaxis.range', [null, null]]},
        {label: 'OOS Only', method: 'relayout', args: ['xaxis.range', ['2019-01-04', '2025-11-20']]}
    ]
}]
```

---

## 4. Database Schema

### New Tables for Supabase

```sql
-- ===========================================
-- Table 1: Portfolio Definitions
-- ===========================================
CREATE TABLE portfolio_definitions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    config JSONB NOT NULL,  -- All backtest parameters
    source VARCHAR(50),     -- "Topweights", "TR20", etc.
    start_date DATE NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE portfolio_definitions IS 'Tracked alpha portfolios with their backtest configurations';

-- ===========================================
-- Table 2: Portfolio Holdings (point-in-time)
-- ===========================================
CREATE TABLE portfolio_holdings (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolio_definitions(id) ON DELETE CASCADE,
    effective_date DATE NOT NULL,
    ticker VARCHAR(20) NOT NULL,
    shares DECIMAL(15,4),
    weight DECIMAL(8,6),
    entry_price DECIMAL(15,4),
    UNIQUE(portfolio_id, effective_date, ticker)
);

CREATE INDEX idx_holdings_portfolio_date ON portfolio_holdings(portfolio_id, effective_date);

-- ===========================================
-- Table 3: Portfolio Daily NAV
-- ===========================================
CREATE TABLE portfolio_daily_nav (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolio_definitions(id) ON DELETE CASCADE,
    trade_date DATE NOT NULL,
    variant VARCHAR(30) NOT NULL,  -- 'raw', 'conservative', 'trend_regime_v2', etc.
    nav DECIMAL(15,4) NOT NULL,
    daily_return DECIMAL(12,10),
    cumulative_return DECIMAL(12,10),
    equity_allocation DECIMAL(8,6),  -- 0.0 to 1.0 for overlay variants
    cash_allocation DECIMAL(8,6),
    UNIQUE(portfolio_id, trade_date, variant)
);

CREATE INDEX idx_nav_portfolio_date ON portfolio_daily_nav(portfolio_id, trade_date);
CREATE INDEX idx_nav_variant ON portfolio_daily_nav(variant, trade_date);

-- ===========================================
-- Table 4: Overlay Signals (daily)
-- ===========================================
CREATE TABLE overlay_signals (
    id SERIAL PRIMARY KEY,
    trade_date DATE NOT NULL,
    model VARCHAR(30) NOT NULL,  -- 'conservative', 'trend_regime_v2', 'future_model_x'
    target_allocation DECIMAL(8,6),
    actual_allocation DECIMAL(8,6),
    trade_required BOOLEAN,
    signals JSONB,  -- RSI, VIX, momentum, etc.
    impacts JSONB,  -- Per-factor allocation impacts
    UNIQUE(trade_date, model)
);

CREATE INDEX idx_signals_date_model ON overlay_signals(trade_date, model);

-- ===========================================
-- Table 5: Pre-computed Metrics
-- ===========================================
CREATE TABLE portfolio_metrics (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolio_definitions(id) ON DELETE CASCADE,
    variant VARCHAR(30) NOT NULL,
    period_type VARCHAR(10) NOT NULL,  -- 'week', 'month', 'quarter', 'year', 'ytd', 'all'
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_return DECIMAL(12,10),
    sharpe_ratio DECIMAL(10,6),
    sortino_ratio DECIMAL(10,6),
    cagr DECIMAL(12,10),
    max_drawdown DECIMAL(12,10),
    calmar_ratio DECIMAL(10,6),
    volatility DECIMAL(12,10),
    win_rate DECIMAL(8,6),
    UNIQUE(portfolio_id, variant, period_type, period_start)
);

CREATE INDEX idx_metrics_lookup ON portfolio_metrics(portfolio_id, variant, period_type);
```

### SQLModel Classes

```python
# AlphaMachine_core/tracking/models.py

from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import date, datetime
from decimal import Decimal

class PortfolioDefinition(SQLModel, table=True):
    __tablename__ = "portfolio_definitions"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, unique=True)
    description: Optional[str] = None
    config: dict = Field(sa_column_kwargs={"type_": "JSONB"})
    source: Optional[str] = Field(max_length=50)
    start_date: date
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PortfolioHolding(SQLModel, table=True):
    __tablename__ = "portfolio_holdings"

    id: Optional[int] = Field(default=None, primary_key=True)
    portfolio_id: int = Field(foreign_key="portfolio_definitions.id")
    effective_date: date
    ticker: str = Field(max_length=20)
    shares: Optional[Decimal] = None
    weight: Optional[Decimal] = None
    entry_price: Optional[Decimal] = None


class PortfolioDailyNAV(SQLModel, table=True):
    __tablename__ = "portfolio_daily_nav"

    id: Optional[int] = Field(default=None, primary_key=True)
    portfolio_id: int = Field(foreign_key="portfolio_definitions.id")
    trade_date: date
    variant: str = Field(max_length=30)  # 'raw', 'conservative', 'trend_regime_v2'
    nav: Decimal
    daily_return: Optional[Decimal] = None
    cumulative_return: Optional[Decimal] = None
    equity_allocation: Optional[Decimal] = None
    cash_allocation: Optional[Decimal] = None


class OverlaySignal(SQLModel, table=True):
    __tablename__ = "overlay_signals"

    id: Optional[int] = Field(default=None, primary_key=True)
    trade_date: date
    model: str = Field(max_length=30)
    target_allocation: Optional[Decimal] = None
    actual_allocation: Optional[Decimal] = None
    trade_required: Optional[bool] = None
    signals: Optional[dict] = Field(sa_column_kwargs={"type_": "JSONB"})
    impacts: Optional[dict] = Field(sa_column_kwargs={"type_": "JSONB"})


class PortfolioMetric(SQLModel, table=True):
    __tablename__ = "portfolio_metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
    portfolio_id: int = Field(foreign_key="portfolio_definitions.id")
    variant: str = Field(max_length=30)
    period_type: str = Field(max_length=10)  # week, month, year, all
    period_start: date
    period_end: date
    total_return: Optional[Decimal] = None
    sharpe_ratio: Optional[Decimal] = None
    sortino_ratio: Optional[Decimal] = None
    cagr: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    calmar_ratio: Optional[Decimal] = None
    volatility: Optional[Decimal] = None
    win_rate: Optional[Decimal] = None
```

---

## 5. File Structure

```
asm-engine/
├── AlphaMachine_core/
│   ├── ... (existing modules)
│   │
│   ├── tracking/                      # NEW: Portfolio tracking module
│   │   ├── __init__.py
│   │   ├── models.py                  # SQLModel definitions (see above)
│   │   ├── tracker.py                 # PortfolioTracker class
│   │   ├── metrics.py                 # Performance metric calculations
│   │   ├── overlay_adapter.py         # Adapter for applying overlays
│   │   ├── s3_adapter.py              # S3 data loading from asm-models
│   │   └── scheduled_update.py        # Daily NAV calculation script
│   │
│   └── ui/                            # NEW: Shared UI components
│       ├── __init__.py
│       ├── dashboard_styles.py        # CSS/HTML patterns from comprehensive dashboard
│       ├── components.py              # Reusable Streamlit components
│       └── charts.py                  # Plotly chart configurations
│
├── app/
│   ├── streamlit_app.py               # MODIFY: Add 4th page with try/except
│   └── pages/                         # NEW: Multi-page structure
│       └── performance_tracker.py     # Main tracking page
│
├── scripts/
│   ├── scheduled_price_update.py      # EXISTING
│   ├── scheduled_nav_update.py        # NEW: Daily NAV calculation
│   └── migrate_tracking_tables.py     # NEW: Create new tables
│
├── docs/
│   └── PORTFOLIO_TRACKING_IMPLEMENTATION_PLAN.md  # This document
│
└── requirements.txt                   # ADD: boto3 for S3 access
```

---

## 6. Implementation Phases

### Phase 1: Database & Models (Critical)
**Estimated Complexity:** Low
**Dependencies:** None

**Tasks:**
1. Create `AlphaMachine_core/tracking/models.py` with SQLModel classes
2. Create migration script `scripts/migrate_tracking_tables.py`
3. Run migration on Supabase
4. Verify tables created with correct indexes

**Deliverables:**
- [ ] SQLModel classes for all 5 tables
- [ ] Migration script
- [ ] Tables created in Supabase

---

### Phase 2: S3 Adapter (Critical)
**Estimated Complexity:** Medium
**Dependencies:** Phase 1

**Tasks:**
1. Create `AlphaMachine_core/tracking/s3_adapter.py`
2. Implement loading of features_latest.parquet
3. Implement loading of allocation_history.csv per model
4. Implement loading of SPY prices
5. Add S3 credentials to Streamlit secrets
6. Test connectivity

**Deliverables:**
- [ ] S3DataLoader class
- [ ] Credential configuration
- [ ] Unit tests for S3 loading

---

### Phase 3: Overlay Adapter (Critical)
**Estimated Complexity:** High
**Dependencies:** Phase 2

**Tasks:**
1. Create `AlphaMachine_core/tracking/overlay_adapter.py`
2. Port allocation logic from asm-models (or call via import)
3. Implement `apply_conservative_overlay(nav, date, features)`
4. Implement `apply_trend_regime_v2_overlay(nav, date, features)`
5. Design for extensibility (easy to add future overlays)

**Key Decision:** Import asm-models code or replicate?
- **Recommendation:** Load allocation parameters from S3 config files, replicate core logic
- **Reason:** Avoids git submodule complexity, easier maintenance

**Deliverables:**
- [ ] OverlayAdapter class with apply_overlay() method
- [ ] Support for conservative and trend_regime_v2
- [ ] Registry pattern for adding new overlays

---

### Phase 4: Core Tracking Engine (Critical)
**Estimated Complexity:** High
**Dependencies:** Phase 1, 2, 3

**Tasks:**
1. Create `AlphaMachine_core/tracking/tracker.py`
2. Implement PortfolioTracker class:
   - `register_portfolio(name, config, holdings)`
   - `calculate_daily_nav(portfolio_id, date)` - for all variants
   - `record_nav(portfolio_id, date, variant, nav, ...)`
   - `get_portfolio_performance(portfolio_id, variant, start, end)`
3. Create `AlphaMachine_core/tracking/metrics.py`
4. Implement metric calculations (reuse from engine.py where possible):
   - Daily/weekly/monthly/yearly returns
   - Rolling Sharpe, Sortino
   - CAGR, Max DD, Calmar
   - Volatility

**Deliverables:**
- [ ] PortfolioTracker class
- [ ] Metrics calculation module
- [ ] Integration tests

---

### Phase 5: UI Components (High)
**Estimated Complexity:** Medium
**Dependencies:** None (can parallel with Phases 1-4)

**Tasks:**
1. Create `AlphaMachine_core/ui/dashboard_styles.py`
   - Extract CSS variables from comprehensive dashboard
   - Create style dictionary
2. Create `AlphaMachine_core/ui/components.py`
   - `render_kpi_grid(metrics_dict)`
   - `render_performance_table(df, columns_config)`
   - `render_is_oos_badge(sample_type)`
   - `render_crisis_table(crisis_periods)`
3. Create `AlphaMachine_core/ui/charts.py`
   - `create_cumulative_returns_chart(nav_data)`
   - `create_drawdown_chart(nav_data)`
   - `create_allocation_chart(allocation_data)`

**Deliverables:**
- [ ] dashboard_styles.py with all CSS patterns
- [ ] components.py with reusable Streamlit components
- [ ] charts.py with Plotly chart configs

---

### Phase 6: Streamlit Page (High)
**Estimated Complexity:** High
**Dependencies:** Phase 4, 5

**Tasks:**
1. Create `app/pages/performance_tracker.py`
2. Modify `app/streamlit_app.py` to add 4th page with error handling
3. Implement sidebar filters:
   - Portfolio multi-select
   - Date range picker
   - Variant checkboxes
   - Timeframe selector
4. Implement main content tabs:
   - **Dashboard** - KPI grid, cumulative chart
   - **Comparison** - Side-by-side table (periods × variants)
   - **Annual/Monthly** - Returns tables
   - **Signals** - Overlay signal analysis
   - **Export** - Generate HTML report

**Deliverables:**
- [ ] performance_tracker.py page
- [ ] Modified streamlit_app.py with resilient 4th page
- [ ] All UI tabs implemented

---

### Phase 7: Scheduled Updates (Medium)
**Estimated Complexity:** Medium
**Dependencies:** Phase 4

**Tasks:**
1. Create `scripts/scheduled_nav_update.py`
2. Create GitHub Actions workflow `.github/workflows/scheduled_nav_update.yml`
   - Trigger: After market close (21:00 UTC)
   - Dependency: After scheduled_price_update completes
3. Implement update logic:
   - Fetch latest prices from Supabase
   - Load features from S3
   - Calculate NAV for all portfolios × all variants
   - Store in portfolio_daily_nav
   - Update overlay_signals
   - Compute periodic metrics

**Deliverables:**
- [ ] scheduled_nav_update.py script
- [ ] GitHub Actions workflow
- [ ] Logging and error notifications

---

### Phase 8: Portfolio Registration Workflow (Medium)
**Estimated Complexity:** Low
**Dependencies:** Phase 6

**Tasks:**
1. Add "Track This Portfolio" button to Backtester results
2. Implement registration flow:
   - Save portfolio_definitions entry
   - Save current holdings to portfolio_holdings
   - Backfill NAV from backtest results
3. Add portfolio management in Performance Tracker:
   - View tracked portfolios
   - Deactivate/reactivate
   - Edit description

**Deliverables:**
- [ ] Registration button in Backtester
- [ ] Portfolio management UI
- [ ] Backfill logic

---

## 7. Data Flow

### Daily Update Flow

```
[21:00 UTC GitHub Action: scheduled_nav_update.py]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Load features_latest.parquet from S3                    │
│    - Contains all technical indicators                      │
│    - Updated nightly by asm-data workflow                   │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Load SPY prices from S3                                  │
│    - For overlay calculations (VIX, momentum, etc.)         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. For each tracked portfolio:                              │
│    a. Get current holdings from portfolio_holdings          │
│    b. Get latest prices from price_data (Supabase)          │
│    c. Calculate raw NAV (100% equity)                       │
│    d. Apply Conservative overlay → adjusted NAV             │
│    e. Apply TrendRegimeV2 overlay → adjusted NAV            │
│    f. Store all 3 NAVs in portfolio_daily_nav               │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Store overlay signals in overlay_signals table           │
│    - One row per model per day                              │
│    - Includes signals JSONB and impacts JSONB               │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Compute and cache periodic metrics                       │
│    - Weekly, monthly, quarterly, yearly, YTD, all-time      │
│    - Store in portfolio_metrics                             │
└─────────────────────────────────────────────────────────────┘
```

### UI Query Flow

```
[User opens Performance Tracker page]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Load active portfolios from portfolio_definitions        │
│    - Populate portfolio selector                            │
└─────────────────────────────────────────────────────────────┘
                    │
[User selects portfolios, date range, variants]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Query portfolio_metrics for pre-computed stats           │
│    - Fast: indexed, aggregated                              │
│    - Used for KPI cards, summary tables                     │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Query portfolio_daily_nav for time series                │
│    - Used for charts (cumulative, drawdown)                 │
│    - Filtered by date range                                 │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Query overlay_signals for signal analysis tab            │
│    - Shows daily allocation decisions                       │
│    - Factor-by-factor impact breakdown                      │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Render UI using dashboard_styles and components          │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. UI Components

### Page Layout

```
┌────────────────────────────────────────────────────────────────────────┐
│  SIDEBAR                                                                │
│  ────────                                                               │
│  📁 Select Portfolios                                                   │
│  [□ Alpha20] [✓ Alpha30] [✓ Alpha50]                                   │
│                                                                         │
│  📅 Date Range                                                          │
│  [2024-01-01] to [2025-11-30]                                          │
│                                                                         │
│  📊 Variants                                                            │
│  [✓ Raw] [✓ Conservative] [✓ TrendRegimeV2]                            │
│                                                                         │
│  ⏱️ Timeframe                                                           │
│  [Daily ▼]                                                              │
├────────────────────────────────────────────────────────────────────────┤
│  MAIN CONTENT                                                           │
│  ────────────                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ [Dashboard] [Comparison] [Annual] [Monthly] [Signals] [Export]   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────────────────── Dashboard Tab ────────────────────┐  │
│  │                                                                   │  │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┐                           │  │
│  │  │ CAGR│Sharpe│Sort.│MaxDD│ Vol │Calmar│  ← KPI Grid            │  │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┘                           │  │
│  │                                                                   │  │
│  │  ┌───────────────────────────────────────────────────────────┐   │  │
│  │  │          Cumulative Returns Chart (Plotly)                 │   │  │
│  │  │  [Full History] [OOS Only]                                 │   │  │
│  │  │                                                             │   │  │
│  │  │  Strategy ───  SPY ....                                    │   │  │
│  │  └───────────────────────────────────────────────────────────┘   │  │
│  │                                                                   │  │
│  │  ┌───────────────────────────────────────────────────────────┐   │  │
│  │  │          Drawdown Chart (Plotly)                           │   │  │
│  │  └───────────────────────────────────────────────────────────┘   │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Comparison Tab

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PORTFOLIO COMPARISON                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┬───────────────────────────────────────────────────────┐│
│  │   Period    │  Alpha30      Alpha30      Alpha30       SPY         ││
│  │             │  Raw          +Conservative +TrendV2                  ││
│  ├─────────────┼───────────────────────────────────────────────────────┤│
│  │ Nov 2025    │  +2.34%       +1.87%       +2.12%       +1.95%       ││
│  │ Oct 2025    │  -1.12%       -0.45%       -0.78%       -0.92%       ││
│  │ Sep 2025    │  +4.56%       +3.21%       +4.01%       +3.44%       ││
│  │ ...         │  ...          ...          ...          ...          ││
│  └─────────────┴───────────────────────────────────────────────────────┘│
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  AGGREGATE METRICS (Selected Period)                                ││
│  ├───────────┬───────────┬───────────┬───────────┬───────────┬────────┤│
│  │  Metric   │ Raw       │ +Conserv. │ +TrendV2  │ SPY       │ Best   ││
│  ├───────────┼───────────┼───────────┼───────────┼───────────┼────────┤│
│  │ Sharpe    │ 1.45      │ 1.82*     │ 1.67      │ 0.95      │ +Cons  ││
│  │ Sortino   │ 2.12      │ 2.89*     │ 2.54      │ 1.21      │ +Cons  ││
│  │ CAGR      │ 15.2%*    │ 11.4%     │ 13.8%     │ 12.1%     │ Raw    ││
│  │ Max DD    │ -18%      │ -9%*      │ -12%      │ -15%      │ +Cons  ││
│  │ Calmar    │ 0.84      │ 1.27*     │ 1.15      │ 0.81      │ +Cons  ││
│  └───────────┴───────────┴───────────┴───────────┴───────────┴────────┘│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Error Handling & Resilience

### App Isolation

```python
# app/streamlit_app.py

import streamlit as st

# Existing page navigation
page = st.sidebar.radio(
    "🗂️ Seite wählen",
    ["Backtester", "Optimizer", "Data Mgmt", "Performance Tracker"],
    index=0
)

if page == "Backtester":
    show_backtester_ui()
elif page == "Optimizer":
    show_optimizer_ui()
elif page == "Data Mgmt":
    show_data_ui()
elif page == "Performance Tracker":
    # CRITICAL: Isolate new page with try/except
    try:
        from app.pages.performance_tracker import show_performance_tracker_ui
        show_performance_tracker_ui()
    except ImportError as e:
        st.error("⚠️ Performance Tracker module not found")
        st.info("The Performance Tracker is being developed. Please check back later.")
        st.code(str(e))
    except Exception as e:
        st.error(f"⚠️ Performance Tracker encountered an error: {type(e).__name__}")
        st.warning("The rest of the app is still functional. Please use other pages.")
        with st.expander("Show error details"):
            import traceback
            st.code(traceback.format_exc())
```

### Database Fallbacks

```python
# AlphaMachine_core/tracking/tracker.py

def get_portfolio_performance(self, portfolio_id, variant, start, end):
    """Get performance with fallback to live calculation."""
    try:
        # Try cached metrics first
        cached = self._get_cached_metrics(portfolio_id, variant, start, end)
        if cached is not None:
            return cached
    except Exception as e:
        logger.warning(f"Cache lookup failed: {e}")

    # Fallback to live calculation
    try:
        nav_data = self._get_nav_data(portfolio_id, variant, start, end)
        return self._calculate_metrics(nav_data)
    except Exception as e:
        logger.error(f"Live calculation failed: {e}")
        return None
```

### S3 Fallbacks

```python
# AlphaMachine_core/tracking/s3_adapter.py

def load_features_latest(self) -> pd.DataFrame:
    """Load features with fallback to local cache."""
    try:
        # Try S3 first
        return self._load_from_s3("features_latest.parquet")
    except Exception as e:
        logger.warning(f"S3 load failed: {e}, trying local cache")

        # Try local cache
        cache_path = Path("data/cache/features_latest.parquet")
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        raise RuntimeError("Features data unavailable from both S3 and cache")
```

---

## 10. Future Extensibility

### Adding New Overlays

The system is designed to easily add new overlays beyond Conservative and TrendRegimeV2:

```python
# AlphaMachine_core/tracking/overlay_adapter.py

# Registry pattern for overlays
OVERLAY_REGISTRY = {
    'conservative': {
        'name': 'Conservative Model (OCT16)',
        'config_s3_key': 'models/conservative/config_oct16.json',
        'calculator': calculate_allocation_oct16,
    },
    'trend_regime_v2': {
        'name': 'Trend Regime V2.0',
        'config_s3_key': 'models/v2_regime/config_v2_regime.json',
        'calculator': calculate_allocation_v2_regime,
    },
    # Add new overlays here:
    # 'momentum_v3': {
    #     'name': 'Momentum V3.0',
    #     'config_s3_key': 'models/momentum_v3/config.json',
    #     'calculator': calculate_allocation_momentum_v3,
    # },
}

def get_available_overlays() -> list[str]:
    """Return list of available overlay models."""
    return list(OVERLAY_REGISTRY.keys())

def apply_overlay(model: str, nav: float, date: date, features: pd.DataFrame) -> tuple[float, dict]:
    """Apply specified overlay to NAV."""
    if model not in OVERLAY_REGISTRY:
        raise ValueError(f"Unknown overlay: {model}. Available: {get_available_overlays()}")

    config = load_config_from_s3(OVERLAY_REGISTRY[model]['config_s3_key'])
    calculator = OVERLAY_REGISTRY[model]['calculator']

    allocation, signals, impacts = calculator(date, config, features)
    adjusted_nav = nav * allocation

    return adjusted_nav, {'allocation': allocation, 'signals': signals, 'impacts': impacts}
```

### Database Support for New Overlays

No schema changes needed! The `variant` column in `portfolio_daily_nav` and `model` column in `overlay_signals` are VARCHAR(30), allowing any new overlay name.

---

## Appendix A: Required Dependencies

Add to `requirements.txt`:

```
# S3 Access (for loading asm-models data)
boto3>=1.34.0

# Already present - verify versions
pandas>=2.2.0
plotly>=6.0.0
streamlit>=1.44.0
sqlmodel>=0.0.24
```

---

## Appendix B: Environment Variables

Add to `.streamlit/secrets.toml`:

```toml
# AWS S3 Access (for asm-models data)
AWS_ACCESS_KEY_ID = "AKIA..."
AWS_SECRET_ACCESS_KEY = "..."
AWS_REGION = "eu-central-1"
S3_BUCKET_ASM_MODELS = "asm-models-data"
```

---

## Appendix C: GitHub Actions Workflow

```yaml
# .github/workflows/scheduled_nav_update.yml

name: Daily NAV Update

on:
  schedule:
    # Run at 21:00 UTC (after US market close)
    - cron: '0 21 * * 1-5'  # Mon-Fri
  workflow_dispatch:  # Manual trigger

jobs:
  update-nav:
    runs-on: ubuntu-latest
    needs: [price-update]  # Ensure prices are updated first

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run NAV update
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          python scripts/scheduled_nav_update.py

      - name: Notify on failure
        if: failure()
        run: |
          # Send notification (Slack, email, etc.)
          echo "NAV update failed!"
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-02 | Claude | Initial draft |
| 1.1 | 2025-12-02 | Claude | Implementation complete - all 8 phases done |

---

## Implementation Status

All phases have been implemented:

| Phase | Status | Files Created |
|-------|--------|---------------|
| 1. Database & Models | ✅ Complete | `tracking/models.py`, `scripts/migrate_tracking_tables.py` |
| 2. S3 Adapter | ✅ Complete | `tracking/s3_adapter.py` |
| 3. Overlay Adapter | ✅ Complete | `tracking/overlay_adapter.py` |
| 4. Core Tracking Engine | ✅ Complete | `tracking/tracker.py`, `tracking/metrics.py` |
| 5. UI Components | ✅ Complete | `ui/styles.py`, `ui/components.py`, `ui/charts.py` |
| 6. Streamlit Page | ✅ Complete | `ui/performance_tracker.py`, modified `streamlit_app.py` |
| 7. Scheduled Updates | ✅ Complete | `scripts/scheduled_nav_update.py`, `.github/workflows/scheduled_nav_update.yml` |
| 8. Portfolio Registration | ✅ Complete | `tracking/registration.py`, "Track Portfolio" button in Backtester |

### To Start Using

1. **Run migration:** `python scripts/migrate_tracking_tables.py`
2. **Configure secrets:** Add AWS credentials to `.streamlit/secrets.toml`
3. **Run a backtest** and click "Track this portfolio" in the Dashboard tab
4. **View in Performance Tracker** page

---

*Implementation completed 2025-12-02.*
