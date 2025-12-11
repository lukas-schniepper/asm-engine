import streamlit as st
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import datetime as dt
from pandas.tseries.offsets import BDay
from pandas.io.formats.style import Styler
import tempfile
import os
from sqlmodel import select
import re
from typing import Dict, Optional, List, Any
import plotly.graph_objects as go
from AlphaMachine_core.models import TickerPeriod
from AlphaMachine_core.models import TickerPeriod, TickerInfo, PriceData
from AlphaMachine_core.db import init_db, get_session
from AlphaMachine_core.optimize_params import run_optimizer
from AlphaMachine_core.engine import SharpeBacktestEngine
from AlphaMachine_core.reporting_no_sparklines import export_results_to_excel
from AlphaMachine_core.data_manager import StockDataManager
from AlphaMachine_core.config import (
    OPTIMIZER_METHOD as CFG_OPT_METHOD,
    COV_ESTIMATOR as CFG_COV_EST,
    REBALANCE_FREQUENCY as CFG_REBAL_FREQ,
    CUSTOM_REBALANCE_MONTHS as CFG_CUSTOM_REBAL,
    ENABLE_TRADING_COSTS as CFG_ENABLE_TC,
    FIXED_COST_PER_TRADE as CFG_FIXED_COST,
    VARIABLE_COST_PCT as CFG_VAR_COST,
    BACKTEST_WINDOW_DAYS as CFG_WINDOW,
    OPTIMIZATION_MODE as CFG_OPT_MODE,
    MIN_WEIGHT as CFG_MIN_W,
    MAX_WEIGHT as CFG_MAX_W,
    FORCE_EQUAL_WEIGHT as CFG_FORCE_EQ,
)

# ---------- Fix A : RiskOverlay-Helper --------------------------------------
from AlphaMachine_core.risk_overlay.overlay import RiskOverlay
from AlphaMachine_core import config as CFG

@st.cache_resource
def get_overlay():
    """Overlay-Objekt gemäss globaler CFG laden (einmal pro Session)."""
    if CFG.RISK_OVERLAY.get("enabled", False):
        return RiskOverlay(CFG.RISK_OVERLAY["config_path"])
    return None

init_db()

# ---------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------

def convert_single_ticker_pricedata_to_ohlcv_df(
    price_data_dicts: List[Dict[str, Any]],
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Convert list of PriceData dicts to OHLCV DataFrame for charting.

    Args:
        price_data_dicts: List of dicts with keys: ticker, trade_date, open, high, low, close, volume
        ticker: Ticker symbol (for filtering)
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume
    """
    if not price_data_dicts:
        return pd.DataFrame()

    # Filter for the specific ticker
    ticker_data = [d for d in price_data_dicts if d.get('ticker') == ticker]

    if not ticker_data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(ticker_data)

    # Convert trade_date to datetime and set as index
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date')

    # Rename columns to standard OHLCV format
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    df = df.rename(columns=column_mapping)

    # Select only OHLCV columns
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_cols = [col for col in ohlcv_cols if col in df.columns]
    df = df[available_cols]

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter by date range
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    # Sort by date
    df = df.sort_index()

    return df

# ---------------------------------------------------------------
# Helper Funktion zur Formatierung von Zahlen
# ---------------------------------------------------------------
def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    def _fmt(col: str, val):
        if pd.isna(val):
            return ""
        if isinstance(val, (int, float, np.integer, np.floating)):
            pct_like = (
                "%" in col or
                any(tag in col.lower()
                    for tag in ["return", "drawdown", "weight",
                                "volatility", "cagr"])
            )
            if pct_like:
                return f"{val:.1f}%"
            else:
                return f"{val:,.0f}"
        return val

    for c in df2.columns:
        df2[c] = df2[c].apply(lambda v, col=c: _fmt(col, v))
    return df2 

# -----------------------------------------------------------------------------
# 1) Page-Config
# -----------------------------------------------------------------------------
st.set_page_config("AlphaMachine", layout="wide")

st.markdown(
    """
    <style>
    /*  GLOBAL: alle vertikalen & horizontalen Scrollbars breiter  */
    /*  WebKit-Browser (Chrome, Edge, Opera, Brave, …)            */
    ::-webkit-scrollbar {
        width: 18px;        /* vertikal */
        height: 18px;       /* horizontal */
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(120, 120, 120, 0.6);
        border-radius: 8px;
        border: 3px solid rgba(0,0,0,0);   /* Abstand zum Rand (= “padding”) */
        background-clip: content-box;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(100, 100, 100, 0.9);
    }

    /*  Firefox (Quantum)  */
    html {
        scrollbar-width: thick;          /* thin | auto | <Länge> | none */
        scrollbar-color: rgba(120,120,120,0.6) transparent;
    }

    /*  Optional: nur DataFrames / AgGrid o. Ä. gezielt ansprechen
        (statt global) – Beispiel für die Streamlit-DataFrame-Box:
       [data-testid="stDataFrameScrollable"] {...}                       */
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 2) Passwort-Gate
# -----------------------------------------------------------------------------
pwd = st.sidebar.text_input("Passwort", type="password")
if pwd != st.secrets.get("APP_PW", ""):
    st.warning("🔒 Bitte korrektes Passwort eingeben.")
    st.stop()

# -----------------------------------------------------------------------------
# 3) Navigation-Switcher
# -----------------------------------------------------------------------------
# Session state for optimizer -> backtester parameter passing
if "optimizer_to_backtester" not in st.session_state:
    st.session_state.optimizer_to_backtester = None
if "auto_run_backtest" not in st.session_state:
    st.session_state.auto_run_backtest = False
if "switch_to_backtester" not in st.session_state:
    st.session_state.switch_to_backtester = False

# Check if we need to switch to Backtester - set BEFORE widget renders
if st.session_state.switch_to_backtester:
    st.session_state.page_radio = "Backtester"  # Set widget state before render
    st.session_state.switch_to_backtester = False  # Reset flag

page = st.sidebar.radio(
    "🗂️ Seite wählen",
    ["Backtester", "Optimizer", "Data Mgmt", "Performance Tracker", "Portfolio Selection"],
    key="page_radio"
)

# -----------------------------------------------------------------------------
# 4) CSV-Loader (Session-Cache)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="📂 CSV wird geladen…")
def load_csv(file):
    return pd.read_csv(file, index_col=0, parse_dates=True)


# -----------------------------------------------------------------------------
# Load Prices
# -----------------------------------------------------------------------------
def load_price_df(
    dm: StockDataManager, 
    tickers: List[str], 
    start_date: dt.date, # Annahme: start_date und end_date sind bereits date-Objekte
    end_date: dt.date, 
    window_days: int, 
    lookback_margin_days: int = 20
) -> tuple[pd.DataFrame, dt.date]: # Typ-Hinweis für Rückgabewert
    
    # Berechne das Lookback in HANDELSTAGEN!
    # pd.to_datetime ist hier nötig, falls start_date als dt.date kommt und BDay einen Timestamp erwartet
    price_start_dt = pd.to_datetime(start_date) - BDay(window_days + lookback_margin_days)
    price_start_for_query = price_start_dt.date()  # Konvertiere zu date für get_price_data

    # dm.get_price_data gibt jetzt eine Liste von Dictionaries zurück
    raw_price_dicts: List[Dict[str, Any]] = dm.get_price_data(
        tickers,
        price_start_for_query.strftime("%Y-%m-%d"), # Konvertiere zu String für die Methode
        end_date.strftime("%Y-%m-%d")               # Konvertiere zu String
    )

    if not raw_price_dicts:
        st.warning(f"Keine Rohdaten von get_price_data für Ticker {tickers} im Zeitraum {price_start_for_query} bis {end_date} erhalten.")
        return pd.DataFrame(), price_start_for_query # Gib leeren DF und das berechnete Startdatum zurück

    # Erstelle den DataFrame direkt aus der Liste von Dictionaries
    # Die Dictionaries sollten bereits die korrekten Keys haben: 
    # 'trade_date', 'ticker', 'open', 'high', 'low', 'close', 'volume'
    # wie von StockDataManager.get_price_data zurückgegeben
    try:
        temp_df = pd.DataFrame(raw_price_dicts)
    except Exception as e_df_create:
        st.error(f"Fehler beim Erstellen des DataFrames aus raw_price_dicts: {e_df_create}")
        return pd.DataFrame(), price_start_for_query

    if temp_df.empty:
        st.warning(f"Temporärer DataFrame ist leer nach Konvertierung von raw_price_dicts für Ticker {tickers}.")
        return pd.DataFrame(), price_start_for_query
        
    # Stelle sicher, dass die benötigten Spalten vorhanden sind
    if "trade_date" not in temp_df.columns or "ticker" not in temp_df.columns or "close" not in temp_df.columns:
        st.error(f"Benötigte Spalten ('trade_date', 'ticker', 'close') nicht im DataFrame aus get_price_data gefunden. Vorhandene Spalten: {temp_df.columns.tolist()}")
        return pd.DataFrame(), price_start_for_query

    try:
        price_df_pivoted = (
            temp_df
            .assign(date=lambda d: pd.to_datetime(d["trade_date"])) # Konvertiere 'trade_date' zu datetime
            .pivot(index="date", columns="ticker", values="close")  # Pivot für Close-Preise
            .sort_index()
        )
    except KeyError as ke:
        st.error(f"KeyError beim Pivotieren der Preisdaten (wahrscheinlich fehlende Spalte): {ke}. DataFrame Spalten: {temp_df.columns.tolist()}")
        return pd.DataFrame(), price_start_for_query
    except Exception as e_pivot:
        st.error(f"Allgemeiner Fehler beim Pivotieren der Preisdaten: {e_pivot}")
        return pd.DataFrame(), price_start_for_query

    # Erstelle einen vollständigen Business-Day-Index für den gesamten benötigten Zeitraum
    # price_start_dt ist der Timestamp für den allerersten Tag der benötigten Historie
    # end_date ist das Enddatum des Backtests
    full_idx = pd.date_range(start=price_start_dt.normalize(), end=pd.to_datetime(end_date).normalize(), freq='B') # 'B' für Business Days
    
    price_df_reindexed = price_df_pivoted.reindex(full_idx).ffill() # Vorwärtsfüllen für fehlende Tage

    if price_df_reindexed.empty:
        st.warning(f"Preis-DataFrame ist nach Reindizierung und ffill leer für Ticker {tickers}.")
    
    return price_df_reindexed, price_start_for_query


# =============================================================================
# === Backtester-UI ===
# =============================================================================
def show_backtester_ui():
    st.sidebar.header("📊 Backtest-Parameter")
    dm = StockDataManager()

    # Check if coming from optimizer with pre-filled parameters
    opt_params = st.session_state.get("optimizer_to_backtester") or {}
    from_optimizer = bool(opt_params)

    if from_optimizer:
        st.info("✅ Parameter aus Optimizer übernommen - Backtest wird automatisch gestartet")

    # 0) Backtest-Periode festlegen
    col1, col2 = st.sidebar.columns(2)
    default_start = opt_params.get("start_date", dt.date.today() - dt.timedelta(days=5*365))
    default_end = opt_params.get("end_date", dt.date.today())
    start_date = col1.date_input(
        "Backtest-Startdatum",
        value=default_start,
        max_value=dt.date.today()
    )
    end_date = col2.date_input(
        "Backtest-Enddatum",
        value=default_end,
        min_value=start_date
    )
    if start_date >= end_date:
        st.sidebar.error("Startdatum muss vor dem Enddatum liegen.")
        return

    # 1) Quellen-Auswahl (DB + Defaults)
    with get_session() as session:
        existing = session.exec(select(TickerPeriod.source)).all()
    defaults = ["Topweights","TR20"]
    all_sources = sorted(set(existing + defaults))
    default_sources = opt_params.get("sources", ["Topweights"])
    # Ensure default_sources are valid options
    default_sources = [s for s in default_sources if s in all_sources] or ["Topweights"]
    sources = st.sidebar.multiselect(
        "Datenquellen auswählen",
        options=all_sources,
        default=default_sources
    )

    # 2) Monat wählen
    months = sorted(dm.get_periods_distinct_months(), reverse=True)  # Newest first
    default_month = opt_params.get("month", months[0] if months else None)
    default_month_idx = months.index(default_month) if default_month in months else 0
    month  = st.sidebar.selectbox("Periode wählen (YYYY-MM)", months, index=default_month_idx)

    # 3) Modus: statisch vs. dynamisch
    mode = st.sidebar.radio(
        "Ticker-Universe",
        ["statisch (gesamte Periode)", "dynamisch (monatlich)"]
    )

    # 4) Lookback Days (Backtest-Fenster)
    default_window = opt_params.get("window_days", CFG_WINDOW)
    # Clamp to valid range
    default_window = max(50, min(500, default_window))
    window_days = st.sidebar.slider(
        "Lookback Days",
        min_value=50,
        max_value=500,
        value=default_window,
        step=10
    )

    # — 5) Portfolio- & Optimierungs-Parameter —
    start_balance = st.sidebar.number_input("Startkapital", 10_000, 1_000_000, 100_000, 1_000)

    default_num_stocks = opt_params.get("num_stocks", 20)
    default_num_stocks = max(5, min(50, default_num_stocks))
    num_stocks    = st.sidebar.slider("Aktien pro Portfolio", 5, 50, default_num_stocks)

    opt_methods = ["ledoit-wolf","minvar","hrp"]
    default_opt_method = opt_params.get("optimizer_method", CFG_OPT_METHOD)
    default_opt_idx = opt_methods.index(default_opt_method) if default_opt_method in opt_methods else 0
    opt_method    = st.sidebar.selectbox(
        "Optimierer", opt_methods,
        index=default_opt_idx
    )

    cov_estimators = ["ledoit-wolf","constant-corr","factor-model"]
    default_cov = opt_params.get("cov_estimator", CFG_COV_EST)
    default_cov_idx = cov_estimators.index(default_cov) if default_cov in cov_estimators else 0
    cov_estimator = st.sidebar.selectbox(
        "Kovarianzschätzer", cov_estimators,
        index=default_cov_idx
    )

    opt_modes = ["select-then-optimize","optimize-subset"]
    default_opt_mode = opt_params.get("optimization_mode", CFG_OPT_MODE)
    default_opt_mode_idx = opt_modes.index(default_opt_mode) if default_opt_mode in opt_modes else 0
    opt_mode      = st.sidebar.selectbox(
        "Optimierungsmodus", opt_modes,
        index=default_opt_mode_idx
    )

    rebalance_freq= st.sidebar.selectbox(
        "Rebalance", ["weekly","monthly","custom"],
        index=["weekly","monthly","custom"].index(CFG_REBAL_FREQ)
    )
    custom_months = (
        st.sidebar.slider("Monate zwischen Rebalances", 1, 12, CFG_CUSTOM_REBAL)
        if rebalance_freq=="custom" else 1
    )

    # — 6) Gewicht-Constraints —
    default_min_w = opt_params.get("min_weight", CFG_MIN_W * 100)
    default_min_w = max(0.0, min(5.0, float(default_min_w)))
    min_w    = st.sidebar.slider("Min Weight (%)", 0.0, 5.0, default_min_w, 0.5) / 100.0

    default_max_w = opt_params.get("max_weight", CFG_MAX_W * 100)
    default_max_w = max(5.0, min(50.0, float(default_max_w)))
    max_w    = st.sidebar.slider("Max Weight (%)", 5.0, 50.0, default_max_w, 1.0) / 100.0

    default_force_eq = opt_params.get("force_equal_weight", CFG_FORCE_EQ)
    force_eq = st.sidebar.checkbox("Force Equal Weight", default_force_eq)

    # — 7) Trading-Kosten —
    st.sidebar.subheader("Trading-Kosten")
    enable_tc  = st.sidebar.checkbox("Kosten aktiv", CFG_ENABLE_TC)
    fixed_cost = st.sidebar.number_input("Fixe Kosten pro Trade", 0.0, 100.0, CFG_FIXED_COST)
    var_cost   = st.sidebar.number_input("Variable Kosten (%)", 0.0, 1.0, CFG_VAR_COST*100) / 100.0


    # --- Risk-Overlay: feste Defaults, weil die Widgets entfernt wurden ------------
    overlay_enabled = False          # Overlay komplett aus
    low_thr  = -0.30                 # Default für Three-Band LOW
    high_thr =  0.10                 # Default für Three-Band HIGH


    # --- Buttons / Auto-Run -----------------------------------------------
    # Check for auto-run from optimizer
    if st.session_state.get("auto_run_backtest"):
        st.session_state.auto_run_backtest = False  # Reset flag
        st.session_state.optimizer_to_backtester = None  # Clear params after use
        run_btn = True
    else:
        run_btn = st.sidebar.button("Backtest starten 🚀")

    # Wenn *keiner* gedrückt wurde → zurück
    if not run_btn:
        st.info("Stelle alle Parameter ein und klicke auf einen der Start‑Buttons.")
        return

    # — VALIDIERUNG —
    if not sources:
        st.error("Bitte mindestens eine Quelle auswählen.") 
        return
    if not month:
        st.error("Bitte einen Monat auswählen.")
        return

     # --- Ticker laden ---
    tickers = dm.get_tickers_for(month, sources)
    if not tickers:
        st.error("Keine Ticker für diese Auswahl.")
        return

    # --- Preisdaten laden ---
    price_df, price_start = load_price_df(dm, tickers, start_date, end_date, window_days)
    st.write(f"⏳ Lade Preisdaten von {price_start} bis {end_date}")
    st.write(f"Price-DF nach Load: {price_df.index.min()} bis {price_df.index.max()}")

    if price_df.empty:
        st.error("Keine Preisdaten gefunden.")
        return
    

    # ‣ wenn weniger Ticker da sind als num_stocks, auf available runterschrauben
    orig_num_stocks = num_stocks
    available = price_df.shape[1]
    if available < orig_num_stocks:
        st.warning(
            f"Achtung: nur {available} Aktien verfügbar; "
            f"Backtest wird mit {available} statt {orig_num_stocks} laufen"
        )
        num_stocks = available

    #DEBUG
    #st.write("🔎 price_df shape:", price_df.shape)
    #st.write(price_df.head())
    #----------------------

    with st.spinner("📈 Backtest läuft…"):
        # ------------------------------------------
        # Overlay-Konfiguration *vor* dem Backtest
        # ------------------------------------------
        overlay_cfg_path = Path(CFG.RISK_OVERLAY["config_path"])
        cfg = json.loads(overlay_cfg_path.read_text())

        # ► Mapping-Schwellen übernehmen
        cfg["mapping"]["params"].update({"low": low_thr, "high": high_thr})
        overlay_cfg_path.write_text(json.dumps(cfg, indent=2))

        # -------------------------------------------------
        # 1) BASELINE – **ohne** Risk-Overlay
        # -------------------------------------------------
        engine_baseline = SharpeBacktestEngine(
            price_df,
            start_balance,
            num_stocks,
            start_month=start_date.strftime("%Y-%m-%d"),
            universe_mode="static" if mode.startswith("statisch") else "dynamic",
            optimizer_method=opt_method,
            cov_estimator=cov_estimator,
            rebalance_frequency=rebalance_freq,
            custom_rebalance_months=custom_months,
            window_days=window_days,
            min_weight=min_w,
            max_weight=max_w,
            force_equal_weight=force_eq,
            enable_trading_costs=enable_tc,
            fixed_cost_per_trade=fixed_cost,
            variable_cost_pct=var_cost,
            optimization_mode=opt_mode,
            use_risk_overlay=False,          # <<—— einzig relevante Änderung
        )
        engine_baseline.run_with_next_month_allocation()

        # -------------------------------------------------
        # 2) OVERLAY – Risk-On / Risk-Off (optional)
        # -------------------------------------------------
        engine_overlay = SharpeBacktestEngine(
            price_df,
            start_balance,
            num_stocks,
            start_month=start_date.strftime("%Y-%m-%d"),
            universe_mode="static" if mode.startswith("statisch") else "dynamic",
            optimizer_method=opt_method,
            cov_estimator=cov_estimator,
            rebalance_frequency=rebalance_freq,
            custom_rebalance_months=custom_months,
            window_days=window_days,
            min_weight=min_w,
            max_weight=max_w,
            force_equal_weight=force_eq,
            enable_trading_costs=enable_tc,
            fixed_cost_per_trade=fixed_cost,
            variable_cost_pct=var_cost,
            optimization_mode=opt_mode,
            use_risk_overlay=overlay_enabled,  # <<—— nur aktiv, wenn Checkbox gesetzt
        )
        engine_overlay.run_with_next_month_allocation()

        # -------------------------------------------------
        # 3) Parameter-Dict für den UI-Tab
        # -------------------------------------------------
        ui_params = {
            "Backtest Startdatum": start_date.strftime("%Y-%m-%d"),
            "Backtest Enddatum":   end_date.strftime("%Y-%m-%d"),
            "Quellen":             ", ".join(sources),
            "Periode (YYYY-MM)":    month,
            "Ticker-Universe":      mode,
            "Lookback Days":        window_days,
            "Startkapital":         start_balance,
            "Aktien pro Portfolio": num_stocks,
            "Optimierer":           opt_method,
            "Kovarianzschätzer":    cov_estimator,
            "Optimierungsmodus":    opt_mode,
            "Rebalance":            rebalance_freq,
            "Custom Monate":        custom_months if rebalance_freq == "custom" else "-",
            "Min Weight (%)":       round(min_w * 100, 2),
            "Max Weight (%)":       round(max_w * 100, 2),
            "Force Equal Weight":   force_eq,
            "Trading-Kosten aktiv": enable_tc,
            "Fixe Kosten/Trade":    fixed_cost,
            "Variable Kosten (%)":  round(var_cost * 100, 2),
            "Risk-Overlay aktiv":   overlay_enabled,      # <<—— zum schnellen Nachvollziehen
            "Three-Band Low":       low_thr,
            "Three-Band High":      high_thr,
        }

    # -----------------------------------------------------
    msg = "Backtest fertig ✅"
    if available < orig_num_stocks:
        msg += f"  (Achtung: nur {available} Stocks vorhanden statt {orig_num_stocks})"
    st.success(msg)


    # Tabs
    tabs = st.tabs([
        "Dashboard",
        "Overlay",
        "Daily",
        "Monthly",
        "Yearly",
        "Monthly Allocation",
        "Next Month Allocation",
        "Drawdowns",
        "Trading Costs",
        "Rebalance",
        "Paramter",
        "Logs"
    ])

    # ────────────────────────────────────────────────────────────────
    # Tabs-Block (1:1 ersetzbar)
    # ────────────────────────────────────────────────────────────────
    with tabs[0]:  # Dashboard
        st.subheader("🔍 KPIs")

        # KPI-Tabelle als statische Tabelle (st.table ⇒ Styling bleibt)
        st.table(fmt_df(engine_baseline.performance_metrics))

        st.markdown("---")
        st.subheader("📈 Portfolio-Verlauf")
        st.line_chart(
            pd.DataFrame({"Baseline": engine_baseline.portfolio_value})
        )

        st.markdown("---")
        st.subheader("📆 Monatliche Performance (%)")
        perf_df = pd.DataFrame({
            "Baseline": engine_baseline.monthly_performance
                        .set_index("Date")["Monthly PnL (%)"]
        })
        st.bar_chart(perf_df)

        # Track Portfolio Section
        st.markdown("---")
        st.subheader("📊 Portfolio Tracking")

        with st.expander("Track this portfolio for performance monitoring", expanded=False):
            try:
                from AlphaMachine_core.tracking.registration import (
                    register_portfolio_from_backtest,
                    get_suggested_portfolio_name,
                    check_portfolio_name_exists,
                )

                # Get suggested name
                suggested_name = get_suggested_portfolio_name(
                    source=sources[0] if sources else "Unknown",
                    num_stocks=num_stocks,
                    optimizer=opt_method,
                )

                # Portfolio name input
                portfolio_name = st.text_input(
                    "Portfolio Name",
                    value=suggested_name,
                    help="Unique name for this portfolio"
                )

                portfolio_description = st.text_area(
                    "Description (optional)",
                    value="",
                    help="Optional description of this portfolio strategy"
                )

                # Check if name exists
                name_exists = check_portfolio_name_exists(portfolio_name)
                if name_exists:
                    st.warning(f"⚠️ Portfolio '{portfolio_name}' already exists. Choose a different name.")

                # Get holdings from the next month allocation
                if hasattr(engine_baseline, 'allocation_history') and engine_baseline.allocation_history:
                    last_alloc = engine_baseline.allocation_history[-1]
                    holdings_df = pd.DataFrame([
                        {"ticker": ticker, "weight": weight}
                        for ticker, weight in last_alloc.get("weights", {}).items()
                    ])
                else:
                    holdings_df = pd.DataFrame()

                # Register button
                if st.button("🚀 Start Tracking", disabled=name_exists or not portfolio_name):
                    with st.spinner("Registering portfolio..."):
                        result = register_portfolio_from_backtest(
                            name=portfolio_name,
                            backtest_params=ui_params,
                            holdings_df=holdings_df,
                            nav_history=engine_baseline.portfolio_value,
                            source=sources[0] if sources else "Unknown",
                            description=portfolio_description if portfolio_description else None,
                        )

                        if result.get("success"):
                            st.success(
                                f"✅ Portfolio '{portfolio_name}' registered successfully!\n\n"
                                f"- Portfolio ID: {result.get('portfolio_id')}\n"
                                f"- Holdings: {result.get('holdings_count', 0)}\n"
                                f"- Historical NAV records: {result.get('backfilled_nav_count', 0)}\n\n"
                                f"View it in the **Performance Tracker** page."
                            )
                        else:
                            st.error(f"❌ Failed to register portfolio: {result.get('error')}")

            except ImportError as e:
                st.info(
                    "Portfolio tracking module not available. "
                    "Run the migration script first: `python scripts/migrate_tracking_tables.py`"
                )
            except Exception as e:
                st.error(f"Error loading tracking module: {e}")


    with tabs[1]:  # Overlay
        st.subheader("⚖️ Risk-On / Risk-Off Overlay")
        overlay_obj = engine_overlay.risk_overlay
        if overlay_obj is None:
            st.info("Overlay ist in der config deaktiviert.")
        elif not overlay_obj.score_log:
            st.info("Noch keine Scores berechnet – starte zuerst den Backtest.")
        else:
            df_score = (pd.DataFrame(overlay_obj.score_log)
                        .set_index("date")
                        .rename(columns={"score": "Aggregated Score"}))
            st.line_chart(df_score, height=200)

            last_score  = df_score["Aggregated Score"].iloc[-1]
            last_weight = overlay_obj.map_to_equity_weight(last_score)
            st.markdown(
                f"**Aktuelle Ziel-Aktienquote:** {last_weight:.0%}  "
                f"(Score {last_score:+.2f})"
            )


    with tabs[2]:  # Daily
        st.subheader("📅 Daily Portfolio Baseline")
        df_daily = engine_baseline.daily_df.copy()

        if isinstance(df_daily.index, pd.DatetimeIndex):
            df_daily.index = df_daily.index.date
        elif "Date" in df_daily.columns:
            # Falls Datum als Spalte existiert (selten bei Daily-DFs)
            df_daily["Date"] = pd.to_datetime(df_daily["Date"]).dt.date

        st.dataframe(fmt_df(df_daily), use_container_width=True)


    with tabs[3]:  # Monthly
        st.subheader("🗓️ Monthly Performance Baseline")
        if not engine_baseline.monthly_performance.empty:
            df_mon = engine_baseline.monthly_performance.copy()
            df_mon["Date"] = (pd.to_datetime(df_mon["Date"])
                            .dt.strftime("%Y - %b"))  # «2025 - Jan»
            st.dataframe(fmt_df(df_mon), use_container_width=True)
        else:
            st.info("Keine Monatsdaten für Baseline.")


    with tabs[4]:  # Yearly
        st.subheader("🗓️ Yearly Performance Baseline")
        if not engine_baseline.portfolio_value.empty:
            yearly_balance = engine_baseline.portfolio_value.resample("YE").last()
            yearly_start   = engine_baseline.portfolio_value.resample("YE").first()
            yearly_pnl     = yearly_balance - yearly_start
            yearly_ret     = yearly_balance.pct_change()*100

            monthly_pnl = engine_baseline.monthly_performance.copy()
            monthly_pnl["Year"] = pd.to_datetime(monthly_pnl["Date"]).dt.year
            yearly_monthly_pnl = monthly_pnl.groupby("Year")["Monthly PnL ($)"].sum()

            df_year = pd.DataFrame({
                "Year": yearly_balance.index.year.astype(str),  # als Text ⇒ kein «2,025»
                "Yearly PnL (Portfolio Value)": yearly_pnl.values,
                "Yearly PnL (Sum Monthly)": yearly_monthly_pnl
                                            .reindex(yearly_balance.index.year)
                                            .values,
                "Return (%)": yearly_ret.values,
                "Balance": yearly_balance.values
            }).reset_index(drop=True)

            st.dataframe(fmt_df(df_year), use_container_width=True)
        else:
            st.info("Keine Jahresdaten für Baseline.")


    with tabs[5]:  # Monthly Allocation
        st.subheader("📊 Monthly Allocation Baseline")
        if not engine_baseline.monthly_allocations.empty:
            df_sorted = engine_baseline.monthly_allocations.sort_values(
                by="Rebalance Date", ascending=False
            )
            st.dataframe(fmt_df(df_sorted), use_container_width=True)
        else:
            st.info("Keine Daten für Baseline.")


    with tabs[6]:  # Next-Month Allocation
        st.subheader("🔮 Next Month Allocation Baseline")
        if hasattr(engine_baseline, "next_month_weights"):
            df_next = (engine_baseline.next_month_weights
                    .mul(100)
                    .reset_index())
            df_next.columns = ["Ticker", "Gewicht (%)"]
            st.dataframe(fmt_df(df_next), use_container_width=True)
        else:
            st.info("Keine Auswahl für den Folgemonat (Baseline).")


    with tabs[7]:  # Drawdowns
        st.subheader("📉 Top 10 Drawdowns Baseline")
        df_port = engine_baseline.portfolio_value.to_frame(name="Portfolio")
        df_port["Peak"] = df_port["Portfolio"].cummax()
        df_port["Drawdown"] = df_port["Portfolio"] / df_port["Peak"] - 1

        periods, in_dd = [], False
        for date, row in df_port.iterrows():
            if not in_dd and row["Drawdown"] < 0:
                in_dd, start = True, date
                peak_val = row["Peak"]; trough_val = row["Portfolio"]; trough = date
            elif in_dd:
                if row["Portfolio"] < trough_val:
                    trough_val, trough = row["Portfolio"], date
                if row["Portfolio"] >= peak_val:
                    periods.append({
                        "Start":         start.date(),
                        "Trough":        trough.date(),
                        "End":           date.date(),
                        "Length (Days)": (date-start).days,
                        "Recovery Time": (date-trough).days,
                        "Drawdown (%)":  round((trough_val/peak_val-1)*100, 2),
                    })
                    in_dd = False
        if in_dd:  # offenes DD-Ende
            last_date = df_port.index[-1]
            periods.append({
                "Start":         start.date(),
                "Trough":        trough.date(),
                "End":           last_date.date(),
                "Length (Days)": (last_date-start).days,
                "Recovery Time": None,
                "Drawdown (%)":  round((trough_val/peak_val-1)*100, 2),
            })

        df_dd = pd.DataFrame(periods)
        if "Drawdown (%)" in df_dd.columns and not df_dd.empty:
            df_dd = (df_dd.sort_values(by="Drawdown (%)")
                        .head(10)
                        .reset_index(drop=True))
            st.dataframe(fmt_df(df_dd), use_container_width=True)
        else:
            st.info("Keine Drawdown-Daten für dieses Portfolio.")


    # Tab 5: Trading Costs
    with tabs[8]:
        st.subheader("💸 Trading Costs Baseline")
        if (not engine_baseline.monthly_allocations.empty and
            "Trading Costs" in engine_baseline.monthly_allocations):
            cost_df = (engine_baseline.monthly_allocations
                    .dropna(subset=["Trading Costs"])
                    .groupby("Rebalance Date")["Trading Costs"]
                    .sum()
                    .reset_index(name="Total Trading Costs"))
            cost_df["Rebalance Date"] = pd.to_datetime(cost_df["Rebalance Date"]).dt.date
            st.dataframe(fmt_df(cost_df), use_container_width=True)
        else:
            st.info("Keine Trading-Kosten-Daten für Baseline.")


    # Tab 6: Rebalance Analysis
    with tabs[9]:
        st.subheader("🔁 Rebalance Analysis")
        if (hasattr(engine_baseline, 'selection_details')
            and engine_baseline.selection_details):
            try:
                df_rebalance = pd.DataFrame(engine_baseline.selection_details)
                if "Rebalance Date" in df_rebalance.columns:
                    df_rebalance = df_rebalance[
                        df_rebalance["Rebalance Date"] != "SUMMARY"
                    ].copy()

                if not df_rebalance.empty:
                    if "Rebalance Date" in df_rebalance.columns:
                        df_rebalance.loc[:, "Rebalance Date"] = (
                            pd.to_datetime(df_rebalance["Rebalance Date"],
                                        errors='coerce').dt.date  # Zeit weg
                        )
                    st.dataframe(fmt_df(df_rebalance), use_container_width=True)
                else:
                    st.info("Keine Rebalance-Detaildaten für die Anzeige.")
            except Exception as e:
                st.error(f"Fehler beim Anzeigen der Rebalance-Details: {e}")
                st.write("Struktur von selection_details:",
                        engine_baseline.selection_details[:2])
        else:
            st.info("Keine Rebalance-Detaildaten vorhanden.")


    # Tab 7: Parameters
    with tabs[10]:
        st.subheader("⚙️ Ausgewählte Backtest-Parameter")
        df_params = pd.DataFrame(ui_params.items(),
                                columns=["Parameter", "Wert"])
        df_params["Wert"] = df_params["Wert"].astype(str)
        st.dataframe(fmt_df(df_params), use_container_width=True)


    # Tab 8: Logs
    with tabs[11]:
        st.subheader("🪵 Logs")
        for line in (engine_baseline.ticker_coverage_logs +
                    engine_baseline.log_lines):
            st.text(line)


    # Excel Download: Baseline und Overlay getrennt
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Baseline-Report
        path_baseline = os.path.join(tmp_dir, f"AlphaMachine_Baseline_{dt.date.today()}.xlsx")
        export_results_to_excel(engine_baseline, path_baseline)
        with open(path_baseline, "rb") as f:
            st.download_button(
                "📥 Excel-Report Baseline",
                f.read(),
                file_name=os.path.basename(path_baseline),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    

# =============================================================================
# === Portfolio Holdings Selection UI ===
# =============================================================================
def _show_portfolio_holdings_ui():
    """
    UI for selecting which tickers from the universe (ticker_period)
    should be included in the actual portfolio (portfolio_holding).
    """
    from decimal import Decimal
    from AlphaMachine_core.tracking.models import PortfolioDefinition, PortfolioHolding

    st.subheader("Portfolio Holdings Management")
    st.markdown("""
    Select which stocks from the **universe** (ticker_period) should be included in the **actual portfolio** (portfolio_holding).

    - **Universe**: All candidate stocks loaded for a source/period
    - **Portfolio**: Only the selected stocks that are actually tracked
    """)

    # Load all portfolios - extract data while session is open to avoid DetachedInstanceError
    try:
        with get_session() as session:
            portfolios_raw = session.exec(
                select(PortfolioDefinition).where(PortfolioDefinition.is_active == True)
            ).all()
            # Extract data while session is open
            portfolios = [
                {
                    "id": p.id,
                    "name": p.name,
                    "source": p.source,
                    "start_date": p.start_date,
                }
                for p in portfolios_raw
            ]
    except Exception as e:
        st.error(f"Error loading portfolios: {e}")
        return

    if not portfolios:
        st.warning("No active portfolios found. Create a portfolio first via the Backtester page.")
        return

    # Portfolio selector
    portfolio_options = {p["name"]: p for p in portfolios}
    selected_portfolio_name = st.selectbox(
        "Select Portfolio",
        options=list(portfolio_options.keys()),
        key="holdings_portfolio_selector"
    )
    selected_portfolio = portfolio_options[selected_portfolio_name]

    # Show portfolio info
    st.info(f"**Source**: {selected_portfolio['source'] or 'Unknown'} | **Start Date**: {selected_portfolio['start_date']}")

    portfolio_source = selected_portfolio["source"]
    if not portfolio_source:
        st.warning("This portfolio has no source defined. Cannot load universe tickers.")
        return

    # Get available months for this source
    try:
        with get_session() as session:
            from sqlalchemy import func
            months_result = session.exec(
                select(func.to_char(TickerPeriod.start_date, 'YYYY-MM'))
                .where(TickerPeriod.source == portfolio_source)
                .distinct()
                .order_by(func.to_char(TickerPeriod.start_date, 'YYYY-MM').desc())
            ).all()
            available_months = [str(m) for m in months_result if m]
    except Exception as e:
        st.error(f"Error loading months: {e}")
        return

    if not available_months:
        st.warning(f"No ticker periods found for source '{portfolio_source}'")
        return

    # Month selector
    selected_month = st.selectbox(
        "Select Period (Month)",
        options=available_months,
        key="holdings_month_selector"
    )

    # Parse month to date
    year, month = selected_month.split("-")
    effective_date = dt.date(int(year), int(month), 1)

    # Get universe tickers for this source/month
    try:
        with get_session() as session:
            universe_tickers = session.exec(
                select(TickerPeriod.ticker)
                .where(TickerPeriod.source == portfolio_source)
                .where(func.to_char(TickerPeriod.start_date, 'YYYY-MM') == selected_month)
                .distinct()
                .order_by(TickerPeriod.ticker)
            ).all()
            universe_tickers = sorted([str(t) for t in universe_tickers if t])
    except Exception as e:
        st.error(f"Error loading universe tickers: {e}")
        return

    if not universe_tickers:
        st.warning(f"No tickers in universe for {portfolio_source} / {selected_month}")
        return

    # Get currently selected tickers (already in portfolio_holding)
    try:
        with get_session() as session:
            current_holdings = session.exec(
                select(PortfolioHolding.ticker)
                .where(PortfolioHolding.portfolio_id == selected_portfolio["id"])
                .where(PortfolioHolding.effective_date == effective_date)
            ).all()
            current_selected = set(str(t) for t in current_holdings if t)
    except Exception as e:
        st.error(f"Error loading current holdings: {e}")
        current_selected = set()

    st.markdown("---")
    st.markdown(f"### Universe: {len(universe_tickers)} stocks | Currently Selected: {len(current_selected)}")

    # Create checkboxes for each ticker
    st.markdown("**Select stocks for portfolio:**")

    # Use columns for better layout (4 columns)
    num_cols = 4
    cols = st.columns(num_cols)

    # Track selections
    selected_tickers = []
    for i, ticker in enumerate(universe_tickers):
        col_idx = i % num_cols
        with cols[col_idx]:
            is_checked = st.checkbox(
                ticker,
                value=ticker in current_selected,
                key=f"ticker_select_{selected_portfolio['id']}_{selected_month}_{ticker}"
            )
            if is_checked:
                selected_tickers.append(ticker)

    st.markdown("---")

    # Summary and action buttons
    col_info, col_save, col_clear = st.columns([2, 1, 1])

    with col_info:
        st.markdown(f"**Selected: {len(selected_tickers)} of {len(universe_tickers)} stocks**")
        if len(selected_tickers) > 0:
            weight = 1.0 / len(selected_tickers)
            st.markdown(f"Equal weight per stock: **{weight*100:.2f}%**")

    with col_save:
        if st.button("Save Holdings", type="primary", key="save_holdings_btn"):
            try:
                with get_session() as session:
                    # Delete existing holdings for this portfolio/date
                    existing = session.exec(
                        select(PortfolioHolding)
                        .where(PortfolioHolding.portfolio_id == selected_portfolio["id"])
                        .where(PortfolioHolding.effective_date == effective_date)
                    ).all()

                    for h in existing:
                        session.delete(h)

                    # Add new holdings
                    weight = Decimal("1") / Decimal(str(len(selected_tickers))) if selected_tickers else Decimal("0")

                    for ticker in selected_tickers:
                        new_holding = PortfolioHolding(
                            portfolio_id=selected_portfolio["id"],
                            effective_date=effective_date,
                            ticker=ticker,
                            weight=weight,
                        )
                        session.add(new_holding)

                    session.commit()

                st.success(f"Saved {len(selected_tickers)} holdings for {selected_month}!")
                st.info("Note: You may need to re-run NAV backfill to update historical returns with the new holdings.")
                st.rerun()

            except Exception as e:
                st.error(f"Error saving holdings: {e}")

    with col_clear:
        if len(current_selected) > 0:
            if st.button("Clear All", type="secondary", key="clear_holdings_btn", help="Delete all holdings for this month"):
                try:
                    with get_session() as session:
                        # Delete all holdings for this portfolio/date
                        existing = session.exec(
                            select(PortfolioHolding)
                            .where(PortfolioHolding.portfolio_id == selected_portfolio["id"])
                            .where(PortfolioHolding.effective_date == effective_date)
                        ).all()

                        deleted_count = len(existing)
                        for h in existing:
                            session.delete(h)

                        session.commit()

                    st.success(f"Cleared {deleted_count} holdings for {selected_month}!")
                    st.rerun()

                except Exception as e:
                    st.error(f"Error clearing holdings: {e}")

    # Show current holdings table
    with st.expander("View Current Holdings Details", expanded=False):
        try:
            with get_session() as session:
                holdings = session.exec(
                    select(PortfolioHolding)
                    .where(PortfolioHolding.portfolio_id == selected_portfolio["id"])
                    .where(PortfolioHolding.effective_date == effective_date)
                    .order_by(PortfolioHolding.ticker)
                ).all()

                if holdings:
                    holdings_data = [
                        {
                            "Ticker": h.ticker,
                            "Weight": f"{float(h.weight)*100:.2f}%" if h.weight else "-",
                            "Entry Price": f"${float(h.entry_price):.2f}" if h.entry_price else "-",
                        }
                        for h in holdings
                    ]
                    st.dataframe(pd.DataFrame(holdings_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No holdings saved for this period yet.")
        except Exception as e:
            st.error(f"Error loading holdings details: {e}")

    # NAV Update Section
    st.markdown("---")
    st.markdown("### NAV Update")
    st.markdown("Run NAV calculation for this portfolio to see performance immediately.")

    # Date range inputs
    col_start, col_end = st.columns(2)
    with col_start:
        nav_start_date = st.date_input(
            "Start Date",
            value=selected_portfolio["start_date"],
            key="nav_update_start_date"
        )
    with col_end:
        nav_end_date = st.date_input(
            "End Date",
            value=dt.date.today(),
            key="nav_update_end_date"
        )

    if st.button("Run NAV Update", type="secondary", key="run_nav_update_btn"):
        _run_nav_update_with_progress(
            selected_portfolio["name"],
            nav_start_date,
            nav_end_date
        )


def _show_delete_portfolios_sources_ui():
    """
    UI for deleting portfolios or sources.

    - Delete Portfolio: Removes portfolio and ALL related data (holdings, NAV, metrics)
    - Delete Source: Removes all ticker_period entries for that source
    """
    from AlphaMachine_core.tracking.models import (
        PortfolioDefinition, PortfolioHolding, PortfolioDailyNAV, PortfolioMetric
    )
    from sqlalchemy import delete

    st.subheader("Delete Portfolios / Sources")

    st.warning("This action is **permanent** and cannot be undone!")

    delete_type = st.radio(
        "What do you want to delete?",
        ["Portfolio", "Source"],
        key="delete_type_radio"
    )

    if delete_type == "Portfolio":
        st.markdown("---")
        st.markdown("### Delete Portfolio")
        st.markdown("""
        Deleting a portfolio will remove:
        - Portfolio definition
        - All holdings (all periods)
        - All daily NAV data (all variants)
        - All periodic metrics
        - Audit log entries
        """)

        # Load all portfolios
        try:
            with get_session() as session:
                portfolios_raw = session.exec(select(PortfolioDefinition)).all()
                portfolios = [
                    {"id": p.id, "name": p.name, "source": p.source, "is_active": p.is_active}
                    for p in portfolios_raw
                ]
        except Exception as e:
            st.error(f"Error loading portfolios: {e}")
            return

        if not portfolios:
            st.info("No portfolios found.")
            return

        # Show portfolio list
        portfolios_df = pd.DataFrame(portfolios)
        st.dataframe(portfolios_df, use_container_width=True, height=min(300, len(portfolios_df)*38 + 58))

        # Portfolio selector
        portfolio_options = {p["name"]: p for p in portfolios}
        selected_portfolio_name = st.selectbox(
            "Select Portfolio to Delete",
            options=["-- Select --"] + list(portfolio_options.keys()),
            key="delete_portfolio_select"
        )

        if selected_portfolio_name != "-- Select --":
            selected_portfolio = portfolio_options[selected_portfolio_name]
            portfolio_id = selected_portfolio["id"]

            # Count related records
            try:
                from sqlalchemy import func
                with get_session() as session:
                    holdings_count = session.exec(
                        select(func.count()).select_from(PortfolioHolding).where(
                            PortfolioHolding.portfolio_id == portfolio_id
                        )
                    ).one()
                    nav_count = session.exec(
                        select(func.count()).select_from(PortfolioDailyNAV).where(
                            PortfolioDailyNAV.portfolio_id == portfolio_id
                        )
                    ).one()
                    metrics_count = session.exec(
                        select(func.count()).select_from(PortfolioMetric).where(
                            PortfolioMetric.portfolio_id == portfolio_id
                        )
                    ).one()

                st.markdown(f"""
                **Records to be deleted:**
                - Holdings: {holdings_count}
                - NAV records: {nav_count}
                - Metrics: {metrics_count}
                """)
            except Exception as e:
                st.warning(f"Could not count records: {e}")

            # Confirmation
            confirm_text = st.text_input(
                f"Type **{selected_portfolio_name}** to confirm deletion:",
                key="confirm_portfolio_delete"
            )

            if st.button("Delete Portfolio", type="primary", key="btn_delete_portfolio"):
                if confirm_text == selected_portfolio_name:
                    try:
                        with get_session() as session:
                            # Delete in order (child tables first)
                            session.exec(delete(PortfolioHolding).where(
                                PortfolioHolding.portfolio_id == portfolio_id
                            ))
                            session.exec(delete(PortfolioDailyNAV).where(
                                PortfolioDailyNAV.portfolio_id == portfolio_id
                            ))
                            session.exec(delete(PortfolioMetric).where(
                                PortfolioMetric.portfolio_id == portfolio_id
                            ))
                            # Delete audit log if exists
                            try:
                                from sqlalchemy import text
                                session.exec(text(
                                    "DELETE FROM nav_audit_log WHERE portfolio_id = :pid"
                                ), {"pid": portfolio_id})
                            except:
                                pass  # Table might not exist
                            # Delete portfolio definition
                            portfolio_to_delete = session.get(PortfolioDefinition, portfolio_id)
                            if portfolio_to_delete:
                                session.delete(portfolio_to_delete)
                            session.commit()

                        st.success(f"Portfolio '{selected_portfolio_name}' and all related data deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting portfolio: {e}")
                else:
                    st.error("Confirmation text does not match. Please type the exact portfolio name.")

    else:  # Delete Source
        st.markdown("---")
        st.markdown("### Delete Source")
        st.markdown("""
        Deleting a source will remove all **ticker_period** entries for that source.

        This does NOT delete:
        - Portfolios using this source
        - Stock price data
        """)

        # Load all sources
        try:
            with get_session() as session:
                from AlphaMachine_core.models import TickerPeriod
                sources_raw = session.exec(select(TickerPeriod.source).distinct()).all()
                sources = sorted([str(s) for s in sources_raw if s])
        except Exception as e:
            st.error(f"Error loading sources: {e}")
            return

        if not sources:
            st.info("No sources found.")
            return

        # Show source counts
        try:
            with get_session() as session:
                from sqlalchemy import func
                source_counts = session.exec(
                    select(TickerPeriod.source, func.count(TickerPeriod.id))
                    .group_by(TickerPeriod.source)
                ).all()
                source_df = pd.DataFrame(source_counts, columns=["Source", "Ticker Count"])
                st.dataframe(source_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load source counts: {e}")

        # Source selector
        selected_source = st.selectbox(
            "Select Source to Delete",
            options=["-- Select --"] + sources,
            key="delete_source_select"
        )

        if selected_source != "-- Select --":
            # Count records to delete
            try:
                from sqlalchemy import func
                with get_session() as session:
                    from AlphaMachine_core.models import TickerPeriod
                    count = session.exec(
                        select(func.count()).select_from(TickerPeriod).where(
                            TickerPeriod.source == selected_source
                        )
                    ).one()

                st.markdown(f"**{count} ticker_period records will be deleted.**")
            except Exception as e:
                st.warning(f"Could not count records: {e}")

            # Confirmation
            confirm_text = st.text_input(
                f"Type **{selected_source}** to confirm deletion:",
                key="confirm_source_delete"
            )

            if st.button("Delete Source", type="primary", key="btn_delete_source"):
                if confirm_text == selected_source:
                    try:
                        with get_session() as session:
                            from AlphaMachine_core.models import TickerPeriod
                            session.exec(delete(TickerPeriod).where(
                                TickerPeriod.source == selected_source
                            ))
                            session.commit()

                        st.success(f"Source '{selected_source}' and all ticker_period entries deleted successfully!")
                        # Clear cache
                        if 'all_periods_for_filters_cached_data_ui' in st.session_state:
                            del st.session_state['all_periods_for_filters_cached_data_ui']
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting source: {e}")
                else:
                    st.error("Confirmation text does not match. Please type the exact source name.")


def _show_cleanup_unused_tickers_ui():
    """
    UI for cleaning up price data for tickers that are truly orphaned.

    A ticker is SAFE to delete only if:
    1. It has NEVER been in any portfolio holdings (current or historical)
    2. It's not part of any active source

    This preserves:
    - Tickers currently in portfolios
    - Tickers that were historically in portfolios
    - Tickers in active screening sources (Topweights, etc.)
    """
    from AlphaMachine_core.tracking.models import PortfolioHolding
    from AlphaMachine_core.models import TickerPeriod
    from sqlalchemy import delete, func

    st.subheader("Cleanup Unused Price Data")
    st.info("""
    This tool identifies **truly orphaned** tickers that can be safely deleted:
    - Never been in any portfolio (current or historical)
    - **Not** in the selected period (default: latest month)

    **Preserved data:**
    - Tickers currently in portfolios
    - Tickers historically in portfolios (for backtesting)
    - Tickers in the selected period (for screening)
    """)

    # Step 1: Gather all data
    try:
        with get_session() as session:
            # Get ALL tickers ever in portfolio holdings (current + historical)
            portfolio_tickers_raw = session.exec(
                select(PortfolioHolding.ticker).distinct()
            ).all()
            portfolio_tickers = set(str(t) for t in portfolio_tickers_raw if t)

            # Get all unique tickers from ticker_period (price data)
            price_tickers_raw = session.exec(
                select(TickerPeriod.ticker).distinct()
            ).all()
            price_tickers = set(str(t) for t in price_tickers_raw if t)

            # Get all active sources
            active_sources_raw = session.exec(
                select(TickerPeriod.source).distinct()
            ).all()
            active_sources = set(str(s) for s in active_sources_raw if s)

    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        return

    # Calculate categories
    used_in_portfolios = price_tickers & portfolio_tickers
    never_in_portfolios = price_tickers - portfolio_tickers

    # Get all unique periods from ticker_period.start_date
    from sqlalchemy import func
    all_periods = []
    try:
        with get_session() as session:
            periods_raw = session.exec(
                select(func.to_char(TickerPeriod.start_date, 'YYYY-MM').label('period'))
                .distinct()
                .order_by(func.to_char(TickerPeriod.start_date, 'YYYY-MM').desc())
            ).all()
            all_periods = [str(p) for p in periods_raw if p]
    except Exception:
        pass

    # Period selection dropdown (default to latest)
    st.markdown("### Select Current Period")
    if all_periods:
        selected_period = st.selectbox(
            "Protect tickers from this period (keep for screening):",
            options=all_periods,
            index=0,  # Default to latest
            key="cleanup_period_selector",
            help="Tickers with start_date in this period will NOT be deleted"
        )
    else:
        selected_period = None
        st.warning("No periods found in database")

    # Get tickers from selected period
    tickers_in_current_period = set()
    if selected_period:
        try:
            with get_session() as session:
                tickers_raw = session.exec(
                    select(TickerPeriod.ticker).distinct().where(
                        func.to_char(TickerPeriod.start_date, 'YYYY-MM') == selected_period
                    )
                ).all()
                tickers_in_current_period.update(str(t) for t in tickers_raw if t)
        except Exception:
            pass

    # Truly orphaned = never in portfolios AND not in current period
    truly_orphaned = never_in_portfolios - tickers_in_current_period
    in_current_period_only = never_in_portfolios & tickers_in_current_period

    # Display summary
    st.markdown("### Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickers in Price DB", len(price_tickers))
    col2.metric("Ever in Portfolios", len(used_in_portfolios),
                help="These are SAFE - used or were used in portfolios")
    col3.metric(f"In {selected_period or 'N/A'}", len(in_current_period_only),
                help=f"Never in portfolios but in {selected_period} period - KEPT for screening")
    col4.metric("Truly Orphaned", len(truly_orphaned),
                help="Never in portfolios AND not in current period - SAFE to delete", delta=f"-{len(truly_orphaned)}" if truly_orphaned else None, delta_color="normal")

    if not truly_orphaned:
        st.success("✅ No truly orphaned tickers found. All tickers are either in portfolios or the selected period.")
        if in_current_period_only:
            st.info(f"ℹ️ {len(in_current_period_only)} tickers are in {selected_period} but not in portfolios - these are preserved for screening.")
        return

    # Show tickers that are truly orphaned
    st.markdown("---")
    st.markdown("### Truly Orphaned Tickers")
    st.warning(f"""
    The following **{len(truly_orphaned)}** tickers:
    - Have **never** been in any portfolio (current or historical)
    - Are **not** in the selected period ({selected_period})

    These are from **older periods** and are safe to delete.
    """)

    # Get ticker info with sources
    try:
        with get_session() as session:
            # Get all ticker_period entries for truly orphaned tickers
            ticker_source_data = []
            source_summary = {}

            for ticker in sorted(truly_orphaned):
                # Get sources for this ticker
                sources = session.exec(
                    select(TickerPeriod.source).distinct().where(
                        TickerPeriod.ticker == ticker
                    )
                ).all()
                sources_list = [str(s) for s in sources if s]
                sources_str = ", ".join(sources_list) if sources_list else "Unknown"

                # Count records
                count = session.exec(
                    select(func.count()).select_from(TickerPeriod).where(
                        TickerPeriod.ticker == ticker
                    )
                ).one()

                ticker_source_data.append({
                    "Ticker": ticker,
                    "Sources": sources_str,
                    "Records": count
                })

                # Track by source for summary
                for src in sources_list:
                    if src not in source_summary:
                        source_summary[src] = {"tickers": set(), "records": 0}
                    source_summary[src]["tickers"].add(ticker)
                    source_summary[src]["records"] += count

            # Show source summary first
            st.markdown("#### Summary by Source (Old Sources Only)")
            st.caption("These are OLD sources (not current month) with orphaned tickers:")
            source_df = pd.DataFrame([
                {"Source": src, "Tickers (never in portfolio)": len(data["tickers"]), "Records": data["records"]}
                for src, data in sorted(source_summary.items())
            ])
            st.dataframe(source_df, use_container_width=True)

            # Show detailed ticker list
            st.markdown("#### Detailed Ticker List")
            ticker_df = pd.DataFrame(ticker_source_data)
            total_records = ticker_df["Records"].sum()
            st.dataframe(ticker_df, use_container_width=True, height=300)
            st.markdown(f"**Total records: {total_records:,}**")

    except Exception as e:
        st.warning(f"Could not load source info: {e}")
        st.write(sorted(truly_orphaned))
        source_summary = {}

    # Deletion options
    st.markdown("---")
    st.markdown("### Delete Options")
    st.caption("These tickers are safe to delete - they are from old sources and were never in any portfolio.")

    delete_option = st.radio(
        "What do you want to delete?",
        ["Delete by SOURCE (recommended)", "Delete specific tickers", "Delete ALL orphaned tickers"],
        key="cleanup_delete_option"
    )

    tickers_to_delete = set()

    if delete_option == "Delete by SOURCE (recommended)":
        if source_summary:
            sources_to_delete = st.multiselect(
                "Select SOURCES to delete (tickers never in portfolios from these sources):",
                options=sorted(source_summary.keys()),
                key="cleanup_select_sources",
                help="Only deletes tickers from selected sources that have NEVER been in any portfolio"
            )
            for src in sources_to_delete:
                tickers_to_delete.update(source_summary[src]["tickers"])
            if sources_to_delete:
                st.info(f"Selected {len(tickers_to_delete)} tickers from {len(sources_to_delete)} sources")
        else:
            st.warning("Could not load source information")

    elif delete_option == "Delete specific tickers":
        selected_to_delete = st.multiselect(
            "Select tickers to delete:",
            options=sorted(truly_orphaned),
            key="cleanup_select_tickers"
        )
        tickers_to_delete = set(selected_to_delete)

    else:  # Delete ALL orphaned
        st.info(f"This will delete ALL {len(truly_orphaned)} truly orphaned tickers (from old sources, never in portfolios).")
        tickers_to_delete = truly_orphaned

    if not tickers_to_delete:
        st.info("Select tickers to delete.")
        return

    # Confirmation
    st.markdown("---")
    st.warning(f"⚠️ You are about to delete price data for **{len(tickers_to_delete)}** tickers. This action is **permanent**!")

    confirm_text = st.text_input(
        f"Type **DELETE {len(tickers_to_delete)}** to confirm:",
        key="confirm_cleanup_delete"
    )

    if st.button("🗑️ Delete Unused Price Data", type="primary", key="btn_cleanup_delete"):
        expected_confirm = f"DELETE {len(tickers_to_delete)}"
        if confirm_text == expected_confirm:
            try:
                from sqlalchemy import text

                tickers_list = sorted(tickers_to_delete)
                total_tickers = len(tickers_list)
                batch_size = 50  # Delete 50 tickers per batch for efficiency

                progress_bar = st.progress(0.0)
                status_text = st.empty()
                records_deleted = 0
                tickers_deleted = 0

                with get_session() as session:
                    # Process in batches for efficiency
                    for batch_start in range(0, total_tickers, batch_size):
                        batch_end = min(batch_start + batch_size, total_tickers)
                        batch = tickers_list[batch_start:batch_end]

                        status_text.info(f"Deleting batch {batch_start//batch_size + 1}/{(total_tickers + batch_size - 1)//batch_size} ({batch_start+1}-{batch_end} of {total_tickers} tickers)...")

                        # Use raw SQL with IN clause for much faster deletion
                        placeholders = ', '.join([f':t{i}' for i in range(len(batch))])
                        params = {f't{i}': ticker for i, ticker in enumerate(batch)}

                        # First count records to be deleted (for reporting)
                        count_result = session.exec(
                            text(f"SELECT COUNT(*) FROM ticker_period WHERE ticker IN ({placeholders})"),
                            params
                        ).one()
                        batch_records = count_result[0] if isinstance(count_result, tuple) else count_result

                        # Delete the batch
                        session.exec(
                            text(f"DELETE FROM ticker_period WHERE ticker IN ({placeholders})"),
                            params
                        )
                        session.commit()

                        records_deleted += batch_records
                        tickers_deleted += len(batch)
                        progress_bar.progress(batch_end / total_tickers)

                    deleted_count = tickers_deleted

                status_text.empty()
                st.success(f"✅ Successfully deleted **{records_deleted:,}** records for **{deleted_count}** tickers!")
                # Clear cache
                if 'all_periods_for_filters_cached_data_ui' in st.session_state:
                    del st.session_state['all_periods_for_filters_cached_data_ui']
                st.rerun()

            except Exception as e:
                st.error(f"Error deleting tickers: {e}")
        else:
            st.error(f"Confirmation text does not match. Please type exactly: {expected_confirm}")


def _show_ticker_analysis_ui():
    """
    Ticker Analysis UI with two tabs:
    - Tab 1: Cross-portfolio ticker matrix (which ticker is in which portfolio for a month)
    - Tab 2: Monthly ticker history for a single portfolio with retention highlighting
    """
    from AlphaMachine_core.tracking.models import PortfolioDefinition, PortfolioHolding
    from sqlalchemy import func

    st.subheader("Ticker Analysis")

    # Load all active portfolios
    try:
        with get_session() as session:
            portfolios_raw = session.exec(
                select(PortfolioDefinition).where(PortfolioDefinition.is_active == True)
            ).all()
            portfolios = [
                {"id": p.id, "name": p.name, "source": p.source, "start_date": p.start_date}
                for p in portfolios_raw
            ]
    except Exception as e:
        st.error(f"Error loading portfolios: {e}")
        return

    if not portfolios:
        st.warning("No active portfolios found.")
        return

    # Helper to clean portfolio names for display
    def clean_name(name):
        return name.replace("_EqualWeight", "").replace("_", " ")

    tab1, tab2 = st.tabs(["Cross-Portfolio Matrix", "Monthly History"])

    # =====================================================================
    # TAB 1: Cross-Portfolio Matrix
    # =====================================================================
    with tab1:
        st.markdown("**Select portfolios and month to see which tickers are in each portfolio.**")

        # Multi-select for portfolios (display clean names, map back to original)
        portfolio_names = [p["name"] for p in portfolios]
        display_names = [clean_name(p["name"]) for p in portfolios]
        display_to_original = {clean_name(p["name"]): p["name"] for p in portfolios}

        selected_display_names = st.multiselect(
            "Select Portfolios",
            options=display_names,
            default=display_names[:3] if len(display_names) >= 3 else display_names,
            key="ticker_analysis_portfolio_multiselect"
        )
        # Map back to original names
        selected_portfolio_names = [display_to_original[d] for d in selected_display_names]

        if not selected_portfolio_names:
            st.info("Please select at least one portfolio.")
        else:
            selected_portfolios = [p for p in portfolios if p["name"] in selected_portfolio_names]
            selected_portfolio_ids = [p["id"] for p in selected_portfolios]

            # Get available months across selected portfolios
            try:
                with get_session() as session:
                    months_result = session.exec(
                        select(func.to_char(PortfolioHolding.effective_date, 'YYYY-MM'))
                        .where(PortfolioHolding.portfolio_id.in_(selected_portfolio_ids))
                        .distinct()
                        .order_by(func.to_char(PortfolioHolding.effective_date, 'YYYY-MM').desc())
                    ).all()
                    available_months = [str(m) for m in months_result if m]
            except Exception as e:
                st.error(f"Error loading months: {e}")
                available_months = []

            if not available_months:
                st.warning("No holdings data found for selected portfolios.")
            else:
                selected_month = st.selectbox(
                    "Select Month",
                    options=available_months,
                    key="ticker_analysis_month_select"
                )

                # Load holdings for all selected portfolios for this month
                try:
                    with get_session() as session:
                        holdings_data = session.exec(
                            select(
                                PortfolioHolding.portfolio_id,
                                PortfolioHolding.ticker
                            )
                            .where(PortfolioHolding.portfolio_id.in_(selected_portfolio_ids))
                            .where(func.to_char(PortfolioHolding.effective_date, 'YYYY-MM') == selected_month)
                        ).all()

                    # Build portfolio_id -> name mapping
                    id_to_name = {p["id"]: p["name"] for p in selected_portfolios}

                    # Build ticker -> portfolios mapping
                    ticker_portfolios = {}
                    for portfolio_id, ticker in holdings_data:
                        if ticker not in ticker_portfolios:
                            ticker_portfolios[ticker] = set()
                        ticker_portfolios[ticker].add(id_to_name[portfolio_id])

                    if not ticker_portfolios:
                        st.info(f"No holdings found for {selected_month}.")
                    else:
                        # Create matrix DataFrame with clean column names
                        all_tickers = sorted(ticker_portfolios.keys())
                        matrix_data = []

                        for ticker in all_tickers:
                            row = {"Ticker": ticker}
                            for pname in selected_portfolio_names:
                                # Use clean name for column header
                                row[clean_name(pname)] = "X" if pname in ticker_portfolios[ticker] else ""
                            row["Total"] = len(ticker_portfolios[ticker])
                            matrix_data.append(row)

                        df_matrix = pd.DataFrame(matrix_data)

                        # Sort by Total descending
                        df_matrix = df_matrix.sort_values("Total", ascending=False)

                        # Style the dataframe
                        def highlight_x(val):
                            if val == "X":
                                return "background-color: #90EE90; font-weight: bold"
                            return ""

                        styled_df = df_matrix.style.applymap(
                            highlight_x,
                            subset=[c for c in df_matrix.columns if c not in ["Ticker", "Total"]]
                        )

                        st.markdown(f"### {selected_month}: {len(all_tickers)} unique tickers across {len(selected_portfolio_names)} portfolios")
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)

                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Unique Tickers", len(all_tickers))
                        with col2:
                            multi_portfolio = sum(1 for t in ticker_portfolios if len(ticker_portfolios[t]) > 1)
                            st.metric("In Multiple Portfolios", multi_portfolio)
                        with col3:
                            avg_count = sum(len(v) for v in ticker_portfolios.values()) / len(ticker_portfolios) if ticker_portfolios else 0
                            st.metric("Avg Portfolios per Ticker", f"{avg_count:.1f}")

                except Exception as e:
                    st.error(f"Error loading holdings data: {e}")

    # =====================================================================
    # TAB 2: Monthly History
    # =====================================================================
    with tab2:
        st.markdown("**Select a portfolio to see monthly ticker changes. Green = retained from previous month.**")

        # Single portfolio selector (display clean names)
        selected_display_name_tab2 = st.selectbox(
            "Select Portfolio",
            options=display_names,
            key="ticker_analysis_single_portfolio"
        )
        # Map back to original name
        selected_portfolio_name_tab2 = display_to_original.get(selected_display_name_tab2, selected_display_name_tab2)

        selected_portfolio = next((p for p in portfolios if p["name"] == selected_portfolio_name_tab2), None)

        if selected_portfolio:
            # Get all months for this portfolio
            try:
                with get_session() as session:
                    months_result = session.exec(
                        select(func.to_char(PortfolioHolding.effective_date, 'YYYY-MM'))
                        .where(PortfolioHolding.portfolio_id == selected_portfolio["id"])
                        .distinct()
                        .order_by(func.to_char(PortfolioHolding.effective_date, 'YYYY-MM').desc())
                    ).all()
                    portfolio_months = [str(m) for m in months_result if m]
            except Exception as e:
                st.error(f"Error loading months: {e}")
                portfolio_months = []

            if not portfolio_months:
                st.warning(f"No holdings found for {selected_display_name_tab2}.")
            else:
                # Load all holdings for this portfolio
                try:
                    with get_session() as session:
                        all_holdings = session.exec(
                            select(
                                func.to_char(PortfolioHolding.effective_date, 'YYYY-MM').label("month"),
                                PortfolioHolding.ticker
                            )
                            .where(PortfolioHolding.portfolio_id == selected_portfolio["id"])
                            .order_by(func.to_char(PortfolioHolding.effective_date, 'YYYY-MM'))
                        ).all()

                    # Build month -> tickers mapping
                    month_tickers = {}
                    for month, ticker in all_holdings:
                        if month not in month_tickers:
                            month_tickers[month] = set()
                        month_tickers[month].add(ticker)

                    sorted_months = sorted(month_tickers.keys())
                    max_tickers = max(len(tickers) for tickers in month_tickers.values()) if month_tickers else 0

                    # Build vertical ticker lists per month with retention info
                    # Each column = one month, rows = tickers sorted alphabetically
                    month_columns = {}
                    prev_month_tickers = set()

                    for month in sorted_months:
                        current_tickers = sorted(month_tickers.get(month, set()))
                        # Store tuple (ticker, is_retained)
                        month_columns[month] = [
                            (ticker, ticker in prev_month_tickers)
                            for ticker in current_tickers
                        ]
                        prev_month_tickers = set(current_tickers)

                    # Create DataFrame with months as columns, tickers as values
                    # Pad shorter columns with empty strings
                    df_data = {}
                    for month in sorted_months:
                        tickers_with_status = month_columns[month]
                        # Just store ticker names for display
                        ticker_list = [t[0] for t in tickers_with_status]
                        # Pad to max length
                        ticker_list.extend([""] * (max_tickers - len(ticker_list)))
                        df_data[month] = ticker_list

                    df_monthly = pd.DataFrame(df_data)

                    # Build retention status for styling
                    retention_status = {}
                    for month in sorted_months:
                        retention_status[month] = {
                            ticker: is_retained
                            for ticker, is_retained in month_columns[month]
                        }

                    def style_vertical_tickers(df):
                        """Style tickers: green if retained from previous month."""
                        styles = pd.DataFrame("", index=df.index, columns=df.columns)
                        for col in df.columns:
                            for idx in df.index:
                                ticker = df.loc[idx, col]
                                if ticker:
                                    is_retained = retention_status.get(col, {}).get(ticker, False)
                                    if is_retained:
                                        styles.loc[idx, col] = "background-color: #90EE90; font-weight: bold"
                        return styles

                    styled_monthly = df_monthly.style.apply(style_vertical_tickers, axis=None)

                    all_unique_tickers = set(t for tickers in month_tickers.values() for t in tickers)
                    st.markdown(f"### {selected_display_name_tab2}: {len(all_unique_tickers)} tickers across {len(sorted_months)} months")
                    st.markdown("**Legend:** Green = retained from previous month")
                    st.dataframe(styled_monthly, use_container_width=True, hide_index=True, height=600)

                    # Monthly stats
                    st.markdown("---")
                    st.markdown("### Monthly Statistics")

                    stats_data = []
                    prev_tickers = set()
                    for month in sorted_months:
                        current_tickers = month_tickers.get(month, set())
                        retained = len(current_tickers & prev_tickers)
                        new_tickers = len(current_tickers - prev_tickers)
                        dropped = len(prev_tickers - current_tickers)

                        # Turnover: percentage of positions that changed
                        # First month has no previous, so N/A
                        if not prev_tickers:
                            turnover_str = "-"
                        else:
                            # Turnover = changed positions / current positions (capped at 100%)
                            turnover = new_tickers / len(current_tickers) if len(current_tickers) > 0 else 0
                            turnover_str = f"{turnover*100:.1f}%"

                        stats_data.append({
                            "Month": month,
                            "Total": len(current_tickers),
                            "Retained": retained,
                            "New": new_tickers,
                            "Dropped": dropped,
                            "Turnover": turnover_str
                        })
                        prev_tickers = current_tickers

                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(df_stats, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Error loading holdings: {e}")


def _get_trading_days(start_date, end_date):
    """Get list of trading days between start and end dates."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    trading_days = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    return [d.date() for d in trading_days]


def _update_portfolio_nav(tracker, portfolio, trade_date, price_data, prev_price_data=None):
    """Update NAV for a single portfolio on a specific date."""
    from datetime import timedelta
    from AlphaMachine_core.tracking import Variants

    try:
        # Get holdings as of trade date
        holdings = tracker.get_holdings(portfolio.id, trade_date)

        if not holdings:
            return False

        # Get previous NAV for return calculation
        prev_nav_df = tracker.get_nav_series(
            portfolio.id,
            Variants.RAW,
            end_date=trade_date - timedelta(days=1),
        )

        if prev_nav_df.empty:
            previous_raw_nav = 100.0
            initial_nav = 100.0
        else:
            previous_raw_nav = prev_nav_df["nav"].iloc[-1]
            initial_nav = prev_nav_df["nav"].iloc[0]

        # Calculate raw NAV
        raw_nav = tracker.calculate_raw_nav(holdings, price_data, previous_raw_nav, prev_price_data)

        # Update NAV for all variants
        tracker.update_daily_nav(
            portfolio_id=portfolio.id,
            trade_date=trade_date,
            raw_nav=raw_nav,
            previous_raw_nav=previous_raw_nav,
            initial_nav=initial_nav,
        )
        return True

    except Exception:
        return False


def _run_nav_update_with_progress(portfolio_name: str, start_date, end_date):
    """Run NAV update with progress bar."""
    from datetime import timedelta

    # Import required modules
    try:
        from AlphaMachine_core.tracking import get_tracker
        from AlphaMachine_core.data_manager import StockDataManager
    except ImportError as e:
        st.error(f"Could not import modules: {e}")
        return

    # Get trading days
    dates_to_process = _get_trading_days(start_date, end_date)

    if not dates_to_process:
        st.warning("No trading days in the selected date range.")
        return

    st.info(f"Processing {len(dates_to_process)} trading days from {start_date} to {end_date}")

    # Initialize tracker and data manager
    tracker = get_tracker()
    dm = StockDataManager()

    # Get portfolio
    portfolio = tracker.get_portfolio_by_name(portfolio_name)
    if not portfolio:
        st.error(f"Portfolio '{portfolio_name}' not found")
        return

    # Collect all tickers needed
    all_tickers = set()
    for d in dates_to_process:
        holdings = tracker.get_holdings(portfolio.id, d)
        all_tickers.update(h.ticker for h in holdings)

    if not all_tickers:
        st.warning("No holdings found for any date in the range.")
        return

    # Get price data
    status_text = st.empty()
    status_text.text(f"Loading prices for {len(all_tickers)} tickers...")

    min_date = min(dates_to_process)
    max_date = max(dates_to_process)

    # Include previous days for return calculation
    prev_min_date = min_date - timedelta(days=7)

    price_dicts = dm.get_price_data(
        list(all_tickers),
        prev_min_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
    )

    if not price_dicts:
        st.error("No price data available. Make sure prices are loaded for these tickers.")
        return

    price_df = pd.DataFrame(price_dicts)
    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"]).dt.date

    # Progress tracking
    progress_bar = st.progress(0)
    stats_container = st.empty()

    successful = 0
    failed = 0

    # Process each date
    for i, process_date in enumerate(dates_to_process):
        status_text.text(f"Processing {process_date}...")

        # Get prices for this date
        date_prices = price_df[price_df["trade_date"] == process_date]
        price_data = dict(zip(date_prices["ticker"], date_prices["close"]))

        if not price_data:
            failed += 1
            progress_bar.progress((i + 1) / len(dates_to_process))
            continue

        # Get previous day's prices
        prev_dates = price_df[price_df["trade_date"] < process_date]
        prev_prices_data = {}
        if not prev_dates.empty:
            latest_prev_date = prev_dates["trade_date"].max()
            prev_day_prices = price_df[price_df["trade_date"] == latest_prev_date]
            prev_prices_data = dict(zip(prev_day_prices["ticker"], prev_day_prices["close"]))

        # Update NAV
        success = _update_portfolio_nav(tracker, portfolio, process_date, price_data, prev_prices_data)
        if success:
            successful += 1
        else:
            failed += 1

        # Update progress
        progress_bar.progress((i + 1) / len(dates_to_process))
        stats_container.text(f"Successful: {successful} | Failed/Skipped: {failed}")

    # Compute metrics
    status_text.text("Computing periodic metrics...")
    try:
        metrics = tracker.compute_and_store_metrics(portfolio.id)
        status_text.text(f"Computed {len(metrics)} metrics")
    except Exception as e:
        status_text.warning(f"Metrics computation failed: {e}")

    # Final results
    progress_bar.progress(1.0)
    status_text.empty()

    if successful > 0:
        st.success(f"NAV Update Complete: {successful} days updated, {failed} days skipped/failed")
    else:
        st.warning(f"NAV Update Complete: No days were successfully updated ({failed} failed/skipped)")


# =============================================================================
# === Data-Management-UI ===
# =============================================================================
def show_data_ui():
    st.header("Data Management")

    if 'sdm_instance' not in st.session_state:
        try:
            st.session_state.sdm_instance = StockDataManager()
        except Exception as e:
            st.error(f"Fehler Initialisierung StockDataManager: {e}"); st.exception(e); st.stop()
    dm = st.session_state.sdm_instance

    mode = st.radio("Modus", ["Add/Update", "View/Delete", "Portfolio Holdings", "Ticker Analysis", "Delete Portfolios/Sources", "Cleanup Unused Tickers"], index=0, key="data_ui_mode_radio_main_v5")

    if mode == "Add/Update":
        st.subheader("Ticker einfuegen & Daten updaten")
        tickers_text = st.text_area("Tickers (eine pro Zeile)", height=120, key="data_ui_tickers_text_main_v2")
        col_date, col_source = st.columns(2)
        month_dt_input = col_date.date_input("Monat wählen", value=dt.date.today().replace(day=1), key="data_ui_month_date_main_v2")
        start_period = month_dt_input.replace(day=1)
        end_period = (pd.to_datetime(start_period) + pd.offsets.MonthEnd(0)).date()
        col_date.write(f"Zeitraum: {start_period} bis {end_period}")
        
        existing_sources_db = []
        try:
            with get_session() as session: 
                results = session.exec(select(TickerPeriod.source).distinct()).all()
                existing_sources_db = [str(s) for s in results if s] 
        except Exception as e_es: st.error(f"Fehler Laden existierender Quellen: {e_es}")
        
        default_sources = ["Topweights", "Manual"]
        source_options = sorted(list(set(existing_sources_db + default_sources))); source_options.append("Andere…")
        
        source_selected_add = col_source.selectbox("Quelle", source_options, key="data_ui_source_select_main_v2")
        if source_selected_add == "Andere…":
            custom_source_add = col_source.text_input("Neue Quelle eingeben", key="data_ui_custom_source_main_v2")
            if custom_source_add: source_selected_add = custom_source_add
        
        if st.button("➕ Ticker zu Periode hinzufügen", key="data_ui_add_button_main_v2"):
            tickers_list = [t.strip().upper() for t in tickers_text.splitlines() if t.strip()]
            if tickers_list and source_selected_add not in ["Andere…", ""]:
                added = dm.add_tickers_for_period(tickers_list, start_period.strftime("%Y-%m-%d"), end_period.strftime("%Y-%m-%d"), source_selected_add)
                st.success(f"{len(added)} Ticker zur Periode hinzugefügt ({source_selected_add}, {start_period.strftime('%Y-%m')}).")
                if 'all_periods_for_filters_cached_data_ui' in st.session_state: del st.session_state['all_periods_for_filters_cached_data_ui']
                st.rerun() 
            else: st.warning("Bitte Ticker und eine gültige Quelle eingeben.")

        if st.button("🔄 Alle Preise für DB-Ticker updaten", key="data_ui_update_button_main_v2"):
            tickers_in_db_update: List[str] = []
            try:
                with get_session() as session: 
                    results = session.exec(select(TickerPeriod.ticker).distinct()).all()
                    tickers_in_db_update = sorted(list(set(str(t) for t in results if t)))
            except Exception as e_gt_db: st.error(f"Fehler Holen Ticker für Update: {e_gt_db}")
            
            if not tickers_in_db_update: st.info("Keine Ticker in DB zum Updaten.")
            else:
                import time
                start_time = time.time()
                estimated_time = len(tickers_in_db_update) * 0.5 / 60  # 0.5s per ticker
                st.info(f"🔄 Update für {len(tickers_in_db_update)} Ticker startet...")
                st.info(f"⏱️ Geschätzte Dauer: ~{estimated_time:.1f} Minuten")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                stats_text = st.empty()
                updated_count = 0
                skipped_count = 0

                for idx, tk_str in enumerate(tickers_in_db_update):
                    elapsed = time.time() - start_time
                    remaining = (elapsed / (idx + 1)) * (len(tickers_in_db_update) - idx - 1) if idx > 0 else 0

                    # Show ticker being processed BEFORE calling update
                    status_text.info(f"📡 Verarbeite {tk_str} ({idx+1}/{len(tickers_in_db_update)})...")

                    result = dm.update_ticker_data(tickers=[tk_str])

                    # Show detailed result info AFTER update completes
                    details = result.get('details', {})
                    if tk_str in details:
                        detail = details[tk_str]
                        if detail.get('status') == 'success':
                            saved_info = detail.get('saved', 'Daten gespeichert')
                            status_text.info(f"✅ {tk_str}: {saved_info} | ⏱️ ~{remaining/60:.1f} min verbleibend")
                            updated_count += 1
                        elif detail.get('status') == 'skipped':
                            reason = detail.get('reason', 'Keine neuen Daten')
                            status_text.info(f"⏭️ {tk_str}: {reason} | ⏱️ ~{remaining/60:.1f} min verbleibend")
                            skipped_count += 1
                        elif detail.get('status') == 'loading':
                            # This shouldn't happen, but show loading info if present
                            load_info = detail.get('info', '')
                            status_text.info(f"📥 {tk_str}: Lade {load_info}")
                    else:
                        # Fallback if no details returned
                        if result.get('updated'):
                            status_text.info(f"✅ {tk_str}: Aktualisiert | ⏱️ ~{remaining/60:.1f} min verbleibend")
                            updated_count += 1
                        else:
                            status_text.info(f"⏭️ {tk_str}: Übersprungen | ⏱️ ~{remaining/60:.1f} min verbleibend")
                            skipped_count += 1

                    # Update stats every 10 tickers
                    if (idx + 1) % 10 == 0:
                        stats_text.info(f"📊 Aktualisiert: {updated_count} | Übersprungen: {skipped_count}")

                    progress_bar.progress((idx + 1) / len(tickers_in_db_update))

                total_time = time.time() - start_time
                status_text.success(f"✅ Update abgeschlossen in {total_time/60:.1f} Minuten")
                stats_text.success(f"Aktualisiert: {updated_count} | Uebersprungen: {skipped_count} | Gesamt: {len(tickers_in_db_update)}")
        return

    # --- Portfolio Holdings Mode ---
    elif mode == "Portfolio Holdings":
        _show_portfolio_holdings_ui()
        return

    # --- Ticker Analysis Mode ---
    elif mode == "Ticker Analysis":
        _show_ticker_analysis_ui()
        return

    # --- Delete Portfolios/Sources Mode ---
    elif mode == "Delete Portfolios/Sources":
        _show_delete_portfolios_sources_ui()
        return

    # --- Cleanup Unused Tickers Mode ---
    elif mode == "Cleanup Unused Tickers":
        _show_cleanup_unused_tickers_ui()
        return

    # --- View/Delete Mode ---
    st.subheader("View/Delete Ticker Periods")
    all_periods_for_filters: List[Dict[str, Any]]
    if 'all_periods_for_filters_cached_data_ui' not in st.session_state:
        with st.spinner("Lade Periodendaten für Filter..."):
            temp_all_periods_for_filters_calc = []
            try:
                with get_session() as session: 
                    db_objects = session.exec(select(TickerPeriod).order_by(TickerPeriod.start_date.desc(), TickerPeriod.source, TickerPeriod.ticker)).all()
                    temp_all_periods_for_filters_calc = [p.model_dump() for p in db_objects]
                st.session_state.all_periods_for_filters_cached_data_ui = temp_all_periods_for_filters_calc
            except Exception as e_load_all_p:
                st.error(f"Fehler beim Laden aller Perioden für Filter: {e_load_all_p}")
                st.session_state.all_periods_for_filters_cached_data_ui = []
    all_periods_for_filters = st.session_state.get('all_periods_for_filters_cached_data_ui', [])
    temp_monate_set = set(); temp_quellen_set = set()
    if not all_periods_for_filters: st.info("Keine TickerPeriod-Einträge."); Monate, Quellen = ["(Keine)"], ["(Keine)"]
    else:
        for p_dict in all_periods_for_filters:
            start_date_val = p_dict.get('start_date')
            if start_date_val:
                current_date_obj = None
                if isinstance(start_date_val, str): 
                    try: current_date_obj = pd.to_datetime(start_date_val).date()
                    except: continue 
                elif isinstance(start_date_val, dt.date):
                    current_date_obj = start_date_val
                
                if current_date_obj: temp_monate_set.add(current_date_obj.strftime("%Y-%m"))

            if p_dict.get('source'): temp_quellen_set.add(str(p_dict['source']))
        
        Monate  = sorted(list(temp_monate_set), reverse=True) if temp_monate_set else ["(Keine)"]
        Quellen = sorted(list(temp_quellen_set)) if temp_quellen_set else ["(Keine)"]
        
    col_v_month, col_v_source = st.columns(2)
    month_selected_view = col_v_month.selectbox("Monat anzeigen/filtern", Monate, key="view_del_month_sb_page_v5")
    source_selected_view = col_v_source.selectbox("Quelle anzeigen/filtern", Quellen, key="view_del_source_sb_page_v5")
    
    periods_to_display_view: List[Dict[str, Any]] = []
    if month_selected_view not in ["(Keine)", "(Fehler)"] and source_selected_view not in ["(Keine)", "(Fehler)"]:
        periods_to_display_view = [
            p_dict for p_dict in all_periods_for_filters 
            if (p_dict.get('start_date') and 
                ((isinstance(p_dict['start_date'], dt.date) and p_dict['start_date'].strftime("%Y-%m") == month_selected_view) or 
                 (isinstance(p_dict['start_date'], str) and p_dict['start_date'].startswith(month_selected_view))))
            and p_dict.get('source') == source_selected_view
        ]

    if periods_to_display_view:
        dfp_view = pd.DataFrame(periods_to_display_view)
        display_cols_view = ['id', 'ticker', 'start_date', 'end_date', 'source']
        dfp_display_view = dfp_view[[col for col in display_cols_view if col in dfp_view.columns]]
        if 'id' in dfp_display_view.columns:
            st.dataframe(dfp_display_view.set_index('id'), use_container_width=True, height=min(300, len(dfp_display_view)*38 + 58))
            available_ids_del_view = dfp_display_view['id'].tolist()
            if available_ids_del_view:
                to_del_view = st.multiselect("Perioden zum Löschen (IDs)", available_ids_del_view, key="del_ms_data_ui_page_v5")
                if st.button("🗑️ Ausgewählte löschen", key="del_btn_data_ui_page_v5"):
                    deleted_count = 0
                    for pid_to_delete in to_del_view:
                        if dm.delete_period(int(pid_to_delete)): deleted_count += 1
                    st.success(f"{deleted_count} von {len(to_del_view)} Einträgen gelöscht.")
                    if 'all_periods_for_filters_cached_data_ui' in st.session_state: del st.session_state['all_periods_for_filters_cached_data_ui']
                    st.rerun()
            else: st.info("Keine IDs zum Löschen.")
        else: st.warning("DFP hat keine 'id'-Spalte."); st.dataframe(dfp_display_view, use_container_width=True)
    elif month_selected_view not in ["(Keine)", "(Fehler)"]: st.info(f"Keine Period-Einträge für '{month_selected_view}' / '{source_selected_view}'.")

    st.markdown("---"); st.subheader("Ticker Info Übersicht")
    info_dicts_ui: List[Dict[str, Any]] = []; dfi_for_chart = pd.DataFrame() 
    try: info_dicts_ui = dm.get_ticker_info()
    except Exception as e_get_ti_ui: st.error(f"Fehler Laden TickerInfo: {e_get_ti_ui}")
    if not info_dicts_ui: st.info("Keine TickerInfo vorhanden.")
    else:
        dfi_for_chart = pd.DataFrame(info_dicts_ui) 
        if 'id' not in dfi_for_chart.columns: st.error("Spalte 'id' fehlt in TickerInfo."); st.dataframe(dfi_for_chart, use_container_width=True)
        else:
            all_cols_dfi_view = list(dfi_for_chart.columns)
            filter_col_dfi_view = st.selectbox("Filter-Spalte TickerInfo", ["(kein)"] + all_cols_dfi_view, index=0, key="ti_filter_col_data_ui_v5")
            
            df_to_display_ticker_info = dfi_for_chart.copy() # Für die Anzeige, Original für Chart-Ticker-Liste behalten
            if filter_col_dfi_view != "(kein)" and filter_col_dfi_view in df_to_display_ticker_info.columns:
                unique_choices_dfi_view = sorted(df_to_display_ticker_info[filter_col_dfi_view].dropna().astype(str).unique())
                if unique_choices_dfi_view:
                    sel_dfi_view = st.multiselect(f"Werte «{filter_col_dfi_view}»", unique_choices_dfi_view, default=unique_choices_dfi_view, key="ti_ms_data_ui_v5")
                    df_to_display_ticker_info = df_to_display_ticker_info[df_to_display_ticker_info[filter_col_dfi_view].astype(str).isin(sel_dfi_view)]
                else: st.info(f"Keine Werte in '{filter_col_dfi_view}' zum Filtern.")
            
            if 'id' in df_to_display_ticker_info.columns:
                st.dataframe(df_to_display_ticker_info.set_index("id"), use_container_width=True, height=min(300, len(df_to_display_ticker_info)*38 + 58))
            elif not df_to_display_ticker_info.empty :
                 st.dataframe(df_to_display_ticker_info, use_container_width=True, height=min(300, len(df_to_display_ticker_info)*38 + 58))
            else: # df_to_display_ticker_info ist leer (z.B. nach Filterung)
                if filter_col_dfi_view != "(kein)": st.info("Keine TickerInfo-Daten nach Filterung.")
                # Ansonsten wurde schon "Keine TickerInfo vorhanden" angezeigt, wenn info_dicts_ui leer war


    # --- Price Chart ---
    st.markdown("---"); st.subheader("📈 Price Chart (Linienchart, max. letzte 10 Jahre)") # Titel angepasst
    available_tickers_pc_list = []
    # dfi_for_chart sollte aus dem vorherigen "Ticker Info Übersicht"-Block korrekt initialisiert sein
    if 'dfi_for_chart' in locals() and isinstance(dfi_for_chart, pd.DataFrame) and not dfi_for_chart.empty and 'ticker' in dfi_for_chart.columns:
        available_tickers_pc_list = sorted(dfi_for_chart["ticker"].dropna().unique())
    
    if not available_tickers_pc_list:
        st.info("Keine Ticker für Chart verfügbar (aus TickerInfo).")
    else:
        if 'chart_ticker_sel_data_ui_v7' not in st.session_state or \
           st.session_state.chart_ticker_sel_data_ui_v7 not in available_tickers_pc_list:
            st.session_state.chart_ticker_sel_data_ui_v7 = available_tickers_pc_list[0]

        def reset_chart_dates_on_ticker_change_v2(): # Neuer Name für Eindeutigkeit
            new_ticker = st.session_state.sb_chart_ticker_key_v7 
            st.session_state.chart_start_date_key_v7 = f"chart_start_{new_ticker}"
            st.session_state.chart_end_date_key_v7 = f"chart_end_{new_ticker}"
            
            _max_date_limit_cb = dt.date.today()
            # *** ÄNDERUNG: Default Start auf max. 10 Jahre in der Vergangenheit ***
            _min_date_for_chart_default = _max_date_limit_cb - pd.DateOffset(years=10)
            _min_date_limit_cb = dt.date(1990,1,1) # Globales Minimum bleibt

            _def_start_cb = _min_date_for_chart_default.date()
            _def_end_cb = _max_date_limit_cb
            
            if 'dfi_for_chart' in locals() and isinstance(dfi_for_chart, pd.DataFrame) and not dfi_for_chart.empty:
                _info_df_cb = dfi_for_chart[dfi_for_chart['ticker'] == new_ticker]
                if not _info_df_cb.empty:
                    _s_val_cb = _info_df_cb["actual_start_date"].min()
                    _e_val_cb = _info_df_cb["actual_end_date"].max()
                    if pd.notna(_s_val_cb): 
                        # Nimm das spätere von (Ticker-Start oder Default-10-Jahre-Start)
                        _def_start_cb = max(pd.to_datetime(_s_val_cb).date(), _min_date_for_chart_default.date())
                    if pd.notna(_e_val_cb): 
                        _def_end_cb = min(pd.to_datetime(_e_val_cb).date(), _max_date_limit_cb)
            
            if _def_start_cb > _def_end_cb: 
                _def_start_cb = max(_min_date_limit_cb, _def_end_cb - pd.Timedelta(days=1))
            
            st.session_state[st.session_state.chart_start_date_key_v7] = _def_start_cb
            st.session_state[st.session_state.chart_end_date_key_v7] = _def_end_cb
            
            cache_key_to_delete = f"chart_df_{new_ticker}_{_def_start_cb}_{_def_end_cb}"
            if cache_key_to_delete in st.session_state:
                del st.session_state[cache_key_to_delete]
        
        ticker_sel_for_chart = st.selectbox(
            "Ticker für Chart", 
            available_tickers_pc_list, 
            index=available_tickers_pc_list.index(st.session_state.chart_ticker_sel_data_ui_v7),
            key="sb_chart_ticker_key_v7",
            on_change=reset_chart_dates_on_ticker_change_v2
        )

        start_date_widget_key = st.session_state.get('chart_start_date_key_v7', f"chart_start_{ticker_sel_for_chart}")
        end_date_widget_key = st.session_state.get('chart_end_date_key_v7', f"chart_end_{ticker_sel_for_chart}")

        if start_date_widget_key not in st.session_state or end_date_widget_key not in st.session_state:
            reset_chart_dates_on_ticker_change_v2() 
            start_date_widget_key = st.session_state.chart_start_date_key_v7
            end_date_widget_key = st.session_state.chart_end_date_key_v7

        val_start = st.session_state[start_date_widget_key]
        val_end = st.session_state[end_date_widget_key]

        col_chart_date_start, col_chart_date_end = st.columns(2)
        # *** ÄNDERUNG: min_value für Startdatum auf 10 Jahre vor Enddatum begrenzen ***
        min_selectable_start_date = val_end - pd.DateOffset(years=10)
        min_selectable_start_date = max(dt.date(1990,1,1), min_selectable_start_date.date())


        selected_start_date = col_chart_date_start.date_input("Startdatum Chart", 
                                                               value=val_start, 
                                                               min_value=min_selectable_start_date, # Begrenzt Auswahl
                                                               max_value=val_end, # Kann nicht nach Enddatum sein
                                                               key=f"di_start_v7_{ticker_sel_for_chart}")
        selected_end_date = col_chart_date_end.date_input("Enddatum Chart", 
                                                          value=val_end, 
                                                          min_value=selected_start_date, # Muss nach Startdatum sein
                                                          max_value=dt.date.today(), 
                                                          key=f"di_end_v7_{ticker_sel_for_chart}")

        if selected_start_date != val_start or selected_end_date != val_end:
            st.session_state[start_date_widget_key] = selected_start_date
            st.session_state[end_date_widget_key] = selected_end_date
            # Invaliere Cache, indem du den spezifischen Chart-Cache-Key löschst
            # (Die Keys im Cache werden mit den *neuen* Daten gebildet, daher wird alter Cache nicht getroffen)
            st.rerun() 

        final_start_for_query = st.session_state[start_date_widget_key]
        final_end_for_query = st.session_state[end_date_widget_key]
        
        @st.cache_data(ttl=300) 
        def get_and_convert_chart_data_line(_ticker_arg, _start_date_str_arg, _end_date_str_arg):
            _sdm = st.session_state.sdm_instance
            if _sdm is None: return pd.DataFrame()
            _raw_dicts = _sdm.get_price_data([_ticker_arg], _start_date_str_arg, _end_date_str_arg)
            if not _raw_dicts: return pd.DataFrame()
            _chart_df = convert_single_ticker_pricedata_to_ohlcv_df(
                _raw_dicts, _ticker_arg, 
                pd.to_datetime(_start_date_str_arg), pd.to_datetime(_end_date_str_arg)
            )
            return _chart_df # Gibt immer noch den OHLCV-DataFrame zurück

        chart_df_to_display = get_and_convert_chart_data_line(ticker_sel_for_chart, 
                                                       final_start_for_query.strftime("%Y-%m-%d"), 
                                                       final_end_for_query.strftime("%Y-%m-%d"))

        if chart_df_to_display.empty or 'Close' not in chart_df_to_display.columns or chart_df_to_display['Close'].isna().all():
            st.info(f"Keine gültigen Schlusskursdaten für {ticker_sel_for_chart} im Zeitraum {final_start_for_query} bis {final_end_for_query} gefunden.")
        else:
            # *** ÄNDERUNG: Line Chart für Schlusskurse ***
            fig_price_chart = go.Figure()
            fig_price_chart.add_trace(go.Scatter(
                x=chart_df_to_display.index, 
                y=chart_df_to_display["Close"], 
                mode='lines', 
                name=f"{ticker_sel_for_chart} Close"
            ))
            fig_price_chart.update_layout(
                title=f"Linienchart (Schlusskurs) für {ticker_sel_for_chart}",
                xaxis_title="Date",
                yaxis_title="Price"
            )
            st.plotly_chart(fig_price_chart, use_container_width=True)

            # Fehlende Tage Logik (bleibt, da sie auf dem Index basiert)
            st.markdown("---") 
            # Statt Subheader, vielleicht ein Expander, wenn es nicht immer relevant ist
            with st.expander("Analyse fehlender Handelstage im Chart-Zeitraum", expanded=False):
                try:
                    us_bd_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())
                    full_bday_range = pd.date_range(final_start_for_query, final_end_for_query, freq=us_bd_cal)
                    # Verwende den Index des DataFrames, der tatsächlich geplottet wird
                    missing_bdays = full_bday_range.difference(chart_df_to_display.dropna(subset=['Close']).index) 
                    
                    if not missing_bdays.empty:
                        st.warning(f"⚠️ {len(missing_bdays)} Handelstage ohne Daten ({final_start_for_query.strftime('%Y-%m-%d')} bis {final_end_for_query.strftime('%Y-%m-%d')}):")
                        st.write(missing_bdays.strftime("%Y-%m-%d").tolist()[:20]) 
                    else:
                        st.success(f"Keine fehlenden Handelstage im Zeitraum ({final_start_for_query.strftime('%Y-%m-%d')} bis {final_end_for_query.strftime('%Y-%m-%d')}) gefunden.")
                except Exception as e_mdays:
                    st.warning(f"Fehler bei Prüfung fehlender Handelstage: {e_mdays}")

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
def show_optimizer_ui():
    st.header("⚙️ Hyperparameter-Optimizer")

    # ---------- Daten-Selektion ------------------------------------
    dm = StockDataManager()
    months_sorted = sorted(dm.get_periods_distinct_months(), reverse=True)  # Newest first
    month   = st.selectbox("Start-Monat (Universe)", months_sorted)
    with get_session() as session:
        existing = session.exec(select(TickerPeriod.source)).all()
    defaults = ["Topweights", "TR20"]
    sources  = st.multiselect("Quellen", sorted(set(existing + defaults)), default=defaults)
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Backtest-Start", value=dt.date.today() - dt.timedelta(days=5*365))
    end_date   = col2.date_input("Backtest-Ende",  value=dt.date.today(), min_value=start_date)

    # Preise laden
    # ----- Preise laden -----------------------------------------
    tickers = dm.get_tickers_for(month, sources)
    if not tickers:
        st.warning("⚠️ Keine Ticker für diese Auswahl.")
        st.stop()

    MAX_LOOKBACK = 1000  # größtes window_days im Suchraum
    price_df, price_start = load_price_df(
        dm, tickers, start_date, end_date, MAX_LOOKBACK
    )

    # ---------- Suchraum-Editor ------------------------------------
    PARAMS = {
        "num_stocks":        ("Anzahl Aktien", 5, 50, 1),
        "window_days":       ("Lookback Tage", 50, 500, 10),
        "min_weight":        ("Min-Weight %", 0.0, 5.0, 0.5),
        "max_weight":        ("Max-Weight %", 5.0, 50.0, 1.0),
        "force_equal_weight":("Equal-Weight", [False, True]),
        "optimization_mode": ("Mode", ["select-then-optimize", "optimize-subset"]),
        "optimizer_method":  ("Optimizer", ["ledoit-wolf", "minvar", "hrp"]),
        "cov_estimator":     ("Cov-Estimator", ["ledoit-wolf", "constant-corr", "factor-model"]),
    }

    search_space = {}
    with st.expander("🔧 Suchraum definieren", expanded=True):
        for key, meta in PARAMS.items():
            label = meta[0]
            if not st.checkbox(f"{label} optimieren", key=f"chk_{key}"):
                continue

            if isinstance(meta[1], (int, float)):
                lo, hi, step = meta[1:]
                lo_val, hi_val = st.slider(label, lo, hi, (lo, hi), step=step, key=f"sl_{key}")
                kind = "int" if isinstance(lo, int) else "float"
                search_space[key] = (kind, lo_val, hi_val, step)
            else:
                opts = meta[1]
                sel  = st.multiselect(f"{label} – Kandidaten", opts, opts, key=f"ms_{key}")
                search_space[key] = ("categorical", sel)

    st.info(f"🎯 Aktueller Suchraum:  {search_space}")

    # Defaults für fixe Parameter, falls nicht optimiert
    start_date_str = start_date.strftime("%Y-%m-%d")
    base_kwargs = {
        "start_balance":          100_000,
        "start_month":            start_date_str,
        "universe_mode":          "static",
        "rebalance_frequency":    "monthly",
        "custom_rebalance_months": 1,
        "enable_trading_costs":   False,
        "use_risk_overlay": False, 
    }
    if "num_stocks"  not in search_space:
        base_kwargs["num_stocks"] = st.number_input("Anzahl Aktien (fix)", 5, 50, 20, key="fix_num")
    if "window_days" not in search_space:
        base_kwargs["window_days"] = st.slider("Lookback Tage (fix)", 50, 500, 200, 10, key="fix_win")

    with st.expander("🎯 Objective-Gewichte"):
        kpi_weights = {
            "Sharpe Ratio": st.slider("Sharpe", 0.0, 3.0, 1.0, 0.1),
            "Ulcer Index":  -st.slider("Ulcer Index", 0.0, 3.0, 1.0, 0.1),
            "CAGR (%)":      st.slider("CAGR", 0.0, 3.0, 1.0, 0.1),
        }
    n_trials = st.number_input("Trials", 10, 500, 100, 10)

    if st.button("🚀 Suche starten"):
        study = run_optimizer(price_df, base_kwargs, search_space, kpi_weights, n_trials)
        # Pass optimizer context for "Run in Backtester" feature
        optimizer_context = {
            "sources": sources,
            "start_date": start_date,
            "end_date": end_date,
            "month": month,
        }
        # Store in session state for persistence across reruns
        st.session_state.optimizer_results = {
            "study": study,
            "kpi_weights": kpi_weights,
            "price_df": price_df,
            "base_kwargs": base_kwargs,
            "optimizer_context": optimizer_context,
        }

    # Display results if available in session state
    if "optimizer_results" in st.session_state and st.session_state.optimizer_results:
        results = st.session_state.optimizer_results
        show_study_results(
            results["study"],
            results["kpi_weights"],
            results["price_df"],
            results["base_kwargs"],
            results["optimizer_context"]
        )

def show_study_results(study, kpi_weights, price_df, fixed_kwargs, optimizer_context=None):
    
    # ------- A) Trials-DataFrame aufbereiten -----------------------
    df = study.trials_dataframe()

    # Optuna ≥ 4 → MultiIndex flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            sec if main in ("params", "user_attrs") else main
            for main, sec in df.columns.to_list()
        ]

    # Optuna ≤ 3 → user_attrs aufsplitten
    if "user_attrs" in df.columns:
        df = pd.concat(
            [df.drop(columns=["user_attrs"]), df["user_attrs"].apply(pd.Series)],
            axis=1
        )

    # Präfixe entfernen
    df = df.rename(columns=lambda c: re.sub(r"^(param_|params_|user_attrs?_)", "", c))

    # KPI-Spalten ermitteln
    kpi_map = {"Sharpe Ratio": "Sharpe", "CAGR (%)": "CAGR", "Ulcer Index": "Ulcer Index"}
    kpis    = [kpi_map[k] for k in kpi_weights if kpi_map[k] in df.columns]

    # ------- B) Top 50 Runs ----------------------------------------
    cols_top = ["number", "value"] + kpis + [
        c for c in sorted(df.columns) if c not in ("number", "value", *kpis)
    ]
    top_df = df[cols_top].sort_values("value", ascending=False).head(50)

    st.subheader("🏆 Top 50 Runs")
    st.dataframe(top_df.style.hide(axis="index"), use_container_width=True)

    # Auswahl eines Runs für Backtest
    run_numbers = top_df["number"].tolist()
    selected = st.selectbox("Wähle Run-Nummer zum Backtesten", run_numbers)

    backtester_btn = st.button("📊 Im Backtester öffnen", key="open_in_backtester_btn")

    # Handle "Open in Backtester" button
    if backtester_btn and optimizer_context:
        sel_row = df[df["number"] == selected].iloc[0]
        params = {c: sel_row[c] for c in sel_row.index if c not in ("number", "value", *kpis)}

        # Store parameters for backtester
        st.session_state.optimizer_to_backtester = {
            # Optimized/fixed parameters
            "num_stocks": int(params.get("num_stocks", fixed_kwargs.get("num_stocks", 20))),
            "window_days": int(params.get("window_days", fixed_kwargs.get("window_days", 252))),
            "min_weight": float(params.get("min_weight", 0.0)),
            "max_weight": float(params.get("max_weight", 20.0)),
            "force_equal_weight": bool(params.get("force_equal_weight", False)),
            "optimization_mode": params.get("optimization_mode", "select-then-optimize"),
            "optimizer_method": params.get("optimizer_method", "ledoit-wolf"),
            "cov_estimator": params.get("cov_estimator", "ledoit-wolf"),
            # Context from optimizer
            "sources": optimizer_context.get("sources", ["Topweights"]),
            "start_date": optimizer_context.get("start_date"),
            "end_date": optimizer_context.get("end_date"),
            "month": optimizer_context.get("month"),
        }
        st.session_state.auto_run_backtest = True
        st.session_state.switch_to_backtester = True  # Flag for page switch
        st.rerun()

    # ------- C) Best-Run erneut ausführen --------------------------
    best_params = study.best_params
    run_kwargs  = {**fixed_kwargs, **best_params}
    if "num_stocks" not in run_kwargs:
        run_kwargs["num_stocks"] = fixed_kwargs.get("num_stocks")
    if "window_days" not in run_kwargs:
        run_kwargs["window_days"] = fixed_kwargs.get("window_days")

    eng_best = SharpeBacktestEngine(price_df, **run_kwargs)
    eng_best.run_with_next_month_allocation()

    st.markdown("---")
    st.write("Engine-Start:", eng_best.user_start_date)
    st.header("🚀 Details des Best-Runs")
    render_engine_tabs(eng_best)

    # ------- D) Best-Run KPIs & Parameter -------------------------
    best = top_df.iloc[0]
    param_cols = [c for c in best.index if c not in ("number", "value", *kpis)]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🥇 Best-Run KPIs")
        st.table(
            best[kpis]
              .rename_axis("KPI")
              .to_frame("Wert")
        )
    with col2:
        st.subheader("⚙️ Best-Run Parameter")
        st.table(
            best[param_cols]
              .dropna()
              .rename_axis("Parameter")
              .to_frame("Wert")
        )

    # ------- E) Performance & Balance pro Jahr --------------------
    yearly_bal     = eng_best.portfolio_value.resample("YE").last()
    yearly_ret_pct = yearly_bal.pct_change().mul(100).round(1)

    df_year = pd.DataFrame({
        "Year":        yearly_bal.index.year,
        "Return (%)":  yearly_ret_pct,
        "Balance":     yearly_bal.round(0).astype(int),
    })
    df_year["Return (%)"] = df_year["Return (%)"].fillna("")

    st.subheader("📈 Performance & Balance pro Jahr des Best-Runs")
    st.table(df_year.astype({"Year": int}).reset_index(drop=True))

def render_engine_tabs(engine):
    tabs = st.tabs(["Dashboard", "Daily", "Monthly", "Yearly", "Drawdown"])

    # --- Dashboard ------------------------------------------------
    with tabs[0]:
        st.subheader("🔍 KPI-Übersicht")
        st.dataframe(engine.performance_metrics, hide_index=True, use_container_width=True)
        st.line_chart(engine.portfolio_value, height=250)

    # --- Daily ----------------------------------------------------
    with tabs[1]:
        st.subheader("📅 Daily Portfolio")
        st.dataframe(engine.daily_df, use_container_width=True)

    # --- Monthly --------------------------------------------------
    with tabs[2]:
        st.subheader("🗓️ Monthly Performance")
        st.dataframe(engine.monthly_performance, use_container_width=True)

    # --- Yearly ---------------------------------------------------
    with tabs[3]:
        yearly_bal = engine.portfolio_value.resample("YE").last()
        yearly_ret = yearly_bal.pct_change()*100
        df_year = pd.DataFrame({"Year": yearly_bal.index.year,
                                "Return (%)": yearly_ret,
                                "Balance": yearly_bal})
        st.dataframe(df_year.reset_index(drop=True), use_container_width=True)

    # --- Drawdown -------------------------------------------------
    with tabs[4]:
        df_port = engine.portfolio_value.to_frame("Portfolio")
        df_port["Peak"] = df_port["Portfolio"].cummax()
        df_port["Drawdown"] = df_port["Portfolio"] / df_port["Peak"] - 1
        dd = (df_port["Drawdown"]*100).round(2)
        st.line_chart(dd, height=250)


# =============================================================================
# === DATENLADE- UND KONVERTIERUNGSFUNKTIONEN (JETZT MAXIMAL ANGEPASST) ===
# =============================================================================

# =============================================================================
# === Performance Tracker (isolated with error handling) ===
# =============================================================================
def show_performance_tracker_safe():
    """
    Wrapper to load Performance Tracker page with error isolation.

    This ensures the main app continues to work even if there are
    issues with the tracking module or database.
    """
    try:
        from AlphaMachine_core.ui.performance_tracker import show_performance_tracker_ui
        show_performance_tracker_ui()
    except ImportError as e:
        st.error(
            "Performance Tracker module could not be loaded.\n\n"
            f"Error: {e}\n\n"
            "Please ensure all required packages are installed."
        )
    except Exception as e:
        st.error(
            f"An error occurred while loading the Performance Tracker:\n\n{e}\n\n"
            "The rest of the application continues to work normally."
        )


# =============================================================================
# === Portfolio Selection Page ===
# =============================================================================
def show_portfolio_selection_ui():
    """
    Portfolio Selection page - helps select optimal portfolios for live trading.

    Features:
    - Risk-adjusted metrics ranking
    - Correlation analysis
    - Optimal combination finder
    - Combined portfolio simulation
    """
    import numpy as np
    import pandas as pd
    from datetime import date, timedelta

    st.title("📊 Portfolio Selection")
    st.markdown("Select optimal portfolios for live trading based on risk-adjusted metrics and diversification.")

    try:
        from AlphaMachine_core.tracking.tracker import PortfolioTracker
        from AlphaMachine_core.tracking import Variants
        from AlphaMachine_core.selection import (
            find_optimal_portfolio_combination,
            simulate_combined_portfolio,
            calculate_candidate_metrics,
            optimize_portfolio_weights,
            walk_forward_validation,
            OPTIMIZATION_PRESETS,
            DEFAULT_WEIGHTS,
            WEIGHT_METHODS,
        )
        from AlphaMachine_core.ui.charts import create_correlation_heatmap, create_nav_chart
    except ImportError as e:
        st.error(f"Could not import required modules: {e}")
        return

    # Initialize tracker
    tracker = PortfolioTracker()

    # Get all portfolios
    all_portfolios = tracker.list_portfolios(active_only=True)
    if not all_portfolios:
        st.warning("No portfolios available.")
        return

    portfolio_map = {p.name: p.id for p in sorted(all_portfolios, key=lambda x: x.name)}

    # ==========================================================================
    # SIDEBAR CONTROLS
    # ==========================================================================
    st.sidebar.markdown("### Selection Settings")

    # Variant selection
    variant_options = Variants.all()
    selected_variants = st.sidebar.multiselect(
        "Variants to Include",
        options=variant_options,
        default=[Variants.RAW],
        help="Select which strategy variants to analyze",
    )

    if not selected_variants:
        st.warning("Please select at least one variant.")
        return

    # Date range - find available dates
    all_min_dates = []
    all_max_dates = []
    for p in all_portfolios:
        nav_df = tracker.get_nav_series(p.id, Variants.RAW)
        if not nav_df.empty:
            all_min_dates.append(nav_df.index.min().date())
            all_max_dates.append(nav_df.index.max().date())

    if not all_min_dates:
        st.warning("No NAV data available.")
        return

    available_min = min(all_min_dates)
    available_max = max(all_max_dates)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Date Range")

    start_date = st.sidebar.date_input(
        "Start Date",
        value=max(available_min, available_max - timedelta(days=365)),
        min_value=available_min,
        max_value=available_max,
        key="ps_start_date",
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=available_max,
        min_value=available_min,
        max_value=available_max,
        key="ps_end_date",
    )

    # Number of portfolios to select
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Optimization")

    n_select = st.sidebar.slider(
        "Portfolios to Select",
        min_value=2,
        max_value=5,
        value=3,
        help="Number of portfolios to recommend",
    )

    # Weight allocation method (simplified to 2 options)
    selected_weight_method = st.sidebar.radio(
        "Weight Allocation",
        options=["equal", "optimized"],
        format_func=lambda x: WEIGHT_METHODS.get(x, x),
        index=0,
        help="How to allocate weights to selected portfolios:\n"
             "- **Equal Weight**: All portfolios get same weight (1/N)\n"
             "- **Optimized Weight**: Find weights that maximize your metric preferences",
    )

    # Max position constraint (only for optimized)
    max_position = 1.0
    if selected_weight_method == "optimized":
        max_position_pct = st.sidebar.slider(
            "Max Position %",
            min_value=30,
            max_value=100,
            value=100,
            step=5,
            help="Maximum weight any single portfolio can have.\n"
                 "100% = no limit, 50% = no portfolio > 50%",
            key="ps_max_position",
        )
        max_position = max_position_pct / 100.0

    # Preset selection with tooltips showing slider values
    preset_options = ["Custom"] + list(OPTIMIZATION_PRESETS.keys())

    # Build tooltip showing what each preset does
    preset_tooltips = {
        "Custom": "Configure metric weights manually using the sliders below",
        "Risk-Adjusted Focus": "Sharpe: 3, Sortino: 2, Calmar: 2, UPI: 2, CAGR: 1",
        "Absolute Returns": "Sharpe: 1, Sortino: 1, Calmar: 1, UPI: 1, CAGR: 3",
        "Capital Preservation": "Sharpe: 1, Sortino: 2, Calmar: 3, UPI: 3, CAGR: 0",
        "Balanced": "Sharpe: 2, Sortino: 2, Calmar: 2, UPI: 2, CAGR: 2",
    }

    selected_preset = st.sidebar.radio(
        "Optimization Preset",
        options=preset_options,
        index=0,
        help="Pre-configured metric weights:\n\n" + "\n".join(
            f"**{k}**: {v}" for k, v in preset_tooltips.items()
        ),
    )

    # Show current preset configuration
    if selected_preset != "Custom":
        st.sidebar.caption(f"📋 {preset_tooltips.get(selected_preset, '')}")

    # Weight sliders
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Metric Weights (0-3)")
    st.sidebar.caption("Higher = more important in selection")

    if selected_preset != "Custom":
        weights = OPTIMIZATION_PRESETS[selected_preset].copy()
        st.sidebar.info(f"Using **{selected_preset}** preset weights")
    else:
        weights = {}

    # Always show sliders (disabled for presets, enabled for custom)
    is_custom = selected_preset == "Custom"

    # 5 metrics for COMBINED portfolio optimization
    # Note: Diversification benefit is already captured in combined Sharpe/Sortino
    weight_labels = {
        "sharpe": ("Sharpe Ratio", "Risk-adjusted return (symmetric)"),
        "sortino": ("Sortino Ratio", "Downside risk-adjusted return"),
        "calmar": ("Calmar Ratio", "Return vs max drawdown"),
        "upi": ("Ulcer Perf. Index", "Return vs drawdown pain"),
        "cagr": ("CAGR", "Pure return preference"),
    }

    for key, (label, help_text) in weight_labels.items():
        default_val = weights.get(key, DEFAULT_WEIGHTS.get(key, 2))
        if is_custom:
            weights[key] = st.sidebar.slider(
                label,
                min_value=0,
                max_value=3,
                value=default_val,
                help=help_text,
                key=f"ps_weight_{key}",
            )
        else:
            st.sidebar.markdown(f"**{label}**: {default_val}")

    # Walk-Forward Validation controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Walk-Forward Validation")

    run_walk_forward = st.sidebar.checkbox(
        "Run Walk-Forward Validation",
        value=False,
        help="Test strategy robustness with out-of-sample validation.\n"
             "This validates that the selection strategy works on unseen data.",
        key="ps_run_wf",
    )

    wf_anchored = True
    wf_min_train_months = 6
    wf_test_months = 2

    if run_walk_forward:
        wf_type = st.sidebar.radio(
            "Walk-Forward Type",
            options=["Anchored (Expanding)", "Rolling"],
            index=0,
            help="**Anchored**: Training window grows over time (uses all history)\n"
                 "**Rolling**: Fixed-size training window that slides forward",
            key="ps_wf_type",
        )
        wf_anchored = wf_type == "Anchored (Expanding)"

        wf_min_train_months = st.sidebar.slider(
            "Min Training (months)",
            min_value=3,
            max_value=12,
            value=6,
            help="Minimum training period before first test",
            key="ps_wf_train",
        )

        wf_test_months = st.sidebar.slider(
            "Test Period (months)",
            min_value=1,
            max_value=6,
            value=2,
            help="Out-of-sample test period length",
            key="ps_wf_test",
        )

    # ==========================================================================
    # COLLECT CANDIDATE DATA
    # ==========================================================================
    st.markdown("---")

    # Build candidates: portfolio + variant combinations
    candidates = {}
    candidate_navs = {}

    def clean_name(name: str) -> str:
        return name.replace("_EqualWeight", "").replace("_", " ")

    with st.spinner("Loading portfolio data..."):
        for portfolio_name, portfolio_id in portfolio_map.items():
            for variant in selected_variants:
                nav_df = tracker.get_nav_series(portfolio_id, variant, start_date, end_date)

                if nav_df.empty or "nav" not in nav_df.columns:
                    continue

                nav_series = nav_df["nav"]
                if len(nav_series) < 30:
                    continue

                # Get returns
                if "daily_return" in nav_df.columns:
                    returns = nav_df["daily_return"].dropna()
                else:
                    returns = nav_series.pct_change().dropna()

                if len(returns) < 30:
                    continue

                # Build candidate name
                variant_short = {
                    "raw": "Raw",
                    "conservative": "Cons",
                    "trend_regime_v2": "Trend",
                }.get(variant, variant)

                candidate_name = f"{clean_name(portfolio_name)} ({variant_short})"
                candidates[candidate_name] = returns
                candidate_navs[candidate_name] = nav_series

    # ==========================================================================
    # SECTION 1: CANDIDATE OVERVIEW
    # ==========================================================================
    st.markdown("### 📋 Candidate Overview")

    if len(candidates) < 2:
        st.warning(
            f"Only {len(candidates)} candidate(s) found with 30+ days of data. "
            "Need at least 2 candidates for optimization. "
            "Try selecting more variants or adjusting the date range."
        )
        return

    st.info(f"**{len(candidates)}** portfolio+variant combinations with 30+ days of data")

    # Show candidate list in expander
    with st.expander("View All Candidates", expanded=False):
        candidate_list = sorted(candidates.keys())
        cols = st.columns(3)
        for i, name in enumerate(candidate_list):
            cols[i % 3].markdown(f"• {name}")

    # ==========================================================================
    # SECTION 2: METRICS RANKING TABLE
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 📈 Risk-Adjusted Metrics Ranking")

    # Calculate metrics for all candidates
    metrics_data = []
    for name, returns in candidates.items():
        nav = candidate_navs[name]
        metrics = calculate_candidate_metrics(nav, returns)
        metrics_data.append({
            "Candidate": name,
            "Sharpe": metrics["sharpe"],
            "Sortino": metrics["sortino"],
            "Calmar": metrics["calmar"],
            "UPI": metrics["upi"],
            "CAGR": metrics["cagr"],
            "Max DD": metrics["max_dd"],
            "Volatility": metrics["volatility"],
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Format for display
    def format_metrics_df(df):
        formatted = df.copy()
        formatted["Sharpe"] = formatted["Sharpe"].apply(lambda x: f"{x:.2f}")
        formatted["Sortino"] = formatted["Sortino"].apply(lambda x: f"{x:.2f}")
        formatted["Calmar"] = formatted["Calmar"].apply(lambda x: f"{x:.2f}")
        formatted["UPI"] = formatted["UPI"].apply(lambda x: f"{x:.2f}")
        formatted["CAGR"] = formatted["CAGR"].apply(lambda x: f"{x*100:.1f}%")
        formatted["Max DD"] = formatted["Max DD"].apply(lambda x: f"{x*100:.1f}%")
        formatted["Volatility"] = formatted["Volatility"].apply(lambda x: f"{x*100:.1f}%")
        return formatted

    # Sort by Sharpe by default
    metrics_df_sorted = metrics_df.sort_values("Sharpe", ascending=False)
    st.dataframe(
        format_metrics_df(metrics_df_sorted),
        use_container_width=True,
        hide_index=True,
    )

    # ==========================================================================
    # SECTION 3: CORRELATION MATRIX
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 🔗 Correlation Matrix")

    # Build correlation matrix
    returns_df = pd.DataFrame(candidates)
    returns_df = returns_df.dropna()
    corr_matrix = returns_df.corr()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Heatmap
        heatmap = create_correlation_heatmap(
            corr_matrix,
            title="Returns Correlation",
            height=max(400, 30 * len(corr_matrix)),
        )
        st.plotly_chart(heatmap, use_container_width=True)

    with col2:
        # Low correlation pairs
        st.markdown("**Low Correlation Pairs** (<0.4)")

        low_corr_pairs = []
        candidate_names = corr_matrix.columns.tolist()
        for i in range(len(candidate_names)):
            for j in range(i + 1, len(candidate_names)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val < 0.4:
                    low_corr_pairs.append({
                        "Pair": f"{candidate_names[i][:15]}... & {candidate_names[j][:15]}...",
                        "Corr": f"{corr_val:.2f}",
                    })

        if low_corr_pairs:
            low_corr_pairs.sort(key=lambda x: float(x["Corr"]))
            st.dataframe(
                pd.DataFrame(low_corr_pairs[:10]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No pairs with correlation < 0.4 found.")

    # ==========================================================================
    # SECTION 4: OPTIMAL COMBINATION FINDER
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 🎯 Optimal Portfolio Combination")

    with st.spinner("Finding optimal combinations..."):
        result = find_optimal_portfolio_combination(
            candidates,
            n_select=n_select,
            weights=weights,
            min_data_points=30,
        )

    if "error" in result:
        st.error(result["error"])
        return

    st.success(
        f"Evaluated **{result['combinations_evaluated']}** combinations "
        f"from {result['candidates_count']} candidates"
    )

    # Show top recommendation
    recommended = result["recommended"]
    if recommended:
        st.markdown("#### 🏆 Top Recommendation")

        # Show combination
        combo_names = recommended["combination"]
        st.markdown("**Selected Portfolios:**")
        for i, name in enumerate(combo_names, 1):
            metrics = result["portfolio_metrics"][name]
            st.markdown(
                f"{i}. **{name}** — "
                f"Sharpe: {metrics['sharpe']:.2f}, "
                f"CAGR: {metrics['cagr']*100:.1f}%, "
                f"MaxDD: {metrics['max_dd']*100:.1f}%"
            )

        # Show COMBINED portfolio metrics (this is what matters!)
        combined = recommended.get("combined_metrics", {})
        st.markdown("**Combined Portfolio Metrics:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Sharpe", f"{combined.get('sharpe', 0):.2f}")
        col2.metric("Sortino", f"{combined.get('sortino', 0):.2f}")
        col3.metric("Calmar", f"{combined.get('calmar', 0):.2f}")
        col4.metric("CAGR", f"{combined.get('cagr', 0)*100:.1f}%")
        col5.metric("Max DD", f"{combined.get('max_dd', 0)*100:.1f}%")

        # Show correlation info
        st.caption(f"Avg pairwise correlation: {recommended['avg_correlation']:.2f} | Score: {recommended['total_score']:.3f}")

        # Show alternatives
        if result["alternatives"]:
            with st.expander("View Alternative Combinations"):
                for i, alt in enumerate(result["alternatives"], 2):
                    combo_str = " + ".join(alt["combination"])
                    alt_combined = alt.get("combined_metrics", {})
                    st.markdown(
                        f"**#{i}**: {combo_str}  \n"
                        f"Combined Sharpe: {alt_combined.get('sharpe', 0):.2f}, "
                        f"CAGR: {alt_combined.get('cagr', 0)*100:.1f}%, "
                        f"Corr: {alt['avg_correlation']:.2f}"
                    )

    # ==========================================================================
    # SECTION 5: COMBINED PORTFOLIO SIMULATION
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 📉 Combined Portfolio Simulation")

    if recommended:
        selected_combo = list(recommended["combination"])

        # Calculate optimized weights
        weight_result = optimize_portfolio_weights(
            candidates,
            selected_combo,
            method=selected_weight_method,
            metric_weights=weights,
            max_position=max_position,
        )

        if "error" in weight_result:
            st.warning(f"Weight optimization failed: {weight_result['error']}. Using equal weights.")
            optimized_weights = {name: 1.0 / len(selected_combo) for name in selected_combo}
            weight_method_name = "Equal Weight"
        else:
            optimized_weights = weight_result["weights"]
            weight_method_name = weight_result["method"]

        # Show weight method info
        st.markdown(f"**Weight Method:** {weight_method_name}")
        if "metrics" in weight_result and "description" in weight_result.get("metrics", {}):
            st.caption(weight_result["metrics"]["description"])

        # Weight sliders (initialized with optimized weights, but adjustable)
        st.markdown("**Portfolio Weights:**")
        weight_cols = st.columns(len(selected_combo))
        sim_weights = []

        for i, (col, name) in enumerate(zip(weight_cols, selected_combo)):
            default_w = optimized_weights.get(name, 1.0 / len(selected_combo))
            w = col.slider(
                name[:20],
                min_value=0.0,
                max_value=1.0,
                value=float(default_w),
                step=0.05,
                key=f"ps_sim_weight_{i}",
            )
            sim_weights.append(w)

        # Normalize weights
        total_weight = sum(sim_weights)
        if total_weight > 0:
            sim_weights = [w / total_weight for w in sim_weights]

        # Show extra metrics for advanced weight methods
        if selected_weight_method != "equal" and "metrics" in weight_result:
            metrics_info = weight_result["metrics"]
            if "expected_sharpe" in metrics_info:
                st.caption(f"Expected Sharpe: {metrics_info['expected_sharpe']:.2f}, "
                          f"Expected Vol: {metrics_info['expected_vol']*100:.1f}%")
            elif "portfolio_vol" in metrics_info:
                st.caption(f"Optimized Portfolio Vol: {metrics_info['portfolio_vol']*100:.1f}%")

        # Run simulation
        sim_result = simulate_combined_portfolio(
            candidates,
            selected_combo,
            weights=sim_weights,
        )

        if "error" not in sim_result:
            # Show combined metrics
            col1, col2, col3, col4 = st.columns(4)
            metrics = sim_result["metrics"]
            col1.metric("Combined Sharpe", f"{metrics['sharpe']:.2f}")
            col2.metric("Combined CAGR", f"{metrics['cagr']*100:.1f}%")
            col3.metric("Combined Max DD", f"{metrics['max_dd']*100:.1f}%")
            col4.metric("Aligned Days", sim_result["aligned_days"])

            # NAV chart
            st.markdown("**Combined vs Individual NAV:**")

            # Build chart data
            chart_data = {"Combined": sim_result["nav"]}
            for name in selected_combo:
                if name in candidate_navs:
                    # Normalize to start at 100
                    nav = candidate_navs[name]
                    chart_data[name] = nav / nav.iloc[0] * 100

            chart_df = pd.DataFrame(chart_data)
            st.line_chart(chart_df)

            # Metrics comparison
            with st.expander("View Detailed Metrics Comparison"):
                comparison_data = [
                    {
                        "Portfolio": "**Combined**",
                        "Sharpe": f"{metrics['sharpe']:.2f}",
                        "Sortino": f"{metrics['sortino']:.2f}",
                        "CAGR": f"{metrics['cagr']*100:.1f}%",
                        "Max DD": f"{metrics['max_dd']*100:.1f}%",
                        "Volatility": f"{metrics['volatility']*100:.1f}%",
                    }
                ]
                for name in selected_combo:
                    m = result["portfolio_metrics"][name]
                    comparison_data.append({
                        "Portfolio": name,
                        "Sharpe": f"{m['sharpe']:.2f}",
                        "Sortino": f"{m['sortino']:.2f}",
                        "CAGR": f"{m['cagr']*100:.1f}%",
                        "Max DD": f"{m['max_dd']*100:.1f}%",
                        "Volatility": f"{m['volatility']*100:.1f}%",
                    })
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    # ==========================================================================
    # SECTION 6: ROBUSTNESS ANALYSIS
    # ==========================================================================
    if recommended and "error" not in sim_result:
        with st.expander("📊 Robustness Analysis"):
            st.markdown("**Rolling Sharpe (60-day window):**")

            rolling_sharpe = sim_result.get("rolling_sharpe")
            if rolling_sharpe is not None and len(rolling_sharpe) > 0:
                st.line_chart(rolling_sharpe.dropna())

                sharpe_stability = sim_result.get("sharpe_stability", 0)
                st.metric(
                    "Sharpe Stability (StdDev)",
                    f"{sharpe_stability:.3f}",
                    help="Lower = more consistent performance",
                )
            else:
                st.caption("Insufficient data for rolling analysis.")

    # ==========================================================================
    # SECTION 7: WALK-FORWARD VALIDATION
    # ==========================================================================
    if run_walk_forward and len(candidates) >= n_select:
        st.markdown("---")
        st.markdown("### 🔄 Walk-Forward Validation")

        with st.spinner("Running walk-forward validation... This may take a moment."):
            wf_result = walk_forward_validation(
                portfolio_returns=candidates,
                n_select=n_select,
                metric_weights=weights,
                weight_method=selected_weight_method,
                max_position=max_position,
                test_months=wf_test_months,
                min_train_months=wf_min_train_months,
                anchored=wf_anchored,
            )

        if "error" in wf_result:
            st.warning(f"Walk-forward validation: {wf_result['error']}")
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            oos_metrics = wf_result.get("aggregated_oos_metrics", {})
            is_metrics = wf_result.get("aggregated_is_metrics", {})
            robustness = wf_result.get("robustness_score", 0)

            col1.metric(
                "OOS Sharpe",
                f"{oos_metrics.get('sharpe', 0):.2f}",
                help="Average out-of-sample Sharpe across all windows",
            )
            col2.metric(
                "OOS CAGR",
                f"{oos_metrics.get('cagr', 0)*100:.1f}%",
                help="Average out-of-sample CAGR",
            )
            col3.metric(
                "Windows",
                wf_result.get("total_windows", 0),
                help="Number of walk-forward windows completed",
            )

            # Robustness score with color coding
            robustness_pct = robustness * 100
            if robustness_pct >= 70:
                col4.metric("Robustness", f"{robustness_pct:.0f}%", help="OOS Sharpe / In-sample Sharpe. Above 70% is good.")
            elif robustness_pct >= 50:
                col4.metric("Robustness", f"{robustness_pct:.0f}%", delta="-moderate", help="OOS Sharpe / In-sample Sharpe")
            else:
                col4.metric("Robustness", f"{robustness_pct:.0f}%", delta="-weak", help="OOS Sharpe / In-sample Sharpe")

            # Selection consistency
            consistency = wf_result.get("selection_consistency", {})
            if consistency:
                st.markdown("**Portfolio Selection Consistency:**")
                consistency_data = [
                    {"Portfolio": name, "Selected %": f"{pct:.0f}%"}
                    for name, pct in consistency.items()
                ]
                st.dataframe(
                    pd.DataFrame(consistency_data),
                    use_container_width=True,
                    hide_index=True,
                )

            # OOS equity curve
            oos_equity = wf_result.get("oos_equity_curve")
            if oos_equity is not None and len(oos_equity) > 0:
                st.markdown("**Out-of-Sample Equity Curve:**")
                st.line_chart(oos_equity)

            # Window details
            windows = wf_result.get("windows", [])
            if windows:
                with st.expander("View Window Details"):
                    window_data = []
                    for w in windows:
                        window_data.append({
                            "#": w["window"],
                            "Train": f"{w['train_start']} to {w['train_end']}",
                            "Test": f"{w['test_start']} to {w['test_end']}",
                            "Selected": ", ".join(w["selected"][:2]) + ("..." if len(w["selected"]) > 2 else ""),
                            "IS Sharpe": f"{w['in_sample_sharpe']:.2f}",
                            "OOS Sharpe": f"{w['oos_sharpe']:.2f}",
                        })
                    st.dataframe(pd.DataFrame(window_data), use_container_width=True, hide_index=True)

            # Robustness warning
            if robustness_pct < 70:
                st.warning(
                    f"⚠️ Robustness score ({robustness_pct:.0f}%) is below 70%. "
                    "The strategy may be overfitting to historical data. "
                    "Consider using different metric weights or more conservative settings."
                )


# -----------------------------------------------------------------------------
# 5) Router
# -----------------------------------------------------------------------------
if page == "Backtester":
    show_backtester_ui()
elif page == "Optimizer":
    show_optimizer_ui()
elif page == "Performance Tracker":
    show_performance_tracker_safe()
elif page == "Portfolio Selection":
    show_portfolio_selection_ui()
else:
    show_data_ui()

