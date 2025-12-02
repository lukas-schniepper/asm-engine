#!/usr/bin/env python3
"""
Migration script to create Portfolio Tracking tables in Supabase.

This script creates the following tables:
- portfolio_definitions
- portfolio_holdings
- portfolio_daily_nav
- overlay_signals
- portfolio_metrics

Usage:
    python scripts/migrate_tracking_tables.py

Environment:
    DATABASE_URL must be set (or in .streamlit/secrets.toml)
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load DATABASE_URL directly to avoid streamlit dependency
import toml
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Try loading from secrets.toml
    secrets_path = project_root / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        secrets = toml.load(secrets_path)
        DATABASE_URL = secrets.get("DATABASE_URL")

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found in environment or .streamlit/secrets.toml")
    sys.exit(1)

# Set it in environment so db.py can find it
os.environ["DATABASE_URL"] = DATABASE_URL

# Now we can safely import - but we need to avoid the config import
# So we create engine directly
from sqlalchemy import create_engine, text
from sqlmodel import SQLModel

engine = create_engine(DATABASE_URL, echo=False)

# Import models directly by loading the module to avoid the __init__.py chain
import importlib.util
models_path = project_root / "AlphaMachine_core" / "tracking" / "models.py"
spec = importlib.util.spec_from_file_location("models", models_path)
models_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_module)

PortfolioDefinition = models_module.PortfolioDefinition
PortfolioHolding = models_module.PortfolioHolding
PortfolioDailyNAV = models_module.PortfolioDailyNAV
OverlaySignal = models_module.OverlaySignal
PortfolioMetric = models_module.PortfolioMetric


def create_tables():
    """Create all tracking tables using SQLModel."""
    print("=" * 60)
    print("Portfolio Tracking - Database Migration")
    print("=" * 60)

    # Create tables via SQLModel
    print("\n[1/3] Creating tables via SQLModel...")
    SQLModel.metadata.create_all(engine)
    print("      Tables created successfully")

    # Add unique constraints and indexes via raw SQL
    print("\n[2/3] Adding constraints and indexes...")

    # Execute DO blocks separately since they contain semicolons
    do_blocks = [
        """DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_portfolio_holdings_portfolio_date_ticker'
    ) THEN
        ALTER TABLE portfolio_holdings
        ADD CONSTRAINT uq_portfolio_holdings_portfolio_date_ticker
        UNIQUE (portfolio_id, effective_date, ticker);
    END IF;
END $$""",
        """DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_portfolio_daily_nav_portfolio_date_variant'
    ) THEN
        ALTER TABLE portfolio_daily_nav
        ADD CONSTRAINT uq_portfolio_daily_nav_portfolio_date_variant
        UNIQUE (portfolio_id, trade_date, variant);
    END IF;
END $$""",
        """DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_overlay_signals_date_model'
    ) THEN
        ALTER TABLE overlay_signals
        ADD CONSTRAINT uq_overlay_signals_date_model
        UNIQUE (trade_date, model);
    END IF;
END $$""",
        """DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'uq_portfolio_metrics_portfolio_variant_period_start'
    ) THEN
        ALTER TABLE portfolio_metrics
        ADD CONSTRAINT uq_portfolio_metrics_portfolio_variant_period_start
        UNIQUE (portfolio_id, variant, period_type, period_start);
    END IF;
END $$""",
    ]

    with engine.connect() as conn:
        for do_block in do_blocks:
            try:
                conn.execute(text(do_block))
                conn.commit()
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"      Warning: {e}")
                conn.rollback()

    # Indexes as separate statements
    index_statements = [
        """CREATE INDEX IF NOT EXISTS idx_nav_portfolio_variant_date
           ON portfolio_daily_nav(portfolio_id, variant, trade_date)""",
        """CREATE INDEX IF NOT EXISTS idx_holdings_effective_date
           ON portfolio_holdings(effective_date DESC)""",
        """CREATE INDEX IF NOT EXISTS idx_metrics_period_lookup
           ON portfolio_metrics(portfolio_id, variant, period_type, period_end DESC)""",
    ]

    with engine.connect() as conn:
        for statement in index_statements:
            try:
                conn.execute(text(statement))
                conn.commit()
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"      Warning: {e}")
                conn.rollback()

    print("      Constraints and indexes added")

    # Verify tables
    print("\n[3/3] Verifying tables...")
    verify_sql = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name IN (
        'portfolio_definitions',
        'portfolio_holdings',
        'portfolio_daily_nav',
        'overlay_signals',
        'portfolio_metrics'
    )
    ORDER BY table_name;
    """

    with engine.connect() as conn:
        result = conn.execute(text(verify_sql))
        tables = [row[0] for row in result]

    expected = [
        "overlay_signals",
        "portfolio_daily_nav",
        "portfolio_definitions",
        "portfolio_holdings",
        "portfolio_metrics",
    ]

    if sorted(tables) == sorted(expected):
        print("      All 5 tables verified:")
        for t in tables:
            print(f"        - {t}")
    else:
        missing = set(expected) - set(tables)
        print(f"      WARNING: Missing tables: {missing}")
        return False

    print("\n" + "=" * 60)
    print("Migration completed successfully!")
    print("=" * 60)
    return True


def show_table_info():
    """Show information about the created tables."""
    info_sql = """
    SELECT
        c.table_name,
        c.column_name,
        c.data_type,
        c.is_nullable,
        c.column_default
    FROM information_schema.columns c
    WHERE c.table_schema = 'public'
    AND c.table_name IN (
        'portfolio_definitions',
        'portfolio_holdings',
        'portfolio_daily_nav',
        'overlay_signals',
        'portfolio_metrics'
    )
    ORDER BY c.table_name, c.ordinal_position;
    """

    with engine.connect() as conn:
        result = conn.execute(text(info_sql))
        rows = result.fetchall()

    current_table = None
    for row in rows:
        table, column, dtype, nullable, default = row
        if table != current_table:
            print(f"\n{table}:")
            print("-" * 60)
            current_table = table
        nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
        default_str = f" DEFAULT {default}" if default else ""
        print(f"  {column:25} {dtype:15} {nullable_str}{default_str}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate Portfolio Tracking tables")
    parser.add_argument("--info", action="store_true", help="Show table information after migration")
    args = parser.parse_args()

    success = create_tables()

    if success and args.info:
        print("\n")
        show_table_info()

    sys.exit(0 if success else 1)
