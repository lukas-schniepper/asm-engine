#!/usr/bin/env python3
"""Run database migration to add adjusted_close column."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load secrets
secrets_path = project_root / '.streamlit' / 'secrets.toml'
if secrets_path.exists():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(secrets_path, 'rb') as f:
        secrets = tomllib.load(f)
    for key, value in secrets.items():
        if key not in os.environ and isinstance(value, str):
            os.environ[key] = value

from sqlalchemy import text
from AlphaMachine_core.db import engine

def run_migration():
    """Add adjusted_close column to price_data table."""
    print("Running migration: Adding adjusted_close column to price_data table...")

    with engine.connect() as conn:
        # Check if column already exists
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'price_data' AND column_name = 'adjusted_close'
        """))

        if result.fetchone():
            print("Column 'adjusted_close' already exists. Skipping migration.")
            return

        # Add the column
        conn.execute(text("""
            ALTER TABLE price_data
            ADD COLUMN adjusted_close DOUBLE PRECISION
        """))
        conn.commit()
        print("Successfully added 'adjusted_close' column to price_data table.")

        # Verify
        result = conn.execute(text("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'price_data'
            ORDER BY ordinal_position
        """))

        print("\nCurrent price_data columns:")
        for row in result:
            print(f"  - {row[0]}: {row[1]}")

if __name__ == "__main__":
    run_migration()
