#!/usr/bin/env python3
"""
Create the etoro_stats table in Supabase.

Run this once to create the table:
    python scripts/create_etoro_stats_table.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from AlphaMachine_core.db import engine
from AlphaMachine_core.models import EToroStats
from sqlmodel import SQLModel


def create_table():
    """Create the etoro_stats table."""
    print("Creating etoro_stats table...")

    # Create only the EToroStats table (not all tables)
    EToroStats.metadata.create_all(engine, tables=[EToroStats.__table__])

    print("Table created successfully!")
    print("\nYou can now run the GitHub Actions workflow to populate the table.")


if __name__ == '__main__':
    create_table()
