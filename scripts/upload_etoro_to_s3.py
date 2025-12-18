#!/usr/bin/env python3
"""
Upload eToro scraped data to S3

Uploads:
1. etoro/live_stats.json - Current data (overwritten daily)
2. etoro/history/YYYY-MM-DD.json - Historical data (preserved)

This allows the app to read live data without redeploying,
and keeps history for tracking performance over time.
"""
import json
import boto3
from datetime import datetime
from pathlib import Path

# S3 configuration
S3_BUCKET = "alphamachine-data"
S3_PREFIX = "etoro"

# Local data file (created by update_etoro_data.py)
DATA_FILE = Path(__file__).parent.parent / "data" / "etoro_scraped_data.json"


def upload_to_s3():
    """Upload eToro data to S3."""
    print("=" * 60)
    print("Uploading eToro data to S3")
    print("=" * 60)

    # Check if data file exists
    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print("Run update_etoro_data.py first to scrape data.")
        return False

    # Load the scraped data
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded data from {DATA_FILE}")
    print(f"  Scraped at: {data.get('scraped_at', 'unknown')}")
    print(f"  Investors: {len(data.get('investors', []))}")

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Get today's date for history file
    today = datetime.now().strftime('%Y-%m-%d')

    # Upload as live_stats.json (current data)
    live_key = f"{S3_PREFIX}/live_stats.json"
    print(f"\nUploading to s3://{S3_BUCKET}/{live_key}")

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=live_key,
        Body=json.dumps(data, indent=2),
        ContentType='application/json',
        CacheControl='max-age=300',  # Cache for 5 minutes
    )
    print(f"  Uploaded live_stats.json")

    # Upload to history folder (preserved)
    history_key = f"{S3_PREFIX}/history/{today}.json"
    print(f"\nUploading to s3://{S3_BUCKET}/{history_key}")

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=history_key,
        Body=json.dumps(data, indent=2),
        ContentType='application/json',
    )
    print(f"  Uploaded {today}.json to history")

    print("\n" + "=" * 60)
    print("Upload complete!")
    print(f"Live data URL: https://{S3_BUCKET}.s3.eu-central-1.amazonaws.com/{live_key}")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = upload_to_s3()
    exit(0 if success else 1)
