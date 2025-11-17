# Scheduled Price Updates

## Overview

This directory contains scripts for automated price updates via GitHub Actions.

## Workflow Configuration

**Schedule:**
- **Monday** at 02:00 Swiss time (CET/CEST)
- **Thursday** at 02:00 Swiss time (CET/CEST)

**What it does:**
1. Fetches all tickers from Supabase database
2. Updates prices for all tickers using EODHD API
3. Logs detailed results (updated/skipped tickers)

## GitHub Secrets Setup

The workflow requires two secrets to be configured in your GitHub repository:

### 1. Navigate to Repository Settings

Go to: `https://github.com/lukas-schniepper/asm-engine/settings/secrets/actions`

### 2. Add Required Secrets

Click **"New repository secret"** and add:

#### SECRET 1: `DATABASE_URL`
```
Name: DATABASE_URL
Value: postgresql://postgres.rmjvbadnwrgduojlasas:VeIE5a6afpTw2B3r@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require
```

#### SECRET 2: `EODHD_API_KEY`
```
Name: EODHD_API_KEY
Value: 688642d59e8c34.95661441
```

### 3. Verify Secrets

After adding both secrets, you should see them listed (values are hidden):
- ✅ DATABASE_URL
- ✅ EODHD_API_KEY

## Manual Trigger

You can manually trigger the workflow from GitHub Actions UI:

1. Go to: `https://github.com/lukas-schniepper/asm-engine/actions`
2. Select **"Update Ticker Prices"** workflow
3. Click **"Run workflow"** button
4. Select branch (usually `main`)
5. Click **"Run workflow"**

## Monitoring

### View Workflow Runs

Check execution logs at:
`https://github.com/lukas-schniepper/asm-engine/actions`

### Expected Output

```
================================================================================
🕐 Scheduled Price Update - 2025-11-18 02:00:15
================================================================================

✅ Environment variables verified
✅ DataManager initialized successfully
✅ Found 738 unique tickers

🔄 Starting update for 738 tickers...
⏱️  Estimated duration: ~6.2 minutes

================================================================================
📊 Update Summary
================================================================================
✅ Updated: 650 tickers
⏭️  Skipped: 88 tickers
📈 Total:   738 tickers

Sample updated tickers:
  ✅ AAPL: 1 Tage von 2025-11-18 bis 2025-11-18
  ✅ SPY: 1 Tage von 2025-11-18 bis 2025-11-18
  ...

✅ Scheduled update completed successfully!
```

## Troubleshooting

### Workflow fails with "DATABASE_URL not set"
→ Secret `DATABASE_URL` is not configured. Add it in repository settings.

### Workflow fails with "EODHD_API_KEY not set"
→ Secret `EODHD_API_KEY` is not configured. Add it in repository settings.

### Workflow fails with timeout
→ Increase timeout in workflow file (default is 60 minutes)

### No tickers found in database
→ Database connection issue. Check DATABASE_URL secret value.

## Files

- `.github/workflows/update-prices.yml` - GitHub Actions workflow definition
- `scripts/scheduled_price_update.py` - Python script that performs the update
- `scripts/README.md` - This documentation file

## Notes

- GitHub Actions has a **6-hour execution limit** per workflow run
- With 738 tickers at ~0.5s each, total runtime is ~6-10 minutes (well within limit)
- Failed runs will show in GitHub Actions UI with error logs
- Workflow runs even if repository has no recent commits
