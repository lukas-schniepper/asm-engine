# ASM-ENGINE DEPLOYMENT GUIDE

## 🎉 Migration Complete!

Your codebase has been successfully migrated from yfinance to EODHD API. All changes are committed and pushed to GitHub.

**Repository:** https://github.com/lukas-schniepper/asm-engine

---

## 📊 Summary of Changes

### ✅ Completed

1. **EODHD HTTP Client Created**
   - File: `AlphaMachine_core/data_sources/eodhd_http_client.py` (370 lines)
   - Direct HTTP API calls - no external library needed
   - Uses existing `requests` library from requirements.txt
   - Automatic ticker format transformation (SPY → SPY.US)
   - Built-in retry logic and error handling

2. **Data Manager Updated**
   - File: `AlphaMachine_core/data_manager.py`
   - Replaced yfinance with EODHD client
   - Initializes EODHD client on startup
   - Reads API key from environment or Streamlit secrets

3. **Risk Overlay Simplified**
   - File: `AlphaMachine_core/risk_overlay/overlay_config.json`
   - Removed: AAII_Sentiment_ZScore indicator (not available)
   - Removed: VIX_Above_30_Alert indicator (not needed)
   - Now uses 5 SPY-based indicators only

4. **Streamlit App Cleaned**
   - File: `app/streamlit_app.py`
   - Removed: Risk Overlay Test page (186 lines)
   - Removed: Helper functions for test page
   - Navigation menu now shows only 3 pages

5. **Obsolete Files Deleted**
   - `OBSOLET models.py`
   - `dummy.py`
   - `test_db2.py`
   - `test_overlay_step1.py`
   - `minimal_dm_test.py`
   - `create_test_data_csvs.py`
   - `test_data_spy.csv`, `test_data_vix.csv`

6. **Secrets Template Updated**
   - File: `.streamlit/secrets.toml`
   - Added: `EODHD_API_KEY` placeholder

---

## 🚀 NEXT STEP: Deploy to Streamlit Cloud

### Step 1: Add Your EODHD API Key Locally (Right Now!)

**IMPORTANT:** Before deploying, you need to add your EODHD API key to the local secrets file.

1. Open this file in your code editor:
   ```
   c:\Users\Lukas\codebase\asm-engine\.streamlit\secrets.toml
   ```

2. Find this line:
   ```toml
   EODHD_API_KEY = "PASTE_YOUR_EODHD_API_KEY_HERE"
   ```

3. Replace `PASTE_YOUR_EODHD_API_KEY_HERE` with your actual EODHD API key

4. Save the file

5. **DO NOT commit this file** (it's already in .gitignore)

---

### Step 2: Deploy to Streamlit Cloud

1. **Go to:** https://share.streamlit.io/

2. **Click:** "New app"

3. **Configure:**
   - **Repository:** `lukas-schniepper/asm-engine`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** Choose a custom URL (e.g., `asm-engine.streamlit.app`)

4. **Advanced Settings → Secrets:**

   Copy and paste this (replace with your actual values):

   ```toml
   DATABASE_URL = "postgresql://postgres.rmjvbadnwrgduojlasas:VeIE5a6afpTw2B3r@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?sslmode=require"

   EODHD_API_KEY = "YOUR_ACTUAL_EODHD_API_KEY_HERE"

   APP_PW = "your_app_password_if_you_have_one"
   ```

5. **Click:** "Deploy!"

6. **Wait:** 2-3 minutes for deployment

---

### Step 3: Verify Deployment

Once deployed, test the following:

#### ✅ App Startup
- [ ] App loads without errors
- [ ] Check logs for: `✅ EODHD HTTP client initialized successfully`
- [ ] Navigation shows only 3 pages: Backtester, Optimizer, Data Mgmt

#### ✅ Data Management
- [ ] Navigate to "Data Mgmt" page
- [ ] Add a test ticker (e.g., SPY) for current month
- [ ] Click "🔄 Alle Preise für DB-Ticker updaten"
- [ ] Watch logs - should see EODHD API fetching data
- [ ] Verify no error messages

#### ✅ Backtester
- [ ] Navigate to "Backtester" page
- [ ] Run a simple backtest with SPY
- [ ] Verify RiskOverlay works (should show 5 indicators)
- [ ] Check equity curve displays correctly

#### ✅ Data Quality
- [ ] In Data Mgmt, view PriceData records
- [ ] Spot-check: SPY close price matches market data
- [ ] Verify volume data looks reasonable

---

## 🔧 Troubleshooting

### Issue: "EODHD_API_KEY not found"
**Solution:** Make sure you added the API key to Streamlit Cloud secrets (Step 2.4 above)

### Issue: "Ticker not found in EODHD database"
**Solution:** EODHD uses different ticker formats. The client automatically converts:
- SPY → SPY.US (stocks)
- ^GSPC → GSPC.INDX (indices)

If a ticker doesn't work, check the EODHD ticker search: https://eodhd.com/financial-apis/

### Issue: Empty DataFrames returned
**Solution:**
1. Check date range - EODHD may not have data for very old dates
2. Verify API key is correct and has All-in-One plan access
3. Check Streamlit logs for specific error messages

### Issue: Risk Overlay not working
**Solution:**
- RiskOverlay now uses only SPY-based indicators
- Verify SPY data exists in database
- Check overlay config has 5 indicators (not 7)

---

## 📈 Performance Notes

### EODHD vs yfinance

| Aspect | yfinance | EODHD (All-in-One) |
|--------|----------|---------------------|
| **Rate Limits** | None documented | 100,000 calls/day |
| **Data Quality** | Community-sourced | Professional-grade |
| **Reliability** | Can break unexpectedly | SLA-backed |
| **Cost** | Free | $79.99/month |
| **Support** | Community only | Email support |

**Your Plan:** All-in-One - unlimited for your needs ✅

---

## 🔄 Rollback Plan (If Needed)

If you encounter critical issues and need to revert to the old AlphaMachine:

1. **Keep old app running:** Your original AlphaMachine on Streamlit Cloud still works
2. **Switch back:** Just use the old app URL
3. **Fix issues:** Debug in asm-engine repo
4. **Redeploy:** Once fixed, Streamlit Cloud auto-deploys on git push

---

## 📝 Important Files Modified

| File | Lines Changed | Status |
|------|---------------|--------|
| `AlphaMachine_core/data_manager.py` | ~30 modified | ✅ Migrated |
| `AlphaMachine_core/risk_overlay/overlay_config.json` | 2 indicators removed | ✅ Simplified |
| `app/streamlit_app.py` | 186 lines removed | ✅ Cleaned |
| `AlphaMachine_core/data_sources/eodhd_http_client.py` | 370 lines added | ✅ New |

---

## 🎯 Next Steps After Deployment

### Week 1: Parallel Testing
- Run both apps in parallel (old AlphaMachine + new asm-engine)
- Compare data quality
- Verify backtests produce similar results
- Monitor for any errors or issues

### Week 2: Confidence Building
- Use asm-engine as your primary app
- Continue monitoring old app as backup
- Test edge cases (unusual tickers, date ranges)

### Week 3: Full Migration
- Once confident, delete old AlphaMachine app from Streamlit Cloud
- Optionally rename asm-engine to AlphaMachine
- Archive old repository

---

## 💬 Support

### EODHD Support
- Documentation: https://eodhd.com/financial-apis/
- Email: support@eodhd.com (check their website for current contact)
- Status: Monitor via dashboard

### Code Issues
- GitHub: https://github.com/lukas-schniepper/asm-engine/issues

---

## ✅ Pre-Deployment Checklist

Before clicking "Deploy" on Streamlit Cloud:

- [ ] EODHD API key ready
- [ ] DATABASE_URL copied from old secrets
- [ ] APP_PW copied from old secrets (if you have one)
- [ ] All secrets pasted into Streamlit Cloud "Advanced Settings"
- [ ] Repository is public (required for Streamlit free tier)

---

**Ready to deploy?** Follow Step 2 above and deploy to Streamlit Cloud!

**Questions?** Review this guide or check the troubleshooting section.

---

Generated: 2025-11-17
Migration completed by: Claude Code (Sonnet 4.5)
Repository: https://github.com/lukas-schniepper/asm-engine
