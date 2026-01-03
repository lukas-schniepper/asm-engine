#!/usr/bin/env python3
"""
eToro Data Updater - Scrape live data and update DEMO_DATA

This script uses Selenium to scrape real-time data from eToro profiles
and updates the DEMO_DATA in etoro_scraper.py.

Designed to run via GitHub Actions (has Chrome pre-installed) or locally.

Usage:
    python scripts/update_etoro_data.py

Requirements:
    pip install selenium webdriver-manager
"""
import os
import re
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("ERROR: Selenium not installed. Run: pip install selenium webdriver-manager")
    sys.exit(1)


# Usernames to scrape
USERNAMES_TO_SCRAPE = [
    'alphawizzard',     # Your portfolio
    'thomaspj',         # #1 Thomas Parry Jones
    'jeppekirkbonde',   # #2 Jeppe Kirk Bonde
    'triangulacapital', # #3 Pietari Laurila
    'cphequities',      # #4 Blue Screen Media ApS
    'fundmanagerzech',  # #5 Zechariah Bin Zheng
]


def create_driver():
    """Create a headless Chrome driver (works on GitHub Actions and locally)."""
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    # On GitHub Actions, Chrome is installed at a known location
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        options.binary_location = '/usr/bin/google-chrome'
        # Use system Chrome driver
        driver = webdriver.Chrome(options=options)
    else:
        # Locally, use webdriver-manager to get the driver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

    return driver


def scrape_monthly_returns(driver, username: str) -> dict:
    """
    Scrape monthly returns from eToro stats page.

    Returns dict with monthly returns and other stats.
    """
    url = f"https://www.etoro.com/people/{username}/stats"
    print(f"  Scraping {url}...")

    driver.get(url)
    time.sleep(8)  # Wait for JavaScript to render

    # Get page content
    page_source = driver.page_source
    body = driver.find_element(By.TAG_NAME, 'body')
    visible_text = body.text
    title = driver.title

    result = {
        'username': username,
        'monthly_returns': {},
        'full_name': username,
        'risk_score': 5,
        'copiers': 0,
        'win_ratio': 50.0,
    }

    # Extract full name from title (format: "Full Name @username Stats")
    name_match = re.search(r'^(.+?)\s+@', title)
    if name_match:
        result['full_name'] = name_match.group(1).strip()

    # Extract user ID from avatar URL - try multiple patterns
    user_id_patterns = [
        r'avatars/\d+X\d+/(\d+)/',           # avatars/50X50/12345678/
        r'avatars/\d+x\d+/(\d+)/',           # avatars/50x50/12345678/ (lowercase)
        r'/avatars/[^/]+/(\d{5,})/',         # /avatars/*/12345678/
        r'"cid":(\d{5,})',                   # "cid":12345678 (customer ID in JSON)
        r'"userId":(\d{5,})',                # "userId":12345678
        r'user[_-]?id["\s:=]+(\d{5,})',      # user_id: 12345678
    ]
    for pattern in user_id_patterns:
        user_id_match = re.search(pattern, page_source, re.IGNORECASE)
        if user_id_match:
            result['user_id'] = int(user_id_match.group(1))
            print(f"    Found user_id: {result['user_id']} (pattern: {pattern})")
            break
    else:
        print(f"    WARNING: Could not extract user_id from page")

    # Extract risk score
    risk_match = re.search(r'Risk Score[^\d]*(\d+)', visible_text, re.IGNORECASE)
    if risk_match:
        result['risk_score'] = int(risk_match.group(1))

    # Extract copiers count

    # Pattern 1: "Copiers (12M)\n{number}" - number on next line after label
    copiers_match = re.search(r'[Cc]opiers\s*\(12M\)\s*\n\s*([\d,]+)', visible_text)
    if not copiers_match:
        # Pattern 2: "X Copiers" or "X,XXX Copiers"
        copiers_match = re.search(r'([\d,]+)\s*[Cc]opiers', visible_text)
    if not copiers_match:
        # Pattern 3: "Copiers X" or "Copiers: X"
        copiers_match = re.search(r'[Cc]opiers[:\s]*([\d,]+)', visible_text)
    if not copiers_match:
        # Pattern 4: Look in page source for data attributes
        copiers_match = re.search(r'"copiers"[:\s]*(\d+)', page_source)
    if copiers_match:
        result['copiers'] = int(copiers_match.group(1).replace(',', ''))

    # Extract win ratio (profitable weeks)
    win_match = re.search(r'(\d+\.?\d*)%\s*Profitable', visible_text)
    if win_match:
        result['win_ratio'] = float(win_match.group(1))

    # Parse monthly returns from eToro stats page
    # eToro shows months in a table with format: Dec Nov Oct Sep Aug Jul Jun May Apr Mar Feb Jan
    # Each month has a corresponding percentage value below it
    month_to_num = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
                   'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

    lines = visible_text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    # Get current date info
    now = datetime.now()
    current_year = now.year
    current_month = now.month

    # Strategy 1: Look for month sequence followed by percentage values
    # Find indices where we see month abbreviations
    month_indices = []
    for i, line in enumerate(lines):
        if line in month_to_num:
            month_indices.append((i, line))

    # Find percentage values (format: "3.19%", "-0.25%", "0", "0.0")
    pct_pattern = re.compile(r'^-?\d+\.?\d*%?$')

    # Look for consecutive month names (Dec Nov Oct ... Jan pattern)
    # This is the header row in the Performance table
    if len(month_indices) >= 12:
        expected_order = ['Dec', 'Nov', 'Oct', 'Sep', 'Aug', 'Jul', 'Jun', 'May', 'Apr', 'Mar', 'Feb', 'Jan']

        for start_idx, (line_idx, month_name) in enumerate(month_indices):
            if month_name == 'Dec':
                sequence_months = [m for _, m in month_indices[start_idx:start_idx+12]]
                if sequence_months == expected_order:
                    last_month_idx = month_indices[start_idx + 11][0]

                    # NEW APPROACH: Group values by year label
                    # eToro 2026 layout - year labels come AFTER the values for that year!
                    #   [month headers: Dec Nov ... Jan]
                    #   1.58% 1.58%          <- 2026 values (appear BEFORE "2026" label)
                    #   2026                 <- year label marks END of 2026's row
                    #   4.98% 1.49% 6.63% .. <- 2025 values (appear BEFORE "2025" label)
                    #   2025                 <- year label marks END of 2025's row
                    #   ...
                    #
                    # Within each year row: YTD first, then Dec, Nov, Oct, ..., Jan
                    # (YTD on left in visual table, months right-to-left visually = Dec to Jan in text)

                    # Collect segments: values between year labels
                    segments = []  # [(year_label, [values])]
                    current_values = []
                    last_year_seen = None

                    for i in range(last_month_idx + 1, min(last_month_idx + 100, len(lines))):
                        line = lines[i]

                        # Detect year labels (4-digit years like 2026, 2025)
                        if line.isdigit() and len(line) == 4:
                            year_int = int(line)
                            if 2020 <= year_int <= 2030:
                                # Save the segment that just ended
                                if current_values:
                                    segments.append((year_int, current_values))
                                current_values = []
                                last_year_seen = year_int
                                continue

                        # Stop if we hit another month header section
                        if line in month_to_num:
                            break

                        # Collect percentage values
                        if pct_pattern.match(line):
                            val_str = line.replace('%', '')
                            try:
                                val = float(val_str)
                                current_values.append(val)
                            except ValueError:
                                pass

                    # Handle any remaining values
                    if current_values and last_year_seen:
                        segments.append((last_year_seen - 1, current_values))

                    print(f"    Segments: {segments}")

                    # Map values to year-month keys
                    # Order within each year row: YTD, Dec, Nov, Oct, ..., Jan
                    # So skip first value (YTD), then map Dec-first

                    for year_label, values in segments:
                        if not values:
                            continue

                        # Skip current year's segment - it's handled by pre-year values logic below
                        # (In January 2026, the "2026" segment only has partial data)
                        if year_label == current_year:
                            print(f"    Skipping current year segment ({year_label}) - handled by pre-year logic")
                            continue

                        # Skip values that are clearly not monthly returns (like 100%)
                        if len(values) == 1 and (values[0] > 50 or values[0] < -50):
                            print(f"    Skipping segment for {year_label}: {values} (not monthly data)")
                            continue

                        # First value is YTD, rest are monthly in Dec, Nov, ..., Jan order
                        if len(values) > 1:
                            ytd = values[0]
                            monthly_dec_first = values[1:]  # Skip YTD
                            print(f"    Year {year_label}: YTD={ytd}, monthly={monthly_dec_first}")

                            # Map to Dec, Nov, Oct, ..., Jan
                            for i, month_name in enumerate(expected_order):
                                if i < len(monthly_dec_first):
                                    month_num = month_to_num[month_name]
                                    month_key = f"{year_label}-{month_num}"
                                    result['monthly_returns'][month_key] = monthly_dec_first[i]
                        else:
                            # Single value - might be partial year with just YTD
                            print(f"    Year {year_label}: single value {values[0]} (YTD only?)")

                    # Handle values before first year label (current year's partial data)
                    # Re-scan to find values before first year label
                    pre_year_values = []
                    for i in range(last_month_idx + 1, min(last_month_idx + 100, len(lines))):
                        line = lines[i]
                        if line.isdigit() and len(line) == 4 and 2020 <= int(line) <= 2030:
                            break  # Hit first year label
                        if line in month_to_num:
                            break
                        if pct_pattern.match(line):
                            val_str = line.replace('%', '')
                            try:
                                pre_year_values.append(float(val_str))
                            except ValueError:
                                pass

                    if pre_year_values:
                        print(f"    Pre-year values (current year {current_year}): {pre_year_values}")
                        # These are the current year's values
                        # Format: Jan value(s), then YTD (for partial year)
                        # In January 2026: [1.58, 1.58] = [Jan, YTD]
                        if len(pre_year_values) >= 1:
                            # First value is Jan (current month)
                            for i in range(min(current_month, len(pre_year_values) - 1)):
                                month_num = f"{i + 1:02d}"
                                month_key = f"{current_year}-{month_num}"
                                result['monthly_returns'][month_key] = pre_year_values[i]
                                month_name = list(month_to_num.keys())[list(month_to_num.values()).index(month_num)]
                                print(f"      Mapped {current_year}-{month_name} ({month_key}): {pre_year_values[i]}")

                    if result['monthly_returns']:
                        print(f"    Successfully parsed {len(result['monthly_returns'])} monthly returns")
                    break

    # Fallback: if Strategy 1 didn't work, try simpler approach
    if not result['monthly_returns']:
        print(f"    Using fallback parsing for {username}")
        pct_values = []
        for line in lines:
            if pct_pattern.match(line):
                val_str = line.replace('%', '')
                try:
                    pct_values.append(float(val_str))
                except ValueError:
                    pass

        # Take first 12 percentage-like values
        if len(pct_values) >= 12:
            expected_order = ['Dec', 'Nov', 'Oct', 'Sep', 'Aug', 'Jul', 'Jun', 'May', 'Apr', 'Mar', 'Feb', 'Jan']
            for i, month_name in enumerate(expected_order[:len(pct_values)]):
                if i < 12:
                    month_num = month_to_num[month_name]
                    if month_name == 'Dec':
                        year = current_year if current_month == 12 else current_year - 1
                    else:
                        month_int = int(month_num)
                        year = current_year if month_int <= current_month else current_year - 1
                    month_key = f"{year}-{month_num}"
                    result['monthly_returns'][month_key] = pct_values[i]

    # Calculate YTD and annual returns
    ytd_months = [k for k in result['monthly_returns'].keys() if k.startswith(str(current_year))]
    if ytd_months:
        ytd_values = [result['monthly_returns'][k] for k in ytd_months]
        # Compound returns
        compounded = 1.0
        for r in ytd_values:
            compounded *= (1 + r / 100)
        result['gain_ytd'] = round((compounded - 1) * 100, 2)
    else:
        result['gain_ytd'] = 0.0

    # Calculate 1Y return (use all available monthly returns up to 12)
    all_monthly = list(result['monthly_returns'].values())[:12]
    if all_monthly:
        compounded = 1.0
        for r in all_monthly:
            compounded *= (1 + r / 100)
        result['gain_1y'] = round((compounded - 1) * 100, 2)
    else:
        result['gain_1y'] = result['gain_ytd']

    # Estimate 2Y (approximate)
    result['gain_2y'] = round(result['gain_1y'] * 1.5, 2)

    # Calculate profitable months percentage
    positive_months = sum(1 for v in all_monthly if v > 0)
    total_months = len(all_monthly) if all_monthly else 1
    result['profitable_months_pct'] = round((positive_months / total_months) * 100, 1)

    return result


def update_demo_data_file(scraped_data: list):
    """Update the DEMO_DATA in etoro_scraper.py with ALL scraped data."""
    scraper_path = project_root / 'AlphaMachine_core' / 'data_sources' / 'etoro_scraper.py'

    print(f"\nUpdating {scraper_path}...")

    # Read current file
    content = scraper_path.read_text(encoding='utf-8')

    # For each scraped user, update ALL their data in the file
    for data in scraped_data:
        username = data['username'].lower()
        monthly_returns = data['monthly_returns']

        if not monthly_returns:
            print(f"  Skipping {username} - no monthly returns scraped")
            continue

        # Update numeric fields (risk_score, copiers, gain_1y, etc.)
        numeric_fields = [
            ('risk_score', data.get('risk_score', 5)),
            ('copiers', data.get('copiers', 0)),
            ('gain_1y', data.get('gain_1y', 0.0)),
            ('gain_2y', data.get('gain_2y', 0.0)),
            ('gain_ytd', data.get('gain_ytd', 0.0)),
            ('win_ratio', data.get('win_ratio', 50.0)),
            ('profitable_months_pct', data.get('profitable_months_pct', 50.0)),
        ]

        for field_name, field_value in numeric_fields:
            # Pattern: 'field_name': old_value,  (within this user's block)
            # We need to find the user block first, then update the field
            pattern = rf"('{username}':\s*\{{[^}}]*'{field_name}':\s*)[\d.]+([,\s])"
            replacement = rf"\g<1>{field_value}\2"
            new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.DOTALL)
            if new_content != content:
                content = new_content

        # Sort monthly returns by date
        sorted_months = sorted(monthly_returns.items())

        # Format as Python dict
        monthly_str_parts = []
        prev_year = None
        for month_key, value in sorted_months:
            year = month_key[:4]
            if year != prev_year:
                if monthly_str_parts:
                    monthly_str_parts.append('')  # Add blank line between years
                monthly_str_parts.append(f"                # {year}")
                prev_year = year
            monthly_str_parts.append(f"                '{month_key}': {value},")

        monthly_str = '\n'.join(monthly_str_parts)

        # Find and replace the monthly_returns for this user
        # Pattern: 'username': { ... 'monthly_returns': { ... } }
        pattern = rf"('{username}':\s*\{{[^}}]*'monthly_returns':\s*\{{)[^}}]*(}})"
        replacement = rf"\1\n{monthly_str}\n            \2"

        new_content = re.sub(pattern, replacement, content, flags=re.IGNORECASE | re.DOTALL)

        if new_content != content:
            content = new_content
            print(f"  Updated ALL data for {username}")
        else:
            print(f"  Could not find pattern for {username}")

    # Write updated file
    scraper_path.write_text(content, encoding='utf-8')
    print("\nFile updated successfully!")


def main():
    """Main entry point."""
    print("=" * 60)
    print("eToro Data Updater")
    print("=" * 60)
    print(f"Scraping {len(USERNAMES_TO_SCRAPE)} profiles...")
    print()

    driver = None
    scraped_data = []

    try:
        driver = create_driver()

        for i, username in enumerate(USERNAMES_TO_SCRAPE, 1):
            print(f"[{i}/{len(USERNAMES_TO_SCRAPE)}] {username}")
            try:
                data = scrape_monthly_returns(driver, username)
                scraped_data.append(data)

                # Print summary
                mtd_key = datetime.now().strftime('%Y-%m')
                mtd = data['monthly_returns'].get(mtd_key, 'N/A')
                print(f"  Name: {data['full_name']}")
                print(f"  Risk: {data['risk_score']}, Copiers: {data['copiers']}")
                print(f"  MTD ({mtd_key}): {mtd}")
                print(f"  Monthly returns: {len(data['monthly_returns'])} months")
                print()

                time.sleep(2)  # Rate limiting

            except Exception as e:
                print(f"  ERROR: {e}")
                print()

    finally:
        if driver:
            driver.quit()

    if scraped_data:
        # Save to JSON for reference
        json_path = project_root / 'data' / 'etoro_scraped_data.json'
        json_path.parent.mkdir(exist_ok=True)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'scraped_at': datetime.now().isoformat(),
                'investors': scraped_data
            }, f, indent=2)

        print(f"Saved scraped data to {json_path}")

        # Update DEMO_DATA
        update_demo_data_file(scraped_data)

        print("\n" + "=" * 60)
        print("Done! Now commit and push the changes:")
        print("  git add -A && git commit -m 'chore: update eToro data' && git push")
        print("=" * 60)
    else:
        print("No data scraped!")


if __name__ == '__main__':
    main()
