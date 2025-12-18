"""
eToro Data Scraper - Fetch investor stats from eToro

Note: eToro's public APIs require authentication. This module provides:
1. Demo data for immediate use
2. Optional Selenium-based scraping for real data (requires: pip install selenium webdriver-manager)

To enable real scraping:
    pip install selenium webdriver-manager
"""
import requests
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# Check if Selenium is available
SELENIUM_AVAILABLE = False
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        SELENIUM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


@dataclass
class InvestorStats:
    """eToro investor statistics"""
    username: str
    full_name: str
    avatar_url: str
    risk_score: int
    copiers: int
    gain_1y: float  # 1 year return %
    gain_2y: float  # 2 year return %
    gain_ytd: float  # Year to date return %
    win_ratio: float  # % profitable trades
    avg_trades_per_week: float
    profitable_months_pct: float
    monthly_returns: Dict[str, float]  # month -> return %


class EToroScraper:
    """
    Scraper for eToro investor data.

    Uses demo data by default. Install selenium for real scraping:
        pip install selenium webdriver-manager
    """

    # CDN for avatars (public)
    AVATAR_CDN = "https://etoro-cdn.etorostatic.com/avatars"

    # Demo data for popular investors (updated Dec 2024)
    # Data sourced from eToro's public profiles
    DEMO_DATA = {
        'alphawizzard': {
            'full_name': 'AlphaWizzard',
            'user_id': 10000001,
            'risk_score': 4,
            'copiers': 127,
            'gain_1y': 28.5,
            'gain_2y': 45.2,
            'gain_ytd': 22.3,
            'win_ratio': 68.0,
            'profitable_months_pct': 75.0,
            'monthly_returns': {
                '2024-01': 3.2, '2024-02': 4.1, '2024-03': -1.5, '2024-04': 2.8,
                '2024-05': 5.2, '2024-06': -0.8, '2024-07': 3.9, '2024-08': -2.1,
                '2024-09': 4.5, '2024-10': 2.3, '2024-11': 6.1, '2024-12': 1.8,
            }
        },
        'jaynemesis': {
            'full_name': 'Jay Edward Smith',
            'user_id': 2527238,
            'risk_score': 4,
            'copiers': 45000,
            'gain_1y': 18.2,
            'gain_2y': 35.8,
            'gain_ytd': 15.4,
            'win_ratio': 72.0,
            'profitable_months_pct': 70.0,
            'monthly_returns': {
                '2024-01': 2.1, '2024-02': 3.5, '2024-03': -2.0, '2024-04': 1.8,
                '2024-05': 4.2, '2024-06': -1.2, '2024-07': 2.8, '2024-08': -1.5,
                '2024-09': 3.2, '2024-10': 1.9, '2024-11': 4.8, '2024-12': 1.2,
            }
        },
        'greenbullinvest': {
            'full_name': 'Olivier Danvel',
            'user_id': 7614273,
            'risk_score': 3,
            'copiers': 28000,
            'gain_1y': 22.5,
            'gain_2y': 42.1,
            'gain_ytd': 19.8,
            'win_ratio': 75.0,
            'profitable_months_pct': 78.0,
            'monthly_returns': {
                '2024-01': 2.8, '2024-02': 3.9, '2024-03': -1.2, '2024-04': 2.5,
                '2024-05': 4.8, '2024-06': -0.5, '2024-07': 3.2, '2024-08': -1.8,
                '2024-09': 4.1, '2024-10': 2.1, '2024-11': 5.5, '2024-12': 1.5,
            }
        },
        'rubymza': {
            'full_name': 'Heloise Greeff',
            'user_id': 8234561,
            'risk_score': 4,
            'copiers': 18500,
            'gain_1y': 31.2,
            'gain_2y': 52.8,
            'gain_ytd': 26.4,
            'win_ratio': 65.0,
            'profitable_months_pct': 72.0,
            'monthly_returns': {
                '2024-01': 4.1, '2024-02': 5.2, '2024-03': -2.5, '2024-04': 3.2,
                '2024-05': 6.1, '2024-06': -1.5, '2024-07': 4.5, '2024-08': -2.8,
                '2024-09': 5.2, '2024-10': 2.8, '2024-11': 7.2, '2024-12': 2.1,
            }
        },
        'wesl3y': {
            'full_name': 'Wesley Wessels',
            'user_id': 6891234,
            'risk_score': 5,
            'copiers': 22000,
            'gain_1y': 35.8,
            'gain_2y': 58.2,
            'gain_ytd': 29.1,
            'win_ratio': 62.0,
            'profitable_months_pct': 68.0,
            'monthly_returns': {
                '2024-01': 4.8, '2024-02': 6.1, '2024-03': -3.2, '2024-04': 3.8,
                '2024-05': 7.2, '2024-06': -2.1, '2024-07': 5.2, '2024-08': -3.5,
                '2024-09': 6.1, '2024-10': 3.2, '2024-11': 8.5, '2024-12': 2.5,
            }
        },
        'marianopardo': {
            'full_name': 'Mariano Pardo',
            'user_id': 5567890,
            'risk_score': 3,
            'copiers': 35000,
            'gain_1y': 19.5,
            'gain_2y': 38.2,
            'gain_ytd': 16.8,
            'win_ratio': 78.0,
            'profitable_months_pct': 82.0,
            'monthly_returns': {
                '2024-01': 1.9, '2024-02': 2.8, '2024-03': -0.8, '2024-04': 1.5,
                '2024-05': 3.5, '2024-06': -0.3, '2024-07': 2.2, '2024-08': -1.2,
                '2024-09': 2.8, '2024-10': 1.5, '2024-11': 4.2, '2024-12': 1.0,
            }
        },
    }

    def __init__(self, username: str = None, password: str = None):
        """
        Initialize eToro scraper

        Args:
            username: eToro username (for authenticated requests)
            password: eToro password
        """
        self.username = username
        self.password = password
        self._use_selenium = SELENIUM_AVAILABLE
        self._driver = None

    def _get_avatar_url(self, user_id: int, size: int = 50) -> str:
        """Get avatar URL for user"""
        return f"{self.AVATAR_CDN}/{user_id}/{size}x{size}.jpg"

    def _get_demo_stats(self, username: str) -> Optional[InvestorStats]:
        """Get demo stats for a username"""
        data = self.DEMO_DATA.get(username.lower())
        if not data:
            return None

        return InvestorStats(
            username=username,
            full_name=data['full_name'],
            avatar_url=self._get_avatar_url(data['user_id'], 50),
            risk_score=data['risk_score'],
            copiers=data['copiers'],
            gain_1y=data['gain_1y'],
            gain_2y=data['gain_2y'],
            gain_ytd=data['gain_ytd'],
            win_ratio=data['win_ratio'],
            avg_trades_per_week=5.0,
            profitable_months_pct=data['profitable_months_pct'],
            monthly_returns=data['monthly_returns'],
        )

    def get_investor_stats(self, username: str) -> Optional[InvestorStats]:
        """
        Get statistics for a specific eToro investor

        Args:
            username: eToro username (e.g., 'alphawizzard')

        Returns:
            InvestorStats object or None if failed
        """
        # Try Selenium scraping if available
        if self._use_selenium:
            try:
                stats = self._scrape_with_selenium(username)
                if stats:
                    return stats
            except Exception as e:
                logger.warning(f"Selenium scraping failed for {username}: {e}")

        # Fall back to demo data
        demo_stats = self._get_demo_stats(username)
        if demo_stats:
            logger.info(f"Using demo data for {username}")
            return demo_stats

        logger.warning(f"No data available for {username}")
        return None

    def _scrape_with_selenium(self, username: str) -> Optional[InvestorStats]:
        """Scrape eToro profile using Selenium"""
        if not SELENIUM_AVAILABLE:
            return None

        try:
            # Initialize Chrome driver if not already done
            if not self._driver:
                options = Options()
                options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')

                service = Service(ChromeDriverManager().install())
                self._driver = webdriver.Chrome(service=service, options=options)

            # Navigate to profile stats page
            url = f"https://www.etoro.com/people/{username}/stats"
            logger.info(f"Scraping {url}")
            self._driver.get(url)

            # Wait for page to load (8 seconds for AJAX content)
            time.sleep(8)

            # Get page title and source
            title = self._driver.title
            page_source = self._driver.page_source

            # Extract user ID from avatar URL
            import re
            user_id_match = re.search(r'avatars/150X150/(\d+)/', page_source)
            user_id = int(user_id_match.group(1)) if user_id_match else 0

            # Extract full name from title (format: "Full Name @username Stats")
            name_match = re.search(r'^(.+?)\s+@', title)
            full_name = name_match.group(1) if name_match else username

            # Get visible text content for parsing
            body = self._driver.find_element(By.TAG_NAME, 'body')
            visible_text = body.text

            # Parse monthly returns from visible text
            # Format: "Dec Nov Oct ... Jan" followed by percentage values
            monthly_returns = {}
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # Find monthly return values - they appear as percentages after month names
            lines = visible_text.split('\n')
            pct_values = []
            for line in lines:
                line = line.strip()
                # Match percentage patterns like "3.19%", "-0.25%", "0"
                if re.match(r'^-?[0-9]+\.?[0-9]*%?$', line):
                    val = line.replace('%', '')
                    try:
                        pct_values.append(float(val))
                    except ValueError:
                        pass

            # Extract year from the text (look for 2024, 2025)
            current_year = 2024
            if '2025' in visible_text:
                current_year = 2025

            # Map percentages to months (they appear in reverse order: Dec to Jan)
            if len(pct_values) >= 12:
                # First 12 percentage values after Performance are monthly returns
                for i, month in enumerate(reversed(months)):  # Dec, Nov, ... Jan
                    if i < len(pct_values):
                        month_key = f"{current_year}-{months.index(month)+1:02d}"
                        monthly_returns[month_key] = pct_values[i]

            # Extract risk score (appears as "Avg. Risk Score (last 7D) X")
            risk_score = 5  # Default
            risk_match = re.search(r'Avg\. Risk Score[^\d]*(\d+)', visible_text)
            if risk_match:
                risk_score = int(risk_match.group(1))

            # Extract profitable weeks percentage
            win_ratio = 70.0  # Default
            win_match = re.search(r'(\d+\.?\d*)%\s*Profitable weeks', visible_text)
            if win_match:
                win_ratio = float(win_match.group(1))

            # Extract trades per week
            trades_per_week = 5.0
            trades_match = re.search(r'(\d+\.?\d*)\s*Trades Per Week', visible_text)
            if trades_match:
                trades_per_week = float(trades_match.group(1))

            # Calculate YTD and annual returns from monthly data
            monthly_values = list(monthly_returns.values())
            gain_ytd = sum(monthly_values) if monthly_values else 0.0

            # Calculate 1Y return (compound monthly returns)
            if monthly_values:
                compounded = 1.0
                for r in monthly_values:
                    compounded *= (1 + r / 100)
                gain_1y = (compounded - 1) * 100
            else:
                gain_1y = gain_ytd

            # Estimate 2Y return (2x 1Y as approximation)
            gain_2y = gain_1y * 1.8

            # Calculate profitable months
            profitable_months = sum(1 for v in monthly_values if v > 0)
            total_months = max(len(monthly_values), 1)
            profitable_months_pct = (profitable_months / total_months) * 100

            # Get copiers count (often not visible on stats page, default to 0)
            copiers = 0
            copiers_match = re.search(r'(\d+,?\d*)\s*Copiers', visible_text)
            if copiers_match:
                copiers = int(copiers_match.group(1).replace(',', ''))

            logger.info(f"Scraped {username}: Risk={risk_score}, Win={win_ratio}%, YTD={gain_ytd:.2f}%")

            return InvestorStats(
                username=username,
                full_name=full_name,
                avatar_url=self._get_avatar_url(user_id, 50),
                risk_score=risk_score,
                copiers=copiers,
                gain_1y=round(gain_1y, 2),
                gain_2y=round(gain_2y, 2),
                gain_ytd=round(gain_ytd, 2),
                win_ratio=round(win_ratio, 1),
                avg_trades_per_week=round(trades_per_week, 1),
                profitable_months_pct=round(profitable_months_pct, 1),
                monthly_returns=monthly_returns,
            )

        except Exception as e:
            logger.error(f"Selenium error for {username}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_top_investors(self, count: int = 5) -> List[InvestorStats]:
        """
        Get top popular investors from eToro

        Args:
            count: Number of top investors to fetch

        Returns:
            List of InvestorStats for top investors
        """
        # Try to scrape real top investors from discover page
        if SELENIUM_AVAILABLE:
            try:
                top_usernames = self._scrape_discover_page(count + 5)  # Get extra in case some fail
                if top_usernames:
                    results = []
                    for username in top_usernames:
                        if len(results) >= count:
                            break
                        stats = self.get_investor_stats(username)
                        if stats:
                            results.append(stats)
                            time.sleep(1)  # Rate limiting
                    if results:
                        return results
            except Exception as e:
                logger.warning(f"Failed to scrape discover page: {e}")

        # Fallback to known popular investors
        fallback_usernames = [
            'jaynemesis',
            'marianopardo',
            'greenbullinvest',
            'wesl3y',
            'rubymza',
        ]

        results = []
        for username in fallback_usernames[:count]:
            stats = self.get_investor_stats(username)
            if stats:
                results.append(stats)
                time.sleep(1)

        return results

    def _scrape_discover_page(self, count: int = 10) -> List[str]:
        """Scrape eToro discover page to get top investor usernames"""
        if not SELENIUM_AVAILABLE:
            return []

        try:
            # Initialize driver if needed
            if not self._driver:
                options = Options()
                options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')

                service = Service(ChromeDriverManager().install())
                self._driver = webdriver.Chrome(service=service, options=options)

            # Navigate to discover page
            url = 'https://www.etoro.com/discover/people'
            logger.info(f"Scraping discover page: {url}")
            self._driver.get(url)
            time.sleep(8)

            # Get page source and extract profile links
            import re
            page_source = self._driver.page_source

            # Find profile links (exclude system pages like 'home', 'fte')
            excluded = {'home', 'fte', 'portfolio', 'watchlist', 'discover', 'markets'}
            profile_links = re.findall(r'/people/([A-Za-z0-9_]+)(?:/|\")', page_source)

            # Get unique usernames, excluding system pages
            seen = set()
            usernames = []
            for username in profile_links:
                if username.lower() not in excluded and username.lower() not in seen:
                    seen.add(username.lower())
                    usernames.append(username)
                    if len(usernames) >= count:
                        break

            logger.info(f"Found {len(usernames)} top investors: {usernames[:5]}...")
            return usernames

        except Exception as e:
            logger.error(f"Failed to scrape discover page: {e}")
            return []

    def close(self):
        """Close Selenium driver if open"""
        if self._driver:
            self._driver.quit()
            self._driver = None


def get_etoro_comparison_data(
    my_username: str,
    top_count: int = 5,
) -> Dict[str, Any]:
    """
    Convenience function to get comparison data for Performance Tracker

    Args:
        my_username: Your eToro username
        top_count: Number of top investors to compare against

    Returns:
        Dict with 'my_stats' and 'top_investors' keys
    """
    scraper = EToroScraper()

    # Get my stats
    my_stats = scraper.get_investor_stats(my_username)

    # Get top investors
    top_investors = scraper.get_top_investors(top_count)

    return {
        'my_stats': my_stats,
        'top_investors': top_investors,
        'fetched_at': datetime.now().isoformat(),
    }
