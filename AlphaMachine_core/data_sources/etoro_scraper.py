"""
eToro Data - Fetch investor stats from Supabase

Data is scraped daily by GitHub Actions and stored in Supabase.
This module loads the cached data for display in the Performance Tracker.
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


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
    eToro investor data loader.

    Loads data from Supabase (updated daily by GitHub Actions).
    Falls back to demo data if database is unavailable.
    """

    # CDN for avatars (public)
    AVATAR_CDN = "https://etoro-cdn.etorostatic.com/avatars"

    # Known top investors (for ordering)
    TOP_INVESTORS = [
        'thomaspj',         # #1 Thomas Parry Jones
        'jeppekirkbonde',   # #2 Jeppe Kirk Bonde
        'triangulacapital', # #3 Pietari Laurila
        'cphequities',      # #4 Blue Screen Media ApS
        'fundmanagerzech',  # #5 Zechariah Bin Zheng
    ]

    # Cache for live data (loaded once per session)
    _live_data_cache: Optional[List] = None
    _live_data_loaded_at: Optional[datetime] = None

    def __init__(self):
        """Initialize eToro data loader."""
        pass

    @classmethod
    def _load_live_data(cls) -> List:
        """Load live data from Supabase (cached for 5 minutes)."""
        now = datetime.now()

        # Check if cache is valid (less than 5 minutes old)
        if (cls._live_data_cache is not None and
            cls._live_data_loaded_at is not None and
            (now - cls._live_data_loaded_at).total_seconds() < 300):
            return cls._live_data_cache

        # Try to load from Supabase
        try:
            from AlphaMachine_core.db import get_session
            from AlphaMachine_core.models import EToroStats
            from sqlmodel import select

            with get_session() as session:
                # Get the most recent scraped date
                latest_date_query = select(EToroStats.scraped_date).order_by(
                    EToroStats.scraped_date.desc()
                ).limit(1)
                result = session.exec(latest_date_query).first()

                if not result:
                    logger.warning("No eToro stats in database")
                    return []

                latest_date = result

                # Get all investors for that date
                query = select(EToroStats).where(EToroStats.scraped_date == latest_date)
                stats = session.exec(query).all()

                # Convert to list of dicts
                live_data = []
                for stat in stats:
                    live_data.append({
                        'username': stat.username,
                        'full_name': stat.full_name,
                        'user_id': stat.user_id,
                        'risk_score': stat.risk_score,
                        'copiers': stat.copiers,
                        'gain_1y': stat.gain_1y,
                        'gain_2y': stat.gain_2y,
                        'gain_ytd': stat.gain_ytd,
                        'win_ratio': stat.win_ratio,
                        'profitable_months_pct': stat.profitable_months_pct,
                        'monthly_returns': stat.monthly_returns or {},
                    })

                cls._live_data_cache = live_data
                cls._live_data_loaded_at = now
                logger.info(f"Loaded eToro data from Supabase ({len(live_data)} investors, date: {latest_date})")
                return cls._live_data_cache

        except Exception as e:
            logger.warning(f"Failed to load data from Supabase: {e}")

        return []

    def _get_avatar_url(self, user_id: int, size: int = 50) -> str:
        """Get avatar URL for user via image proxy (bypasses CORS).

        eToro CDN format: avatars/{SIZE}X{SIZE}/{user_id}/3.jpg
        We use images.weserv.nl as proxy to bypass CDN restrictions.
        """
        if not user_id or user_id == 0:
            return ""
        # Use weserv.nl image proxy - pass URL without https://
        return f"https://images.weserv.nl/?url=etoro-cdn.etorostatic.com/avatars/{size}X{size}/{user_id}/3.jpg&w={size}&h={size}&fit=cover&default=1"

    def _get_live_stats(self, username: str) -> Optional[InvestorStats]:
        """Get stats from Supabase data."""
        live_data = self._load_live_data()

        for inv in live_data:
            if inv.get('username', '').lower() == username.lower():
                return InvestorStats(
                    username=username,
                    full_name=inv.get('full_name', username),
                    avatar_url=self._get_avatar_url(inv.get('user_id', 0), 50),
                    risk_score=inv.get('risk_score', 5),
                    copiers=inv.get('copiers', 0),
                    gain_1y=inv.get('gain_1y', 0.0),
                    gain_2y=inv.get('gain_2y', 0.0),
                    gain_ytd=inv.get('gain_ytd', 0.0),
                    win_ratio=inv.get('win_ratio', 50.0),
                    avg_trades_per_week=5.0,
                    profitable_months_pct=inv.get('profitable_months_pct', 50.0),
                    monthly_returns=inv.get('monthly_returns', {}),
                )
        return None

    def get_investor_stats(self, username: str) -> Optional[InvestorStats]:
        """
        Get statistics for a specific eToro investor.

        Args:
            username: eToro username (e.g., 'alphawizzard')

        Returns:
            InvestorStats object or None if not found
        """
        stats = self._get_live_stats(username)
        if stats:
            return stats

        logger.warning(f"No data available for {username}")
        return None

    def get_top_investors(self, count: int = 5) -> List[InvestorStats]:
        """
        Get top popular investors from eToro.

        Args:
            count: Number of top investors to fetch

        Returns:
            List of InvestorStats for top investors
        """
        # Load data from Supabase
        live_data = self._load_live_data()
        if not live_data:
            return []

        results = []
        for username in self.TOP_INVESTORS[:count]:
            stats = self._get_live_stats(username)
            if stats:
                results.append(stats)

        logger.info(f"Loaded {len(results)} top investors from Supabase")
        return results


def get_etoro_comparison_data(
    my_username: str,
    top_count: int = 5,
    custom_usernames: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get comparison data for Performance Tracker.

    Args:
        my_username: Your eToro username
        top_count: Number of top investors to compare against
        custom_usernames: Optional list of specific usernames to compare against

    Returns:
        Dict with 'my_stats' and 'top_investors' keys
    """
    scraper = EToroScraper()

    # Get my stats
    my_stats = scraper.get_investor_stats(my_username)

    # Get comparison investors
    if custom_usernames:
        top_investors = []
        for username in custom_usernames:
            username = username.strip()
            if username and username.lower() != my_username.lower():
                stats = scraper.get_investor_stats(username)
                if stats:
                    top_investors.append(stats)
    else:
        top_investors = scraper.get_top_investors(top_count)

    return {
        'my_stats': my_stats,
        'top_investors': top_investors,
        'fetched_at': datetime.now().isoformat(),
    }
