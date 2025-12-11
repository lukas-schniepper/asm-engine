#!/usr/bin/env python3
"""
Data Quality Monitor - Daily Health Check for Portfolio Tracking.

This script should be run daily (e.g., 7 AM) to detect data issues
BEFORE users see them in dashboards.

What a senior quant dev would want:
1. Overnight anomaly detection
2. Cross-portfolio consistency checks
3. Benchmark divergence alerts
4. Audit trail review
5. Email/Slack alerts for issues

Usage:
    python scripts/data_quality_monitor.py

    # Check specific portfolio
    python scripts/data_quality_monitor.py --portfolio "QQQ"

    # Check last N days
    python scripts/data_quality_monitor.py --days 14

    # Output to JSON for integration with monitoring systems
    python scripts/data_quality_monitor.py --json

    # Send email alerts on failure
    python scripts/data_quality_monitor.py --email your@gmail.com

Environment Variables for Email:
    SMTP_SERVER: SMTP server (default: smtp.gmail.com)
    SMTP_PORT: SMTP port (default: 587)
    SMTP_USER: Email sender address
    SMTP_PASSWORD: Email password or app-specific password
    ALERT_EMAIL: Default recipient email
"""

import os
import sys
import json
import logging
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load secrets
def _load_secrets():
    secrets_path = project_root / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        for key, value in secrets.items():
            if key not in os.environ and isinstance(value, str):
                os.environ[key] = value

_load_secrets()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_nav_anomalies(
    portfolios: List,
    tracker,
    days_back: int = 7,
) -> List[Dict]:
    """
    Check for NAV anomalies across portfolios.

    Detects:
    - Extreme daily returns (>10%)
    - NAV near 100 (potential reset)
    - Missing data points
    - Stale data (same NAV for multiple days)
    """
    from AlphaMachine_core.tracking import Variants

    issues = []
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    for portfolio in portfolios:
        nav_df = tracker.get_nav_series(
            portfolio.id, Variants.RAW, start_date, end_date
        )

        if nav_df.empty:
            issues.append({
                "portfolio": portfolio.name,
                "type": "NO_DATA",
                "severity": "warning",
                "message": f"No NAV data for last {days_back} days",
            })
            continue

        # Check for extreme daily returns
        daily_returns = nav_df["daily_return"] * 100
        extreme_returns = daily_returns[abs(daily_returns) > 10]

        for idx, ret in extreme_returns.items():
            issues.append({
                "portfolio": portfolio.name,
                "date": idx.strftime("%Y-%m-%d"),
                "type": "EXTREME_RETURN",
                "severity": "error" if abs(ret) > 20 else "warning",
                "message": f"Daily return of {ret:+.2f}%",
                "value": float(ret),
            })

        # Check for NAV near 100 (potential reset) when it shouldn't be
        latest_nav = nav_df["nav"].iloc[-1]
        if portfolio.start_date and portfolio.start_date < end_date - timedelta(days=30):
            # Portfolio is >30 days old, NAV near 100 is suspicious
            if 95 < latest_nav < 105:
                issues.append({
                    "portfolio": portfolio.name,
                    "date": nav_df.index[-1].strftime("%Y-%m-%d"),
                    "type": "SUSPICIOUS_NAV",
                    "severity": "warning",
                    "message": f"NAV is {latest_nav:.2f} (near initial 100) for mature portfolio",
                    "value": float(latest_nav),
                })

        # Check for stale data (same NAV for 3+ consecutive days)
        nav_values = nav_df["nav"].values
        if len(nav_values) >= 3:
            for i in range(2, len(nav_values)):
                if nav_values[i] == nav_values[i-1] == nav_values[i-2]:
                    issues.append({
                        "portfolio": portfolio.name,
                        "date": nav_df.index[i].strftime("%Y-%m-%d"),
                        "type": "STALE_DATA",
                        "severity": "warning",
                        "message": f"NAV unchanged for 3+ days at {nav_values[i]:.2f}",
                        "value": float(nav_values[i]),
                    })
                    break  # Only report once

        # Check for missing trading days
        expected_days = len(nav_df)  # Simplified check
        if expected_days < days_back * 0.6:  # Less than 60% of days have data
            issues.append({
                "portfolio": portfolio.name,
                "type": "SPARSE_DATA",
                "severity": "warning",
                "message": f"Only {expected_days} data points in {days_back} days",
            })

    return issues


def check_cross_portfolio_consistency(
    portfolios: List,
    tracker,
) -> List[Dict]:
    """
    Check for consistency issues across portfolios.

    Detects:
    - Portfolios with very different returns on same day
    - Missing updates for some portfolios
    """
    from AlphaMachine_core.tracking import Variants

    issues = []
    yesterday = date.today() - timedelta(days=1)

    # Collect yesterday's returns
    returns = {}
    for portfolio in portfolios:
        nav_df = tracker.get_nav_series(
            portfolio.id, Variants.RAW, yesterday, yesterday
        )
        if not nav_df.empty:
            returns[portfolio.name] = nav_df["daily_return"].iloc[0] * 100

    if len(returns) < 2:
        return issues

    # Check for outliers (>3 std from mean)
    import numpy as np
    values = list(returns.values())
    mean_ret = np.mean(values)
    std_ret = np.std(values)

    if std_ret > 0:
        for name, ret in returns.items():
            z_score = (ret - mean_ret) / std_ret
            if abs(z_score) > 3:
                issues.append({
                    "portfolio": name,
                    "date": yesterday.strftime("%Y-%m-%d"),
                    "type": "OUTLIER_RETURN",
                    "severity": "warning",
                    "message": f"Return {ret:+.2f}% is {z_score:.1f} std devs from mean ({mean_ret:.2f}%)",
                    "value": float(ret),
                })

    return issues


def check_audit_log_issues(days_back: int = 7) -> List[Dict]:
    """Check audit log for blocked updates or suspicious patterns."""
    from AlphaMachine_core.tracking.data_quality import get_audit_log

    issues = []
    audit_log = get_audit_log()

    # Get suspicious changes
    suspicious = audit_log.get_suspicious_changes(
        threshold_pct=10.0,
        days_back=days_back,
    )

    if not suspicious.empty:
        for _, row in suspicious.iterrows():
            issues.append({
                "portfolio": row.get("portfolio_name", "Unknown"),
                "date": str(row.get("trade_date")),
                "type": "AUDIT_SUSPICIOUS",
                "severity": "warning",
                "message": f"NAV changed {row.get('change_pct', 0):.1f}% (source: {row.get('source', 'unknown')})",
                "value": float(row.get("change_pct", 0)),
            })

    return issues


def generate_report(issues: List[Dict]) -> str:
    """Generate human-readable report."""
    if not issues:
        return "✅ All data quality checks passed. No issues detected."

    lines = [
        "=" * 60,
        "DATA QUALITY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]

    # Group by severity
    critical = [i for i in issues if i.get("severity") == "critical"]
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    if critical:
        lines.append("🚨 CRITICAL ISSUES:")
        for issue in critical:
            lines.append(f"  - [{issue['portfolio']}] {issue['message']}")
        lines.append("")

    if errors:
        lines.append("❌ ERRORS:")
        for issue in errors:
            date_str = f" ({issue['date']})" if 'date' in issue else ""
            lines.append(f"  - [{issue['portfolio']}]{date_str} {issue['message']}")
        lines.append("")

    if warnings:
        lines.append("⚠️  WARNINGS:")
        for issue in warnings:
            date_str = f" ({issue['date']})" if 'date' in issue else ""
            lines.append(f"  - [{issue['portfolio']}]{date_str} {issue['message']}")
        lines.append("")

    lines.extend([
        "=" * 60,
        f"Summary: {len(critical)} critical, {len(errors)} errors, {len(warnings)} warnings",
        "=" * 60,
    ])

    return "\n".join(lines)


def send_email_alert(
    recipient: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None,
) -> bool:
    """
    Send email alert using SMTP.

    Requires environment variables:
    - SMTP_SERVER (default: smtp.gmail.com)
    - SMTP_PORT (default: 587)
    - SMTP_USER (sender email)
    - SMTP_PASSWORD (app-specific password for Gmail)

    Returns:
        True if email sent successfully, False otherwise
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        logger.warning(
            "Email not configured. Set SMTP_USER and SMTP_PASSWORD environment variables."
        )
        return False

    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = recipient

        # Plain text version
        msg.attach(MIMEText(body, "plain"))

        # HTML version (if provided)
        if html_body:
            msg.attach(MIMEText(html_body, "html"))

        # Send
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, recipient, msg.as_string())

        logger.info(f"Alert email sent to {recipient}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False


def generate_html_report(issues: List[Dict], result: Dict) -> str:
    """Generate HTML version of the report for email."""
    critical = [i for i in issues if i.get("severity") == "critical"]
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    status_color = "#dc3545" if critical or errors else "#ffc107" if warnings else "#28a745"
    status_text = "CRITICAL" if critical else "ERRORS" if errors else "WARNINGS" if warnings else "PASS"

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: {status_color}; color: white; padding: 15px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .issue {{ padding: 10px; margin: 5px 0; border-left: 4px solid; }}
            .critical {{ border-color: #dc3545; background-color: #f8d7da; }}
            .error {{ border-color: #fd7e14; background-color: #fff3cd; }}
            .warning {{ border-color: #ffc107; background-color: #fffce8; }}
            .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4a5568; color: white; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>🔍 ASM Data Quality Report - {status_text}</h2>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>

        <div class="summary">
            <strong>Summary:</strong> {len(critical)} critical, {len(errors)} errors, {len(warnings)} warnings
            <br>
            <strong>Portfolios Checked:</strong> {result.get('portfolios_checked', 0)}
            <br>
            <strong>Days Analyzed:</strong> {result.get('days_checked', 7)}
        </div>
    """

    if critical:
        html += """
        <div class="section">
            <h3>🚨 Critical Issues</h3>
        """
        for issue in critical:
            html += f"""
            <div class="issue critical">
                <strong>{issue['portfolio']}</strong>
                {f" ({issue['date']})" if 'date' in issue else ""}
                <br>{issue['message']}
            </div>
            """
        html += "</div>"

    if errors:
        html += """
        <div class="section">
            <h3>❌ Errors</h3>
        """
        for issue in errors:
            html += f"""
            <div class="issue error">
                <strong>{issue['portfolio']}</strong>
                {f" ({issue['date']})" if 'date' in issue else ""}
                <br>{issue['message']}
            </div>
            """
        html += "</div>"

    if warnings:
        html += """
        <div class="section">
            <h3>⚠️ Warnings</h3>
        """
        for issue in warnings:
            html += f"""
            <div class="issue warning">
                <strong>{issue['portfolio']}</strong>
                {f" ({issue['date']})" if 'date' in issue else ""}
                <br>{issue['message']}
            </div>
            """
        html += "</div>"

    if not issues:
        html += """
        <div class="section">
            <h3>✅ All Checks Passed</h3>
            <p>No data quality issues detected.</p>
        </div>
        """

    html += """
        <hr>
        <p style="color: #666; font-size: 12px;">
            This is an automated alert from ASM Portfolio Tracking System.
            <br>
            To investigate, check the Performance Tracker → Scraper View.
        </p>
    </body>
    </html>
    """

    return html


def run_monitor(
    portfolio_name: Optional[str] = None,
    days_back: int = 7,
    output_json: bool = False,
    email_recipient: Optional[str] = None,
) -> Dict:
    """
    Run comprehensive data quality monitoring.

    Args:
        portfolio_name: Specific portfolio to check (or all if None)
        days_back: Number of days to check
        email_recipient: Email address to send alerts (on failure)
        output_json: If True, return JSON-serializable dict

    Returns:
        Dictionary with monitoring results
    """
    from AlphaMachine_core.tracking import get_tracker

    logger.info("Starting data quality monitoring...")

    tracker = get_tracker()

    # Get portfolios
    if portfolio_name:
        portfolio = tracker.get_portfolio_by_name(portfolio_name)
        portfolios = [portfolio] if portfolio else []
        if not portfolios:
            logger.error(f"Portfolio '{portfolio_name}' not found")
            return {"error": "Portfolio not found"}
    else:
        portfolios = tracker.list_portfolios(active_only=True)

    logger.info(f"Checking {len(portfolios)} portfolios over last {days_back} days")

    all_issues = []

    # Run checks
    logger.info("Checking NAV anomalies...")
    all_issues.extend(check_nav_anomalies(portfolios, tracker, days_back))

    logger.info("Checking cross-portfolio consistency...")
    all_issues.extend(check_cross_portfolio_consistency(portfolios, tracker))

    logger.info("Checking audit log...")
    all_issues.extend(check_audit_log_issues(days_back))

    # Generate report
    report = generate_report(all_issues)

    if not output_json:
        print("\n" + report)

    # Summary
    result = {
        "timestamp": datetime.now().isoformat(),
        "portfolios_checked": len(portfolios),
        "days_checked": days_back,
        "total_issues": len(all_issues),
        "critical_count": len([i for i in all_issues if i.get("severity") == "critical"]),
        "error_count": len([i for i in all_issues if i.get("severity") == "error"]),
        "warning_count": len([i for i in all_issues if i.get("severity") == "warning"]),
        "issues": all_issues,
        "status": "PASS" if len(all_issues) == 0 else "FAIL",
    }

    if output_json:
        print(json.dumps(result, indent=2))

    # Send email alert if issues found and email configured
    if email_recipient and all_issues:
        has_critical_or_error = (
            result["critical_count"] > 0 or result["error_count"] > 0
        )

        # Determine subject based on severity
        if result["critical_count"] > 0:
            subject = f"🚨 [CRITICAL] ASM Data Quality Alert - {result['critical_count']} critical issues"
        elif result["error_count"] > 0:
            subject = f"❌ [ERROR] ASM Data Quality Alert - {result['error_count']} errors detected"
        else:
            subject = f"⚠️ [WARNING] ASM Data Quality Alert - {result['warning_count']} warnings"

        # Generate email content
        html_body = generate_html_report(all_issues, result)

        # Send email
        email_sent = send_email_alert(
            recipient=email_recipient,
            subject=subject,
            body=report,
            html_body=html_body,
        )

        result["email_sent"] = email_sent
        result["email_recipient"] = email_recipient

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Data quality monitoring for portfolio tracking"
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        help="Specific portfolio to check (default: all)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to check (default: 7)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--email",
        type=str,
        help="Email address to send alerts on failure",
    )
    parser.add_argument(
        "--always-email",
        action="store_true",
        help="Send email even if no issues found (daily digest)",
    )

    args = parser.parse_args()

    # Get email from args or environment
    email_recipient = args.email or os.getenv("ALERT_EMAIL")

    result = run_monitor(
        portfolio_name=args.portfolio,
        days_back=args.days,
        output_json=args.json,
        email_recipient=email_recipient,
    )

    # Send daily digest even if no issues (if requested)
    if args.always_email and email_recipient and not result.get("email_sent"):
        send_email_alert(
            recipient=email_recipient,
            subject="✅ ASM Data Quality - All Checks Passed",
            body=f"Daily data quality check completed.\n\n"
                 f"Portfolios checked: {result.get('portfolios_checked', 0)}\n"
                 f"Days analyzed: {result.get('days_checked', 7)}\n"
                 f"Status: PASS - No issues detected.\n\n"
                 f"Timestamp: {result.get('timestamp')}",
            html_body=generate_html_report([], result),
        )

    # Exit code based on results
    if result.get("error"):
        sys.exit(1)
    if result.get("critical_count", 0) > 0 or result.get("error_count", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
