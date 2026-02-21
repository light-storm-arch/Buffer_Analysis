"""
scraper.py — Innovator ETF Web Scraping Module

This module attempts to scrape current buffer ETF data from Innovator ETFs
(innovatoretfs.com). Innovator publishes daily updated cap and buffer levels
for each of their defined outcome ETFs.

Data we attempt to retrieve for a given ticker:
  - Remaining cap (max upside from current price)
  - Remaining buffer (remaining downside protection)
  - Outcome period start date and reset (end) date
  - Current NAV
  - Starting cap and starting buffer at period inception

Strategy:
  1. First, try the Innovator ETF individual fund page with browser-like headers.
  2. If that fails, try the "define" tool page which shows real-time outcome data.
  3. If all scraping attempts fail, return None so the UI can fall back to manual entry.

Note: Innovator's site uses ASP.NET and renders much of its data via JavaScript,
so simple HTTP scraping may not capture dynamically-loaded content. A production
version might use Selenium or Playwright for full browser rendering. This module
uses requests + BeautifulSoup as a lightweight first attempt.
"""

import re
import logging
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional

import requests
from bs4 import BeautifulSoup

# Configure logging so users can debug scraping issues
logger = logging.getLogger(__name__)

# Common Innovator Buffer ETF tickers organized by month and type.
# This helps validate user input and provide autocomplete suggestions.
INNOVATOR_TICKERS = {
    # U.S. Equity Buffer ETFs (9% buffer) — monthly series
    "BJAN": "Buffer Jan", "BFEB": "Buffer Feb", "BMAR": "Buffer Mar",
    "BAPR": "Buffer Apr", "BMAY": "Buffer May", "BJUN": "Buffer Jun",
    "BJUL": "Buffer Jul", "BAUG": "Buffer Aug", "BSEP": "Buffer Sep",
    "BOCT": "Buffer Oct", "BNOV": "Buffer Nov", "BDEC": "Buffer Dec",
    # U.S. Equity Power Buffer ETFs (15% buffer) — monthly series
    "PJAN": "Power Buffer Jan", "PFEB": "Power Buffer Feb", "PMAR": "Power Buffer Mar",
    "PAPR": "Power Buffer Apr", "PMAY": "Power Buffer May", "PJUN": "Power Buffer Jun",
    "PJUL": "Power Buffer Jul", "PAUG": "Power Buffer Aug", "PSEP": "Power Buffer Sep",
    "POCT": "Power Buffer Oct", "PNOV": "Power Buffer Nov", "PDEC": "Power Buffer Dec",
    # U.S. Equity Ultra Buffer ETFs (30% buffer, 5-35% range) — monthly series
    "UJAN": "Ultra Buffer Jan", "UFEB": "Ultra Buffer Feb", "UMAR": "Ultra Buffer Mar",
    "UAPR": "Ultra Buffer Apr", "UMAY": "Ultra Buffer May", "UJUN": "Ultra Buffer Jun",
    "UJUL": "Ultra Buffer Jul", "UAUG": "Ultra Buffer Aug", "USEP": "Ultra Buffer Sep",
    "UOCT": "Ultra Buffer Oct", "UNOV": "Ultra Buffer Nov", "UDEC": "Ultra Buffer Dec",
}


@dataclass
class ETFData:
    """
    Container for scraped (or manually entered) buffer ETF data.

    All percentage fields are stored as decimals (e.g., 0.15 for 15%).
    Dates are stored as Python date objects.
    """
    ticker: str = ""

    # Outcome period dates
    outcome_period_start: Optional[date] = None
    reset_date: Optional[date] = None

    # Cap levels (as decimals, e.g., 0.15 = 15%)
    starting_cap: Optional[float] = None
    remaining_cap: Optional[float] = None

    # Buffer levels (as decimals)
    starting_buffer: Optional[float] = None
    remaining_buffer: Optional[float] = None

    # NAV values
    starting_nav: Optional[float] = None
    current_nav: Optional[float] = None

    # Metadata
    scrape_successful: bool = False
    scrape_message: str = ""
    scrape_timestamp: Optional[datetime] = None

    # Additional fields that may be available
    fund_name: str = ""
    underlying_index: str = "S&P 500"
    downside_before_buffer: Optional[float] = None  # For ultra buffers (e.g., -5%)

    def days_remaining(self) -> Optional[int]:
        """Calculate trading days remaining until reset date."""
        if self.reset_date is None:
            return None
        delta = self.reset_date - date.today()
        return max(0, delta.days)

    def current_return(self) -> Optional[float]:
        """Calculate the current return since outcome period start."""
        if self.starting_nav and self.current_nav and self.starting_nav > 0:
            return (self.current_nav / self.starting_nav) - 1.0
        return None


def _get_session() -> requests.Session:
    """
    Create a requests session with browser-like headers.

    Innovator's site blocks requests that don't look like they come from
    a real browser. We set common headers to improve our chances, though
    this won't help if the data is loaded via JavaScript after page load.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    return session


def _parse_percentage(text: str) -> Optional[float]:
    """
    Parse a percentage string into a decimal float.

    Examples:
        "15.50%"  -> 0.155
        "-9.00%"  -> -0.09
        "15.50"   -> 0.155   (assumes percentage even without % sign)
        "N/A"     -> None

    Args:
        text: Raw text that may contain a percentage value.

    Returns:
        The percentage as a decimal, or None if parsing fails.
    """
    if not text:
        return None
    # Remove whitespace and common non-numeric characters
    cleaned = text.strip().replace(",", "").replace("%", "")
    try:
        return float(cleaned) / 100.0
    except ValueError:
        return None


def _parse_date(text: str) -> Optional[date]:
    """
    Parse a date string in common formats used by Innovator ETFs.

    Tries multiple formats since the site isn't perfectly consistent:
      - "01/01/2025" (MM/DD/YYYY)
      - "January 1, 2025"
      - "Jan 1, 2025"
      - "2025-01-01" (ISO format)

    Args:
        text: Raw date string from the website.

    Returns:
        A date object, or None if parsing fails.
    """
    if not text:
        return None
    cleaned = text.strip()
    formats = [
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%Y-%m-%d",
        "%m/%d/%y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _parse_nav(text: str) -> Optional[float]:
    """
    Parse a NAV (dollar) value from text.

    Examples:
        "$32.45" -> 32.45
        "32.45"  -> 32.45
        "N/A"    -> None

    Args:
        text: Raw text containing a dollar value.

    Returns:
        The NAV as a float, or None if parsing fails.
    """
    if not text:
        return None
    cleaned = text.strip().replace("$", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def scrape_etf_page(ticker: str) -> ETFData:
    """
    Attempt to scrape buffer ETF data from the Innovator ETF fund page.

    This is our primary scraping strategy. The individual fund page at
    innovatoretfs.com/etf/default.aspx?ticker=XXXX contains key data
    points including cap, buffer, outcome period dates, and NAV.

    Args:
        ticker: The ETF ticker symbol (e.g., "BJUL", "PAPR").

    Returns:
        An ETFData instance. Check scrape_successful to determine if
        data was actually retrieved, or if manual entry is needed.
    """
    result = ETFData(ticker=ticker.upper())
    url = f"https://www.innovatoretfs.com/etf/default.aspx?ticker={ticker.upper()}"

    try:
        session = _get_session()
        response = session.get(url, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Try to extract the fund name from the page title or header
        title_tag = soup.find("title")
        if title_tag:
            result.fund_name = title_tag.get_text(strip=True)

        # Look for data in common patterns used by Innovator's ASP.NET pages.
        # The site uses various div/span elements with specific IDs or classes.
        page_text = soup.get_text(separator="\n")

        # --- Attempt to extract remaining cap ---
        # Look for patterns like "Remaining Cap: 12.50%" or "Remaining Cap 12.50%"
        cap_match = re.search(
            r"(?:remaining\s+cap|cap\s+remaining)[:\s]*([+-]?\d+\.?\d*)\s*%",
            page_text, re.IGNORECASE
        )
        if cap_match:
            result.remaining_cap = float(cap_match.group(1)) / 100.0

        # --- Attempt to extract remaining buffer ---
        buffer_match = re.search(
            r"(?:remaining\s+(?:buffer|protection)|buffer\s+remaining)[:\s]*([+-]?\d+\.?\d*)\s*%",
            page_text, re.IGNORECASE
        )
        if buffer_match:
            result.remaining_buffer = float(buffer_match.group(1)) / 100.0

        # --- Attempt to extract starting cap ---
        starting_cap_match = re.search(
            r"(?:starting\s+cap|cap\s+at\s+start|initial\s+cap)[:\s]*([+-]?\d+\.?\d*)\s*%",
            page_text, re.IGNORECASE
        )
        if starting_cap_match:
            result.starting_cap = float(starting_cap_match.group(1)) / 100.0

        # --- Attempt to extract starting buffer ---
        starting_buffer_match = re.search(
            r"(?:starting\s+buffer|buffer\s+at\s+start|initial\s+buffer)[:\s]*([+-]?\d+\.?\d*)\s*%",
            page_text, re.IGNORECASE
        )
        if starting_buffer_match:
            result.starting_buffer = float(starting_buffer_match.group(1)) / 100.0

        # --- Attempt to extract outcome period dates ---
        # Look for "Outcome Period: MM/DD/YYYY - MM/DD/YYYY" or similar
        date_range_match = re.search(
            r"(?:outcome\s+period)[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})\s*[-–to]+\s*(\d{1,2}/\d{1,2}/\d{2,4})",
            page_text, re.IGNORECASE
        )
        if date_range_match:
            result.outcome_period_start = _parse_date(date_range_match.group(1))
            result.reset_date = _parse_date(date_range_match.group(2))

        # --- Attempt to extract NAV ---
        nav_match = re.search(
            r"(?:NAV|net\s+asset\s+value)[:\s]*\$?\s*(\d+\.?\d*)",
            page_text, re.IGNORECASE
        )
        if nav_match:
            result.current_nav = float(nav_match.group(1))

        # --- Look for data in table cells ---
        # Innovator often puts data in labeled table rows
        _extract_table_data(soup, result)

        # --- Look for JSON data embedded in script tags ---
        _extract_script_data(soup, result)

        # Determine if we got enough data to be useful
        has_cap = result.remaining_cap is not None
        has_buffer = result.remaining_buffer is not None
        has_dates = result.outcome_period_start is not None and result.reset_date is not None

        if has_cap or has_buffer or has_dates:
            result.scrape_successful = True
            result.scrape_message = "Data scraped from Innovator ETFs website."
            missing = []
            if not has_cap:
                missing.append("remaining cap")
            if not has_buffer:
                missing.append("remaining buffer")
            if not has_dates:
                missing.append("outcome period dates")
            if result.current_nav is None:
                missing.append("current NAV")
            if missing:
                result.scrape_message += f" Missing fields (enter manually): {', '.join(missing)}."
        else:
            result.scrape_successful = False
            result.scrape_message = (
                "Could not extract buffer ETF data from the Innovator website. "
                "The site may use JavaScript to load data dynamically, which "
                "requires a full browser engine to render. Please enter values manually."
            )

        result.scrape_timestamp = datetime.now()

    except requests.exceptions.HTTPError as e:
        result.scrape_successful = False
        if e.response is not None and e.response.status_code == 403:
            result.scrape_message = (
                "Access denied by Innovator ETFs website (HTTP 403). "
                "The site may be blocking automated requests. "
                "Please enter values manually."
            )
        else:
            result.scrape_message = f"HTTP error while scraping: {e}. Please enter values manually."
        logger.warning("HTTP error scraping %s: %s", ticker, e)

    except requests.exceptions.ConnectionError:
        result.scrape_successful = False
        result.scrape_message = (
            "Could not connect to innovatoretfs.com. "
            "Check your internet connection and try again, or enter values manually."
        )
        logger.warning("Connection error scraping %s", ticker)

    except requests.exceptions.Timeout:
        result.scrape_successful = False
        result.scrape_message = (
            "Request to innovatoretfs.com timed out. "
            "The site may be slow or unavailable. Please enter values manually."
        )
        logger.warning("Timeout scraping %s", ticker)

    except Exception as e:
        result.scrape_successful = False
        result.scrape_message = f"Unexpected error during scraping: {e}. Please enter values manually."
        logger.exception("Unexpected error scraping %s", ticker)

    return result


def _extract_table_data(soup: BeautifulSoup, result: ETFData) -> None:
    """
    Extract data from HTML table elements on the Innovator ETF page.

    Innovator's fund pages use labeled table rows to display key metrics.
    This function scans all table cells looking for recognizable labels
    and extracts the associated values.

    Args:
        soup: Parsed HTML of the fund page.
        result: ETFData instance to populate with any found values.
    """
    # Find all table rows and look for label-value pairs
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        label = cells[0].get_text(strip=True).lower()
        value = cells[1].get_text(strip=True)

        # Map common labels to ETFData fields
        if "remaining cap" in label and result.remaining_cap is None:
            result.remaining_cap = _parse_percentage(value)
        elif "remaining buffer" in label and result.remaining_buffer is None:
            result.remaining_buffer = _parse_percentage(value)
        elif "remaining protection" in label and result.remaining_buffer is None:
            result.remaining_buffer = _parse_percentage(value)
        elif "starting cap" in label and result.starting_cap is None:
            result.starting_cap = _parse_percentage(value)
        elif "starting buffer" in label and result.starting_buffer is None:
            result.starting_buffer = _parse_percentage(value)
        elif "nav" in label and result.current_nav is None:
            result.current_nav = _parse_nav(value)
        elif ("start date" in label or "period start" in label) and result.outcome_period_start is None:
            result.outcome_period_start = _parse_date(value)
        elif ("end date" in label or "reset date" in label or "period end" in label) and result.reset_date is None:
            result.reset_date = _parse_date(value)
        elif "downside before buffer" in label and result.downside_before_buffer is None:
            result.downside_before_buffer = _parse_percentage(value)


def _extract_script_data(soup: BeautifulSoup, result: ETFData) -> None:
    """
    Look for JSON data embedded in <script> tags on the page.

    Some ASP.NET pages embed data in JavaScript variables or JSON objects
    within script tags. This function searches for recognizable patterns
    like `var fundData = {...}` or inline JSON that contains our target fields.

    Args:
        soup: Parsed HTML of the fund page.
        result: ETFData instance to populate with any found values.
    """
    import json

    for script in soup.find_all("script"):
        text = script.string
        if not text:
            continue

        # Look for JSON-like objects containing fund data
        # Pattern: variable assignment with JSON object
        json_matches = re.findall(r'\{[^{}]*"(?:cap|buffer|nav|outcome)"[^{}]*\}', text, re.IGNORECASE)
        for match in json_matches:
            try:
                data = json.loads(match)
                # Try to extract values from the parsed JSON
                for key, value in data.items():
                    key_lower = key.lower()
                    if "remaining_cap" in key_lower or "remainingcap" in key_lower:
                        if result.remaining_cap is None:
                            result.remaining_cap = _parse_percentage(str(value))
                    elif "remaining_buffer" in key_lower or "remainingbuffer" in key_lower:
                        if result.remaining_buffer is None:
                            result.remaining_buffer = _parse_percentage(str(value))
                    elif "starting_cap" in key_lower or "startingcap" in key_lower:
                        if result.starting_cap is None:
                            result.starting_cap = _parse_percentage(str(value))
                    elif "starting_buffer" in key_lower or "startingbuffer" in key_lower:
                        if result.starting_buffer is None:
                            result.starting_buffer = _parse_percentage(str(value))
                    elif key_lower == "nav":
                        if result.current_nav is None:
                            result.current_nav = _parse_nav(str(value))
            except (json.JSONDecodeError, ValueError):
                continue

        # Look for individual variable assignments
        # Pattern: var remainingCap = "15.50%";
        var_patterns = [
            (r'(?:remaining|rem)(?:_)?[Cc]ap\s*[=:]\s*["\']?([+-]?\d+\.?\d*)\s*%?["\']?',
             "remaining_cap"),
            (r'(?:remaining|rem)(?:_)?[Bb]uffer\s*[=:]\s*["\']?([+-]?\d+\.?\d*)\s*%?["\']?',
             "remaining_buffer"),
            (r'(?:starting|start)(?:_)?[Cc]ap\s*[=:]\s*["\']?([+-]?\d+\.?\d*)\s*%?["\']?',
             "starting_cap"),
            (r'(?:starting|start)(?:_)?[Bb]uffer\s*[=:]\s*["\']?([+-]?\d+\.?\d*)\s*%?["\']?',
             "starting_buffer"),
        ]
        for pattern, field_name in var_patterns:
            match = re.search(pattern, text)
            if match and getattr(result, field_name) is None:
                setattr(result, field_name, float(match.group(1)) / 100.0)


def get_ticker_suggestions(query: str) -> list[str]:
    """
    Return matching Innovator ETF tickers for autocomplete suggestions.

    Filters the known ticker list based on a partial query string.

    Args:
        query: Partial ticker string entered by the user.

    Returns:
        List of matching ticker symbols, sorted alphabetically.
    """
    query = query.upper().strip()
    if not query:
        return sorted(INNOVATOR_TICKERS.keys())
    return sorted(
        ticker for ticker in INNOVATOR_TICKERS
        if ticker.startswith(query) or query in INNOVATOR_TICKERS[ticker].upper()
    )


def get_ticker_description(ticker: str) -> str:
    """
    Return a human-readable description for a known Innovator ETF ticker.

    Args:
        ticker: The ETF ticker symbol.

    Returns:
        Description string, or "Unknown ticker" if not in our list.
    """
    return INNOVATOR_TICKERS.get(ticker.upper(), "Unknown ticker")
