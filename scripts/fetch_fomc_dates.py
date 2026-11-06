"""Build data/events/fomc_decision_dates.csv from the Federal Reserve's own website.

Run (needs internet):  python scripts/fetch_fomc_dates.py

Sources (both on federalreserve.gov):
  * 2000-2020: one historical page per year (fomchistorical<year>.htm)
  * 2021 onward: the FOMC calendar page (fomccalendars.htm), which also lists
    future scheduled meetings.

What counts as a date:
  The DECISION day of a regularly SCHEDULED meeting (the last day of a one- or
  two-day meeting, when the policy statement is released). These dates are
  published long in advance, so "an FOMC decision is tomorrow" is knowable at
  the close of today.

What is excluded (because it was not known in advance, or had no decision):
  conference calls, "(unscheduled)" meetings, "(notation vote)" decisions,
  "(cancelled)" meetings, and entries with no policy Statement (for example a
  2003 pre-meeting session listed the day before the real meeting).
"""

import collections
import datetime as dt
import html
import re
import subprocess
import time
from pathlib import Path

BASE = "https://www.federalreserve.gov/monetarypolicy/"
OUT = Path(__file__).resolve().parent.parent / "data" / "events" / "fomc_decision_dates.csv"
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def month_number(name: str) -> int:
    """'Feb' or 'February' -> 2 (matched on the first three letters)."""
    return next(i for i, m in enumerate(MONTHS, 1) if m.startswith(name[:3]))


def fetch(url: str) -> str:
    """Download a page with the system `curl`.

    curl uses the operating system's trusted certificates. Python's own urllib
    fails with CERTIFICATE_VERIFY_FAILED on a default python.org install on macOS
    (its certificates have to be installed separately), so curl is more reliable here.
    """
    result = subprocess.run(["curl", "-sS", "--fail", "-m", "30", url],
                            capture_output=True, check=True)
    return result.stdout.decode("utf-8", errors="replace")


def text_of(fragment: str) -> str:
    """Strip HTML tags and tidy whitespace."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def parse_historical_year(page: str, year: int) -> list:
    """Decision dates from one 2000-2020 year page."""
    dates = []
    heads = list(re.finditer(r"<h5[^>]*>(.*?)</h5>", page, re.S))
    for k, m in enumerate(heads):
        title = text_of(m.group(1))
        end = heads[k + 1].start() if k + 1 < len(heads) else len(page)
        body = text_of(page[m.end():end])
        if "Meeting" not in title or re.search(r"unscheduled|cancel|notation", title, re.I):
            continue
        if "Statement" not in body:  # no policy statement = no decision that day
            continue
        # Titles look like "June 29-30 Meeting", "April/May 30-1 Meeting" or "March 22 Meeting".
        mm = re.match(r"(?:[A-Z][a-z]+/)?([A-Z][a-z]+)\s+(\d+)(?:\s*-\s*(?:([A-Z][a-z]+)\s+)?(\d+))?", title)
        first_month, first_day, second_month, last_day = mm.groups()
        month = month_number(second_month or first_month)
        day = int(last_day) if last_day else int(first_day)
        dates.append(dt.date(year, month, day))
    return dates


def parse_calendar_page(page: str) -> list:
    """Decision dates from the 2021-onward calendar page."""
    dates = []
    panels = list(re.finditer(r'<h4><a id="\d+">(\d{4}) FOMC Meetings</a></h4>', page))
    for k, p in enumerate(panels):
        year = int(p.group(1))
        if year < 2021:
            continue
        end = panels[k + 1].start() if k + 1 < len(panels) else len(page)
        # Each meeting is one row; quarterly meetings use a "shaded" variant of the row class.
        for row in re.split(r'<div class="(?:fomc-meeting--shaded )?row fomc-meeting[^"]*"', page[p.end():end])[1:]:
            month = text_of(re.search(r"fomc-meeting__month[^>]*>(.*?)</div>", row, re.S).group(1))
            date = text_of(re.search(r"fomc-meeting__date[^>]*>(.*?)</div>", row, re.S).group(1))
            if re.search(r"unscheduled|cancel|notation", month + date, re.I):
                continue
            mm = re.match(r"(\d+)(?:\s*-\s*(\d+))?", date)  # e.g. "27-28", "16-17*", "31-1"
            day = int(mm.group(2) or mm.group(1))
            dates.append(dt.date(year, month_number(month.split("/")[-1].strip()), day))
    return dates


def main() -> None:
    dates = []
    for year in range(2000, 2021):
        dates += parse_historical_year(fetch(f"{BASE}fomchistorical{year}.htm"), year)
        time.sleep(0.5)  # be polite to the server
    dates += parse_calendar_page(fetch(f"{BASE}fomccalendars.htm"))
    dates = sorted(set(dates))

    # Sanity checks: fail loudly instead of writing bad data.
    assert not [d for d in dates if d.weekday() >= 5], "a date fell on a weekend"
    per_year = collections.Counter(d.year for d in dates)
    odd = {y: n for y, n in per_year.items() if n != 8}
    print("Years without exactly 8 scheduled meetings (2020 should be 7):", odd)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("date\n" + "\n".join(d.isoformat() for d in dates) + "\n")
    print(f"Wrote {len(dates)} dates to {OUT}")


if __name__ == "__main__":
    main()
