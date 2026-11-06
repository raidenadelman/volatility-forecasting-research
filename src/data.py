"""Download, cache and validate daily S&P 500 (^GSPC) price data.

Usage from other code:
    from src.data import load_data
    df = load_data()               # uses the cached CSV if it exists
    df = load_data(refresh=True)   # forces a fresh download

Usage from the command line:
    python -m src.data --refresh
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

TICKER = "^GSPC"
START_DATE = "2000-01-01"
# Project root = the folder that contains src/. Works no matter where you run from.
ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "data" / "raw" / "gspc.csv"

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


def download_data(start: str = START_DATE, end: str | None = None) -> pd.DataFrame:
    """Download daily OHLC data from Yahoo Finance via the yfinance library.

    Raises a RuntimeError with a plain-language message if the download fails,
    rather than returning fake or empty data.
    """
    import yfinance as yf  # imported here so cached runs work even without network libs

    end = end or date.today().isoformat()
    try:
        # auto_adjust=False keeps the raw Open/High/Low/Close (an index has no
        # dividends/splits to adjust anyway).
        raw = yf.download(TICKER, start=start, end=end, auto_adjust=False,
                          progress=False)
    except Exception as exc:  # network errors, rate limits, etc.
        raise RuntimeError(
            f"yfinance download failed: {exc}\n"
            "Check your internet connection and try again. If you have a cached "
            f"file at {CACHE_PATH}, run without --refresh to use it."
        ) from exc

    if raw is None or raw.empty:
        raise RuntimeError(
            "yfinance returned no data (likely a network problem or Yahoo "
            "rate-limiting). Wait a minute and retry with --refresh."
        )

    # Newer yfinance versions return two-level column names like ("Close", "^GSPC").
    # Flatten them to plain "Close".
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[PRICE_COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


def validate_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Clean and check the price table, printing what (if anything) was dropped.

    Checks: no duplicate dates, sorted index, no missing High/Low/Close,
    High >= Low.
    """
    n_start = len(df)
    dropped = []  # list of (reason, number of rows)

    # 1. Sorted index (sorting loses no rows, so it is not counted as a drop).
    df = df.sort_index()

    # 2. Duplicate dates: keep the first occurrence of each date.
    dup_mask = df.index.duplicated(keep="first")
    if dup_mask.any():
        dropped.append(("duplicate dates", int(dup_mask.sum())))
        df = df[~dup_mask]

    # 3. Missing High / Low / Close.
    missing_mask = df[["High", "Low", "Close"]].isna().any(axis=1)
    if missing_mask.any():
        dropped.append(("missing High/Low/Close", int(missing_mask.sum())))
        df = df[~missing_mask]

    # 4. High must be >= Low (otherwise the row is a data error).
    bad_range = df["High"] < df["Low"]
    if bad_range.any():
        dropped.append(("High < Low", int(bad_range.sum())))
        df = df[~bad_range]

    # Final assertions: if any of these fail, something above is wrong.
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert not df[["High", "Low", "Close"]].isna().any().any()
    assert (df["High"] >= df["Low"]).all()

    if verbose:
        print("--- Data summary ---")
        print(f"Date range : {df.index.min().date()} to {df.index.max().date()}")
        print(f"Rows       : {len(df)} (downloaded/cached: {n_start})")
        if dropped:
            for reason, n in dropped:
                print(f"Dropped    : {n} row(s) because of {reason}")
        else:
            print("Dropped    : 0 rows (all validation checks passed)")
    return df


def load_data(refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Return validated daily OHLC data, using the CSV cache when possible."""
    if CACHE_PATH.exists() and not refresh:
        if verbose:
            print(f"Loading cached data from {CACHE_PATH}")
        df = pd.read_csv(CACHE_PATH, index_col="Date", parse_dates=True)
    else:
        if verbose:
            print(f"Downloading {TICKER} from {START_DATE} via yfinance ...")
        df = download_data()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH)
        if verbose:
            print(f"Saved cache to {CACHE_PATH}")
    return validate_data(df, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download/cache S&P 500 data.")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache and download fresh data")
    args = parser.parse_args()
    load_data(refresh=args.refresh)
