import io
import os
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(".env")

SHARADAR_URL = "https://api.sharadar.com/v1.0/data/funds"  # SFP (Fund Prices)
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
PAGE_SIZE = 10000
MAX_FFILL_BDAYS = 5


def _get_sharadar(ticker: str, start_date: str, api_key: str) -> pd.DataFrame:
    frames, offset = [], 0
    while True:
        resp = requests.get(
            SHARADAR_URL,
            headers={"x-api-key": api_key},
            params={"ticker": ticker, "from": start_date, "fields": "ticker,date,open,close,closeadj",
                    "format": "csv", "limit": PAGE_SIZE, "offset": offset},
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Sharadar API error {resp.status_code} for {ticker}: {resp.text[:300]}")
        page = pd.read_csv(io.StringIO(resp.text))
        frames.append(page)
        if len(page) < PAGE_SIZE:
            return pd.concat(frames, ignore_index=True)
        offset += PAGE_SIZE


def fetch_sharadar_etf_data(
    tickers: list[str] = ["SPY", "QQQ", "BIL"],
    start_date: str = "1999-01-01",
    cache_path: str = "data/sharadar_etfs_ohlc.parquet"
) -> pd.DataFrame:
    """
    Raw long-format rows (ticker, date, open, close, closeadj) from the Sharadar funds (SFP) table.
    Cached locally to avoid redundant API queries.
    """
    if os.path.exists(cache_path):
        print(f"[CACHE] Loading historical data from {cache_path}...")
        return pd.read_parquet(cache_path)

    api_key = os.getenv("SHARADAR_API_KEY")
    if not api_key:
        raise ValueError("Missing SHARADAR_API_KEY. Set it in your .env file.")

    print(f"[SHARADAR] Querying funds (SFP) table for {tickers} starting {start_date}...")
    raw = pd.concat([_get_sharadar(t, start_date, api_key) for t in tickers], ignore_index=True)
    missing = set(tickers) - set(raw["ticker"])
    if missing:
        raise ValueError(f"Missing tickers in Sharadar funds response: {missing}")
    raw["date"] = pd.to_datetime(raw["date"])

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    raw.to_parquet(cache_path)
    print(f"[CACHE] Saved {len(raw)} rows to {cache_path}")
    return raw


def build_price_panels(raw: pd.DataFrame, tickers: list[str]) -> dict:
    """
    Pivot to date x ticker panels on the common trading calendar (rows where all tickers exist).
    adj_open = open * closeadj / close (same row). Bad opens fall back to that day's adjusted close.
    """
    raw = raw[raw["ticker"].isin(tickers)]
    if (raw["close"] <= 0).any() or (raw["closeadj"] <= 0).any() or raw[["close", "closeadj"]].isna().any().any():
        raise ValueError("Non-positive or missing close/closeadj in Sharadar data")

    piv = raw.pivot(index="date", columns="ticker").sort_index()
    first_common = max(piv["closeadj"][t].first_valid_index() for t in tickers)
    piv = piv.loc[first_common:]

    # Interior gaps (a ticker missing on a date another ticker traded) are forward-filled and counted.
    gap_mask = piv["closeadj"][tickers].isna()
    n_ffilled = int(gap_mask.sum().sum())

    adj_close = piv["closeadj"][tickers].ffill()
    close = piv["close"][tickers].ffill()
    open_raw = piv["open"][tickers]
    adj_open = open_raw * adj_close / close

    bad_open = open_raw.isna() | (open_raw <= 0) | gap_mask
    fallback_log = [(adj_open.index[i].date(), tickers[j]) for i, j in zip(*np.where(bad_open.to_numpy()))]
    adj_open = adj_open.mask(bad_open, adj_close)
    open_raw = open_raw.mask(bad_open, close)

    ratio = adj_open / adj_close
    if ((ratio < 0.5) | (ratio > 2.0)).any().any():
        raise ValueError("Impossible open/close ratio found after adjustment")

    return {"adj_close": adj_close, "adj_open": adj_open, "open": open_raw, "close": close,
            "n_ffilled": n_ffilled, "open_fallbacks": fallback_log}


def fetch_fred_rf(index: pd.DatetimeIndex, series: str = "DTB3",
                  cache_path: str = "data/fred_dtb3.parquet") -> tuple[pd.Series, pd.Series, list]:
    """
    3M T-bill (annual %, FRED) aligned to the trading calendar.
    Returns (annual_pct, daily_rf, ffill_log). Fills over gaps > MAX_FFILL_BDAYS are logged.
    """
    if os.path.exists(cache_path):
        fred = pd.read_parquet(cache_path)["rate"]
    else:
        resp = requests.get(FRED_URL.format(series), timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), na_values=".")
        fred = pd.Series(df[series].values, index=pd.to_datetime(df["observation_date"]), name="rate")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        fred.to_frame().to_parquet(cache_path)

    fred = fred.loc[index[0] - pd.Timedelta(days=30): index[-1]]
    bdays = pd.bdate_range(fred.index[0], index[-1])
    on_bdays = fred.reindex(bdays)

    # Log every run of missing business days longer than the allowed fill window.
    ffill_log = []
    missing = on_bdays.isna()
    run_id = (~missing).cumsum()
    for _, run in on_bdays[missing].groupby(run_id[missing]):
        if len(run) > MAX_FFILL_BDAYS:
            ffill_log.append((run.index[0].date(), run.index[-1].date(), len(run)))

    annual = on_bdays.ffill().reindex(index)
    if annual.isna().any():
        raise ValueError(f"Risk-free rate missing on {annual.isna().sum()} trading days")
    daily = (1 + annual / 100) ** (1 / 252) - 1
    return annual, daily, ffill_log
