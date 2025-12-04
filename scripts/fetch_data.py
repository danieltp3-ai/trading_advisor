import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from config import COIN_ID, CACHE_FILE, CACHE_TTL_HOURS

# Coinbase Exchange candles endpoint
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product_id}/candles"
# maximum candles per request for Coinbase Exchange API is 300
COINBASE_MAX_CANDLES = 300
# hourly granularity seconds
GRANULARITY = 3600

HEADERS = {
    "User-Agent": "trading-advisor/1.0",
    "Accept": "application/json",
}


def _coinbase_request(product_id: str, start_iso: str, end_iso: str, granularity: int = GRANULARITY):
    """Low-level request wrapper with retries for Coinbase exchange candles endpoint."""
    url = COINBASE_CANDLES_URL.format(product_id=product_id)
    params = {"start": start_iso, "end": end_iso, "granularity": granularity}

    for attempt in range(4):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"⚠️ network error (attempt {attempt+1}): {e}; sleeping {wait}s")
            time.sleep(wait)
            continue

        if r.status_code == 429:
            wait = 20 * (attempt + 1)
            print(f"⏳ Coinbase rate-limited (429). attempt {attempt+1}, sleeping {wait}s...")
            time.sleep(wait)
            continue

        if r.status_code >= 500:
            wait = 5 * (attempt + 1)
            print(f"⚠️ Coinbase server error {r.status_code}. attempt {attempt+1}, sleeping {wait}s...")
            time.sleep(wait)
            continue

        if r.status_code != 200:
            # include body (truncated) for debugging
            body = r.text[:500]
            raise RuntimeError(f"❌ Coinbase HTTP {r.status_code}: {body}")

        return r.json()

    raise RuntimeError("Too many retries requesting Coinbase candles.")


def _parse_coinbase_candles(raw):
    """
    Coinbase returns arrays: [time, low, high, open, close, volume]
    We convert to DataFrame with columns: timestamp (UTC), open, high, low, close, volume
    """
    if not raw:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
    # time is epoch seconds in UTC
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # reorder and cast numerics
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    return df


def fetch_from_coinbase(days: int = 365, product_suffix: str = "-USD"):
    """
    Fetch hourly OHLCV from Coinbase Exchange by paging backwards in 300-candle windows.
    Returns DataFrame with columns: timestamp (tz-aware UTC), open, high, low, close, volume.
    """
    product_id = f"{COIN_ID.upper()}{product_suffix}"
    hours_needed = int(days * 24)
    all_chunks = []

    # Coinbase returns at most 300 candles per request. We'll request windows of 300 hours.
    window_hours = COINBASE_MAX_CANDLES  # 300
    end_dt = datetime.now(timezone.utc)  # fetch up to 'now' UTC

    print(f"📡 Fetching {hours_needed} hours of OHLCV for {product_id} from Coinbase... (granularity=1h)")

    while hours_needed > 0:
        this_hours = min(window_hours, hours_needed)
        start_dt = end_dt - timedelta(hours=this_hours)

        # Coinbase expects ISO8601 with seconds, e.g. 2025-11-12T00:00:00Z
        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()

        raw = _coinbase_request(product_id, start_iso, end_iso, GRAnULARITY := GRANULARITY)

        # parse and append
        df_chunk = _parse_coinbase_candles(raw)
        if df_chunk.empty:
            # no more history or API returned empty set
            print("⚠️ Coinbase returned no candles for window; stopping early.")
            break

        all_chunks.append(df_chunk)

        # move window backward
        oldest_ts = df_chunk["timestamp"].min()
        end_dt = oldest_ts - timedelta(seconds=1)
        hours_needed -= len(df_chunk)

        # polite short sleep to avoid triggering rate limits
        time.sleep(0.25)

    if not all_chunks:
        raise RuntimeError("Coinbase returned no candle data for the requested period.")

    df = pd.concat(all_chunks, ignore_index=True)
    # drop duplicates, sort ascending
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    print(f"✅ Retrieved {len(df)} hourly candles from Coinbase.")
    return df


def fetch_coinbase_data(days: int = 365, product_suffix: str = "-USD"):
    """
    Cached fetcher for Coinbase OHLCV. Respects CACHE_TTL_HOURS and appends new hourly rows.
    Returns DataFrame with columns: timestamp (UTC-aware), open, high, low, close, volume
    """
    now = pd.Timestamp.utcnow().tz_convert("UTC")

    # If cache exists and not stale, return it
    if Path(CACHE_FILE).exists():
        df_cache = pd.read_parquet(CACHE_FILE)
        # ensure tz-aware timestamps
        df_cache["timestamp"] = pd.to_datetime(df_cache["timestamp"], utc=True)
        last_ts = df_cache["timestamp"].max()
        age_hours = (now - last_ts).total_seconds() / 3600.0

        if age_hours < CACHE_TTL_HOURS:
            print(f"📦 Using cached Coinbase data ({age_hours:.2f}h old) → {CACHE_FILE}")
            return df_cache

        # fetch latest ~3 days to be safe and append only new rows
        print(f"♻️ Cache stale ({age_hours:.2f}h). Fetching recent data since {last_ts} ...")
        new_raw = fetch_from_coinbase(days=3, product_suffix=product_suffix)
        new_raw = new_raw[new_raw["timestamp"] > last_ts]

        if new_raw.empty:
            print("ℹ️ No new rows returned by Coinbase; returning existing cache.")
            return df_cache

        combined = pd.concat([df_cache, new_raw], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        combined.to_parquet(CACHE_FILE, index=False)
        print(f"✅ Appended {len(new_raw)} new hourly rows and updated cache at {CACHE_FILE}")
        return combined

    # No cache: fetch full history and save
    print("🆕 No cache found — fetching full Coinbase history...")
    df = fetch_from_coinbase(days=days, product_suffix=product_suffix)

    # ensure hourly spacing (resample if necessary) - force 1H candles using last price and sum volume
    df = df.set_index("timestamp").resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()

    df.to_parquet(CACHE_FILE, index=False)
    print(f"✅ Cached {len(df)} hourly candles → {CACHE_FILE}")
    return df