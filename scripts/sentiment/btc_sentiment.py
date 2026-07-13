from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import requests
import pandas as pd

FNG_URL = "https://api.alternative.me/fng/"

SENTIMENT_CACHE = Path("/Users/dpowers01/trading_advisor/data/btc_sentiment.json")
SENTIMENT_TTL_HOURS = 6


# ---------- Cache helpers ----------

def _load_cached_fng():
    if not SENTIMENT_CACHE.exists():
        return None

    try:
        with open(SENTIMENT_CACHE, "r") as f:
            data = json.load(f)

        ts = datetime.fromisoformat(data["timestamp"])
        if datetime.now(timezone.utc) - ts > timedelta(hours=SENTIMENT_TTL_HOURS):
            return None

        return int(data["fng_value"])
    except Exception:
        return None


def _save_fng(fng_value: int):
    SENTIMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(SENTIMENT_CACHE, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "fng_value": int(fng_value),
            },
            f,
            indent=2,
        )


# ---------- Live sentiment ----------

def get_btc_fng() -> int:
    """
    Returns latest BTC Fear & Greed Index (0–100).
    Cached for stability + rate limiting.
    """
    cached = _load_cached_fng()
    if cached is not None:
        return cached

    fng = _fetch_fng_live()
    _save_fng(fng)
    return fng


def _fetch_fng_live() -> int:
    try:
        r = requests.get(FNG_URL, timeout=10)
        r.raise_for_status()

        data = r.json().get("data", [])
        if not data:
            return 50  # neutral fallback

        value = int(data[0]["value"])
        print(f"📊 BTC Fear & Greed Index: {value}")
        return value

    except Exception as e:
        print(f"⚠️ Failed to fetch Fear & Greed index: {e}")
        return 50


# ---------- Historical sentiment ----------

def fetch_historical_btc_sentiment(days: int = 730) -> pd.DataFrame:
    """
    Fetch BTC Fear & Greed history.

    Returns:
      timestamp (UTC, daily aligned)
      fng (int 0–100)
    """
    try:
        limit = min(days, 2000)

        r = requests.get(f"{FNG_URL}?limit={limit}", timeout=10)
        r.raise_for_status()

        data = r.json()["data"]
        df = pd.DataFrame(data)

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

        # Normalize to daily boundary (critical for merge_asof)
        df["timestamp"] = df["timestamp"].dt.floor("D")

        df["fng"] = df["value"].astype(int)

        return (
            df[["timestamp", "fng"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    except Exception as e:
        print(f"⚠️ Failed to fetch historical FNG: {e}")
        return pd.DataFrame(columns=["timestamp", "fng"])


# ---------- Utility for feature pipelines ----------

def merge_fng_feature(df: pd.DataFrame, df_fng: pd.DataFrame) -> pd.DataFrame:
    """
    Safely merge FNG into hourly price data.

    - Uses backward merge (no lookahead)
    - Forward-fills missing values
    - Defaults to neutral (50)
    """
    df = df.sort_values("timestamp")

    df = pd.merge_asof(
        df,
        df_fng,
        on="timestamp",
        direction="backward"
    )

    # Fill missing values safely
    df["fng"] = df["fng"].ffill().fillna(50)

    return df


# ---------- Optional: confidence shaping ----------

# def sentiment_conf_penalty(fng: int) -> float:
#     """
#     Convert FNG → confidence adjustment.

#     Negative = easier to buy
#     Positive = harder to buy
#     """
#     if fng <= 10:
#         return 0.30
#     elif fng <= 25:
#         return 0.20
#     elif fng <= 40:
#         return 0.10
#     elif fng <= 60:
#         return 0.00
#     elif fng <= 75:
#         return -0.05
#     else:
#         return -0.10
