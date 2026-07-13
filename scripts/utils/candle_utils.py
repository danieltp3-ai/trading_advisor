import pandas as pd

def fetch_and_clean_candles(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures:
    - Only fully closed hourly candles
    - No missing hours (forward-filled)
    - Monotonic, clean time index
    """

    df = raw_df.copy()

    # --- normalize timestamps ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    # --- drop current in-progress candle ---
    now_hour = pd.Timestamp.utcnow().floor("H")
    df = df[df["timestamp"] < now_hour]

    # --- set hourly index & backfill gaps ---
    df = df.set_index("timestamp")

    df = df.asfreq("1H")

    # Forward-fill OHLCV for missing hours
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    df[ohlcv_cols] = df[ohlcv_cols].ffill()

    # Drop any rows still incomplete (start-of-series edge)
    df = df.dropna(subset=["close"])

    return df.reset_index()