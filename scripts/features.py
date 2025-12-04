import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: df with columns:
        ['timestamp','open','high','low','close','volume']
        spaced hourly.
    Returns: df including engineered features.
    """

    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Basic Returns ---
    df["return_1h"] = df["close"].pct_change()
    df["return_open_close"] = (df["close"] - df["open"]) / df["open"]
    df["return_high_low"] = (df["high"] - df["low"]) / df["low"]

    # --- Candlestick Structural Features ---
    df["candle_body"] = df["close"] - df["open"]
    df["candle_range"] = df["high"] - df["low"]
    df["candle_body_pct"] = df["candle_body"] / df["candle_range"].replace(0, np.nan)

    df["upper_wick"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_wick"] = df[["close", "open"]].min(axis=1) - df["low"]

    df["upper_wick_pct"] = df["upper_wick"] / df["candle_range"].replace(0, np.nan)
    df["lower_wick_pct"] = df["lower_wick"] / df["candle_range"].replace(0, np.nan)

    # --- Moving Averages ---
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    # --- Volatility ---
    df["volatility_10"] = df["return_1h"].rolling(10).std()

    # --- RSI (Close-based) ---
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD (Close-based) ---
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- Volume ---
    df["volume_roc"] = df["volume"].pct_change()

    # --- ATR (true volatility) ---
    df["prev_close"] = df["close"].shift(1)

    true_range = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["prev_close"]).abs(),
        (df["low"] - df["prev_close"]).abs(),
    ], axis=1).max(axis=1)

    df["atr_14"] = true_range.rolling(14).mean()

    df.drop(columns=["prev_close"], inplace=True)

    # Final clean
    df = df.dropna().reset_index(drop=True)
    return df