import numpy as np
import pandas as pd

def _dynamic_buy_threshold(atr_pct: float) -> float:
    """
    Adaptive probability threshold based on volatility regime.
    
    atr_pct: ATR normalized by last close (e.g., 0.01 = 1% movement)

    Returns:
        threshold between 0.44 and 0.58 controlling ML confidence needed to trade.
    """

    # Volatility bounds
    LOW_VOL = 0.005   # < 0.5% movement = chop → be picky
    HIGH_VOL = 0.015  # > 1.5% movement = trending → more aggressive

    # Default midpoint
    base_threshold = 0.50

    if atr_pct >= HIGH_VOL:
        # High volatility trending → reduce threshold (more trades)
        threshold = base_threshold - 0.06  # 0.44
    elif atr_pct <= LOW_VOL:
        # Sideways conditions → require stronger evidence
        threshold = base_threshold + 0.06  # 0.56
    else:
        # Scale linearly in transition regime
        scale = (atr_pct - LOW_VOL) / (HIGH_VOL - LOW_VOL)
        threshold = base_threshold + (0.06 * (1 - scale) * (-1 if scale > 0 else 1))

    # Safety clamp
    return float(np.clip(threshold, 0.40, 0.60))


def _trend_direction(row: pd.Series) -> int:
    """
    Basic trend filter using short vs long EMA.
    
    Expects row to include:
        - ema_fast (e.g., 10)
        - ema_slow (e.g., 30)
        If missing, falls back to neutral trend = 0.

    Returns:
        1   → Uptrend (long bias)
        -1  → Downtrend (short bias)
        0   → Neutral / No strong signal
    """
    
    fast = row.get("ema_fast", np.nan)
    slow = row.get("ema_slow", np.nan)

    if np.isnan(fast) or np.isnan(slow):
        return 0

    if fast > slow:
        return 1
    elif fast < slow:
        return -1
    else:
        return 0
