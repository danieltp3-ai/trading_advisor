import json
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sentiment.btc_sentiment import get_btc_fng, sentiment_conf_penalty


from config import (
    TRADE_FEE, TAKE_PROFIT, STOP_LOSS, ATR_EXIT_THRESHOLD,
    BASE_DIR
)
from utils.signal_logic import _dynamic_buy_threshold

# Paths
STATE_PATH = BASE_DIR / "state" / "state.json"
TRADES_LOG = BASE_DIR / "data" / "trades_log.csv"

# Ensure directories exist
STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
TRADES_LOG.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, data: Dict):
    """Write JSON atomically to avoid corruption if process interrupted."""
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="tmp_state_", dir=str(path.parent))
    with open(tmp_fd, "w") as f:
        json.dump(data, f, default=str, indent=2)
    Path(tmp_path).replace(path)


def load_state(path: Path = STATE_PATH) -> Dict:
    """Load persisted state or return default initial state."""
    if not path.exists():
        state = {
            "position": 0,           # 0 = flat, 1 = long
            "entry_price": None,
            "entry_ts": None,
            "equity": 1000.0,        # simulated equity base
            "cash": 1000.0,
            "last_action": None,
            "last_action_ts": None,
            "notes": ""
        }
        _atomic_write_json(path, state)
        return state
    with open(path, "r") as f:
        return json.load(f)


def save_state(state: Dict, path: Path = STATE_PATH):
    state["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(path, state)


def append_trade_log(trade: Dict, path: Path = TRADES_LOG):
    """Append one trade row to CSV in an atomic-friendly way."""
    df = pd.DataFrame([trade])
    if path.exists():
        df_existing = pd.read_csv(path)
        df_out = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_out = df
    # atomic write
    tmp = path.with_suffix(".tmp")
    df_out.to_csv(tmp, index=False)
    tmp.replace(path)


def evaluate_and_update_state(
    df: pd.DataFrame,
    model,
    feature_cols,
    state_path: Optional[Path] = None
) -> Tuple[int, Dict, Optional[Dict]]:
    """
    Evaluate the latest row using backtest-equivalent logic, update persisted state,
    and return (action, new_state, trade_record_or_None).

    action:  1 -> BUY executed
             -1 -> SELL executed
             0 -> HOLD / no trade
    new_state: the updated state dict
    trade_record_or_None: trade dict if a trade occurred, else None
    """
    if state_path is None:
        state_path = STATE_PATH

    state = load_state(state_path)

    # safety
    df = df.copy().reset_index(drop=True)
    if df.empty:
        return 0, state, None

    # Make sure features are numeric and required columns exist
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # predictions (raw) and probs if available
    pred = model.predict(df[feature_cols])
    try:
        probs = model.predict_proba(df[feature_cols])
        classes = list(model.classes_)
        if 1 in classes:
            idx_buy = classes.index(1)
            buy_conf_series = probs[:, idx_buy]
        else:
            buy_conf_series = probs[:, -1]
    except Exception:
        buy_conf_series = np.ones(len(df))

    df["pred_signal"] = pred.astype(int)
    df["buy_conf"] = buy_conf_series.astype(float)

    # Ensure ATR present
    if "atr_14" not in df.columns:
        df["atr_14"] = df["close"].rolling(14).apply(lambda x: (x.max() - x.min()), raw=False).fillna(0)
    df["atr_pct"] = df["atr_14"] / df["close"].replace(0, np.nan)

    # Use the last row as "current"
    last = df.iloc[-1]
    ts = pd.to_datetime(last["timestamp"])
    price = float(last["close"])
    raw_signal = int(last["pred_signal"])
    buy_conf = float(last["buy_conf"]) if not np.isnan(last["buy_conf"]) else 1.0
    atr_pct = float(last["atr_pct"]) if not np.isnan(last["atr_pct"]) else 0.0

    # compute dynamic threshold (same helper as backtest)
    dynamic_thr = _dynamic_buy_threshold(atr_pct)

    # adjust threshold based on sentiment
    btc_sentiment_fng_int = get_btc_fng()
    sentiment_penalty = sentiment_conf_penalty(btc_sentiment_fng_int)
    adjusted_thr = dynamic_thr + sentiment_penalty
    adjusted_thr = min(max(adjusted_thr, 0.0), 0.95)

    action = 0
    trade_record = None

    # Use same trading rules as backtest (long-only gating)
    if state["position"] == 0:
        # Attempt to open long
        if raw_signal == 1:
            # require buy_conf >= adjusted_thr and a sensible atr_pct
            if buy_conf >= adjusted_thr and atr_pct >= 0.0:
                # BUY: use all equity to buy, after fee
                cash_after_fee = state["equity"] * (1 - TRADE_FEE)
                position_qty = cash_after_fee / price
                state["position"] = 1
                state["entry_price"] = price
                state["entry_ts"] = ts.isoformat()
                state["cash"] = 0.0
                # we keep equity as total account value until exit (like backtest),
                # but store equity_after_fee for record purposes
                state["equity_after_fee"] = cash_after_fee
                action = 1
                trade_record = {
                    "timestamp": ts.isoformat(),
                    "action": "BUY",
                    "price": price,
                    "equity_before": state.get("equity", 0.0),
                    "equity_after_fee": cash_after_fee,
                    "buy_conf": buy_conf,
                    "atr_pct": atr_pct,
                    "adjusted_thr": adjusted_thr
                }
                state["last_action"] = "BUY"
                state["last_action_ts"] = ts.isoformat()

    else:
        # If currently long, evaluate exit conditions (match backtest)
        entry_price = float(state["entry_price"]) if state.get("entry_price") else price
        gain = (price - entry_price) / entry_price if entry_price > 0 else 0.0

        # ATR exit (forced)
        if atr_pct <= ATR_EXIT_THRESHOLD:
            proceeds = (state["equity_after_fee"] / entry_price) * price * (1 - TRADE_FEE) if state.get("equity_after_fee") else 0.0
            pnl = proceeds - state.get("equity", 0.0)
            # settle
            state["equity"] = proceeds
            state["cash"] = proceeds
            state["position"] = 0
            state["entry_price"] = None
            state["entry_ts"] = None
            action = -1
            trade_record = {
                "timestamp": ts.isoformat(),
                "action": "SELL (ATR_EXIT)",
                "price": price,
                "entry_price": entry_price,
                "gain": gain,
                "pnl": pnl,
                "atr_pct": atr_pct
            }
            state["last_action"] = "SELL (ATR_EXIT)"
            state["last_action_ts"] = ts.isoformat()

        # stop loss
        elif gain <= STOP_LOSS:
            proceeds = (state["equity_after_fee"] / entry_price) * price * (1 - TRADE_FEE) if state.get("equity_after_fee") else 0.0
            pnl = proceeds - state.get("equity", 0.0)
            state["equity"] = proceeds
            state["cash"] = proceeds
            state["position"] = 0
            state["entry_price"] = None
            state["entry_ts"] = None
            action = -1
            trade_record = {
                "timestamp": ts.isoformat(),
                "action": "SELL (STOPLOSS)",
                "price": price,
                "entry_price": entry_price,
                "gain": gain,
                "pnl": pnl
            }
            state["last_action"] = "SELL (STOPLOSS)"
            state["last_action_ts"] = ts.isoformat()

        # take profit -> only sell if model no longer encourages holding (raw_signal != 1)
        elif gain >= TAKE_PROFIT and raw_signal != 1:
            proceeds = (state["equity_after_fee"] / entry_price) * price * (1 - TRADE_FEE) if state.get("equity_after_fee") else 0.0
            pnl = proceeds - state.get("equity", 0.0)
            state["equity"] = proceeds
            state["cash"] = proceeds
            state["position"] = 0
            state["entry_price"] = None
            state["entry_ts"] = None
            action = -1
            trade_record = {
                "timestamp": ts.isoformat(),
                "action": "SELL (TAKEPROFIT)",
                "price": price,
                "entry_price": entry_price,
                "gain": gain,
                "pnl": pnl
            }
            state["last_action"] = "SELL (TAKEPROFIT)"
            state["last_action_ts"] = ts.isoformat()

        # model SELL signal
        elif raw_signal == -1:
            proceeds = (state["equity_after_fee"] / entry_price) * price * (1 - TRADE_FEE) if state.get("equity_after_fee") else 0.0
            pnl = proceeds - state.get("equity", 0.0)
            state["equity"] = proceeds
            state["cash"] = proceeds
            state["position"] = 0
            state["entry_price"] = None
            state["entry_ts"] = None
            action = -1
            trade_record = {
                "timestamp": ts.isoformat(),
                "action": "SELL (MODEL)",
                "price": price,
                "entry_price": entry_price,
                "gain": gain,
                "pnl": pnl
            }
            state["last_action"] = "SELL (MODEL)"
            state["last_action_ts"] = ts.isoformat()

    # If trade happened, append to trades log and persist
    if trade_record:
        append_trade_log(trade_record)
    # Save updated state
    save_state(state, state_path)

    return action, state, trade_record
