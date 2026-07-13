import pandas as pd
import numpy as np
from config import TRADE_FEE, TAKE_PROFIT, STOP_LOSS
from config import BASE_BUY_CONF, ATR_SCALE, MAX_CONF_ADJUST, MIN_BUY_CONF, MAX_BUY_CONF, ATR_EXIT_THRESHOLD
from utils.signal_logic import _dynamic_buy_threshold
from sentiment.btc_sentiment import fetch_historical_btc_sentiment, sentiment_conf_penalty

def classify_sentiment_impact(trades_df: pd.DataFrame) -> pd.DataFrame:
    df = trades_df.copy()

    # Identify BUY trades
    df = df[df["action"] == "BUY"].copy()

    # Trade blocked by sentiment if:
    # buy_conf >= dynamic_thr BUT < adjusted_thr
    df["blocked_by_sentiment"] = (
        (df["buy_conf"] >= df["dynamic_thr"]) &
        (df["buy_conf"] < df["adjusted_thr"])
    )

    return df

def estimate_blocked_trade_cost(trades_df: pd.DataFrame) -> pd.DataFrame:
    buys = classify_sentiment_impact(trades_df)

    # Find corresponding SELLs
    sells = trades_df[trades_df["action"].str.contains("SELL")].copy()

    trade_pairs = []
    for _, buy in buys.iterrows():
        sell = sells[sells["timestamp"] > buy["timestamp"]].head(1)
        if sell.empty:
            continue

        trade_pairs.append({
            "timestamp": buy["timestamp"],
            "blocked_by_sentiment": buy["blocked_by_sentiment"],
            "btc_fng": buy["btc_sentiment"],
            "expected_pnl": sell.iloc[0]["pnl"]
        })

    return pd.DataFrame(trade_pairs)

def summarize_sentiment_impact(trade_impact_df: pd.DataFrame):
    summary = {}

    blocked = trade_impact_df[trade_impact_df["blocked_by_sentiment"]]
    allowed = trade_impact_df[~trade_impact_df["blocked_by_sentiment"]]

    summary["blocked_trades"] = len(blocked)
    summary["allowed_trades"] = len(allowed)

    summary["blocked_total_pnl"] = blocked["expected_pnl"].sum()
    summary["allowed_total_pnl"] = allowed["expected_pnl"].sum()

    summary["blocked_avg_pnl"] = blocked["expected_pnl"].mean()
    summary["allowed_avg_pnl"] = allowed["expected_pnl"].mean()

    summary["sentiment_net_harm"] = summary["blocked_total_pnl"] < 0

    return summary

def analyze_fng_ranges(trade_impact_df: pd.DataFrame) -> pd.DataFrame:
    df = trade_impact_df.copy()

    bins = [0, 10, 25, 40, 55, 70, 85, 100]
    labels = [
        "Extreme Fear",
        "High Fear",
        "Fear",
        "Neutral",
        "Greed",
        "High Greed",
        "Extreme Greed"
    ]

    df["fng_bucket"] = pd.cut(df["btc_fng"], bins=bins, labels=labels)

    agg = df.groupby("fng_bucket").agg(
        blocked_trades=("blocked_by_sentiment", "sum"),
        avg_blocked_pnl=("expected_pnl", "mean"),
        total_blocked_pnl=("expected_pnl", "sum")
    ).reset_index()

    return agg.sort_values("total_blocked_pnl")


def backtest_with_trade_log_and_accuracy(df, model, feature_cols,
                                         start_equity=1000.0,
                                         trade_fee=TRADE_FEE,
                                         take_profit=TAKE_PROFIT,
                                         stop_loss=STOP_LOSS):
    """
    Backtest with:
      - ATR-adaptive buy gating (applies only to LONG buys)
      - Forced exit when ATR drops below ATR_EXIT_THRESHOLD
      - Keeps behavior consistent with live adjusted signals
    """
    df = df.copy()
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # If an adjusted 'signal' column exists (from live logic), use it for SELL
    # But we'll compute pred and buy_conf inside backtest to apply ATR gating consistently.
    df["pred_signal"] = model.predict(df[feature_cols])

    # try predict_proba if available (for buy confidence). fallback to 1.0 when not available.
    buy_conf_vals = None
    try:
        probs = model.predict_proba(df[feature_cols])
        # find column index for class '1' (BUY)
        # if model uses labels [-1,0,1] ordering may be different; handle generically
        classes = list(model.classes_)
        if 1 in classes:
            idx_buy = classes.index(1)
            buy_conf_vals = probs[:, idx_buy]
        else:
            # fallback: try last column meaning 'positive'
            buy_conf_vals = probs[:, -1]
    except Exception:
        # model doesn't support predict_proba or failed — assume neutral confidence 1.0 for buys
        buy_conf_vals = np.ones(len(df))

    df["buy_conf"] = buy_conf_vals
    # Ensure ATR exists in df (expect 'atr_14' from compute_features). compute atr_pct
    if "atr_14" not in df.columns:
        df["atr_14"] = df["close"].rolling(14).apply(lambda x: (x.max() - x.min()), raw=False).fillna(0)
    df["atr_pct"] = df["atr_14"] / df["close"].replace(0, np.nan)

    # compute true signal for accuracy (future-looking)
    df["future_return_3h"] = df["close"].shift(-3) / df["close"] - 1
    df["true_signal"] = np.where(df["future_return_3h"] > 0.01, 1,
                          np.where(df["future_return_3h"] < -0.005, -1, 0))
    df = df.dropna(subset=["true_signal"]).reset_index(drop=True)
    accuracy = (df["pred_signal"] == df["true_signal"]).mean()

    # ---- BTC sentiment (2 years max) ----
    try:
        df_sent = fetch_historical_btc_sentiment(days=730)

        df = pd.merge_asof(
            df.sort_values("timestamp"),
            df_sent,
            on="timestamp",
            direction="backward"
        )

    except Exception as e:
        print(f"⚠️ BTC sentiment unavailable, defaulting to NEUTRAL ({e})")
        df["btc_sentiment"] = 50  # neutral


    # Trading sim (long-only gating)
    equity = start_equity
    position = 0.0
    entry_price = 0.0
    entry_ts = None
    target_reached = False
    trades = []
    equity_curve = []

    for idx, row in df.iterrows():
        ts = row["timestamp"]
        price = float(row["close"])
        # model predicted signal (raw)
        raw_signal = int(row["pred_signal"])
        buy_conf = float(row["buy_conf"]) if not np.isnan(row["buy_conf"]) else 1.0
        atr_pct = float(row["atr_pct"]) if not np.isnan(row["atr_pct"]) else 0.0

        # compute dynamic buy threshold (only relevant to LONG buys)
        dynamic_thr = _dynamic_buy_threshold(atr_pct)

        # adjust threshold based on sentiment
        fng = int(row["fng"]) if not np.isnan(row["fng"]) else 50
        sentiment_penalty = sentiment_conf_penalty(fng)
        adjusted_thr = dynamic_thr + sentiment_penalty
        adjusted_thr = min(max(adjusted_thr, 0.0), 0.95)

        # BUY logic (only when flat)
        if (
            raw_signal == 1
            and position == 0
        ):

            # require ATR to be non-zero and buy_conf >= adjusted threshold
            if buy_conf >= adjusted_thr and atr_pct >= 0.0:
                cash_after_fee = equity * (1 - trade_fee)
                position = cash_after_fee / price
                entry_price = price
                entry_ts = ts
                target_reached = False
                trades.append({
                    "timestamp": ts,
                    "action": "BUY",
                    "price": price,
                    "equity_before": equity,
                    "equity_after_fee": cash_after_fee,
                    "buy_conf": buy_conf,
                    "atr_pct": atr_pct,
                    "dynamic_thr": dynamic_thr,
                    "adjusted_thr": adjusted_thr,
                    "sentiment_penalty": sentiment_penalty,
                    "btc_sentiment": fng
                })

        # If we hold a position, check forced-exit and other sells
        elif position > 0:
            gain = (price - entry_price) / entry_price if entry_price > 0 else 0.0

            # If ATR drops below the exit threshold -> forced sell (2B)
            if atr_pct <= ATR_EXIT_THRESHOLD:
                proceeds = position * price * (1 - trade_fee)
                pnl = proceeds - equity
                trades.append({
                    "timestamp": ts,
                    "action": "SELL (ATR_EXIT)",
                    "price": price,
                    "entry_price": entry_price,
                    "gain": gain,
                    "pnl": pnl,
                    "atr_pct": atr_pct,
                    "btc_sentiment": fng
                })
                equity = proceeds
                position = 0.0
                entry_price = 0.0
                entry_ts = None
                target_reached = False
                # continue to next row after exit
                equity_curve.append(equity)
                continue

            # Normal take-profit logic (as before)
            if gain >= take_profit:
                target_reached = True

            # stop-loss
            if gain <= stop_loss:
                proceeds = position * price * (1 - trade_fee)
                pnl = proceeds - equity
                trades.append({
                    "timestamp": ts,
                    "action": "SELL (STOPLOSS)",
                    "price": price,
                    "entry_price": entry_price,
                    "gain": gain,
                    "pnl": pnl,
                    "btc_sentiment": fng
                })
                equity = proceeds
                position = 0.0
                entry_price = 0.0
                entry_ts = None
                target_reached = False

            # take-profit: only sell if model isn't encouraging to continue (raw_signal != 1)
            elif target_reached and (raw_signal != 1):
                proceeds = position * price * (1 - trade_fee)
                pnl = proceeds - equity
                trades.append({
                    "timestamp": ts,
                    "action": "SELL (TAKEPROFIT)",
                    "price": price,
                    "entry_price": entry_price,
                    "gain": gain,
                    "pnl": pnl,
                    "btc_sentiment": fng
                })
                equity = proceeds
                position = 0.0
                entry_price = 0.0
                entry_ts = None
                target_reached = False

            # model SELL
            elif raw_signal == -1:
                proceeds = position * price * (1 - trade_fee)
                pnl = proceeds - equity
                trades.append({
                    "timestamp": ts,
                    "action": "SELL (MODEL)",
                    "price": price,
                    "entry_price": entry_price,
                    "gain": gain,
                    "pnl": pnl,
                    "btc_sentiment": fng
                })
                equity = proceeds
                position = 0.0
                entry_price = 0.0
                entry_ts = None
                target_reached = False

        # Track equity (current)
        current_equity = equity + position * price
        equity_curve.append(current_equity)

    # finalize
    df["equity"] = equity_curve
    trades_df = pd.DataFrame(trades)
    final_equity = equity_curve[-1] if equity_curve else start_equity

    # compute win rate properly (SELL trades)
    if not trades_df.empty:
        sell_trades = trades_df[trades_df["action"].str.contains("SELL", na=False)]
        if not sell_trades.empty:
            winning_trades = sell_trades[sell_trades["pnl"] > 0]
            win_rate = len(winning_trades) / len(sell_trades)
            winning_trades_df = winning_trades
        else:
            win_rate = 0.0
            winning_trades_df = pd.DataFrame()
    else:
        win_rate = 0.0
        winning_trades_df = pd.DataFrame()

    trade_impact = estimate_blocked_trade_cost(trades_df)

    summary = summarize_sentiment_impact(trade_impact)
    print("=== SENTIMENT IMPACT SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    fng_analysis = analyze_fng_ranges(trade_impact)
    print("\n=== FNG RANGE IMPACT ===")
    print(fng_analysis)

    return df, final_equity, trades_df, accuracy, winning_trades_df, win_rate