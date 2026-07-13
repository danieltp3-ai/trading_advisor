import pandas as pd
import datetime
from zoneinfo import ZoneInfo
import os
from dotenv import load_dotenv

from fetch_data import fetch_coinbase_data
from features import compute_features
from model import train_or_load_model
from discord_alert import send_discord_alert
from config import LOG_PATH
from trading_core import evaluate_and_update_state
from sentiment.btc_sentiment import get_btc_fng


def generate_signal():
    now_est = datetime.datetime.now(ZoneInfo("America/New_York"))
    print(f"🚀 MAMO Cloud Signal Check @ {now_est} EST")

    # Load environment variables
    load_dotenv(os.getenv("DOTENV_PATH", ".env"))

    # Load trained model
    model, feature_cols = train_or_load_model()

    # --------------------------------------------------
    # Fetch & prepare data (CLOSED candles only)
    # --------------------------------------------------
    df_raw = fetch_coinbase_data(days=60)

    # Drop the most recent candle if it's still forming
    last_ts = df_raw["timestamp"].max()
    if last_ts >= pd.Timestamp.utcnow().floor("1h"):
        df_raw = df_raw.iloc[:-1]

    df = compute_features(df_raw).dropna().reset_index(drop=True)

    if df.empty:
        print("⚠️ No usable candles after feature computation — aborting run.")
        return

    # Use last CLOSED candle timestamp for logging
    candle_ts = df["timestamp"].iloc[-1]

    # --------------------------------------------------
    # Core trading logic (stateful)
    # --------------------------------------------------
    action, new_state, trade = evaluate_and_update_state(df, model, feature_cols)
    btc_sentiment = get_btc_fng()
    print(f"🌍 BTC Sentiment: {btc_sentiment}")

    signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
    signal_str = signal_map[action]

    print(
        f"💡 Signal: {signal_str} | "
        f"Equity=${new_state['equity']:.2f} | "
        f"Candle={candle_ts}"
    )

    # --------------------------------------------------
    # Alerts
    # --------------------------------------------------
    if trade:
        send_discord_alert(
            signal_str=signal_str,
            btc_sentiment=btc_sentiment,
            equity=new_state["equity"],
            trade_pnl=trade.get("pnl", 0.0),
            price=trade.get("price"),
            reason=trade.get("reason"),
        )
    else:
        send_discord_alert(
            signal_str=signal_str,
            btc_sentiment=btc_sentiment,
            equity=new_state["equity"],
            trade_pnl=0.0,
        )

    # --------------------------------------------------
    # Logging (candle-aligned)
    # --------------------------------------------------
    log_entry = {
        "timestamp": candle_ts,
        "signal": signal_str,
        "equity": new_state["equity"],
    }

    if LOG_PATH.exists():
        df_log = pd.concat(
            [pd.read_csv(LOG_PATH), pd.DataFrame([log_entry])],
            ignore_index=True,
        )
    else:
        df_log = pd.DataFrame([log_entry])

    df_log.to_csv(LOG_PATH, index=False)
    print("✔ Logged signal")

    print("✨ Run complete\n" + ("-" * 50))


if __name__ == "__main__":
    generate_signal()

