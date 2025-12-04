import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Render to files instead of GUI windows
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
from fetch_data import fetch_coinbase_data
from features import compute_features
from model import train_or_load_model
from backtest import backtest_with_trade_log_and_accuracy
from alert import send_sms_via_email, send_daily_summary
from utils.signal_logic import _dynamic_buy_threshold, _trend_direction
from config import LOG_PATH, MODEL_PATH
from dotenv import load_dotenv  # install with: pip install python-dotenv
import subprocess
from pathlib import Path


def generate_latest_signal_and_backtest(days_window=60):
    print("🚀 Running MAMO Hourly Signal Generator + Backtest...")
    start_time = datetime.datetime.now()
    print(f"🕐 MAMO daemon started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    load_dotenv(os.getenv("DOTENV_PATH", ".env"))

    PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY")
    if not PRIVATE_KEY:
        raise ValueError("❌ WALLET_PRIVATE_KEY not found in environment variables.")

    # Load model + feature set
    model, feature_cols = train_or_load_model()

    # Fetch + features
    df = compute_features(fetch_coinbase_data(days=days_window)).dropna().reset_index(drop=True)

    # --- Model probability for LONG direction ---
    probs = model.predict_proba(df[feature_cols])
    idx_buy = list(model.classes_).index(1) if 1 in model.classes_ else -1
    df["buy_conf"] = probs[:, idx_buy].astype(float)

    # --- Trend Filter ---
    df["trend"] = df.apply(lambda r: _trend_direction(r), axis=1)

    # --- ATR volatility regime ---
    atr = df["atr_14"].iloc[-1]
    close = df["close"].iloc[-1]
    atr_pct = atr / close if close > 0 else 0.0

    # Dynamic threshold
    dynamic_thr = _dynamic_buy_threshold(atr_pct)

    # --- Generate Hourly Trading Signals ---
    signal = 0
    latest_conf = df["buy_conf"].iloc[-1]
    latest_trend = df["trend"].iloc[-1]

    if latest_conf >= dynamic_thr and latest_trend == 1:
        signal = 1  # Long entry or maintain long
    elif latest_conf <= (1 - dynamic_thr) and latest_trend == -1:
        signal = -1  # Short entry or maintain short
    else:
        signal = 0  # Stay flat if uncertain / no edge

    df["signal"] = 0
    df.loc[df.index[-1], "signal"] = signal

    # Debug print
    print(f"DEBUG: conf={latest_conf:.3f}, thr={dynamic_thr:.3f}, trend={latest_trend}, signal={signal}")

    # Backtest for full window context
    df_bt, final_equity, trades_df, accuracy, winning_trades, win_rate = \
        backtest_with_trade_log_and_accuracy(df, model, feature_cols)

    # Log + charts + notifications
    latest = df.iloc[-1]
    ts, price = latest["timestamp"], latest["close"]
    signal_str = {1: "BUY", -1: "SELL", 0: "HOLD"}[signal]
    print(f"💡 Latest Signal: {signal_str} at ${price:.6f}")

    print(f"💰 Final equity: ${final_equity:.2f}")
    print(f"🎯 Accuracy: {accuracy*100:.2f}% | 📈 Win rate: {win_rate*100:.2f}% ({len(trades_df)} trades)")

    log_entry = pd.DataFrame([{
        "timestamp": ts, "signal": signal_str, "price": price,
        "final_equity": final_equity, "accuracy": accuracy
    }])
    if LOG_PATH.exists():
        df_log = pd.concat([pd.read_csv(LOG_PATH), log_entry], ignore_index=True)
    else:
        df_log = log_entry
    df_log.to_csv(LOG_PATH, index=False)

    # Existing call
    send_sms_via_email(signal_str, price, ts, accuracy, final_equity)

    # New addition (runs only at noon)
    send_daily_summary(final_equity, len(trades_df), win_rate)

    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), constrained_layout=True)

    # --- Full duration plot ---
    ax1 = axes[0]
    ax1.plot(df_bt["timestamp"], df_bt["equity"], color="tab:blue", label="Equity Curve", linewidth=2)
    ax1.set_ylabel("Equity ($)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(df_bt["timestamp"], df_bt["close"], color="tab:orange", label="Coin Price", linewidth=2, alpha=0.7)
    ax2.set_ylabel("Price (USD)", color="tab:orange")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title("📈 MAMO Model Performance — Full Backtest Period")

    # --- Last 7 days plot ---
    last_week = df_bt[df_bt["timestamp"] >= (df_bt["timestamp"].max() - pd.Timedelta(days=7))]
    ax3 = axes[1]
    ax3.plot(last_week["timestamp"], last_week["equity"], color="tab:blue", label="Equity Curve", linewidth=2)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Equity ($)", color="tab:blue")
    ax4 = ax3.twinx()
    ax4.plot(last_week["timestamp"], last_week["close"], color="tab:orange", label="Coin Price", linewidth=2, alpha=0.7)
    ax4.set_ylabel("Price (USD)", color="tab:orange")
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.set_title("📊 MAMO Model Performance — Last 7 Days")

    # Save the figure to files instead of showing
    full_chart_path = "/Users/dpowers01/trading_advisor/logs/mamo_full_backtest.png"
    last7_chart_path = "/Users/dpowers01/trading_advisor/logs/mamo_last7days.png"

    # Save full duration plot separately
    fig_full, ax_full = plt.subplots(figsize=(12,6))
    ax_full.plot(df_bt["timestamp"], df_bt["equity"], color="tab:blue", linewidth=2, label="Equity Curve")
    ax_full.set_ylabel("Equity ($)", color="tab:blue")
    ax_full2 = ax_full.twinx()
    ax_full2.plot(df_bt["timestamp"], df_bt["close"], color="tab:orange", linewidth=2, alpha=0.7, label="Coin Price")
    ax_full2.set_ylabel("Price (USD)", color="tab:orange")
    ax_full.legend(loc="upper left")
    ax_full.grid(True, linestyle="--", alpha=0.5)
    ax_full.set_title("MAMO Model Performance — Full Backtest Period")
    fig_full.tight_layout()
    fig_full.savefig(full_chart_path)
    plt.close(fig_full)

    # Save last 7 days plot separately
    fig_week, ax_week = plt.subplots(figsize=(12,6))
    ax_week.plot(last_week["timestamp"], last_week["equity"], color="tab:blue", linewidth=2, label="Equity Curve")
    ax_week.set_xlabel("Time")
    ax_week.set_ylabel("Equity ($)", color="tab:blue")
    ax_week2 = ax_week.twinx()
    ax_week2.plot(last_week["timestamp"], last_week["close"], color="tab:orange", linewidth=2, alpha=0.7, label="Coin Price")
    ax_week2.set_ylabel("Price (USD)", color="tab:orange")
    ax_week.legend(loc="upper left")
    ax_week.grid(True, linestyle="--", alpha=0.5)
    ax_week.set_title("MAMO Model Performance — Last 7 Days")
    fig_week.tight_layout()
    fig_week.savefig(last7_chart_path)
    plt.close(fig_week)

    print(f"✅ Saved full backtest chart → {full_chart_path}")
    print(f"✅ Saved last 7 days chart → {last7_chart_path}")

    end_time = datetime.datetime.now()
    print(f"✅ MAMO daemon finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("—" * 60)

    return df_bt, trades_df, final_equity

def write_heartbeat():
    heartbeat_path = "/Users/dpowers01/trading_advisor/logs/heartbeat.log"
    os.makedirs(os.path.dirname(heartbeat_path), exist_ok=True)
    with open(heartbeat_path, "a") as f:
        f.write(f"✅ MAMO ran successfully at {datetime.datetime.now().now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def get_private_key():
    """
    Securely retrieves the private key.
    Tries macOS Keychain first, then falls back to .env.
    """
    # Try macOS Keychain
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-a", os.getlogin(), "-s", "TRADING_ADVISOR_PRIVATE_KEY", "-w"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            key = result.stdout.strip()
            if key:
                return key
    except Exception as e:
        print(f"[WARN] Could not access Keychain: {e}")

    # Fallback: .env file
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        key = os.getenv("WALLET_PRIVATE_KEY")
        if key:
            return key

    raise ValueError("❌ Private key not found in Keychain or .env file.")

if __name__ == "__main__":
    df_bt, trades_df, final_equity = generate_latest_signal_and_backtest()
    write_heartbeat()

    from datetime import datetime
    current_hour = datetime.now().hour

    # Optional: put Mac to sleep after running at specific hours
    # print("💤 Job run complete — putting Mac to sleep.")
    # os.system("pmset sleepnow")
    import sys
    sys.exit(0)
