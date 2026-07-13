import os
import requests

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def send_discord_alert(signal: str, price: float, win_rate: float, equity: float):
    if not WEBHOOK_URL:
        print("⚠️ No Discord webhook URL found in environment variables!")
        return

    emoji = {
        "BUY": "🔵",
        "SELL": "🔴",
        "HOLD": "⚪"
    }.get(signal, "ℹ️")

    message = (
        f"**{emoji} MAMO Signal Alert**\n"
        f"Signal: **{signal}**\n"
        f"Price: `${price:.5f}`\n"
        f"Win Rate: `{win_rate:.2f}%`\n"
        f"Equity: `${equity:,.2f}`"
    )

    payload = {"content": message}

    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        if response.status_code == 204:
            print("📨 Discord alert sent successfully!")
        else:
            print(f"⚠️ Discord error: {response.status_code} → {response.text}")
    except Exception as e:
        print(f"❌ Discord alert failed: {e}")
