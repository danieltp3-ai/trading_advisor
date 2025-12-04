import smtplib
import json
import os
from datetime import datetime
from email.mime.text import MIMEText
from config import EMAIL_SENDER, EMAIL_PASSWORD, TO_SMS

STATE_FILE = "/Users/dpowers01/trading_advisor/state.json"

ALERT_HOURS = [8, 10, 12, 14, 16, 18, 20, 22]  # 9AM, 12PM, 3PM, 6PM, 9PM

def send_sms(subject, body):
    """Internal helper for sending text messages."""
    if not (EMAIL_SENDER and EMAIL_PASSWORD and TO_SMS):
        print("⚠️ SMS not configured; set ALERT_EMAIL_SENDER, ALERT_EMAIL_PASSWORD, ALERT_SMS")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_SENDER
    msg["To"] = TO_SMS

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, TO_SMS, msg.as_string())
        print(f"📱 Sent SMS: {subject}")
    except Exception as e:
        print("⚠️ SMS send failed:", e)


def send_sms_via_email(signal, price, timestamp, accuracy, final_equity):
    """Send trading signal alerts, but only at specific hours."""
    current_hour = datetime.now().hour
    if signal != 0:
        if current_hour not in ALERT_HOURS:
            print(f"⏰ Skipping alert ({current_hour}h not in {ALERT_HOURS})")
            return

    subject = f"MAMO {signal} ALERT ({accuracy*100:.1f}%)"
    body = (
        f"MAMO Signal: {signal}\n"
        f"Price: {price:.6f}\n"
        f"Time: {timestamp}\n"
        f"Model Accuracy: {accuracy*100:.2f}%\n"
        f"Equity: ${final_equity:.2f}"
    )
    send_sms(subject, body)


def send_daily_summary(current_equity, total_trades, win_rate):
    """Send a noon summary with daily P/L change."""
    now = datetime.now()
    if now.hour != 12:
        return  # only at noon

    # Load previous equity
    prev_equity = None
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                prev_equity = json.load(f).get("equity")
        except Exception:
            prev_equity = None

    # Save current equity for tomorrow
    with open(STATE_FILE, "w") as f:
        json.dump({"equity": current_equity}, f)

    # Compute change
    if prev_equity is not None:
        change = current_equity - prev_equity
        pct = (change / prev_equity) * 100 if prev_equity != 0 else 0
        change_str = f"{'+' if change >= 0 else ''}${change:.2f} ({pct:+.2f}%)"
    else:
        change_str = "N/A (first day)"

    subject = "📊 MAMO Daily Summary"
    body = (
        f"Total Equity: ${current_equity:.2f}\n"
        f"Daily P/L: {change_str}\n"
        f"Total Trades: {total_trades}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Time: {now.strftime('%Y-%m-%d %H:%M')}"
    )

    send_sms(subject, body)
