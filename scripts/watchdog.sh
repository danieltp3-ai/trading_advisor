LOGFILE="/Users/dpowers01/trading_advisor/logs/watchdog.log"
AGENT_LOG="/Users/dpowers01/trading_advisor/logs/stdout.log"
PLIST="/Users/dpowers01/Library/LaunchAgents/com.trading_advisor.hourly.plist"

# If log hasn't been updated in the last 2 hours, reload agent
if [ $(find "$AGENT_LOG" -mmin +70 2>/dev/null | wc -l) -gt 0 ]; then
  echo "$(date): ⚠️ main.py appears inactive — restarting LaunchAgent..." >> "$LOGFILE"
  /usr/bin/env launchctl unload "$PLIST" 2>/dev/null
  /usr/bin/env launchctl load "$PLIST"
else
  echo "$(date): ✅ main.py ran recently — no action needed." >> "$LOGFILE"
fi
