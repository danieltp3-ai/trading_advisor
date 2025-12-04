#!/bin/bash
set -e

HOME=/Users/dpowers01
PROJECT_ROOT="$HOME/trading_advisor"
PYTHON_BIN="$PROJECT_ROOT/venv/bin/python3"
PLIST_PATH="$HOME/Library/LaunchAgents/com.trading_advisor.hourly.plist"
WATCHDOG_PATH="$HOME/Library/LaunchAgents/com.dpowers.trading_advisor.watchdog.plist"
ENV_FILE="$PROJECT_ROOT/.env"

echo "🚀 Setting up Trading Advisor LaunchAgent..."

# Create project root if missing
mkdir -p "$PROJECT_ROOT"

# Create virtual environment if missing
if [ ! -d "$PROJECT_ROOT/venv" ]; then
  echo "🐍 Creating virtual environment..."
  python3 -m venv "$PROJECT_ROOT/venv"
fi

# Create .env file if missing
if [ ! -f "$ENV_FILE" ]; then
  echo "🔐 Creating .env template..."
  cat <<EOF > "$ENV_FILE"
# Environment variables for Trading Advisor
# Never commit this file or share your private key!
WALLET_PRIVATE_KEY="your_private_key_here"
EOF
  chmod 600 "$ENV_FILE"
  echo "⚠️  Please update $ENV_FILE with your actual private key."
fi

# Install dependencies
echo "📦 Installing dependencies..."
source "$PROJECT_ROOT/venv/bin/activate"
pip install -U pip
pip install lightgbm pandas numpy requests joblib matplotlib python-dotenv
deactivate

# Create LaunchAgent plist
echo "🧩 Creating LaunchAgent plist at $PLIST_PATH..."
cat <<EOF > "$PLIST_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.trading_advisor.hourly</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_BIN</string>
        <string>-u</string>
        <string>$PROJECT_ROOT/scripts/main.py</string>
    </array>

    <!-- Load environment variables from .env -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>DOTENV_PATH</key>
        <string>$ENV_FILE</string>
    </dict>

    <!-- Run every hour at minute 0 -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>WakeOnDemand</key>
    <true/>
    <key>TimeOut</key>
    <integer>900</integer>

    <!-- Logging -->
    <key>StandardOutPath</key>
    <string>/Users/dpowers01/trading_advisor/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/dpowers01/trading_advisor/logs/stderr.log</string>
    <key>ThrottleInterval</key>
    <integer>60</integer>

</dict>
</plist>
EOF

# Create Watchdog plist
echo "🧩 Creating Watchdog plist at $WATCHDOG_PATH..."
cat <<EOF > "$WATCHDOG_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.dpowers.trading_advisor.watchdog</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>/Users/dpowers01/trading_advisor/scripts/watchdog.sh</string>
  </array>

  <key>StartInterval</key>
  <integer>600</integer> <!-- every 10 minutes -->

  <key>StandardOutPath</key>
  <string>/Users/dpowers01/trading_advisor/logs/watchdog_stdout.log</string>

  <key>StandardErrorPath</key>
  <string>/Users/dpowers01/trading_advisor/logs/watchdog_stderr.log</string>

  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>
EOF


# Ensure permissions
mkdir -p "$PROJECT_ROOT/logs"
chmod 644 "$PLIST_PATH"
chmod 644 "$WATCHDOG_PATH"

# Load and start agent
echo "🔄 Loading LaunchAgent..."
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load "$PLIST_PATH"

# Load and start watchdog
echo "🔄 Loading Watchdog..."
launchctl unload "$WATCHDOG_PATH" 2>/dev/null || true
launchctl load "$WATCHDOG_PATH"

echo "✅ Trading Advisor LaunchAgent successfully installed!"
echo "It will run every hour on the hour while you are logged in."
echo "Logs: $PROJECT_ROOT/logs/"
