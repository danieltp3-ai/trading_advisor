from pathlib import Path
import os

# ---------------------------------------------------------------------
# Core asset / directories
# ---------------------------------------------------------------------
COIN_ID = "mamo"   # Coinbase asset ID

BASE_DIR = Path("/Users/dpowers01/trading_advisor")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

CACHE_FILE = DATA_DIR / f"{COIN_ID}_hourly_cache.parquet"
MODEL_PATH = MODELS_DIR / f"{COIN_ID}_hourly_model.pkl"
LOG_PATH = DATA_DIR / f"{COIN_ID}_signals_log.csv"

# Coinbase OHLCV endpoint (public)
COINBASE_OHLCV_URL = (
    "https://api.exchange.coinbase.com/products/"
    f"{COIN_ID.upper()}-USD/candles"
)

# How long cached OHLCV is valid (1 hour)
CACHE_TTL_HOURS = 1.0

# Training params
TRAIN_TEST_SPLIT = 0.8 # 80% train / 20% test split
HORIZON_HOURS = 3     # predict 3 hours ahead

# ---------------------------------------------------------------------
# Alerts: Email → SMS
# ---------------------------------------------------------------------
EMAIL_SENDER = os.getenv("ALERT_EMAIL_SENDER") or "jeffyspeffy@gmail.com"
EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD") or "xkpn fqfb yknl vopa"

# Verizon SMS gateway by default
TO_SMS = os.getenv("ALERT_SMS") or "4439753869@vtext.com"

# ---------------------------------------------------------------------
# Trading defaults (backtests + live)
# ---------------------------------------------------------------------
TRADE_FEE = 0.01        # 1%
TAKE_PROFIT = 0.06      # +6%
STOP_LOSS = -0.10       # -10%

# ATR / confidence adaptive params (tweak these)
# base minimum probability required for a BUY (before ATR scaling)
BASE_BUY_CONF = 0.55

# ATR expressed as fraction of price (atr / close). ATR_SCALE defines typical ATR
# used to compute dynamic adjustment. e.g. ATR_SCALE=0.02 means 2% ATR maps to
# full allowed adjustment.
ATR_SCALE = 0.02

# How much the threshold can change (positive value). The dynamic threshold becomes:
# threshold = BASE_BUY_CONF + clamp((ATR_SCALE - atr_pct)/ATR_SCALE * MAX_CONF_ADJUST, -MAX_CONF_ADJUST, MAX_CONF_ADJUST)
MAX_CONF_ADJUST = 0.20

# Hard floor/ceiling for threshold
MIN_BUY_CONF = 0.30
MAX_BUY_CONF = 0.90

# ATR exit threshold (if atr_pct drops below this while holding -> forced exit)
ATR_EXIT_THRESHOLD = 0.003  # 0.3%

# ---------------------------------------------------------------------
# Ensure folders exist
# ---------------------------------------------------------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
