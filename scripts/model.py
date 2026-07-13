import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from config import MODEL_PATH, TRAIN_TEST_SPLIT, HORIZON_HOURS
from fetch_data import fetch_coinbase_data  # or the cached wrapper you use
from features import compute_features       # hourly features function you already have

def _make_dynamic_thresholds(df, lookback_hours=48, up_mult=1.5, down_mult=1.0):
    """
    Compute adaptive up/down thresholds using recent volatility.
    lookback_hours: number of hourly candles to compute rolling std (e.g. 48 = 2 days)
    up_mult/down_mult: multipliers for threshold scaling
    """
    vol = df["close"].pct_change().rolling(lookback_hours).std()
    up_thresh = vol * up_mult
    down_thresh = -vol * down_mult
    return up_thresh, down_thresh

def train_or_load_model(feature_cols=None, retrain=False):
    """
    Train (or load) an hourly model with:
      - dynamic thresholds for target labeling
      - chronological train/validation split (no leakage)
    Returns: (model, feature_cols)
    """
    if feature_cols is None:
        feature_cols = [
            "return_1h", "ma_5", "ma_20",
            "volatility_10", "rsi_14",
            "macd", "macd_signal", "volume_roc", "atr_14"
        ]

    # load existing model unless retrain requested
    if MODEL_PATH.exists() and not retrain:
        print("✅ Loaded existing model.")
        model = joblib.load(MODEL_PATH)
        return model, feature_cols

    print("🧠 Training model on hourly dataset (no leakage)…")

    # fetch hourly history (1 year default)
    df = fetch_coinbase_data(days=365)
    df = compute_features(df)   # ensure features present; drops initial NANs

    # --- Targets: adaptive thresholds based on recent vol ---
    df = df.sort_values("timestamp").reset_index(drop=True)
    up_thresh, down_thresh = _make_dynamic_thresholds(df, lookback_hours=48, up_mult=1.5, down_mult=1.0)
    df["up_thresh"] = up_thresh
    df["down_thresh"] = down_thresh

    # future return over HORIZON_HOURS
    horizon = int(HORIZON_HOURS)  # typically 3
    df["future_return_3h"] = df["close"].shift(-horizon) / df["close"] - 1

    # label: 1 = buy, -1 = sell, 0 = hold based on adaptive thresholds
    df["target_dir"] = np.where(
        df["future_return_3h"] > df["up_thresh"], 1,
        np.where(df["future_return_3h"] < df["down_thresh"], -1, 0)
    )

    # drop the last 'horizon' rows (they peek into the future) and NA rows
    df = df.iloc[:-horizon].dropna().reset_index(drop=True)

    print("Prepared training data:", df.shape)
    print(df[["timestamp", "close", "future_return_3h", "up_thresh", "down_thresh", "target_dir"]].tail(3))

    # chronological split
    cutoff = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:cutoff]
    test_df = df.iloc[cutoff:]

    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = train_df["target_dir"]
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_test = test_df["target_dir"]

    # choose LightGBM if available, otherwise fallback to RandomForest
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH, protocol=4)
    print(f"💾 Model saved → {MODEL_PATH}")

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_test, y_test)
    print(f"✅ Train acc: {train_acc*100:.2f}% | Val acc: {val_acc*100:.2f}%")

    return model, feature_cols
