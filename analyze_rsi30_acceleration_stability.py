
import numpy as np
import pandas as pd
from pathlib import Path

SIGNAL_FILE = Path("data/BTCUSDT_signals.csv")
PRICE_15M_FILE = Path("data/BTCUSDT_15m.csv")
PRICE_1M_FILE = Path("data/BTCUSDT_1m.csv")
FEATURE_FILE = Path("data/backtest_results/strategy_development_2021_2023.csv")
OUTPUT_DIR = Path("data/backtest_results")

DEV_START = pd.Timestamp("2021-01-01", tz="UTC")
DEV_END = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")

RSI_LEVEL = 70.0
TARGET_PCT = -2.0
STOP_PCT = 2.0
HORIZON_HOURS = 24

# Pre-declared, deliberately small threshold grid.
THRESHOLDS = [0.0, 2.0, 4.0, 6.0, 8.0]


def load_ohlcv(path):
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_rsi(price_15m):
    cols = ["RSI_15m", "RSI_30m", "RSI_1h", "RSI_4h"]
    if all(c in price_15m.columns for c in cols):
        return price_15m[["open_time"] + cols].copy()

    rsi = pd.read_csv("data/BTCUSDT_RSI.csv")
    rsi["open_time"] = pd.to_datetime(rsi["open_time"], utc=True)
    missing = [c for c in cols if c not in rsi.columns]
    if missing:
        raise ValueError(f"BTCUSDT_RSI.csv missing columns: {missing}")
    return rsi[["open_time"] + cols].copy()


def find_gaps(price_1m):
    ts = price_1m["open_time"].sort_values().reset_index(drop=True)
    diffs = ts.diff().dt.total_seconds().div(60)
    return [(ts.iloc[i-1], ts.iloc[i]) for i in range(1, len(ts)) if diffs.iloc[i] > 1]


def gap_between(gaps, start, end):
    for before, after in gaps:
        if after > start and before < end:
            return True
    return False


def backtest(entry_time, price_1m, gaps):
    if pd.isna(entry_time):
        return {"status": "NO_ENTRY"}

    arr = price_1m["open_time"].array
    i = arr.searchsorted(entry_time, side="left")

    if i >= len(price_1m) or price_1m.loc[i, "open_time"] != entry_time:
        return {"status": "NO_ENTRY_DATA"}

    horizon_end = entry_time + pd.Timedelta(hours=HORIZON_HOURS)

    if gap_between(gaps, entry_time, horizon_end):
        return {"status": "GAP_BLOCKED"}

    j = arr.searchsorted(horizon_end, side="right")
    w = price_1m.iloc[i:j].copy()

    if w.empty:
        return {"status": "NO_DATA"}

    entry = float(w.iloc[0]["open"])
    target = entry * (1 + TARGET_PCT / 100)
    stop = entry * (1 + STOP_PCT / 100)

    lows = w["low"].to_numpy(float)
    highs = w["high"].to_numpy(float)

    target_idx = np.flatnonzero(lows <= target)
    stop_idx = np.flatnonzero(highs >= stop)

    t = int(target_idx[0]) if len(target_idx) else None
    s = int(stop_idx[0]) if len(stop_idx) else None

    if t is not None and s is not None:
        if t < s:
            outcome = "TARGET_FIRST"
        elif s < t:
            outcome = "STOP_FIRST"
        else:
            outcome = "AMBIGUOUS"
    elif t is not None:
        outcome = "TARGET_FIRST"
    elif s is not None:
        outcome = "STOP_FIRST"
    else:
        outcome = "NEITHER"

    last_close = float(w.iloc[-1]["close"])
    return {
        "status": "COMPLETED",
        "outcome": outcome,
        "raw_short_return_pct": (entry / last_close - 1) * 100,
        "mfe_pct": (entry / float(np.min(lows)) - 1) * 100,
        "mae_pct": (entry / float(np.max(highs)) - 1) * 100,
    }


def prepare_signal_entries():
    signals = pd.read_csv(SIGNAL_FILE)
    signals["open_time"] = pd.to_datetime(signals["open_time"], utc=True)
    signals = signals[signals["open_time"].between(DEV_START, DEV_END)].copy()
    signals = signals.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    p15 = load_ohlcv(PRICE_15M_FILE)
    rsi = load_rsi(p15)
    p15 = p15.merge(rsi, on="open_time", how="left", suffixes=("", "_rsi"))

    if "RSI_15m_rsi" in p15.columns:
        p15["RSI_15m"] = p15["RSI_15m"].fillna(p15["RSI_15m_rsi"])
        p15.drop(columns=["RSI_15m_rsi"], inplace=True)

    p15["rsi_cross"] = (
        p15["RSI_15m"].notna()
        & p15["RSI_15m"].lt(RSI_LEVEL)
        & p15["RSI_15m"].shift(1).ge(RSI_LEVEL)
    )

    features = pd.read_csv(FEATURE_FILE)
    features["signal_time"] = pd.to_datetime(features["signal_time"], utc=True)
    features = features[features["signal_time"].between(DEV_START, DEV_END)].copy()

    data = signals.merge(
        features[["signal_time", "RSI30_delta_4"]],
        left_on="open_time",
        right_on="signal_time",
        how="left",
        validate="one_to_one",
    ).drop(columns=["signal_time"])

    idx_map = {t: i for i, t in enumerate(p15["open_time"])}

    rows = []
    for _, row in data.iterrows():
        idx = idx_map.get(row["open_time"])
        entry = pd.NaT

        if idx is not None:
            for j in range(idx + 1, len(p15)):
                # Never cross a real 15m data gap while waiting for confirmation.
                if p15.loc[j, "open_time"] - p15.loc[j-1, "open_time"] != pd.Timedelta(minutes=15):
                    break
                if bool(p15.loc[j, "rsi_cross"]):
                    if j + 1 < len(p15):
                        entry = p15.loc[j + 1, "open_time"]
                    break

        rows.append({
            "signal_time": row["open_time"],
            "RSI30_delta_4": pd.to_numeric(row["RSI30_delta_4"], errors="coerce"),
            "entry_time": entry,
        })

    return pd.DataFrame(rows)


def summarize(df, label):
    completed = df[df["status"].eq("COMPLETED")].copy()
    target = completed["outcome"].eq("TARGET_FIRST")
    stop = completed["outcome"].eq("STOP_FIRST")
    resolved = target | stop

    return {
        "group": label,
        "signals": len(df),
        "completed": len(completed),
        "target_first": int(target.sum()),
        "stop_first": int(stop.sum()),
        "neither": int(completed["outcome"].eq("NEITHER").sum()),
        "ambiguous": int(completed["outcome"].eq("AMBIGUOUS").sum()),
        "resolved_win_rate": (
            float(target[resolved].mean()) if resolved.any() else np.nan
        ),
        "mean_raw_short_return_pct": completed["raw_short_return_pct"].mean(),
        "median_raw_short_return_pct": completed["raw_short_return_pct"].median(),
        "median_mfe_pct": completed["mfe_pct"].median(),
        "median_mae_pct": completed["mae_pct"].median(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = prepare_signal_entries()
    p1 = load_ohlcv(PRICE_1M_FILE)
    gaps = find_gaps(p1)

    # Backtest the RSI-cross entry for all development signals once.
    tested = []
    for _, row in entries.iterrows():
        bt = backtest(row["entry_time"], p1, gaps)
        tested.append({**row.to_dict(), **bt})

    trades = pd.DataFrame(tested)
    trades["rsi_accel_q"] = pd.qcut(
        trades["RSI30_delta_4"],
        q=4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
        duplicates="drop",
    )

    # Remove any accidental rows without the feature from bucket calculations.
    quantile_groups = []
    for label, group in trades.groupby("rsi_accel_q", observed=True):
        quantile_groups.append(summarize(group, str(label)))

    quantile_summary = pd.DataFrame(quantile_groups)

    threshold_groups = []
    for threshold in THRESHOLDS:
        group = trades[trades["RSI30_delta_4"] >= threshold].copy()
        threshold_groups.append(summarize(group, f"RSI30_delta_4 >= {threshold:g}"))

    threshold_summary = pd.DataFrame(threshold_groups)

    print("=" * 72)
    print("DEVELOPMENT-ONLY RSI30 ACCELERATION STABILITY TEST")
    print("=" * 72)
    print(f"Development signals: {len(trades)}")
    print()
    print("QUARTILE BEHAVIOR")
    print(quantile_summary.to_string(index=False))
    print()
    print("PRE-DECLARED THRESHOLD GRID")
    print(threshold_summary.to_string(index=False))

    q_path = OUTPUT_DIR / "rsi30_acceleration_quartile_summary_development.csv"
    t_path = OUTPUT_DIR / "rsi30_acceleration_threshold_summary_development.csv"
    d_path = OUTPUT_DIR / "rsi30_acceleration_trade_dataset_development.csv"

    quantile_summary.to_csv(q_path, index=False)
    threshold_summary.to_csv(t_path, index=False)
    trades.to_csv(d_path, index=False)

    print()
    print("OUTPUTS")
    print(q_path)
    print(t_path)
    print(d_path)
    print()
    print("RULE")
    print("This is still development analysis. Do not use 2024/2025 to modify")
    print("the threshold grid or reinterpret a threshold after seeing validation.")


if __name__ == "__main__":
    main()
