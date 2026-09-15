
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
    return rsi[["open_time"] + cols].copy()


def find_gaps(price_1m):
    ts = price_1m["open_time"].sort_values().reset_index(drop=True)
    diff_min = ts.diff().dt.total_seconds().div(60)
    return [(ts.iloc[i-1], ts.iloc[i]) for i in range(1, len(ts)) if diff_min.iloc[i] > 1]


def gap_between(gaps, start, end):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
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

    tpos = np.flatnonzero(lows <= target)
    spos = np.flatnonzero(highs >= stop)

    t = int(tpos[0]) if len(tpos) else None
    s = int(spos[0]) if len(spos) else None

    if t is not None and s is not None:
        if t < s:
            outcome = "TARGET_FIRST"
        elif s < t:
            outcome = "STOP_FIRST"
        else:
            outcome = "AMBIGUOUS"
        outcome_pos = min(t, s)
    elif t is not None:
        outcome = "TARGET_FIRST"
        outcome_pos = t
    elif s is not None:
        outcome = "STOP_FIRST"
        outcome_pos = s
    else:
        outcome = "NEITHER"
        outcome_pos = None

    last_close = float(w.iloc[-1]["close"])
    raw_short_return = (entry / last_close - 1) * 100

    best_low = float(np.min(lows))
    worst_high = float(np.max(highs))

    return {
        "status": "COMPLETED",
        "entry_price": entry,
        "outcome": outcome,
        "raw_short_return_pct": raw_short_return,
        "mfe_pct": (entry / best_low - 1) * 100,
        "mae_pct": (entry / worst_high - 1) * 100,
        "minutes_to_outcome": (
            (w.iloc[outcome_pos]["open_time"] - entry_time).total_seconds() / 60
            if outcome_pos is not None else np.nan
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    signals = pd.read_csv(SIGNAL_FILE)
    signals["open_time"] = pd.to_datetime(signals["open_time"], utc=True)
    signals = signals[
        signals["open_time"].between(DEV_START, DEV_END)
    ].sort_values("open_time").reset_index(drop=True)

    p15 = load_ohlcv(PRICE_15M_FILE)
    rsi = load_rsi(p15)
    p15 = p15.merge(rsi, on="open_time", how="left", suffixes=("", "_rsi"))

    if "RSI_15m_rsi" in p15.columns:
        p15["RSI_15m"] = p15["RSI_15m"].fillna(p15["RSI_15m_rsi"])
        p15.drop(columns=["RSI_15m_rsi"], inplace=True)

    # Confirmation candle: first 15m candle after original signal whose RSI
    # crosses from >=70 to <70.
    p15["rsi_cross"] = (
        p15["RSI_15m"].notna()
        & p15["RSI_15m"].lt(RSI_LEVEL)
        & p15["RSI_15m"].shift(1).ge(RSI_LEVEL)
    )

    # Development-only feature thresholds.
    f = pd.read_csv(FEATURE_FILE)
    f["signal_time"] = pd.to_datetime(f["signal_time"], utc=True)
    f = f[f["signal_time"].between(DEV_START, DEV_END)].copy()

    rsi_q75 = pd.to_numeric(f["RSI30_delta_4"], errors="coerce").quantile(0.75)
    ret12_q25 = pd.to_numeric(f["return_12h_pct"], errors="coerce").quantile(0.25)

    sig = signals.merge(
        f[["signal_time", "RSI30_delta_4", "return_12h_pct"]],
        left_on="open_time",
        right_on="signal_time",
        how="left",
        validate="one_to_one",
    ).drop(columns="signal_time")

    idx_map = {t: i for i, t in enumerate(p15["open_time"])}

    # Find confirmation entry for every original signal once.
    rows = []
    for _, s in sig.iterrows():
        signal_time = s["open_time"]
        idx = idx_map.get(signal_time)
        confirmation = None

        if idx is not None:
            for j in range(idx + 1, len(p15)):
                if p15.loc[j, "open_time"] - p15.loc[j-1, "open_time"] != pd.Timedelta(minutes=15):
                    break
                if bool(p15.loc[j, "rsi_cross"]):
                    if j + 1 < len(p15):
                        confirmation = p15.loc[j + 1, "open_time"]
                    break

        rows.append({
            "signal_time": signal_time,
            "b_entry_time": confirmation,
            "RSI30_delta_4": pd.to_numeric(s["RSI30_delta_4"], errors="coerce"),
            "return_12h_pct": pd.to_numeric(s["return_12h_pct"], errors="coerce"),
            "rsi30_high": (
                pd.notna(s["RSI30_delta_4"]) and s["RSI30_delta_4"] >= rsi_q75
            ),
            "return12_low": (
                pd.notna(s["return_12h_pct"]) and s["return_12h_pct"] <= ret12_q25
            ),
        })

    base = pd.DataFrame(rows)

    # D1 and D2 are component tests; D3 is their locked intersection.
    candidates = {
        "D1_RSI_ACCEL_PLUS_CROSS": base["rsi30_high"],
        "D2_LOW_12H_RETURN_PLUS_CROSS": base["return12_low"],
        "D3_BOTH_EXHAUSTION_FILTERS_PLUS_CROSS": (
            base["rsi30_high"] & base["return12_low"]
        ),
    }

    price_gaps = find_gaps(load_ohlcv(PRICE_1M_FILE))
    p1 = load_ohlcv(PRICE_1M_FILE)

    results = []
    summaries = []

    for strategy, mask in candidates.items():
        subset = base[mask].copy()

        for _, row in subset.iterrows():
            bt = backtest(row["b_entry_time"], p1, price_gaps)
            results.append({
                "strategy": strategy,
                "signal_time": row["signal_time"],
                "entry_time": row["b_entry_time"],
                "RSI30_delta_4": row["RSI30_delta_4"],
                "return_12h_pct": row["return_12h_pct"],
                **bt,
            })

    result_df = pd.DataFrame(results)

    for strategy in candidates:
        r = result_df[result_df["strategy"] == strategy]
        c = r[r["status"] == "COMPLETED"]
        good = c["outcome"].eq("TARGET_FIRST")
        bad = c["outcome"].eq("STOP_FIRST")
        resolved = good | bad

        summaries.append({
            "strategy": strategy,
            "signals_qualifying": len(r),
            "entries_completed": len(c),
            "target_first": int(good.sum()),
            "stop_first": int(bad.sum()),
            "neither": int(c["outcome"].eq("NEITHER").sum()),
            "ambiguous": int(c["outcome"].eq("AMBIGUOUS").sum()),
            "target_first_rate_resolved": (
                float(good[resolved].mean()) if resolved.any() else np.nan
            ),
            "mean_raw_short_return_pct": c["raw_short_return_pct"].mean(),
            "median_raw_short_return_pct": c["raw_short_return_pct"].median(),
            "median_mfe_pct": c["mfe_pct"].median(),
            "median_mae_pct": c["mae_pct"].median(),
        })

    summary = pd.DataFrame(summaries)

    print("=" * 72)
    print("DEVELOPMENT-ONLY EXHAUSTION COMPONENT TEST")
    print("=" * 72)
    print(f"Development signals: {len(signals)}")
    print(f"RSI30_delta_4 Q75: {rsi_q75:.4f}")
    print(f"return_12h_pct Q25: {ret12_q25:.4f}")
    print()
    print(summary.to_string(index=False))

    result_path = OUTPUT_DIR / "exhaustion_component_trade_results_development.csv"
    summary_path = OUTPUT_DIR / "exhaustion_component_summary_development.csv"

    result_df.to_csv(result_path, index=False)
    summary.to_csv(summary_path, index=False)

    print()
    print("OUTPUTS")
    print(result_path)
    print(summary_path)
    print()
    print("RULE: These results are exploratory. Do not change thresholds based on")
    print("2024/2025 performance after this point.")


if __name__ == "__main__":
    main()
