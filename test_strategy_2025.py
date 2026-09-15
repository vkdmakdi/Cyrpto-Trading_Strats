
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# FINAL 2025 TEST
# ---------------------------------------------------------------------
SIGNAL_FILE = Path("data/BTCUSDT_signals.csv")
PRICE_15M_FILE = Path("data/BTCUSDT_15m.csv")
PRICE_1M_FILE = Path("data/BTCUSDT_1m.csv")
FEATURE_FILE = Path("data/backtest_results/strategy_test_2025.csv")

OUTPUT_DIR = Path("data/backtest_results")

VALIDATION_START = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
VALIDATION_END = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")

TARGET_PCT = -2.0
STOP_PCT = 2.0
HORIZON_HOURS = 24

RSI_LEVEL = 70.0
FROZEN_RSI30_DELTA_4_THRESHOLD = 4.0
SWING_LOOKBACK = 4


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "open_time" not in df.columns:
        raise ValueError(f"{path} is missing open_time")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = (
        df.sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_rsi(price_15m: pd.DataFrame) -> pd.DataFrame:
    cols = ["RSI_15m", "RSI_30m", "RSI_1h", "RSI_4h"]

    if all(c in price_15m.columns for c in cols):
        return price_15m[["open_time"] + cols].copy()

    rsi_path = Path("data/BTCUSDT_RSI.csv")
    if not rsi_path.exists():
        raise FileNotFoundError(
            "15m file does not contain RSI columns and data/BTCUSDT_RSI.csv was not found."
        )

    rsi = pd.read_csv(rsi_path)
    rsi["open_time"] = pd.to_datetime(rsi["open_time"], utc=True)

    missing = [c for c in cols if c not in rsi.columns]
    if missing:
        raise ValueError(f"BTCUSDT_RSI.csv missing columns: {missing}")

    return (
        rsi[["open_time"] + cols]
        .sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )


def build_15m_frame() -> pd.DataFrame:
    p15 = load_ohlcv(PRICE_15M_FILE)
    rsi = load_rsi(p15)

    # Avoid duplicate RSI columns if they are already present in the price file.
    if all(c in p15.columns for c in ["RSI_15m", "RSI_30m", "RSI_1h", "RSI_4h"]):
        out = p15.copy()
    else:
        out = p15.merge(rsi, on="open_time", how="left")

    out["prior_4_low"] = out["low"].shift(1).rolling(SWING_LOOKBACK).min()

    # IMPORTANT:
    # "cross below 70" is a transition from >=70 on the prior completed
    # 15m candle to <70 on the current completed 15m candle.
    out["rsi_cross_below_70"] = (
        out["RSI_15m"].notna()
        & out["RSI_15m"].lt(RSI_LEVEL)
        & out["RSI_15m"].shift(1).ge(RSI_LEVEL)
    )

    # C uses simultaneous RSI reversal + structure break on the same
    # confirmation candle, matching the development definition.
    out["swing_low_break"] = (
        out["prior_4_low"].notna()
        & out["close"].lt(out["prior_4_low"])
    )

    return out


def load_2024_signals() -> pd.DataFrame:
    sig = pd.read_csv(SIGNAL_FILE)

    if "open_time" not in sig.columns:
        raise ValueError(f"{SIGNAL_FILE} is missing open_time")

    sig["open_time"] = pd.to_datetime(sig["open_time"], utc=True)

    sig = sig[
        sig["open_time"].between(VALIDATION_START, VALIDATION_END)
    ].copy()

    sig = (
        sig.sort_values("open_time")
        .drop_duplicates("open_time")
        .reset_index(drop=True)
    )

    return sig


def load_validation_features() -> pd.DataFrame:
    f = pd.read_csv(FEATURE_FILE)

    if "signal_time" not in f.columns:
        raise ValueError(f"{FEATURE_FILE} is missing signal_time")

    f["signal_time"] = pd.to_datetime(f["signal_time"], utc=True)

    # Hard guard: this file must truly be 2025 test data.
    if not f["signal_time"].between(VALIDATION_START, VALIDATION_END).all():
        raise ValueError(
            "Test feature file contains rows outside 2025. "
            "Aborting to prevent contamination."
        )

    if "RSI30_delta_4" not in f.columns:
        raise ValueError(f"{FEATURE_FILE} is missing RSI30_delta_4")

    return (
        f[["signal_time", "RSI30_delta_4"]]
        .sort_values("signal_time")
        .drop_duplicates("signal_time")
        .reset_index(drop=True)
    )


def find_global_1m_gaps(price_1m: pd.DataFrame):
    ts = price_1m["open_time"].sort_values().reset_index(drop=True)
    diffs = ts.diff().dt.total_seconds().div(60).to_numpy()

    gaps = []
    for i in range(1, len(ts)):
        if diffs[i] > 1:
            gaps.append((ts.iloc[i - 1], ts.iloc[i]))
    return gaps


def has_gap_between(gaps, start, end):
    for before, after in gaps:
        if after > start and before < end:
            return True
    return False


def backtest_entry(entry_time, price_1m, gaps):
    if pd.isna(entry_time):
        return {"status": "NO_ENTRY"}

    arr = price_1m["open_time"].array
    entry_idx = arr.searchsorted(entry_time, side="left")

    if (
        entry_idx >= len(price_1m)
        or price_1m.loc[entry_idx, "open_time"] != entry_time
    ):
        return {"status": "NO_ENTRY_DATA"}

    horizon_end = entry_time + pd.Timedelta(hours=HORIZON_HOURS)

    if has_gap_between(gaps, entry_time, horizon_end):
        return {"status": "GAP_BLOCKED"}

    end_idx = arr.searchsorted(horizon_end, side="right")
    w = price_1m.iloc[entry_idx:end_idx].copy()

    if w.empty:
        return {"status": "NO_DATA"}

    entry_price = float(w.iloc[0]["open"])
    target_price = entry_price * (1 + TARGET_PCT / 100.0)
    stop_price = entry_price * (1 + STOP_PCT / 100.0)

    lows = w["low"].to_numpy(dtype=float)
    highs = w["high"].to_numpy(dtype=float)

    target_idx = np.flatnonzero(lows <= target_price)
    stop_idx = np.flatnonzero(highs >= stop_price)

    t = int(target_idx[0]) if len(target_idx) else None
    s = int(stop_idx[0]) if len(stop_idx) else None

    if t is not None and s is not None:
        if t < s:
            outcome = "TARGET_FIRST"
            outcome_idx = t
        elif s < t:
            outcome = "STOP_FIRST"
            outcome_idx = s
        else:
            outcome = "AMBIGUOUS"
            outcome_idx = t
    elif t is not None:
        outcome = "TARGET_FIRST"
        outcome_idx = t
    elif s is not None:
        outcome = "STOP_FIRST"
        outcome_idx = s
    else:
        outcome = "NEITHER"
        outcome_idx = None

    horizon_close = float(w.iloc[-1]["close"])
    raw_short_return_pct = (entry_price / horizon_close - 1.0) * 100.0

    result = {
        "status": "COMPLETED",
        "entry_price": entry_price,
        "target_price": target_price,
        "stop_price": stop_price,
        "outcome": outcome,
        "raw_short_return_pct": raw_short_return_pct,
        "mfe_pct": (entry_price / float(np.min(lows)) - 1.0) * 100.0,
        "mae_pct": (entry_price / float(np.max(highs)) - 1.0) * 100.0,
    }

    if outcome_idx is not None:
        result["minutes_to_outcome"] = (
            w.iloc[outcome_idx]["open_time"] - entry_time
        ).total_seconds() / 60.0
    else:
        result["minutes_to_outcome"] = np.nan

    return result


def summarize(df: pd.DataFrame, strategy: str) -> dict:
    completed = df[df["status"].eq("COMPLETED")].copy()

    target = completed["outcome"].eq("TARGET_FIRST")
    stop = completed["outcome"].eq("STOP_FIRST")
    neither = completed["outcome"].eq("NEITHER")
    ambiguous = completed["outcome"].eq("AMBIGUOUS")

    resolved = target | stop

    # A simple mark-to-horizon P&L convention:
    # target-first = +2%, stop-first = -2%, neither/ambiguous use actual
    # horizon-close short return. This is gross and excludes all costs.
    pnl = completed["raw_short_return_pct"].copy()
    pnl.loc[target] = abs(TARGET_PCT)
    pnl.loc[stop] = -STOP_PCT

    expectancy = pnl.mean() if len(pnl) else np.nan

    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()

    return {
        "strategy": strategy,
        "signals_considered": int(len(df)),
        "entries_completed": int(len(completed)),
        "target_first": int(target.sum()),
        "stop_first": int(stop.sum()),
        "neither": int(neither.sum()),
        "ambiguous": int(ambiguous.sum()),
        "gap_blocked": int(df["status"].eq("GAP_BLOCKED").sum()),
        "no_entry": int(df["status"].eq("NO_ENTRY").sum()),
        "no_entry_data": int(df["status"].eq("NO_ENTRY_DATA").sum()),
        "resolved_win_rate": (
            float(target[resolved].mean()) if resolved.any() else np.nan
        ),
        "mean_raw_short_return_pct": completed["raw_short_return_pct"].mean(),
        "median_raw_short_return_pct": completed["raw_short_return_pct"].median(),
        "expectancy_gross_pct": expectancy,
        "profit_factor_mark_to_horizon": (
            gains / losses if losses > 0 else np.nan
        ),
        "median_mfe_pct": completed["mfe_pct"].median(),
        "median_mae_pct": completed["mae_pct"].median(),
        "mean_minutes_to_outcome": completed["minutes_to_outcome"].mean(),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FINAL 2025 TEST")
    print("=" * 72)
    print("Period: 2025-01-01 through 2025-12-31 UTC")
    print("Horizon: 24h")
    print("Target: -2% BTC move")
    print("Stop: +2% BTC move")
    print(f"Frozen RSI30_delta_4 threshold: >= {FROZEN_RSI30_DELTA_4_THRESHOLD:g}")
    print()

    signals = load_2024_signals()
    p15 = build_15m_frame()
    p1 = load_ohlcv(PRICE_1M_FILE)
    features = load_validation_features()

    print(f"2025 original all-RSI signals: {len(signals)}")

    # Attach validation-only RSI acceleration feature.
    signals = signals.merge(
        features,
        left_on="open_time",
        right_on="signal_time",
        how="left",
        validate="one_to_one",
    ).drop(columns=["signal_time"])

    if signals["RSI30_delta_4"].isna().any():
        n = int(signals["RSI30_delta_4"].isna().sum())
        raise ValueError(f"{n} 2025 signals are missing RSI30_delta_4.")

    # Fast timestamp lookup.
    p15_idx = {t: i for i, t in enumerate(p15["open_time"])}

    candidates = []

    for _, s in signals.iterrows():
        signal_time = s["open_time"]
        idx = p15_idx.get(signal_time)

        baseline_entry = signal_time + pd.Timedelta(minutes=15)
        b_entry = pd.NaT
        c_entry = pd.NaT

        if idx is not None:
            # Look only after the original signal candle has completed.
            for j in range(idx + 1, len(p15)):
                # Do not cross a real 15m gap.
                if (
                    p15.loc[j, "open_time"]
                    - p15.loc[j - 1, "open_time"]
                    != pd.Timedelta(minutes=15)
                ):
                    break

                if bool(p15.loc[j, "rsi_cross_below_70"]):
                    if j + 1 < len(p15):
                        b_entry = p15.loc[j + 1, "open_time"]
                    break

            # C: first candle satisfying both RSI cross and swing break.
            for j in range(idx + 1, len(p15)):
                if (
                    p15.loc[j, "open_time"]
                    - p15.loc[j - 1, "open_time"]
                    != pd.Timedelta(minutes=15)
                ):
                    break

                if (
                    bool(p15.loc[j, "rsi_cross_below_70"])
                    and bool(p15.loc[j, "swing_low_break"])
                ):
                    if j + 1 < len(p15):
                        c_entry = p15.loc[j + 1, "open_time"]
                    break

        # D is the frozen strategy:
        # all-RSI signal + RSI30_delta_4 >= 4, then B-style RSI reversal.
        d_qualifies = (
            float(s["RSI30_delta_4"])
            >= FROZEN_RSI30_DELTA_4_THRESHOLD
        )
        d_entry = b_entry if d_qualifies else pd.NaT

        candidates.append({
            "signal_time": signal_time,
            "RSI30_delta_4": float(s["RSI30_delta_4"]),
            "baseline_entry_time": baseline_entry,
            "b_entry_time": b_entry,
            "c_entry_time": c_entry,
            "d_entry_time": d_entry,
            "d_qualifies": bool(d_qualifies),
        })

    candidates = pd.DataFrame(candidates)

    gaps = find_global_1m_gaps(p1)

    all_trade_rows = []
    summary_rows = []

    strategy_entry_cols = {
        "A_ORIGINAL_IMMEDIATE": "baseline_entry_time",
        "B_RSI_CROSS": "b_entry_time",
        "C_RSI_CROSS_PLUS_SWING_BREAK": "c_entry_time",
        "D_FROZEN_RSI_ACCEL_PLUS_CROSS": "d_entry_time",
    }

    for strategy, entry_col in strategy_entry_cols.items():
        strategy_rows = []

        source_df = candidates.copy()

        # D is intentionally limited to signals passing the frozen filter.
        if strategy == "D_FROZEN_RSI_ACCEL_PLUS_CROSS":
            source_df = source_df[source_df["d_qualifies"]].copy()

        for _, row in source_df.iterrows():
            bt = backtest_entry(row[entry_col], p1, gaps)

            trade = {
                "strategy": strategy,
                "signal_time": row["signal_time"],
                "entry_time": row[entry_col],
                "RSI30_delta_4": row["RSI30_delta_4"],
                **bt,
            }

            strategy_rows.append(trade)
            all_trade_rows.append(trade)

        strategy_df = pd.DataFrame(strategy_rows)
        summary_rows.append(summarize(strategy_df, strategy))

    summary = pd.DataFrame(summary_rows)
    trades = pd.DataFrame(all_trade_rows)

    # Extra D diagnostic: show the raw frozen filter count independently.
    diagnostics = pd.DataFrame([{
        "test_year": 2025,
        "original_all_RSI_signals": len(candidates),
        "frozen_RSI_accel_qualified_signals": int(candidates["d_qualifies"].sum()),
        "frozen_threshold": FROZEN_RSI30_DELTA_4_THRESHOLD,
        "threshold_source": "FROZEN_FROM_DEVELOPMENT_2021_2023",
    }])

    summary_path = OUTPUT_DIR / "strategy_test_2025_summary.csv"
    trades_path = OUTPUT_DIR / "strategy_test_2025_trades.csv"
    diagnostics_path = OUTPUT_DIR / "strategy_test_2025_diagnostics.csv"

    summary.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    print()
    print("RESULTS")
    print(summary.to_string(index=False))
    print()
    print("DIAGNOSTICS")
    print(diagnostics.to_string(index=False))
    print()
    print("OUTPUTS")
    print(f"  {summary_path}")
    print(f"  {trades_path}")
    print(f"  {diagnostics_path}")
    print()
    print("VALIDATION RULE")
    print("2024 results are out-of-sample. Do not change the frozen threshold,")
    print("entry definition, target, stop, or horizon based on these results.")


if __name__ == "__main__":
    main()
