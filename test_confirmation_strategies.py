
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Locked research inputs
# ---------------------------------------------------------------------
SIGNAL_FILE = Path("data/BTCUSDT_signals.csv")
PRICE_15M_FILE = Path("data/BTCUSDT_15m.csv")
PRICE_1M_FILE = Path("data/BTCUSDT_1m.csv")
FEATURE_FILE = Path("data/backtest_results/strategy_development_2021_2023.csv")
OUTPUT_DIR = Path("data/backtest_results")

DEVELOPMENT_START = pd.Timestamp("2021-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")

HORIZON_HOURS = 24
TARGET_PCT = -2.0          # short return convention: BTC falls 2% => +2% gross
STOP_PCT = 2.0             # BTC rises 2% => -2% gross
SWING_LOOKBACK = 4         # prior completed 15m candles
RSI_LEVEL = 70.0


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "open_time" not in df.columns:
        raise ValueError(f"{path} is missing open_time")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_signal_data() -> pd.DataFrame:
    df = pd.read_csv(SIGNAL_FILE)
    if "open_time" not in df.columns:
        raise ValueError(f"{SIGNAL_FILE} is missing open_time")

    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    return df


def first_global_gap_minutes(df_1m: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    # Keep timestamps explicitly UTC-aware throughout the backtest.
    ts = pd.to_datetime(df_1m["open_time"], utc=True).sort_values().reset_index(drop=True)
    diffs = ts.diff().dt.total_seconds().div(60).to_numpy()

    gaps = []
    for i in range(1, len(ts)):
        if diffs[i] > 1:
            gaps.append((ts.iloc[i - 1], ts.iloc[i]))
    return gaps


def has_gap_between(
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    for before, after in gaps:
        before = pd.Timestamp(before).tz_convert("UTC")
        after = pd.Timestamp(after).tz_convert("UTC")
        if after > start and before < end:
            return True
    return False


def build_rsi_frame(price_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Build the minimal 15m confirmation frame.

    The saved 15m file is expected to contain the same RSI columns used by
    BTCUSDT_RSI.csv / signal_features. If those columns are absent, fall back
    to BTCUSDT_RSI.csv.
    """
    rsi_cols = ["RSI_15m", "RSI_30m", "RSI_1h", "RSI_4h"]

    if all(c in price_15m.columns for c in rsi_cols):
        out = price_15m[["open_time"] + rsi_cols].copy()
        return out

    rsi_file = Path("data/BTCUSDT_RSI.csv")
    if not rsi_file.exists():
        raise ValueError(
            "15m price file does not contain RSI columns and BTCUSDT_RSI.csv was not found."
        )

    rsi = pd.read_csv(rsi_file)
    if "open_time" not in rsi.columns:
        raise ValueError("BTCUSDT_RSI.csv is missing open_time")
    rsi["open_time"] = pd.to_datetime(rsi["open_time"], utc=True)

    missing = [c for c in rsi_cols if c not in rsi.columns]
    if missing:
        raise ValueError(f"BTCUSDT_RSI.csv missing columns: {missing}")

    return rsi[["open_time"] + rsi_cols].sort_values("open_time").drop_duplicates("open_time")


def build_confirmation_candidates(
    signals: pd.DataFrame,
    price_15m: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one possible confirmation entry per original signal.

    Strategy B:
      original all-four-RSI>70 signal
      -> first 15m RSI cross from >=70 to <70
      -> enter at the next 15m candle open

    Strategy C:
      same RSI reversal
      -> confirmation candle must also close below the LOW of the
         previous 4 completed 15m candles
      -> enter at the next 15m candle open

    Strategy D:
      development-only exhaustion filters at the ORIGINAL signal:
        RSI30_delta_4 >= development 75th percentile
        return_12h_pct <= development 25th percentile
      -> then Strategy B confirmation.
    """
    p = price_15m.copy().sort_values("open_time").reset_index(drop=True)

    required = ["open_time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in p.columns]
    if missing:
        raise ValueError(f"15m price file missing: {missing}")

    rsi = build_rsi_frame(p)
    p = p.merge(rsi, on="open_time", how="left", suffixes=("", "_rsi"))

    # Use the RSI columns from the merged frame and drop accidental duplicates.
    if "RSI_15m_rsi" in p.columns:
        p["RSI_15m"] = p["RSI_15m"].fillna(p["RSI_15m_rsi"])
        p = p.drop(columns=["RSI_15m_rsi"])

    # Previous 4 completed 15m lows for the current candle.
    p["prior_4_low"] = p["low"].shift(1).rolling(SWING_LOOKBACK).min()

    p["rsi_cross_below_70"] = (
        p["RSI_15m"].notna()
        & p["RSI_15m"].lt(RSI_LEVEL)
        & p["RSI_15m"].shift(1).ge(RSI_LEVEL)
    )

    p["swing_low_break"] = (
        p["prior_4_low"].notna()
        & p["close"].lt(p["prior_4_low"])
    )

    # Development-derived exhaustion thresholds.
    features = pd.read_csv(FEATURE_FILE)
    features["signal_time"] = pd.to_datetime(features["signal_time"], utc=True)

    if not features["signal_time"].between(DEVELOPMENT_START, DEVELOPMENT_END).all():
        raise ValueError("Development feature file contains rows outside 2021-2023.")

    for col in ["RSI30_delta_4", "return_12h_pct"]:
        if col not in features.columns:
            raise ValueError(f"Development feature file missing {col}")

    rsi30_q75 = pd.to_numeric(features["RSI30_delta_4"], errors="coerce").quantile(0.75)
    ret12_q25 = pd.to_numeric(features["return_12h_pct"], errors="coerce").quantile(0.25)

    # Signal rows are the original all-four-RSI>70 events.
    sig = signals.copy().sort_values("open_time").reset_index(drop=True)
    sig = sig[
        (sig["open_time"] >= DEVELOPMENT_START)
        & (sig["open_time"] <= DEVELOPMENT_END)
    ].copy()

    if "RSI_15m" not in sig.columns:
        # Merge raw RSI into signal file if the signal file itself lacks it.
        srsi = build_rsi_frame(p)
        sig = sig.merge(srsi, on="open_time", how="left", suffixes=("", "_rsi"))

    # Use exact development feature values at the original signal.
    feature_subset = features[
        ["signal_time", "RSI30_delta_4", "return_12h_pct"]
    ].copy()
    sig = sig.merge(
        feature_subset,
        left_on="open_time",
        right_on="signal_time",
        how="left",
        validate="one_to_one",
    ).drop(columns=["signal_time"])

    # Fast lookup by timestamp.
    time_to_idx = {t: i for i, t in enumerate(p["open_time"])}

    rows = []

    for _, s in sig.iterrows():
        signal_time = s["open_time"]
        start_idx = time_to_idx.get(signal_time)

        if start_idx is None:
            continue

        # Confirmation must occur AFTER the original signal candle has closed.
        cross_idx = None
        swing_idx = None

        for j in range(start_idx + 1, len(p)):
            # Need an actual continuous sequence.
            # If data is missing before confirmation, abandon this signal.
            if p.loc[j, "open_time"] - p.loc[j - 1, "open_time"] != pd.Timedelta(minutes=15):
                break

            if bool(p.loc[j, "rsi_cross_below_70"]):
                cross_idx = j
                break

        # Strategy C searches for the first candle that satisfies both
        # confirmation conditions, rather than requiring the first RSI
        # cross candle to also break structure.
        for j in range(start_idx + 1, len(p)):
            if p.loc[j, "open_time"] - p.loc[j - 1, "open_time"] != pd.Timedelta(minutes=15):
                break
            if bool(p.loc[j, "rsi_cross_below_70"]) and bool(p.loc[j, "swing_low_break"]):
                swing_idx = j
                break

        base = {
            "signal_time": signal_time,
            "rsi30_delta_4": pd.to_numeric(s.get("RSI30_delta_4"), errors="coerce"),
            "return_12h_pct": pd.to_numeric(s.get("return_12h_pct"), errors="coerce"),
            "rsi30_q75_threshold": rsi30_q75,
            "return12_q25_threshold": ret12_q25,
        }

        if cross_idx is not None and cross_idx + 1 < len(p):
            base["b_entry_time"] = p.loc[cross_idx + 1, "open_time"]
            base["b_confirmation_time"] = p.loc[cross_idx, "open_time"]
            base["b_confirmation_close"] = p.loc[cross_idx, "close"]
        else:
            base["b_entry_time"] = pd.NaT
            base["b_confirmation_time"] = pd.NaT
            base["b_confirmation_close"] = np.nan

        if swing_idx is not None and swing_idx + 1 < len(p):
            base["c_entry_time"] = p.loc[swing_idx + 1, "open_time"]
            base["c_confirmation_time"] = p.loc[swing_idx, "open_time"]
            base["c_confirmation_close"] = p.loc[swing_idx, "close"]
        else:
            base["c_entry_time"] = pd.NaT
            base["c_confirmation_time"] = pd.NaT
            base["c_confirmation_close"] = np.nan

        d_filter = (
            pd.notna(base["rsi30_delta_4"])
            and pd.notna(base["return_12h_pct"])
            and base["rsi30_delta_4"] >= rsi30_q75
            and base["return_12h_pct"] <= ret12_q25
        )
        base["d_exhaustion_filter"] = bool(d_filter)
        base["d_entry_time"] = base["b_entry_time"] if d_filter else pd.NaT
        base["d_confirmation_time"] = base["b_confirmation_time"] if d_filter else pd.NaT
        base["d_confirmation_close"] = (
            base["b_confirmation_close"] if d_filter else np.nan
        )

        rows.append(base)

    return pd.DataFrame(rows)


def backtest_entry(
    entry_time: pd.Timestamp,
    price_1m: pd.DataFrame,
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict:
    if pd.isna(entry_time):
        return {
            "status": "NO_ENTRY",
        }

    # Avoid np.datetime64 on timezone-aware timestamps; compare using
    # pandas Timestamp values instead.
    entry_times = price_1m["open_time"].array
    entry_idx = entry_times.searchsorted(entry_time, side="left")

    if entry_idx >= len(price_1m) or price_1m.loc[entry_idx, "open_time"] != entry_time:
        return {
            "status": "NO_ENTRY_DATA",
        }

    entry_price = float(price_1m.loc[entry_idx, "open"])
    horizon_end = entry_time + pd.Timedelta(hours=HORIZON_HOURS)

    if has_gap_between(gaps, entry_time, horizon_end):
        return {
            "status": "GAP_BLOCKED",
            "entry_price": entry_price,
        }

    end_idx = price_1m["open_time"].array.searchsorted(
        horizon_end,
        side="right",
    )

    window = price_1m.iloc[entry_idx:end_idx].copy()

    if window.empty:
        return {
            "status": "NO_DATA",
            "entry_price": entry_price,
        }

    # Short thresholds:
    # target price is entry * 0.98, stop price is entry * 1.02.
    target_price = entry_price * (1.0 + TARGET_PCT / 100.0)
    stop_price = entry_price * (1.0 + STOP_PCT / 100.0)

    target_hits = window["low"].to_numpy(dtype=float) <= target_price
    stop_hits = window["high"].to_numpy(dtype=float) >= stop_price

    target_pos = np.flatnonzero(target_hits)
    stop_pos = np.flatnonzero(stop_hits)

    target_first_pos = int(target_pos[0]) if len(target_pos) else None
    stop_first_pos = int(stop_pos[0]) if len(stop_pos) else None

    if target_first_pos is not None and stop_first_pos is not None:
        if target_first_pos < stop_first_pos:
            outcome = "TARGET_FIRST"
        elif stop_first_pos < target_first_pos:
            outcome = "STOP_FIRST"
        else:
            outcome = "AMBIGUOUS"
        outcome_pos = min(target_first_pos, stop_first_pos)
    elif target_first_pos is not None:
        outcome = "TARGET_FIRST"
        outcome_pos = target_first_pos
    elif stop_first_pos is not None:
        outcome = "STOP_FIRST"
        outcome_pos = stop_first_pos
    else:
        outcome = "NEITHER"
        outcome_pos = None

    # Short return at horizon: positive is profitable.
    close_at_horizon = float(window.iloc[-1]["close"])
    raw_short_return_pct = (entry_price / close_at_horizon - 1.0) * 100.0

    lows = window["low"].to_numpy(dtype=float)
    highs = window["high"].to_numpy(dtype=float)
    mfe_pct = (entry_price / np.minimum.accumulate(lows)[-1] - 1.0) * 100.0
    mae_pct = (entry_price / np.maximum.accumulate(highs)[-1] - 1.0) * 100.0

    result = {
        "status": "COMPLETED",
        "entry_price": entry_price,
        "target_price": target_price,
        "stop_price": stop_price,
        "outcome": outcome,
        "raw_short_return_pct": raw_short_return_pct,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
    }

    if outcome_pos is not None:
        result["minutes_to_outcome"] = (
            window.iloc[outcome_pos]["open_time"] - entry_time
        ).total_seconds() / 60.0
    else:
        result["minutes_to_outcome"] = np.nan

    return result


def summarize(df: pd.DataFrame, strategy: str) -> dict:
    completed = df[df["status"].eq("COMPLETED")].copy()
    good = completed["outcome"].eq("TARGET_FIRST")
    bad = completed["outcome"].eq("STOP_FIRST")

    resolved = good | bad

    return {
        "strategy": strategy,
        "signals_considered": len(df),
        "entries_completed": len(completed),
        "target_first": int(good.sum()),
        "stop_first": int(bad.sum()),
        "neither": int(completed["outcome"].eq("NEITHER").sum()),
        "ambiguous": int(completed["outcome"].eq("AMBIGUOUS").sum()),
        "gap_blocked": int(df["status"].eq("GAP_BLOCKED").sum()),
        "no_entry": int(df["status"].eq("NO_ENTRY").sum()),
        "target_first_rate_resolved": (
            float(good[resolved].mean()) if resolved.any() else np.nan
        ),
        "mean_raw_short_return_pct": completed["raw_short_return_pct"].mean(),
        "median_raw_short_return_pct": completed["raw_short_return_pct"].median(),
        "median_mfe_pct": completed["mfe_pct"].median(),
        "median_mae_pct": completed["mae_pct"].median(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Development-only test of RSI confirmation strategies B/C/D."
    )
    parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DEVELOPMENT-ONLY CONFIRMATION STRATEGY TEST")
    print("=" * 72)
    print("Baseline exits: -2% target / +2% stop")
    print("Horizon: 24h")
    print()

    signals = load_signal_data()
    price_15m = read_ohlcv(PRICE_15M_FILE)
    price_1m = read_ohlcv(PRICE_1M_FILE)

    # Confirm the historical test data really covers only the development period.
    min_signal = signals["open_time"].min()
    max_signal = signals["open_time"].max()
    print(f"Original signal range: {min_signal} -> {max_signal}")

    candidates = build_confirmation_candidates(signals, price_15m)

    print()
    print("DEVELOPMENT INPUTS")
    print(f"  Candidate signals: {len(candidates)}")
    print(
        f"  RSI30_delta_4 Q75: "
        f"{candidates['rsi30_q75_threshold'].dropna().iloc[0]:.4f}"
    )
    print(
        f"  return_12h_pct Q25: "
        f"{candidates['return12_q25_threshold'].dropna().iloc[0]:.4f}"
    )
    print()

    gaps = first_global_gap_minutes(price_1m)
    results = []
    summary_rows = []

    entry_columns = {
        "B_RSI_CROSS": "b_entry_time",
        "C_RSI_CROSS_PLUS_SWING_BREAK": "c_entry_time",
        "D_EXHAUSTION_PLUS_RSI_CROSS": "d_entry_time",
    }

    for strategy, entry_col in entry_columns.items():
        strategy_rows = []

        for _, row in candidates.iterrows():
            bt = backtest_entry(
                row[entry_col],
                price_1m,
                gaps,
            )

            out = {
                "strategy": strategy,
                "signal_time": row["signal_time"],
                "entry_time": row[entry_col],
                "rsi30_delta_4": row["rsi30_delta_4"],
                "return_12h_pct": row["return_12h_pct"],
                "d_exhaustion_filter": row["d_exhaustion_filter"],
                "rsi30_q75_threshold": row["rsi30_q75_threshold"],
                "return12_q25_threshold": row["return12_q25_threshold"],
                **bt,
            }
            strategy_rows.append(out)
            results.append(out)

        strategy_df = pd.DataFrame(strategy_rows)
        summary_rows.append(summarize(strategy_df, strategy))

    result_df = pd.DataFrame(results)
    summary_df = pd.DataFrame(summary_rows)

    result_path = OUTPUT_DIR / "confirmation_strategy_trade_results_development.csv"
    summary_path = OUTPUT_DIR / "confirmation_strategy_summary_development.csv"

    result_df.to_csv(result_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # A compact entry diagnostic: how often each confirmation happened.
    entry_diag = []
    for strategy, entry_col in entry_columns.items():
        if strategy.startswith("D_"):
            mask = candidates["d_exhaustion_filter"]
        else:
            mask = candidates[entry_col].notna()

        entry_diag.append(
            {
                "strategy": strategy,
                "signals_considered": len(candidates) if not strategy.startswith("D_") else int(mask.sum()),
                "confirmed_entries": int(candidates.loc[mask, entry_col].notna().sum()),
                "entry_rate_pct": (
                    100.0 * candidates.loc[mask, entry_col].notna().mean()
                    if mask.any()
                    else np.nan
                ),
            }
        )

    pd.DataFrame(entry_diag).to_csv(
        OUTPUT_DIR / "confirmation_strategy_entry_diagnostics_development.csv",
        index=False,
    )

    print("RESULTS")
    print(summary_df.to_string(index=False))
    print()
    print("OUTPUTS")
    print(f"  {result_path}")
    print(f"  {summary_path}")
    print(
        f"  {OUTPUT_DIR / 'confirmation_strategy_entry_diagnostics_development.csv'}"
    )
    print()
    print("RULE")
    print(
        "These are development-stage results only. Do not alter thresholds or "
        "strategy definitions after seeing 2024 validation results."
    )


if __name__ == "__main__":
    main()
