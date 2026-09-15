import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/BTCUSDT_15m.csv"
OUTPUT_FILE = "data/BTCUSDT_RSI.csv"

RSI_PERIOD = 14

BASE_INTERVAL = pd.Timedelta(minutes=15)


# ============================================================
# RSI — WILDER / EMA STYLE
# ============================================================

def calculate_rsi_segmented(close, period=14, expected_delta=None):
    """
    Calculate RSI independently inside each continuous data segment.

    This is important because a real data gap means we cannot
    pretend that the missing candle(s) existed.
    """

    result = pd.Series(
        np.nan,
        index=close.index,
        dtype="float64"
    )

    if expected_delta is None:
        expected_delta = BASE_INTERVAL

    # Identify the beginning of every continuous segment.
    diffs = close.index.to_series().diff()

    new_segment = (
        diffs.isna()
        | (diffs != expected_delta)
    )

    segment_id = new_segment.cumsum()

    for _, segment in close.groupby(segment_id):

        if len(segment) < period:
            continue

        delta = segment.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        ).mean()

        rs = avg_gain / avg_loss

        rsi = 100 - (100 / (1 + rs))

        # Edge cases
        rsi = rsi.where(
            ~((avg_gain == 0) & (avg_loss == 0)),
            50
        )

        rsi = rsi.where(
            avg_loss != 0,
            100
        )

        rsi = rsi.where(
            avg_gain != 0,
            0
        )

        result.loc[segment.index] = rsi

    return result


# ============================================================
# LOAD 15M DATA
# ============================================================

print("Loading 15m data...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["open_time"]
)

df = df.sort_values("open_time")
df = df.set_index("open_time")

print(f"Loaded {len(df):,} 15m candles")


# ============================================================
# BASIC VALIDATION
# ============================================================

if not df.index.is_monotonic_increasing:
    raise ValueError("ERROR: timestamps are not sorted.")

if df.index.duplicated().any():
    raise ValueError("ERROR: duplicate timestamps detected.")

required_columns = [
    "open",
    "high",
    "low",
    "close",
    "volume"
]

for col in required_columns:
    if df[col].isna().any():
        raise ValueError(
            f"ERROR: missing values in {col}"
        )


# ============================================================
# DETECT BASE-TIMEFRAME GAPS
# ============================================================

print("\n===== BASE DATA GAP CHECK =====")

base_diffs = df.index.to_series().diff()

base_gaps = base_diffs[
    base_diffs > BASE_INTERVAL
]

print(
    f"15m gaps greater than 15 minutes: "
    f"{len(base_gaps)}"
)

if len(base_gaps) > 0:
    for timestamp, gap in base_gaps.items():
        print(
            f"  {timestamp - gap} → {timestamp} "
            f"({gap})"
        )


# ============================================================
# 15M RSI
# ============================================================

print("\nCalculating 15m RSI...")

df["RSI_15m"] = calculate_rsi_segmented(
    df["close"],
    RSI_PERIOD,
    BASE_INTERVAL
)


# ============================================================
# GAP-AWARE HIGHER TIMEFRAME BUILDER
# ============================================================

def build_complete_timeframe(data, timeframe, expected_candles):
    """
    Build a higher timeframe only when every expected 15m
    candle exists inside the bin.

    Example:
        30m -> exactly 2 x 15m candles
        1h  -> exactly 4 x 15m candles
        4h  -> exactly 16 x 15m candles
    """

    ohlcv = data.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })

    counts = data["close"].resample(timeframe).count()

    complete = counts == expected_candles

    result = ohlcv.loc[complete].copy()

    # Remove any accidental incomplete OHLC records
    result = result.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    rejected = (~complete).sum()

    print(
        f"{timeframe}: "
        f"{len(result):,} complete candles | "
        f"{rejected:,} incomplete bins rejected"
    )

    return result


# ============================================================
# BUILD HIGHER TIMEFRAMES
# ============================================================

print("\n===== BUILDING HIGHER TIMEFRAMES =====")

df_30m = build_complete_timeframe(
    df,
    "30min",
    2
)

df_1h = build_complete_timeframe(
    df,
    "1h",
    4
)

df_4h = build_complete_timeframe(
    df,
    "4h",
    16
)


# ============================================================
# CALCULATE HIGHER-TIMEFRAME RSI
# ============================================================

print("\n===== CALCULATING HIGHER-TIMEFRAME RSI =====")

df_30m["RSI_30m"] = calculate_rsi_segmented(
    df_30m["close"],
    RSI_PERIOD,
    pd.Timedelta(minutes=30)
)

df_1h["RSI_1h"] = calculate_rsi_segmented(
    df_1h["close"],
    RSI_PERIOD,
    pd.Timedelta(hours=1)
)

df_4h["RSI_4h"] = calculate_rsi_segmented(
    df_4h["close"],
    RSI_PERIOD,
    pd.Timedelta(hours=4)
)


# ============================================================
# ALIGN COMPLETED HIGHER-TIMEFRAME RSI
# ============================================================

def align_completed_rsi(
    base_index,
    htf_rsi,
    timeframe
):
    """
    For each 15m timestamp, use the RSI of the LAST COMPLETED
    higher-timeframe candle.

    Example for 1h:

        20:00  -> use RSI from 19:00 candle
        20:15  -> use RSI from 19:00 candle
        20:30  -> use RSI from 19:00 candle
        20:45  -> use RSI from 19:00 candle
        21:00  -> use RSI from 20:00 candle

    Importantly, if the 19:00 candle was incomplete and
    rejected, all four corresponding 15m rows receive NaN.
    We do NOT silently carry an older RSI across the gap.
    """

    # Timestamp at which each HTF candle becomes available.
    completed_timestamp = (
        htf_rsi.index + timeframe
    )

    completed = pd.Series(
        htf_rsi.to_numpy(),
        index=completed_timestamp,
        name=htf_rsi.name
    )

    # Map each base candle to the relevant completed HTF candle.
    base_period_start = (
        base_index.floor(timeframe)
    )

    lookup_timestamp = (
        base_period_start
        - timeframe
    )

    aligned = pd.Series(
        completed.reindex(
            lookup_timestamp
        ).to_numpy(),
        index=base_index,
        name=htf_rsi.name
    )

    return aligned


# ============================================================
# ALIGN
# ============================================================

print("\n===== ALIGNING COMPLETED HTF RSI =====")

df["RSI_30m"] = align_completed_rsi(
    df.index,
    df_30m["RSI_30m"],
    pd.Timedelta(minutes=30)
)

df["RSI_1h"] = align_completed_rsi(
    df.index,
    df_1h["RSI_1h"],
    pd.Timedelta(hours=1)
)

df["RSI_4h"] = align_completed_rsi(
    df.index,
    df_4h["RSI_4h"],
    pd.Timedelta(hours=4)
)


# ============================================================
# RSI RANGE CHECK
# ============================================================

print("\n===== RSI RANGE CHECK =====")

rsi_columns = [
    "RSI_15m",
    "RSI_30m",
    "RSI_1h",
    "RSI_4h"
]

for col in rsi_columns:

    valid = df[col].dropna()

    if len(valid) == 0:
        print(f"{col}: NO VALID VALUES")
        continue

    minimum = valid.min()
    maximum = valid.max()

    invalid = (
        (valid < 0) |
        (valid > 100)
    ).sum()

    print(
        f"{col}: "
        f"min={minimum:.6f}, "
        f"max={maximum:.6f}, "
        f"invalid={invalid}"
    )

    if invalid:
        raise ValueError(
            f"ERROR: {col} contains invalid RSI values."
        )


# ============================================================
# WARM-UP CHECK
# ============================================================

print("\n===== WARM-UP CHECK =====")

for col in rsi_columns:

    first_valid = df[col].first_valid_index()

    print(
        f"{col}: first valid = {first_valid}"
    )


# ============================================================
# LOOK-AHEAD SANITY CHECK
# ============================================================

print("\n===== LOOK-AHEAD SANITY CHECK =====")

test_timestamp = pd.Timestamp(
    "2025-12-31 20:00:00+00:00"
)

window = df.loc[
    test_timestamp - pd.Timedelta(hours=1):
    test_timestamp + pd.Timedelta(hours=1),
    rsi_columns
]

print(window.to_string())


# ============================================================
# GAP SAFETY CHECK
# ============================================================

print("\n===== GAP SAFETY CHECK =====")

for timestamp, gap in base_gaps.items():

    before_time = timestamp - BASE_INTERVAL

    before = df.loc[
        df.index <= before_time
    ].tail(1)

    after = df.loc[
        df.index >= timestamp
    ].head(1)

    print(
        f"\nGap: {before_time} → {timestamp} "
        f"({gap})"
    )

    if not before.empty:
        print("Before:")
        print(
            before[rsi_columns].to_string()
        )

    if not after.empty:
        print("After:")
        print(
            after[rsi_columns].to_string()
        )


# ============================================================
# FINAL MISSING-VALUE CHECK
# ============================================================

print("\n===== FINAL RSI DATA CHECK =====")

for col in rsi_columns:

    missing = df[col].isna().sum()

    print(
        f"{col}: {missing:,} missing"
    )


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=True
)

print("\n========================================")
print("GAP-AWARE RSI CALCULATION COMPLETE")
print("========================================")

print(f"Rows: {len(df):,}")

print("\nColumns:")
print(df.columns.tolist())

print("\nLast 10 rows:")
print(
    df.tail(10).to_string()
)

print(f"\nSaved to: {OUTPUT_FILE}")