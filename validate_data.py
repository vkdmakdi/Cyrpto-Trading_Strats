import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FILE = "data/BTCUSDT_1m.csv"

EXPECTED_INTERVAL = pd.Timedelta(minutes=1)

EXPECTED_START = pd.Timestamp("2021-01-01", tz="UTC")
EXPECTED_END = pd.Timestamp("2026-01-01", tz="UTC")


# ============================================================
# LOAD DATA
# ============================================================

print(f"\nLoading: {FILE}")

df = pd.read_csv(FILE)

df["open_time"] = pd.to_datetime(
    df["open_time"],
    utc=True
)

df = (
    df.sort_values("open_time")
    .reset_index(drop=True)
)


# ============================================================
# BASIC INFO
# ============================================================

print("\n===== BASIC INFO =====")

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# DATE RANGE
# ============================================================

print("\n===== DATE RANGE =====")

start = df["open_time"].min()
end = df["open_time"].max()

print(f"Start: {start}")
print(f"End:   {end}")


# ============================================================
# EXPECTED CANDLE COUNT
# ============================================================

print("\n===== EXPECTED CANDLE COUNT =====")

expected_rows = int(
    (EXPECTED_END - EXPECTED_START)
    / EXPECTED_INTERVAL
)

actual_rows = len(df)

print(f"Expected rows: {expected_rows:,}")
print(f"Actual rows:   {actual_rows:,}")

difference = actual_rows - expected_rows

if difference == 0:
    print("✓ Row count is complete")
else:
    print(f"⚠ Row count differs by {difference:,}")


# ============================================================
# TIMESTAMP RANGE CHECK
# ============================================================

print("\n===== TIMESTAMP RANGE CHECK =====")

if start == EXPECTED_START:
    print("✓ Start timestamp is correct")
else:
    print(f"⚠ Expected start: {EXPECTED_START}")

expected_last = EXPECTED_END - EXPECTED_INTERVAL

if end == expected_last:
    print("✓ End timestamp is correct")
else:
    print(f"⚠ Expected end: {expected_last}")


# ============================================================
# DUPLICATES
# ============================================================

print("\n===== DUPLICATES =====")

duplicates = df["open_time"].duplicated().sum()

print(f"Duplicate timestamps: {duplicates}")

if duplicates == 0:
    print("✓ No duplicate timestamps")
else:
    print("⚠ Duplicate timestamps detected")


# ============================================================
# MISSING VALUES
# ============================================================

print("\n===== MISSING VALUES =====")

missing = df.isna().sum()

print(missing)

if missing.sum() == 0:
    print("✓ No missing values")
else:
    print("⚠ Missing values detected")


# ============================================================
# OHLC CHECKS
# ============================================================

print("\n===== OHLC CHECKS =====")

bad_high = (
    df["high"] < df[["open", "close"]].max(axis=1)
).sum()

bad_low = (
    df["low"] > df[["open", "close"]].min(axis=1)
).sum()

bad_ohlc_range = (
    df["high"] < df["low"]
).sum()

bad_nonpositive = (
    df[["open", "high", "low", "close"]] <= 0
).any(axis=1).sum()

print(f"Invalid highs:       {bad_high}")
print(f"Invalid lows:        {bad_low}")
print(f"High < Low:          {bad_ohlc_range}")
print(f"Non-positive prices: {bad_nonpositive}")

if (
    bad_high == 0
    and bad_low == 0
    and bad_ohlc_range == 0
    and bad_nonpositive == 0
):
    print("✓ OHLC data looks valid")
else:
    print("⚠ OHLC issues detected")


# ============================================================
# TIME GAPS
# ============================================================

print("\n===== TIME GAPS =====")

time_diff = df["open_time"].diff()

gaps = df[
    time_diff > EXPECTED_INTERVAL
].copy()

print(f"Unexpected gaps: {len(gaps)}")

if len(gaps) > 0:

    print("\nFirst 20 gaps:")

    for index, row in gaps.head(20).iterrows():

        previous = df.loc[
            index - 1,
            "open_time"
        ]

        current = row["open_time"]

        difference = current - previous

        # Number of missing candles between timestamps
        missing_candles = (
            int(difference / EXPECTED_INTERVAL) - 1
        )

        print(
            f"{previous} → {current} "
            f"({difference}, "
            f"{missing_candles:,} missing candles)"
        )

else:
    print("✓ No unexpected gaps")


# ============================================================
# MONTHLY COVERAGE
# ============================================================

print("\n===== MONTHLY COVERAGE =====")

df["month"] = df["open_time"].dt.to_period("M")

monthly = df.groupby("month").size()

print(monthly.to_string())


# ============================================================
# PRICE RANGE
# ============================================================

print("\n===== PRICE RANGE =====")

print(
    f"Lowest price:  ${df['low'].min():,.2f}"
)

print(
    f"Highest price: ${df['high'].max():,.2f}"
)


# ============================================================
# VOLUME
# ============================================================

print("\n===== VOLUME =====")

print(
    f"Total BTC volume: {df['volume'].sum():,.2f}"
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n===== VALIDATION COMPLETE =====")