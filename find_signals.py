import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "data/BTCUSDT_RSI.csv"
OUTPUT_FILE = "data/BTCUSDT_signals.csv"

RSI_THRESHOLD = 70


# ============================================================
# LOAD DATA
# ============================================================

print("Loading RSI data...")

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["open_time"]
)

df = df.sort_values("open_time")
df = df.set_index("open_time")

print(f"Loaded {len(df):,} candles")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "open",
    "high",
    "low",
    "close",
    "RSI_15m",
    "RSI_30m",
    "RSI_1h",
    "RSI_4h"
]

missing_columns = [
    col for col in required
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# ORIGINAL STRATEGY CONDITION
# ============================================================

df["all_rsi_over_70"] = (
    (df["RSI_15m"] > RSI_THRESHOLD) &
    (df["RSI_30m"] > RSI_THRESHOLD) &
    (df["RSI_1h"] > RSI_THRESHOLD) &
    (df["RSI_4h"] > RSI_THRESHOLD)
)


# ============================================================
# NEW SIGNAL = CONDITION CHANGED FROM FALSE → TRUE
# ============================================================

df["new_signal"] = (
    df["all_rsi_over_70"] &
    ~df["all_rsi_over_70"].shift(
        1,
        fill_value=False
    )
)


# ============================================================
# EXTRACT SIGNALS
# ============================================================

signals = df[
    df["new_signal"]
].copy()


# ============================================================
# SIGNAL NUMBER
# ============================================================

signals.insert(
    0,
    "signal_id",
    range(1, len(signals) + 1)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("SIGNAL RESULTS")
print("========================================")

print(
    f"Total candles: {len(df):,}"
)

print(
    f"Candles with all 4 RSI > 70: "
    f"{df['all_rsi_over_70'].sum():,}"
)

print(
    f"Independent signal events: "
    f"{len(signals):,}"
)


# ============================================================
# SHOW SIGNALS
# ============================================================

if len(signals) > 0:

    print("\n===== SIGNALS =====")

    display_columns = [
        "signal_id",
        "open",
        "high",
        "low",
        "close",
        "RSI_15m",
        "RSI_30m",
        "RSI_1h",
        "RSI_4h"
    ]

    print(
        signals[display_columns].to_string()
    )

else:

    print("\nNo signals found.")


# ============================================================
# YEARLY BREAKDOWN
# ============================================================

print("\n===== SIGNALS BY YEAR =====")

if len(signals) > 0:

    yearly = (
        signals
        .groupby(signals.index.year)
        .size()
    )

    print(yearly.to_string())

else:

    print("No signals.")


# ============================================================
# MONTHLY BREAKDOWN
# ============================================================

print("\n===== SIGNALS BY YEAR / MONTH =====")

if len(signals) > 0:

    monthly = (
        signals
        .groupby([
            signals.index.year,
            signals.index.month
        ])
        .size()
    )

    monthly.index.names = [
        "year",
        "month"
    ]

    print(monthly.to_string())

else:

    print("No signals.")


# ============================================================
# SAVE
# ============================================================

signals.to_csv(
    OUTPUT_FILE,
    index=True
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)