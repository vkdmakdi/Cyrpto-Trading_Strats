import os
import zipfile
import requests
import pandas as pd
from io import BytesIO
from datetime import datetime

# ============================================================
# SETTINGS
# ============================================================

SYMBOL = "BTCUSDT"
INTERVAL = "1m"

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

OUTPUT_FILE = f"data/{SYMBOL}_{INTERVAL}.csv"

BASE_URL = (
    "https://data.binance.vision/data/spot/monthly/"
    f"klines/{SYMBOL}/{INTERVAL}/"
)

# ============================================================
# BINANCE TIMESTAMP CONVERTER
# ============================================================

def convert_binance_timestamp(series):
    """
    Binance Spot timestamps:

    Before 2025:
        milliseconds (~13 digits)

    2025 onward:
        microseconds (~16 digits)

    Detect the unit automatically based on magnitude.
    """

    values = pd.to_numeric(series, errors="coerce")

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns, UTC]"
    )

    # Milliseconds
    ms_mask = (
        values.notna()
        & (values >= 10**12)
        & (values < 10**15)
    )

    # Microseconds
    us_mask = values.notna() & (values >= 10**15)

    if ms_mask.any():
        result.loc[ms_mask] = pd.to_datetime(
            values.loc[ms_mask],
            unit="ms",
            utc=True,
            errors="coerce"
        )

    if us_mask.any():
        result.loc[us_mask] = pd.to_datetime(
            values.loc[us_mask],
            unit="us",
            utc=True,
            errors="coerce"
        )

    return result


# ============================================================
# DOWNLOAD ONE MONTH
# ============================================================

def download_month(year, month):

    month_str = f"{year}-{month:02d}"

    filename = f"{SYMBOL}-{INTERVAL}-{month_str}.zip"
    url = BASE_URL + filename

    print(f"Downloading {filename}")

    response = requests.get(url, timeout=60)

    if response.status_code != 200:
        print(f"  ! Failed: HTTP {response.status_code}")
        return None

    try:
        with zipfile.ZipFile(BytesIO(response.content)) as z:

            csv_files = [
                name for name in z.namelist()
                if name.endswith(".csv")
            ]

            if not csv_files:
                print("  ! No CSV found")
                return None

            with z.open(csv_files[0]) as f:

                df = pd.read_csv(
                    f,
                    header=None
                )

        print(f"  → {len(df):,} raw rows")

        # Binance kline columns
        df.columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ]

        # ====================================================
        # IMPORTANT:
        # Automatically detect ms vs microseconds.
        # ====================================================

        df["open_time"] = convert_binance_timestamp(
            df["open_time"]
        )

        df["close_time"] = convert_binance_timestamp(
            df["close_time"]
        )

        # Check that timestamps survived
        bad_timestamps = df["open_time"].isna().sum()

        if bad_timestamps:
            print(
                f"  ! WARNING: {bad_timestamps:,} "
                f"timestamps failed conversion"
            )

        # Numeric OHLCV columns
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # Remove unusable rows
        df = df.dropna(
            subset=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

        # Keep only required columns
        df = df[
            [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        ]

        df = df.sort_values("open_time")

        if len(df) > 0:
            print(
                f"  → {len(df):,} valid rows | "
                f"{df['open_time'].iloc[0]} → "
                f"{df['open_time'].iloc[-1]}"
            )

        return df

    except Exception as e:
        print(f"  ! ERROR processing file: {e}")
        return None


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs("data", exist_ok=True)

    start = pd.Timestamp(
        START_DATE,
        tz="UTC"
    )

    end = pd.Timestamp(
        END_DATE,
        tz="UTC"
    )

    all_data = []

    current = start

    while current < end:

        year = current.year
        month = current.month

        df = download_month(year, month)

        if df is not None and len(df) > 0:
            all_data.append(df)

        # Move to first day of next month
        if month == 12:
            current = pd.Timestamp(
                year=year + 1,
                month=1,
                day=1,
                tz="UTC"
            )
        else:
            current = pd.Timestamp(
                year=year,
                month=month + 1,
                day=1,
                tz="UTC"
            )

    # ========================================================
    # COMBINE
    # ========================================================

    print("\nCombining monthly data...")

    if not all_data:
        print("ERROR: No data downloaded.")
        return

    data = pd.concat(
        all_data,
        ignore_index=True
    )

    # Remove duplicates
    data = data.drop_duplicates(
        subset=["open_time"]
    )

    # Sort
    data = data.sort_values(
        "open_time"
    ).reset_index(drop=True)

    # Final date restriction
    data = data[
        (data["open_time"] >= start)
        & (data["open_time"] < end)
    ]

    # Save
    data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n========================================")
    print("FINAL DATASET")
    print("========================================")

    print(f"Rows:  {len(data):,}")

    if len(data) > 0:
        print(f"Start: {data['open_time'].iloc[0]}")
        print(f"End:   {data['open_time'].iloc[-1]}")

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()