import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Build point-in-time market features for the 120 RSI signal events."
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="data/backtest_results")
    return p.parse_args()


def true_range(df):
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def add_segmented_indicators(df, gap_minutes=15):
    """
    Calculate indicators separately inside continuous 15m segments.
    This prevents a real data gap from contaminating rolling/EMA calculations.
    """
    df = df.copy()
    t = df["open_time"].array.asi8
    gap_ns = gap_minutes * 60 * 1_000_000_000
    new_segment = np.r_[True, np.diff(t) > gap_ns]
    df["_segment"] = np.cumsum(new_segment)

    pieces = []

    for _, g in df.groupby("_segment", sort=False):
        g = g.copy()

        close = g["close"]
        volume = g["volume"]

        # Trend
        g["EMA_20"] = close.ewm(span=20, adjust=False).mean()
        g["EMA_50"] = close.ewm(span=50, adjust=False).mean()
        g["EMA_200"] = close.ewm(span=200, adjust=False).mean()

        # ATR(14), Wilder-style smoothing
        tr = true_range(g)
        g["ATR_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

        # Bollinger Bands: 20-period SMA, 2 std
        g["BB_mid_20"] = close.rolling(20, min_periods=20).mean()
        g["BB_std_20"] = close.rolling(20, min_periods=20).std(ddof=0)
        g["BB_upper_20"] = g["BB_mid_20"] + 2.0 * g["BB_std_20"]
        g["BB_lower_20"] = g["BB_mid_20"] - 2.0 * g["BB_std_20"]

        # Volume context
        g["volume_SMA_20"] = volume.rolling(20, min_periods=20).mean()

        # Price returns from the completed signal candle
        for n, label in [(4, "1h"), (16, "4h"), (48, "12h"), (96, "24h")]:
            g[f"return_{label}_pct"] = (close / close.shift(n) - 1.0) * 100.0

        # RSI momentum / slope. The RSI columns already exist in BTCUSDT_RSI.csv.
        for col, prefix in [
            ("RSI_15m", "RSI15"),
            ("RSI_30m", "RSI30"),
            ("RSI_1h", "RSI1h"),
            ("RSI_4h", "RSI4h"),
        ]:
            if col in g.columns:
                g[f"{prefix}_delta_1"] = g[col] - g[col].shift(1)
                g[f"{prefix}_delta_4"] = g[col] - g[col].shift(4)

        # Price acceleration over the recent 15m bars
        g["return_1h_delta_pct"] = (
            g["return_1h_pct"] - g["return_1h_pct"].shift(4)
        )

        # Distance to trend measures
        g["close_vs_EMA20_pct"] = (close / g["EMA_20"] - 1.0) * 100.0
        g["close_vs_EMA50_pct"] = (close / g["EMA_50"] - 1.0) * 100.0
        g["close_vs_EMA200_pct"] = (close / g["EMA_200"] - 1.0) * 100.0

        # EMA slopes, measured as percentage change over recent completed bars.
        g["EMA20_slope_4_pct"] = (g["EMA_20"] / g["EMA_20"].shift(4) - 1.0) * 100.0
        g["EMA50_slope_4_pct"] = (g["EMA_50"] / g["EMA_50"].shift(4) - 1.0) * 100.0
        g["EMA200_slope_16_pct"] = (
            g["EMA_200"] / g["EMA_200"].shift(16) - 1.0
        ) * 100.0

        # Volatility
        g["ATR_pct"] = g["ATR_14"] / close * 100.0

        # Bollinger position:
        # 0 = lower band, 0.5 = middle, 1 = upper band.
        band_width = g["BB_upper_20"] - g["BB_lower_20"]
        g["BB_position"] = (close - g["BB_lower_20"]) / band_width
        g["BB_width_pct"] = band_width / g["BB_mid_20"] * 100.0

        # Volume relative to recent average.
        g["volume_ratio_20"] = volume / g["volume_SMA_20"]

        # Price structure using PRIOR completed candles only.
        # Excluding the current candle prevents the current spike from defining
        # the reference level we're comparing against.
        for n in [8, 16, 32, 64]:
            prior = g["high"].shift(1).rolling(n, min_periods=n).max()
            prior_low = g["low"].shift(1).rolling(n, min_periods=n).min()

            g[f"prior_{n}_bar_high"] = prior
            g[f"prior_{n}_bar_low"] = prior_low
            g[f"distance_prior_{n}_bar_high_pct"] = (close / prior - 1.0) * 100.0
            g[f"distance_prior_{n}_bar_low_pct"] = (close / prior_low - 1.0) * 100.0

        # Candle anatomy of the completed signal candle.
        candle_range = g["high"] - g["low"]
        g["candle_range_pct"] = candle_range / close * 100.0
        g["candle_body_pct"] = (g["close"] - g["open"]) / g["open"] * 100.0
        g["upper_wick_pct"] = (
            g["high"] - g[["open", "close"]].max(axis=1)
        ) / g["open"] * 100.0
        g["lower_wick_pct"] = (
            g[["open", "close"]].min(axis=1) - g["low"]
        ) / g["open"] * 100.0

        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)
    return out.drop(columns=["_segment"])


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    price = pd.read_csv(data_dir / "BTCUSDT_15m.csv")
    price["open_time"] = pd.to_datetime(price["open_time"], utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        price[col] = pd.to_numeric(price[col], errors="coerce")

    rsi = pd.read_csv(data_dir / "BTCUSDT_RSI.csv")
    rsi["open_time"] = pd.to_datetime(rsi["open_time"], utc=True)

    signals = pd.read_csv(data_dir / "BTCUSDT_signals.csv")
    signals["open_time"] = pd.to_datetime(signals["open_time"], utc=True)
    signals = signals.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    # Prefer the RSI file as the canonical source for the RSI values.
    base = price.merge(rsi, on="open_time", how="left", suffixes=("", "_rsi"))

    features = add_segmented_indicators(base)

    # A signal is only known after its 15m candle closes.
    signals["entry_time"] = signals["open_time"] + pd.Timedelta(minutes=15)

    # Point-in-time merge: signal timestamp must match the completed 15m candle.
    feature_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "RSI_15m", "RSI_30m", "RSI_1h", "RSI_4h",
        "EMA_20", "EMA_50", "EMA_200",
        "ATR_14", "ATR_pct",
        "BB_mid_20", "BB_upper_20", "BB_lower_20",
        "BB_position", "BB_width_pct",
        "volume_SMA_20", "volume_ratio_20",
        "return_1h_pct", "return_4h_pct", "return_12h_pct", "return_24h_pct",
        "return_1h_delta_pct",
        "close_vs_EMA20_pct", "close_vs_EMA50_pct", "close_vs_EMA200_pct",
        "EMA20_slope_4_pct", "EMA50_slope_4_pct", "EMA200_slope_16_pct",
        "RSI15_delta_1", "RSI15_delta_4",
        "RSI30_delta_1", "RSI30_delta_4",
        "RSI1h_delta_1", "RSI1h_delta_4",
        "RSI4h_delta_1", "RSI4h_delta_4",
        "candle_range_pct", "candle_body_pct",
        "upper_wick_pct", "lower_wick_pct",
    ]

    for n in [8, 16, 32, 64]:
        feature_cols.extend(
            [
                f"prior_{n}_bar_high",
                f"prior_{n}_bar_low",
                f"distance_prior_{n}_bar_high_pct",
                f"distance_prior_{n}_bar_low_pct",
            ]
        )

    available = [c for c in feature_cols if c in features.columns]
    feature_frame = features[available].copy()

    # The signal file already contains OHLC + raw RSI columns. Rename the
    # feature-side copies so the merge never creates *_x / *_y collisions.
    feature_frame = feature_frame.rename(columns={
        "open": "feature_open",
        "high": "feature_high",
        "low": "feature_low",
        "close": "feature_close",
        "volume": "feature_volume",
        "RSI_15m": "feature_RSI_15m",
        "RSI_30m": "feature_RSI_30m",
        "RSI_1h": "feature_RSI_1h",
        "RSI_4h": "feature_RSI_4h",
    })

    signal_features = signals.merge(
        feature_frame,
        on="open_time",
        how="left",
        validate="one_to_one",
    )

    # Add explicit point-in-time sanity flags.
    signal_features["feature_time"] = signal_features["open_time"]
    signal_features["known_before_entry"] = (
        signal_features["feature_time"] < signal_features["entry_time"]
    )

    # A compact categorical description of trend context.
    signal_features["trend_regime_200EMA"] = np.select(
        [
            (signal_features["feature_close"] > signal_features["EMA_200"])
            & (signal_features["EMA200_slope_16_pct"] > 0),
            (signal_features["feature_close"] < signal_features["EMA_200"])
            & (signal_features["EMA200_slope_16_pct"] < 0),
        ],
        ["bull_trend", "bear_trend"],
        default="mixed",
    )

    # Useful binary context flags; these are descriptive, not strategy rules.
    signal_features["price_above_upper_BB"] = (
        signal_features["feature_close"] > signal_features["BB_upper_20"]
    )
    signal_features["volume_above_2x_avg"] = signal_features["volume_ratio_20"] >= 2.0
    signal_features["price_more_than_5pct_above_EMA200"] = (
        signal_features["close_vs_EMA200_pct"] >= 5.0
    )
    signal_features["RSI15_falling_1bar"] = signal_features["RSI15_delta_1"] < 0
    signal_features["RSI15_falling_4bars"] = signal_features["RSI15_delta_4"] < 0

    # Basic integrity report.
    required = [
        "RSI_15m", "RSI_30m", "RSI_1h", "RSI_4h",
        "EMA_200", "ATR_pct", "BB_position", "volume_ratio_20",
    ]
    missing_counts = signal_features[required].isna().sum()

    print(f"15m candles: {len(price):,}")
    print(f"Signal events: {len(signals):,}")
    print(f"Feature rows: {len(signal_features):,}")
    print()
    print("POINT-IN-TIME CHECK")
    print(f"All features known before entry: {signal_features['known_before_entry'].all()}")
    print()
    print("Missing values in core features:")
    print(missing_counts.to_string())
    print()
    print("TREND REGIME COUNTS")
    print(signal_features["trend_regime_200EMA"].value_counts(dropna=False).to_string())
    print()
    print("FEATURES:")
    print(
        "RSI level/slope, EMA trend, ATR, Bollinger Bands, volume, recent returns, "
        "prior price structure, and signal-candle anatomy."
    )

    output = out_dir / "signal_features.csv"
    signal_features.to_csv(output, index=False)

    # A small descriptive summary for quick inspection.
    summary_cols = [
        "RSI_15m", "RSI_30m", "RSI_1h", "RSI_4h",
        "RSI15_delta_4", "close_vs_EMA200_pct", "EMA200_slope_16_pct",
        "ATR_pct", "BB_position", "volume_ratio_20",
        "return_1h_pct", "return_4h_pct", "return_12h_pct",
        "return_24h_pct", "candle_body_pct", "upper_wick_pct",
        "lower_wick_pct",
    ]
    summary = signal_features[summary_cols].describe().T
    summary.to_csv(out_dir / "signal_features_summary.csv")

    print()
    print(f"WROTE: {output}")
    print(f"WROTE: {out_dir / 'signal_features_summary.csv'}")


if __name__ == "__main__":
    main()
