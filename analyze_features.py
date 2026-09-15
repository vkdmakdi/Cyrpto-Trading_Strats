import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_FILE = Path("data/backtest_results/signal_features.csv")
OUTCOME_FILE = Path("data/backtest_results/signal_target_stop_outcomes.csv")
HORIZON_FILE = Path("data/backtest_results/signal_horizon_metrics.csv")
OUTPUT_DIR = Path("data/backtest_results")

# FIRST-PASS ANALYSIS IS LOCKED to the original hypothesis.
ANALYSIS_HORIZON = "24h"
TARGET_PCT = -2.0
STOP_PCT = 2.0

# Features we want to compare.
FEATURES = [
    "RSI_15m",
    "RSI_30m",
    "RSI_1h",
    "RSI_4h",
    "feature_RSI_15m",
    "feature_RSI_30m",
    "feature_RSI_1h",
    "feature_RSI_4h",
    "RSI15_delta_1",
    "RSI15_delta_4",
    "RSI30_delta_1",
    "RSI30_delta_4",
    "RSI1h_delta_1",
    "RSI1h_delta_4",
    "RSI4h_delta_1",
    "RSI4h_delta_4",
    "close_vs_EMA20_pct",
    "close_vs_EMA50_pct",
    "close_vs_EMA200_pct",
    "EMA20_slope_4_pct",
    "EMA50_slope_4_pct",
    "EMA200_slope_16_pct",
    "ATR_pct",
    "BB_position",
    "BB_width_pct",
    "volume_ratio_20",
    "return_1h_pct",
    "return_4h_pct",
    "return_12h_pct",
    "return_24h_pct",
    "return_1h_delta_pct",
    "distance_prior_8_bar_high_pct",
    "distance_prior_16_bar_high_pct",
    "distance_prior_32_bar_high_pct",
    "distance_prior_64_bar_high_pct",
    "distance_prior_8_bar_low_pct",
    "distance_prior_16_bar_low_pct",
    "distance_prior_32_bar_low_pct",
    "distance_prior_64_bar_low_pct",
    "candle_range_pct",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
]

CATEGORICAL_FEATURES = [
    "trend_regime_200EMA",
    "price_above_upper_BB",
    "volume_above_2x_avg",
    "price_more_than_5pct_above_EMA200",
    "RSI15_falling_1bar",
    "RSI15_falling_4bars",
]


def pct(x):
    return f"{x * 100:.2f}%"


def cliffs_delta(x, y):
    """
    Rank-based effect size:
    P(X > Y) - P(X < Y)

    Positive => X tends to be larger than Y.
    Negative => X tends to be smaller than Y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    # Exact pairwise implementation is fine for only 120 observations.
    diff = x[:, None] - y[None, :]
    return float((diff > 0).mean() - (diff < 0).mean())


def standardized_mean_difference(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)

    pooled = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))

    if pooled == 0:
        return 0.0

    return float((np.mean(x) - np.mean(y)) / pooled)


def univariate_auc(x, y):
    """
    AUC via pairwise comparisons.
    Uses X as 'positive' class and Y as 'negative' class.
    0.5 = no separation.
    >0.5 = positive class tends to be larger.
    <0.5 = positive class tends to be smaller.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    diff = x[:, None] - y[None, :]
    greater = (diff > 0).sum()
    equal = (diff == 0).sum()
    return float((greater + 0.5 * equal) / diff.size)


def describe_feature(df, good_mask, bad_mask, neither_mask, feature):
    x_good = pd.to_numeric(df.loc[good_mask, feature], errors="coerce")
    x_bad = pd.to_numeric(df.loc[bad_mask, feature], errors="coerce")
    x_neither = pd.to_numeric(df.loc[neither_mask, feature], errors="coerce")

    good = x_good.dropna().to_numpy()
    bad = x_bad.dropna().to_numpy()
    neither = x_neither.dropna().to_numpy()

    if len(good) == 0 or len(bad) == 0:
        return None

    result = {
        "feature": feature,
        "good_n": len(good),
        "bad_n": len(bad),
        "neither_n": len(neither),
        "good_mean": np.mean(good),
        "bad_mean": np.mean(bad),
        "good_median": np.median(good),
        "bad_median": np.median(bad),
        "good_q25": np.percentile(good, 25),
        "good_q75": np.percentile(good, 75),
        "bad_q25": np.percentile(bad, 25),
        "bad_q75": np.percentile(bad, 75),
        "good_minus_bad_mean": np.mean(good) - np.mean(bad),
        "good_minus_bad_median": np.median(good) - np.median(bad),
        "standardized_mean_difference": standardized_mean_difference(good, bad),
        "auc_good_vs_bad": univariate_auc(good, bad),
        "cliffs_delta_good_vs_bad": cliffs_delta(good, bad),
    }

    if len(neither):
        result.update({
            "neither_mean": np.mean(neither),
            "neither_median": np.median(neither),
        })
    else:
        result.update({
            "neither_mean": np.nan,
            "neither_median": np.nan,
        })

    return result


def build_quantile_table(df, good_mask, bad_mask, feature):
    values = pd.to_numeric(df[feature], errors="coerce")
    temp = pd.DataFrame({
        "value": values,
        "good": good_mask,
        "bad": bad_mask,
    }).dropna(subset=["value"])

    if len(temp) < 10:
        return None

    try:
        temp["quartile"] = pd.qcut(
            temp["value"],
            q=4,
            labels=["Q1_low", "Q2", "Q3", "Q4_high"],
            duplicates="drop",
        )
    except ValueError:
        return None

    rows = []
    for q, g in temp.groupby("quartile", observed=True):
        total = len(g)
        wins = int(g["good"].sum())
        losses = int(g["bad"].sum())
        rows.append({
            "feature": feature,
            "quartile": str(q),
            "signals": total,
            "good_short": wins,
            "bad_short": losses,
            "good_rate_pct": 100.0 * wins / total if total else np.nan,
            "bad_rate_pct": 100.0 * losses / total if total else np.nan,
            "mean_feature_value": g["value"].mean(),
        })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ANALYZING SIGNAL FEATURES VS OUTCOMES")
    print("=" * 72)
    print()
    print("LOCKED FIRST-PASS TEST:")
    print(f"  Horizon: {ANALYSIS_HORIZON}")
    print(f"  Target:  {TARGET_PCT:+.1f}%")
    print(f"  Stop:    {STOP_PCT:+.1f}%")
    print()

    features = pd.read_csv(FEATURE_FILE)
    outcomes = pd.read_csv(OUTCOME_FILE)

    features["open_time"] = pd.to_datetime(features["open_time"], utc=True)
    outcomes["signal_time"] = pd.to_datetime(outcomes["signal_time"], utc=True)

    outcomes = outcomes[
        (outcomes["horizon"] == ANALYSIS_HORIZON)
        & (outcomes["target_pct"] == TARGET_PCT)
        & (outcomes["stop_pct"] == STOP_PCT)
    ].copy()

    # Every signal should appear exactly once for the locked setup.
    signal_keys = features[["signal_time"]].copy() if "signal_time" in features.columns else None

    # build_features.py uses open_time as the signal timestamp.
    if "signal_time" not in features.columns:
        features = features.rename(columns={"open_time": "signal_time"})

    merged = features.merge(
        outcomes[
            [
                "signal_time",
                "outcome",
                "raw_trade_move_pct",
                "mfe_pct",
                "mae_pct",
                "minutes_to_outcome",
                "cluster_id",
                "cluster_size",
            ]
        ],
        on="signal_time",
        how="inner",
        validate="one_to_one",
    )

    print(f"Feature rows: {len(features):,}")
    print(f"Matched outcome rows: {len(merged):,}")
    print()

    if len(merged) != len(features):
        missing = sorted(
            set(features["signal_time"]) - set(merged["signal_time"])
        )
        print(f"WARNING: {len(missing)} feature rows did not match outcomes.")

    # Locked outcome groups.
    good_mask = merged["outcome"].eq("TARGET_FIRST")
    bad_mask = merged["outcome"].eq("STOP_FIRST")
    neither_mask = merged["outcome"].eq("NEITHER")

    ambiguous_mask = merged["outcome"].eq("AMBIGUOUS")
    gap_mask = merged["outcome"].eq("GAP_BLOCKED")

    print("OUTCOME COUNTS")
    print(f"  Good short (target first): {int(good_mask.sum())}")
    print(f"  Bad short (stop first):    {int(bad_mask.sum())}")
    print(f"  Neither:                   {int(neither_mask.sum())}")
    print(f"  Ambiguous:                 {int(ambiguous_mask.sum())}")
    print(f"  Gap blocked:               {int(gap_mask.sum())}")
    print()

    resolved = good_mask | bad_mask
    resolved_n = int(resolved.sum())

    if resolved_n:
        print(
            f"Target-first among resolved: "
            f"{100.0 * good_mask.sum() / resolved_n:.2f}%"
        )
    print()

    numeric_rows = []
    for feature in FEATURES:
        if feature not in merged.columns:
            continue
        row = describe_feature(
            merged,
            good_mask,
            bad_mask,
            neither_mask,
            feature,
        )
        if row is not None:
            numeric_rows.append(row)

    numeric_df = pd.DataFrame(numeric_rows)

    if not numeric_df.empty:
        numeric_df["distance_from_random_auc"] = (
            numeric_df["auc_good_vs_bad"] - 0.5
        ).abs()
        numeric_df = numeric_df.sort_values(
            "distance_from_random_auc",
            ascending=False,
        ).reset_index(drop=True)

    categorical_rows = []
    for feature in CATEGORICAL_FEATURES:
        if feature not in merged.columns:
            continue

        temp = merged.loc[resolved, [feature]].copy()
        temp["good"] = good_mask.loc[resolved].to_numpy()

        for value, g in temp.groupby(feature, dropna=False):
            n = len(g)
            wins = int(g["good"].sum())
            categorical_rows.append({
                "feature": feature,
                "value": value,
                "signals": n,
                "good_short": wins,
                "bad_short": n - wins,
                "good_rate_among_resolved_pct": (
                    100.0 * wins / n if n else np.nan
                ),
            })

    categorical_df = pd.DataFrame(categorical_rows)

    quantile_frames = []
    for feature in FEATURES:
        if feature not in merged.columns:
            continue
        q = build_quantile_table(
            merged,
            good_mask,
            bad_mask,
            feature,
        )
        if q is not None:
            quantile_frames.append(q)

    quantile_df = (
        pd.concat(quantile_frames, ignore_index=True)
        if quantile_frames
        else pd.DataFrame()
    )

    # Simple correlations with a locked binary label.
    # 1 = target first, 0 = stop first. Neither is excluded.
    correlation_rows = []
    for feature in FEATURES:
        if feature not in merged.columns:
            continue

        temp = merged.loc[resolved, [feature]].copy()
        temp[feature] = pd.to_numeric(temp[feature], errors="coerce")
        temp["good_label"] = good_mask.loc[resolved].astype(int).to_numpy()
        temp = temp.dropna()

        if len(temp) >= 5 and temp[feature].nunique() > 1:
            correlation_rows.append({
                "feature": feature,
                "pearson_corr_good_label": temp[feature].corr(
                    temp["good_label"], method="pearson"
                ),
                "spearman_corr_good_label": temp[feature].corr(
                    temp["good_label"], method="spearman"
                ),
                "n": len(temp),
            })

    corr_df = pd.DataFrame(correlation_rows)
    if not corr_df.empty:
        corr_df["abs_spearman"] = corr_df["spearman_corr_good_label"].abs()
        corr_df = corr_df.sort_values(
            "abs_spearman",
            ascending=False,
        ).reset_index(drop=True)

    # Save outputs.
    merged_file = OUTPUT_DIR / "feature_outcome_dataset.csv"
    numeric_file = OUTPUT_DIR / "feature_vs_outcome_numeric.csv"
    categorical_file = OUTPUT_DIR / "feature_vs_outcome_categorical.csv"
    quantile_file = OUTPUT_DIR / "feature_vs_outcome_quantiles.csv"
    correlation_file = OUTPUT_DIR / "feature_vs_outcome_correlations.csv"

    merged.to_csv(merged_file, index=False)
    numeric_df.to_csv(numeric_file, index=False)
    categorical_df.to_csv(categorical_file, index=False)
    quantile_df.to_csv(quantile_file, index=False)
    corr_df.to_csv(correlation_file, index=False)

    print("FILES WRITTEN")
    print(f"  {merged_file}")
    print(f"  {numeric_file}")
    print(f"  {categorical_file}")
    print(f"  {quantile_file}")
    print(f"  {correlation_file}")
    print()

    # Console report: only descriptive / univariate results.
    print("=" * 72)
    print("TOP FEATURES BY AUC SEPARATION")
    print("=" * 72)
    if numeric_df.empty:
        print("No numeric feature results available.")
    else:
        cols = [
            "feature",
            "good_median",
            "bad_median",
            "auc_good_vs_bad",
            "cliffs_delta_good_vs_bad",
            "standardized_mean_difference",
        ]
        print(numeric_df[cols].head(20).to_string(index=False))

    print()
    print("=" * 72)
    print("TOP FEATURES BY SPEARMAN CORRELATION")
    print("=" * 72)
    if corr_df.empty:
        print("No correlation results available.")
    else:
        print(
            corr_df[
                [
                    "feature",
                    "pearson_corr_good_label",
                    "spearman_corr_good_label",
                    "n",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print()
    print("=" * 72)
    print("IMPORTANT")
    print("=" * 72)
    print(
        "These are exploratory univariate associations, not strategy rules."
    )
    print(
        "Do NOT select thresholds from this full 2021-2025 sample."
    )
    print(
        "The next step is to reproduce the analysis on chronological "
        "development/validation/test periods."
    )


if __name__ == "__main__":
    main()
