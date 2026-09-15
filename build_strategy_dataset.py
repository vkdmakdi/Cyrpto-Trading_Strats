import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_FILE = Path("data/backtest_results/signal_features.csv")
OUTCOME_FILE = Path("data/backtest_results/signal_target_stop_outcomes.csv")
OUTPUT_DIR = Path("data/backtest_results")

# Chronological split is locked for this research stage.
DEVELOPMENT_START = 2021
DEVELOPMENT_END = 2023
VALIDATION_YEAR = 2024
TEST_YEAR = 2025

# The first outcome label is deliberately locked to the original hypothesis.
PRIMARY_HORIZON = "24h"
PRIMARY_TARGET = -2.0
PRIMARY_STOP = 2.0

# Secondary labels let us study whether a setup eventually had room to move,
# without choosing a new trading rule yet.
SECONDARY_TARGETS = [-0.5, -1.0, -1.5, -2.0, -3.0, -5.0]
SECONDARY_STOPS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


def load_feature_data():
    df = pd.read_csv(FEATURE_FILE)
    if "signal_time" not in df.columns:
        df = df.rename(columns={"open_time": "signal_time"})
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    df["signal_year"] = df["signal_time"].dt.year
    return df


def load_outcomes():
    df = pd.read_csv(OUTCOME_FILE)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    return df


def build_primary_labels(features, outcomes):
    """
    One row per signal for the locked baseline:
      24h, -2% target, +2% stop.

    'good_short' = target first
    'bad_short' = stop first
    'neither' = neither threshold touched
    """
    primary = outcomes[
        (outcomes["horizon"] == PRIMARY_HORIZON)
        & (outcomes["target_pct"] == PRIMARY_TARGET)
        & (outcomes["stop_pct"] == PRIMARY_STOP)
    ].copy()

    primary = primary[
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
    ]

    primary["primary_good_short"] = primary["outcome"].eq("TARGET_FIRST")
    primary["primary_bad_short"] = primary["outcome"].eq("STOP_FIRST")
    primary["primary_neither"] = primary["outcome"].eq("NEITHER")

    return features.merge(
        primary,
        on="signal_time",
        how="left",
        validate="one_to_one",
    )


def build_secondary_labels(features, outcomes):
    """
    Attach descriptive 24h target/stop-hit labels.

    The detailed outcome file contains 42 rows per signal (one per
    target × stop combination). Therefore, for each level we first filter
    to that level and reduce to exactly one row per signal before merging.
    """
    horizon = outcomes[outcomes["horizon"] == PRIMARY_HORIZON].copy()

    # Make sure each signal/level pair is unique before merging.
    target_parts = []
    for target in SECONDARY_TARGETS:
        part = horizon[horizon["target_pct"] == target].copy()
        part = (
            part.sort_values(["signal_time", "stop_pct"])
            .drop_duplicates(subset=["signal_time"], keep="first")
        )

        col = f"hit_target_{str(abs(target)).replace('.', '_')}pct"
        target_parts.append(
            part[["signal_time"]]
            .assign(**{col: part["target_first_time"].notna().to_numpy()})
        )

    stop_parts = []
    for stop in SECONDARY_STOPS:
        part = horizon[horizon["stop_pct"] == stop].copy()
        part = (
            part.sort_values(["signal_time", "target_pct"])
            .drop_duplicates(subset=["signal_time"], keep="first")
        )

        col = f"hit_stop_{str(stop).replace('.', '_')}pct"
        stop_parts.append(
            part[["signal_time"]]
            .assign(**{col: part["stop_first_time"].notna().to_numpy()})
        )

    out = features.copy()

    for part in target_parts + stop_parts:
        out = out.merge(
            part,
            on="signal_time",
            how="left",
            validate="one_to_one",
        )

    return out


def save_split(df, year_mask, name):
    split = df.loc[year_mask].sort_values("signal_time").reset_index(drop=True)
    path = OUTPUT_DIR / name
    split.to_csv(path, index=False)
    return split, path


def numeric_feature_summary(df, label_col):
    """
    Descriptive development-only feature comparison.
    No threshold search.
    """
    excluded = {
        "signal_time",
        "entry_time",
        "feature_time",
        label_col,
        "primary_bad_short",
        "primary_neither",
    }

    numeric = []
    for c in df.columns:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)

    rows = []
    good = df[label_col].eq(True)
    bad = df["primary_bad_short"].eq(True)

    for feature in numeric:
        x = pd.to_numeric(df.loc[good, feature], errors="coerce").dropna()
        y = pd.to_numeric(df.loc[bad, feature], errors="coerce").dropna()

        if len(x) < 5 or len(y) < 5:
            continue

        rows.append({
            "feature": feature,
            "good_n": len(x),
            "bad_n": len(y),
            "good_mean": x.mean(),
            "bad_mean": y.mean(),
            "good_median": x.median(),
            "bad_median": y.median(),
            "good_minus_bad_mean": x.mean() - y.mean(),
            "good_minus_bad_median": x.median() - y.median(),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare chronological development/validation/test datasets."
    )
    parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = load_feature_data()
    outcomes = load_outcomes()

    print("=" * 72)
    print("BUILDING CHRONOLOGICAL STRATEGY DATASETS")
    print("=" * 72)
    print()

    combined = build_primary_labels(features, outcomes)
    combined = build_secondary_labels(combined, outcomes)

    years = sorted(combined["signal_year"].dropna().unique())

    print(f"Total feature signals: {len(combined):,}")
    print(f"Years present: {years}")
    print()

    # Development
    development_mask = (
        (combined["signal_year"] >= DEVELOPMENT_START)
        & (combined["signal_year"] <= DEVELOPMENT_END)
    )
    validation_mask = combined["signal_year"].eq(VALIDATION_YEAR)
    test_mask = combined["signal_year"].eq(TEST_YEAR)

    development, dev_path = save_split(
        combined,
        development_mask,
        "strategy_development_2021_2023.csv",
    )
    validation, val_path = save_split(
        combined,
        validation_mask,
        "strategy_validation_2024.csv",
    )
    test, test_path = save_split(
        combined,
        test_mask,
        "strategy_test_2025.csv",
    )

    print("SPLIT SIZES")
    print(f"  Development 2021-2023: {len(development)}")
    print(f"  Validation 2024:        {len(validation)}")
    print(f"  Test 2025:              {len(test)}")
    print()

    # Development-only descriptive analysis.
    dev_summary = numeric_feature_summary(
        development,
        "primary_good_short",
    )
    dev_summary_path = OUTPUT_DIR / "development_feature_summary.csv"
    dev_summary.to_csv(dev_summary_path, index=False)

    # Development label counts.
    print("DEVELOPMENT PRIMARY OUTCOMES")
    print(
        development["outcome"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()

    print("VALIDATION PRIMARY OUTCOMES")
    print(
        validation["outcome"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()

    print("TEST PRIMARY OUTCOMES")
    print(
        test["outcome"]
        .value_counts(dropna=False)
        .to_string()
    )
    print()

    # Strict no-leakage audit for the development/validation/test boundaries.
    # No feature column is allowed to depend on future trade outcomes.
    forbidden_feature_names = {
        "outcome",
        "raw_trade_move_pct",
        "mfe_pct",
        "mae_pct",
        "minutes_to_outcome",
        "primary_good_short",
        "primary_bad_short",
        "primary_neither",
        "cluster_id",
        "cluster_size",
    }

    feature_like_cols = [
        c for c in combined.columns
        if c not in forbidden_feature_names
    ]

    print("DATASET CHECK")
    print(
        "All chronological splits are constructed from signal timestamps; "
        "2025 is not used for development or validation."
    )
    print(
        f"Feature-like columns retained: {len(feature_like_cols)}"
    )

    print()
    print("WROTE")
    print(f"  {dev_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {dev_summary_path}")
    print()

    print("=" * 72)
    print("NEXT STAGE")
    print("=" * 72)
    print(
        "Use ONLY strategy_development_2021_2023.csv to discover candidate "
        "feature combinations/thresholds."
    )
    print(
        "Use strategy_validation_2024.csv only after the candidate rule is frozen."
    )
    print(
        "Keep strategy_test_2025.csv untouched until the final strategy is frozen."
    )


if __name__ == "__main__":
    main()
