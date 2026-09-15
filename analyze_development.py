import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path("data/backtest_results/strategy_development_2021_2023.csv")
OUTPUT_DIR = Path("data/backtest_results")

# Locked baseline outcome from the original hypothesis.
GOOD = "TARGET_FIRST"
BAD = "STOP_FIRST"
NEITHER = "NEITHER"

# Keep this deliberately small. We are looking for robust information groups,
# not building a giant decision tree from 70 observations.
MIN_GROUP_N = 10
TOP_FEATURES = 15


ID_LIKE = {
    "signal_time",
    "entry_time",
    "feature_time",
    "signal_year",
    "cluster_id",
    "cluster_size",
}

OUTCOME_LIKE = {
    "outcome",
    "primary_good_short",
    "primary_bad_short",
    "primary_neither",
    "raw_trade_move_pct",
    "mfe_pct",
    "mae_pct",
    "minutes_to_outcome",
}

LABEL_PREFIXES = (
    "hit_target_",
    "hit_stop_",
)


def auc_probability_good(x_good: pd.Series, x_bad: pd.Series) -> float:
    """AUC of feature values for ranking GOOD above BAD."""
    x_good = pd.to_numeric(x_good, errors="coerce").dropna()
    x_bad = pd.to_numeric(x_bad, errors="coerce").dropna()
    if len(x_good) == 0 or len(x_bad) == 0:
        return np.nan

    combined = pd.concat(
        [x_good.rename("x"), x_bad.rename("x")],
        ignore_index=True,
    )
    ranks = combined.rank(method="average")
    r_good = ranks.iloc[: len(x_good)].sum()
    return (r_good - len(x_good) * (len(x_good) + 1) / 2) / (len(x_good) * len(x_bad))


def cliffs_delta(x_good: pd.Series, x_bad: pd.Series) -> float:
    """Cliff's delta using rank statistics, avoiding O(n^2) comparisons."""
    x_good = pd.to_numeric(x_good, errors="coerce").dropna()
    x_bad = pd.to_numeric(x_bad, errors="coerce").dropna()
    if len(x_good) == 0 or len(x_bad) == 0:
        return np.nan

    combined = pd.concat(
        [x_good.rename("x"), x_bad.rename("x")],
        ignore_index=True,
    )
    ranks = combined.rank(method="average")
    r_good = ranks.iloc[: len(x_good)].sum()
    u = r_good - len(x_good) * (len(x_good) + 1) / 2
    return 2 * u / (len(x_good) * len(x_bad)) - 1


def numeric_features(df: pd.DataFrame) -> list[str]:
    features = []
    for col in df.columns:
        if col in ID_LIKE or col in OUTCOME_LIKE:
            continue
        if any(col.startswith(prefix) for prefix in LABEL_PREFIXES):
            continue
        if pd.api.types.is_bool_dtype(df[col]):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)
    return features


def association_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    good = df["outcome"].eq(GOOD)
    bad = df["outcome"].eq(BAD)

    rows = []
    for feature in features:
        x = pd.to_numeric(df.loc[good, feature], errors="coerce")
        y = pd.to_numeric(df.loc[bad, feature], errors="coerce")
        valid_good = x.dropna()
        valid_bad = y.dropna()
        if len(valid_good) < 5 or len(valid_bad) < 5:
            continue

        rows.append(
            {
                "feature": feature,
                "good_n": len(valid_good),
                "bad_n": len(valid_bad),
                "good_mean": valid_good.mean(),
                "bad_mean": valid_bad.mean(),
                "good_median": valid_good.median(),
                "bad_median": valid_bad.median(),
                "mean_diff_good_minus_bad": valid_good.mean() - valid_bad.mean(),
                "median_diff_good_minus_bad": valid_good.median() - valid_bad.median(),
                "auc_good": auc_probability_good(valid_good, valid_bad),
                "cliffs_delta": cliffs_delta(valid_good, valid_bad),
                "good_p25": valid_good.quantile(0.25),
                "good_p75": valid_good.quantile(0.75),
                "bad_p25": valid_bad.quantile(0.25),
                "bad_p75": valid_bad.quantile(0.75),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["auc_distance"] = (out["auc_good"] - 0.5).abs()
    return out.sort_values("auc_distance", ascending=False).reset_index(drop=True)


def quantile_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Development-only quartile analysis.

    The bins are learned separately for each feature using the development set
    only. This is explicitly exploratory and must not be treated as validation.
    """
    rows = []

    for feature in features:
        x = pd.to_numeric(df[feature], errors="coerce")
        if x.notna().sum() < 20 or x.nunique(dropna=True) < 4:
            continue

        try:
            bins = pd.qcut(x, q=4, duplicates="drop")
        except ValueError:
            continue

        tmp = df.assign(_bucket=bins)
        grouped = tmp.groupby("_bucket", observed=True)

        for bucket, g in grouped:
            n = len(g)
            good_n = int(g["outcome"].eq(GOOD).sum())
            bad_n = int(g["outcome"].eq(BAD).sum())
            neither_n = int(g["outcome"].eq(NEITHER).sum())
            resolved_n = good_n + bad_n

            rows.append(
                {
                    "feature": feature,
                    "bucket": str(bucket),
                    "n": n,
                    "feature_median": pd.to_numeric(g[feature], errors="coerce").median(),
                    "target_first_n": good_n,
                    "stop_first_n": bad_n,
                    "neither_n": neither_n,
                    "resolved_n": resolved_n,
                    "target_first_rate_resolved": good_n / resolved_n if resolved_n else np.nan,
                    "mean_raw_trade_move_pct": pd.to_numeric(g["raw_trade_move_pct"], errors="coerce").mean(),
                    "median_mfe_pct": pd.to_numeric(g["mfe_pct"], errors="coerce").median(),
                    "median_mae_pct": pd.to_numeric(g["mae_pct"], errors="coerce").median(),
                }
            )

    return pd.DataFrame(rows)


def boolean_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in df.columns:
        if not pd.api.types.is_bool_dtype(df[feature]):
            continue
        if feature.startswith("primary_") or feature.startswith("hit_"):
            continue

        for value in [False, True]:
            g = df[df[feature].eq(value)]
            n = len(g)
            if n == 0:
                continue
            good_n = int(g["outcome"].eq(GOOD).sum())
            bad_n = int(g["outcome"].eq(BAD).sum())
            resolved_n = good_n + bad_n
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "n": n,
                    "target_first_n": good_n,
                    "stop_first_n": bad_n,
                    "neither_n": int(g["outcome"].eq(NEITHER).sum()),
                    "resolved_n": resolved_n,
                    "target_first_rate_resolved": good_n / resolved_n if resolved_n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def candidate_thresholds(df: pd.DataFrame, association: pd.DataFrame) -> pd.DataFrame:
    """
    Test a small number of development-derived one-feature filters.

    Thresholds are the development quartiles. We deliberately do not search
    every possible threshold, and we require a minimum sample size. These are
    candidate observations for later validation, not final rules.
    """
    rows = []

    selected = association.head(TOP_FEATURES)["feature"].tolist()
    for feature in selected:
        x = pd.to_numeric(df[feature], errors="coerce")
        q25, q50, q75 = x.quantile([0.25, 0.50, 0.75]).tolist()
        if not np.isfinite([q25, q50, q75]).all():
            continue

        masks = {
            "LOW_Q1": x <= q25,
            "HIGH_Q4": x >= q75,
            "LOW_HALF": x <= q50,
            "HIGH_HALF": x >= q50,
        }

        for rule_name, mask in masks.items():
            g = df.loc[mask.fillna(False)].copy()
            n = len(g)
            if n < MIN_GROUP_N:
                continue

            good_n = int(g["outcome"].eq(GOOD).sum())
            bad_n = int(g["outcome"].eq(BAD).sum())
            neither_n = int(g["outcome"].eq(NEITHER).sum())
            resolved_n = good_n + bad_n

            rows.append(
                {
                    "feature": feature,
                    "rule": rule_name,
                    "threshold": {
                        "LOW_Q1": q25,
                        "HIGH_Q4": q75,
                        "LOW_HALF": q50,
                        "HIGH_HALF": q50,
                    }[rule_name],
                    "n": n,
                    "coverage_pct": 100 * n / len(df),
                    "target_first_n": good_n,
                    "stop_first_n": bad_n,
                    "neither_n": neither_n,
                    "resolved_n": resolved_n,
                    "target_first_rate_resolved": good_n / resolved_n if resolved_n else np.nan,
                    "mean_raw_trade_move_pct": pd.to_numeric(g["raw_trade_move_pct"], errors="coerce").mean(),
                    "median_mfe_pct": pd.to_numeric(g["mfe_pct"], errors="coerce").median(),
                    "median_mae_pct": pd.to_numeric(g["mae_pct"], errors="coerce").median(),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Sorting is intentionally secondary to sample size. A tiny high-rate slice
    # should not dominate the research output.
    return out.sort_values(
        ["target_first_rate_resolved", "n"],
        ascending=[False, False],
    ).reset_index(drop=True)


def print_top_results(association: pd.DataFrame, candidates: pd.DataFrame) -> None:
    print("\nTOP DEVELOPMENT FEATURE ASSOCIATIONS")
    if association.empty:
        print("No numeric feature associations available.")
    else:
        cols = [
            "feature",
            "good_median",
            "bad_median",
            "auc_good",
            "cliffs_delta",
        ]
        print(association[cols].head(15).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nTOP DEVELOPMENT-ONLY QUARTILE CANDIDATES")
    if candidates.empty:
        print("No candidate filters met the minimum sample size.")
    else:
        cols = [
            "feature",
            "rule",
            "threshold",
            "n",
            "coverage_pct",
            "target_first_n",
            "stop_first_n",
            "neither_n",
            "target_first_rate_resolved",
            "mean_raw_trade_move_pct",
        ]
        print(candidates[cols].head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only feature analysis for the BTCUSDT short hypothesis."
    )
    parser.add_argument(
        "--input",
        default=str(INPUT_FILE),
        help="Development dataset CSV (2021-2023 only).",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(
            f"Development dataset not found: {path}. Run build_strategy_dataset.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)

    required = {
        "signal_time",
        "outcome",
        "raw_trade_move_pct",
        "mfe_pct",
        "mae_pct",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Hard guard: this file must remain development-only.
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    years = sorted(df["signal_time"].dt.year.dropna().unique().tolist())
    if set(years) - {2021, 2022, 2023}:
        raise ValueError(
            f"Development dataset contains non-development years: {years}. "
            "Do not use 2024/2025 for rule discovery."
        )

    print("=" * 72)
    print("DEVELOPMENT-ONLY FEATURE ANALYSIS")
    print("=" * 72)
    print(f"Rows: {len(df):,}")
    print(f"Years: {years}")
    print("Baseline: 24h, -2% target, +2% stop")
    print()

    print("OUTCOMES")
    print(df["outcome"].value_counts(dropna=False).to_string())

    features = numeric_features(df)
    association = association_table(df, features)
    quantiles = quantile_table(df, features)
    booleans = boolean_table(df)
    candidates = candidate_thresholds(df, association)

    association_path = OUTPUT_DIR / "development_feature_associations.csv"
    quantile_path = OUTPUT_DIR / "development_feature_quantiles.csv"
    boolean_path = OUTPUT_DIR / "development_boolean_features.csv"
    candidate_path = OUTPUT_DIR / "development_rule_candidates.csv"

    association.to_csv(association_path, index=False)
    quantiles.to_csv(quantile_path, index=False)
    booleans.to_csv(boolean_path, index=False)
    candidates.to_csv(candidate_path, index=False)

    print_top_results(association, candidates)

    print("\nOUTPUTS")
    for p in [association_path, quantile_path, boolean_path, candidate_path]:
        print(f"  {p}")

    print("\nRESEARCH RULE")
    print("These results are exploratory. Do not deploy or freeze any threshold")
    print("from this file until it is tested unchanged on 2024 validation data.")


if __name__ == "__main__":
    main()
