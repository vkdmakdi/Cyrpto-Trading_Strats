import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SIGNALS_FILE = Path("data/BTCUSDT_signals.csv")
DATA_FILE = Path("data/BTCUSDT_1m.csv")
OUTPUT_DIR = Path("data/backtest_results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# The signal is generated from a 15-minute candle's RSI.
# That candle is only known once it closes.
SIGNAL_CANDLE_MINUTES = 15


# Test all of these holding periods.
HORIZONS = {
    "24h": 24 * 60,
    "48h": 48 * 60,
    "72h": 72 * 60,
    "7d": 7 * 24 * 60,
}


# Short targets.
# Negative = favorable.
TARGET_LEVELS = np.array([
    -0.005,   # -0.5%
    -0.010,   # -1%
    -0.015,   # -1.5%
    -0.020,   # -2%
    -0.030,   # -3%
    -0.050,   # -5%
])


# Short stops / adverse levels.
# Positive = unfavorable.
STOP_LEVELS = np.array([
     0.005,   # +0.5%
     0.010,   # +1%
     0.015,   # +1.5%
     0.020,   # +2%
     0.030,   # +3%
     0.050,   # +5%
     0.100,   # +10%
])


# Provisional signal-clustering threshold.
# We will NOT use this to remove trades.
CLUSTER_GAP_MINUTES = 4 * 60


# ============================================================
# HELPERS
# ============================================================

def pct_label(value):
    """
    Convert decimal percentage to readable label.

    -0.02 -> "-2%"
     0.02 -> "+2%"
    """
    return f"{value * 100:+g}%"


def first_true_index(mask):
    """
    Given a 1D boolean NumPy array, return the index of
    the first True value, or -1 if none exists.
    """
    positions = np.flatnonzero(mask)

    if len(positions) == 0:
        return -1

    return int(positions[0])


def first_true_indices(mask_matrix):
    """
    mask_matrix shape:
        rows    = candles
        columns = levels

    Returns one first-hit index per column.
    -1 means the level was never reached.

    This is substantially faster than looping through
    every candle and every level.
    """
    n_levels = mask_matrix.shape[1]

    result = np.full(
        n_levels,
        -1,
        dtype=np.int64
    )

    any_hit = mask_matrix.any(axis=0)

    if any_hit.any():

        # argmax returns the first maximum.
        result[any_hit] = (
            mask_matrix[:, any_hit]
            .argmax(axis=0)
        )

    return result


# ============================================================
# LOAD SIGNALS
# ============================================================

print()
print("=" * 72)
print("LOADING SIGNALS")
print("=" * 72)

signals = pd.read_csv(SIGNALS_FILE)

signals["open_time"] = pd.to_datetime(
    signals["open_time"],
    utc=True
)

signals = (
    signals
    .sort_values("open_time")
    .reset_index(drop=True)
)


# ============================================================
# SIGNAL CLUSTERS
# ============================================================

signal_gap_minutes = (
    signals["open_time"]
    .diff()
    .dt.total_seconds()
    .div(60)
)

signals["minutes_since_previous_signal"] = (
    signal_gap_minutes
)

signals["new_cluster"] = (
    signal_gap_minutes.isna()
    | (
        signal_gap_minutes
        > CLUSTER_GAP_MINUTES
    )
)

signals["cluster_id"] = (
    signals["new_cluster"]
    .cumsum()
    .astype(int)
)

cluster_sizes = (
    signals["cluster_id"]
    .value_counts()
)

signals["cluster_size"] = (
    signals["cluster_id"]
    .map(cluster_sizes)
)


# ============================================================
# LOAD 1-MINUTE DATA
# ============================================================

print()
print("=" * 72)
print("LOADING 1-MINUTE DATA")
print("=" * 72)

data = pd.read_csv(DATA_FILE)

data["open_time"] = pd.to_datetime(
    data["open_time"],
    utc=True
)

data = (
    data
    .sort_values("open_time")
    .reset_index(drop=True)
)


# ============================================================
# NUMPY ARRAYS
# ============================================================

print()
print("Converting market data to NumPy arrays...")

times = data["open_time"].array.asi8

opens = data["open"].to_numpy(dtype=np.float64)
highs = data["high"].to_numpy(dtype=np.float64)
lows = data["low"].to_numpy(dtype=np.float64)
closes = data["close"].to_numpy(dtype=np.float64)
volumes = data["volume"].to_numpy(dtype=np.float64)


# ============================================================
# GLOBAL TIME GAP DETECTION
# ============================================================

EXPECTED_MINUTE_NS = (
    60 * 1_000_000_000
)

time_diffs = np.diff(times)

gap_start_indices = (
    np.flatnonzero(
        time_diffs > EXPECTED_MINUTE_NS
    )
    + 1
)

print(
    f"Detected {len(gap_start_indices)} "
    "historical 1-minute data gaps."
)


# ============================================================
# EXACT TIMESTAMP LOOKUP
# ============================================================

def find_timestamp_index(timestamp):
    """
    Find the exact 1-minute candle corresponding to timestamp.
    Returns -1 if absent.
    """

    target_ns = timestamp.value

    index = np.searchsorted(
        times,
        target_ns
    )

    if index >= len(times):
        return -1

    if times[index] != target_ns:
        return -1

    return int(index)


# ============================================================
# FIND NEXT DATA GAP
# ============================================================

def first_gap_after(entry_index):
    """
    Return the index where the first data gap after entry begins.

    If there is no later gap, return len(times).
    """

    position = np.searchsorted(
        gap_start_indices,
        entry_index,
        side="right"
    )

    if position >= len(gap_start_indices):
        return len(times)

    return int(
        gap_start_indices[position]
    )


# ============================================================
# ANALYZE ONE SIGNAL / ONE HORIZON
# ============================================================

def analyze_horizon(
    entry_index,
    entry_time,
    entry_price,
    horizon_minutes,
):
    """
    Analyze one trade over one holding horizon.

    Uses:
        entry = next 15m candle's 1m open

    Price movement for a short:
        favorable = entry -> low
        adverse   = entry -> high
    """

    requested_end_index = min(
        entry_index + horizon_minutes,
        len(times) - 1
    )

    # Find the first historical data gap after entry.
    gap_start = first_gap_after(
        entry_index
    )

    gap_blocked = (
        gap_start <= requested_end_index
    )

    if gap_blocked:

        actual_end_index = (
            gap_start - 1
        )

    else:

        actual_end_index = (
            requested_end_index
        )

    if actual_end_index < entry_index:

        return {
            "status": "GAP_AT_ENTRY",
            "usable_minutes": 0,
            "gap_blocked": True,
            "mfe_pct": np.nan,
            "mae_pct": np.nan,
            "time_to_mfe_minutes": np.nan,
            "time_to_mae_minutes": np.nan,
            "first_any_event": None,
            "first_any_event_time": None,
            "first_any_event_minutes": np.nan,
            "first_any_event_side": None,
            "ambiguous_first_any_event": False,
            "target_first_touch": {},
            "stop_first_touch": {},
            "target_first_time": {},
            "stop_first_time": {},
            "target_first_minutes": {},
            "stop_first_minutes": {},
        }

    # --------------------------------------------------------
    # EXTRACT WINDOW
    # --------------------------------------------------------

    sl = slice(
        entry_index,
        actual_end_index + 1
    )

    local_highs = highs[sl]
    local_lows = lows[sl]
    local_times = times[sl]

    # --------------------------------------------------------
    # PRICE MOVEMENTS
    # --------------------------------------------------------

    favorable = (
        entry_price - local_lows
    ) / entry_price

    adverse = (
        local_highs - entry_price
    ) / entry_price

    # --------------------------------------------------------
    # MFE
    # --------------------------------------------------------

    mfe_index = int(
        np.argmax(favorable)
    )

    mfe = favorable[mfe_index]

    mfe_time = pd.Timestamp(
        local_times[mfe_index],
        tz="UTC"
    )

    time_to_mfe = (
        mfe_time - entry_time
    ).total_seconds() / 60

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae_index = int(
        np.argmax(adverse)
    )

    mae = adverse[mae_index]

    mae_time = pd.Timestamp(
        local_times[mae_index],
        tz="UTC"
    )

    time_to_mae = (
        mae_time - entry_time
    ).total_seconds() / 60

    # --------------------------------------------------------
    # TARGET HIT MATRIX
    # --------------------------------------------------------

    # For each target:
    #
    # favorable >= abs(target)
    #
    # Example:
    # target = -0.02
    # favorable >= 0.02

    target_thresholds = np.abs(
        TARGET_LEVELS
    )

    target_masks = (
        favorable[:, None]
        >= target_thresholds[None, :]
    )

    target_indices = first_true_indices(
        target_masks
    )

    # --------------------------------------------------------
    # STOP HIT MATRIX
    # --------------------------------------------------------

    stop_thresholds = STOP_LEVELS

    stop_masks = (
        adverse[:, None]
        >= stop_thresholds[None, :]
    )

    stop_indices = first_true_indices(
        stop_masks
    )

    # --------------------------------------------------------
    # FIRST TOUCH TIMES FOR EACH TARGET
    # --------------------------------------------------------

    target_first_touch = {}
    target_first_times = {}
    target_first_minutes = {}

    for level_index, level in enumerate(
        TARGET_LEVELS
    ):

        label = pct_label(level)

        candle_index = target_indices[
            level_index
        ]

        if candle_index == -1:

            target_first_touch[label] = False
            target_first_times[label] = None
            target_first_minutes[label] = np.nan

        else:

            target_first_touch[label] = True

            touch_time = pd.Timestamp(
                local_times[candle_index],
                tz="UTC"
            )

            target_first_times[label] = (
                touch_time
            )

            target_first_minutes[label] = (
                touch_time - entry_time
            ).total_seconds() / 60
            

    # --------------------------------------------------------
    # FIRST TOUCH TIMES FOR EACH STOP
    # --------------------------------------------------------

    stop_first_touch = {}
    stop_first_times = {}
    stop_first_minutes = {}

    for level_index, level in enumerate(
        STOP_LEVELS
    ):

        label = pct_label(level)

        candle_index = stop_indices[
            level_index
        ]

        if candle_index == -1:

            stop_first_touch[label] = False
            stop_first_times[label] = None
            stop_first_minutes[label] = np.nan

        else:

            stop_first_touch[label] = True

            touch_time = pd.Timestamp(
                local_times[candle_index],
                tz="UTC"
            )

            stop_first_times[label] = (
                touch_time
            )

            stop_first_minutes[label] = (
                touch_time - entry_time
            ).total_seconds() / 60
            

    # --------------------------------------------------------
    # FIRST ±0.5% EVENT
    # --------------------------------------------------------

    first_target_index = target_indices[0]
    first_stop_index = stop_indices[0]

    if (
        first_target_index == -1
        and first_stop_index == -1
    ):

        first_any_event = None
        first_any_side = None
        first_any_index = -1
        ambiguous = False

    elif first_target_index == -1:

        first_any_event = "+0.5%"
        first_any_side = "ADVERSE"
        first_any_index = first_stop_index
        ambiguous = False

    elif first_stop_index == -1:

        first_any_event = "-0.5%"
        first_any_side = "FAVORABLE"
        first_any_index = first_target_index
        ambiguous = False

    elif (
        first_target_index
        == first_stop_index
    ):

        # Same 1-minute candle crossed both sides.
        # OHLC data cannot tell us which happened first.
        first_any_event = "AMBIGUOUS"
        first_any_side = "BOTH"
        first_any_index = first_target_index
        ambiguous = True

    elif first_target_index < first_stop_index:

        first_any_event = "-0.5%"
        first_any_side = "FAVORABLE"
        first_any_index = first_target_index
        ambiguous = False

    else:

        first_any_event = "+0.5%"
        first_any_side = "ADVERSE"
        first_any_index = first_stop_index
        ambiguous = False

    if first_any_index == -1:

        first_any_time = None
        first_any_minutes = np.nan

    else:

        first_any_time = pd.Timestamp(
            local_times[first_any_index],
            tz="UTC"
        )

        first_any_minutes = (
            first_any_time - entry_time
        ).total_seconds() / 60

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if ambiguous:

        status = "AMBIGUOUS"

    elif first_any_event is None:

        if gap_blocked:
            status = "GAP_BLOCKED"
        else:
            status = "NO_THRESHOLD_HIT"

    else:

        status = "COMPLETED"

    usable_minutes = (
        len(local_times) - 1
    )

    return {
        "status": status,
        "usable_minutes": usable_minutes,
        "gap_blocked": gap_blocked,
        "mfe_pct": mfe * 100,
        "mae_pct": mae * 100,
        "time_to_mfe_minutes": time_to_mfe,
        "time_to_mae_minutes": time_to_mae,
        "first_any_event": first_any_event,
        "first_any_event_time": first_any_time,
        "first_any_event_minutes": first_any_minutes,
        "first_any_event_side": first_any_side,
        "ambiguous_first_any_event": ambiguous,
        "target_first_touch": target_first_touch,
        "stop_first_touch": stop_first_touch,
        "target_first_time": target_first_times,
        "stop_first_time": stop_first_times,
        "target_first_minutes": target_first_minutes,
        "stop_first_minutes": stop_first_minutes,
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

print()
print("=" * 72)
print("RUNNING FULL SIGNAL ANALYSIS")
print("=" * 72)

horizon_rows = []
outcome_rows = []
level_rows = []


for signal_number, signal in signals.iterrows():

    signal_time = signal["open_time"]

    # IMPORTANT:
    #
    # The RSI condition belongs to the 15m candle beginning
    # at signal_time. That RSI uses the candle's close.
    #
    # Therefore we cannot enter until the following 15m
    # candle begins.

    entry_time = (
        signal_time
        + pd.Timedelta(
            minutes=SIGNAL_CANDLE_MINUTES
        )
    )

    entry_index = find_timestamp_index(
        entry_time
    )

    # --------------------------------------------------------
    # ENTRY DATA CHECK
    # --------------------------------------------------------

    if entry_index == -1:

        for horizon_name, horizon_minutes in HORIZONS.items():

            horizon_rows.append({
                "signal_number": signal_number + 1,
                "signal_time": signal_time,
                "entry_time": entry_time,
                "entry_price": np.nan,
                "horizon": horizon_name,
                "horizon_minutes": horizon_minutes,
                "status": "NO_ENTRY_DATA",
                "cluster_id":
                    signal["cluster_id"],
                "cluster_size":
                    signal["cluster_size"],
                "minutes_since_previous_signal":
                    signal["minutes_since_previous_signal"],
            })

        continue

    entry_price = opens[entry_index]

    # --------------------------------------------------------
    # ANALYZE EACH HORIZON
    # --------------------------------------------------------

    for horizon_name, horizon_minutes in HORIZONS.items():

        analysis = analyze_horizon(
            entry_index=entry_index,
            entry_time=entry_time,
            entry_price=entry_price,
            horizon_minutes=horizon_minutes,
        )

        # ----------------------------------------------------
        # HORIZON SUMMARY
        # ----------------------------------------------------

        horizon_rows.append({
            "signal_number":
                signal_number + 1,

            "signal_time":
                signal_time,

            "entry_time":
                entry_time,

            "entry_price":
                entry_price,

            "horizon":
                horizon_name,

            "horizon_minutes":
                horizon_minutes,

            "status":
                analysis["status"],

            "usable_minutes":
                analysis["usable_minutes"],

            "gap_blocked":
                analysis["gap_blocked"],

            "mfe_pct":
                analysis["mfe_pct"],

            "mae_pct":
                analysis["mae_pct"],

            "time_to_mfe_minutes":
                analysis["time_to_mfe_minutes"],

            "time_to_mae_minutes":
                analysis["time_to_mae_minutes"],

            "first_any_event":
                analysis["first_any_event"],

            "first_any_event_side":
                analysis["first_any_event_side"],

            "first_any_event_time":
                analysis["first_any_event_time"],

            "first_any_event_minutes":
                analysis["first_any_event_minutes"],

            "ambiguous_first_any_event":
                analysis["ambiguous_first_any_event"],

            "cluster_id":
                signal["cluster_id"],

            "cluster_size":
                signal["cluster_size"],

            "minutes_since_previous_signal":
                signal["minutes_since_previous_signal"],

        })

        # ----------------------------------------------------
        # ALL TARGET × STOP COMBINATIONS
        # ----------------------------------------------------

        for target_level in TARGET_LEVELS:

            target_label = pct_label(
                target_level
            )

            target_index = None

            if (
                target_label
                in analysis["target_first_touch"]
            ):
                target_touch = analysis[
                    "target_first_touch"
                ][target_label]

                if target_touch:
                    target_time = analysis[
                        "target_first_time"
                    ][target_label]

                    target_minutes = analysis[
                        "target_first_minutes"
                    ][target_label]
                else:
                    target_time = None
                    target_minutes = np.nan

            else:
                target_touch = False
                target_time = None
                target_minutes = np.nan

            for stop_level in STOP_LEVELS:

                stop_label = pct_label(
                    stop_level
                )

                stop_touch = analysis[
                    "stop_first_touch"
                ][stop_label]

                if stop_touch:

                    stop_time = analysis[
                        "stop_first_time"
                    ][stop_label]

                    stop_minutes = analysis[
                        "stop_first_minutes"
                    ][stop_label]

                else:

                    stop_time = None
                    stop_minutes = np.nan

                target_idx = -1
                stop_idx = -1

                # Determine ordering using exact stored
                # first-touch minutes.
                if target_touch:

                    target_idx = (
                        int(
                            round(
                                target_minutes
                            )
                        )
                    )

                if stop_touch:

                    stop_idx = (
                        int(
                            round(
                                stop_minutes
                            )
                        )
                    )

                # ------------------------------------------------
                # OUTCOME
                # ------------------------------------------------

                ambiguous = False

                if (
                    not target_touch
                    and not stop_touch
                ):

                    if analysis["gap_blocked"]:

                        outcome = "GAP_BLOCKED"

                    else:

                        outcome = "NEITHER"

                elif target_touch and not stop_touch:

                    outcome = "TARGET_FIRST"

                elif stop_touch and not target_touch:

                    outcome = "STOP_FIRST"

                else:

                    # Both were reached.
                    #
                    # Equal candle timestamp means both thresholds
                    # were crossed within the same 1m candle.
                    if (
                        target_time
                        == stop_time
                    ):

                        outcome = "AMBIGUOUS"
                        ambiguous = True

                    elif target_time < stop_time:

                        outcome = "TARGET_FIRST"

                    else:

                        outcome = "STOP_FIRST"

                # ------------------------------------------------
                # PROFIT / LOSS AT OUTCOME
                # ------------------------------------------------

                if outcome == "TARGET_FIRST":

                    realized_move_pct = (
                        target_level * 100
                    )

                    outcome_time = target_time

                    minutes_to_outcome = (
                        target_minutes
                    )

                elif outcome == "STOP_FIRST":

                    realized_move_pct = (
                        -stop_level * 100
                    )

                    outcome_time = stop_time

                    minutes_to_outcome = (
                        stop_minutes
                    )

                elif outcome == "AMBIGUOUS":

                    realized_move_pct = np.nan

                    outcome_time = target_time

                    minutes_to_outcome = (
                        target_minutes
                    )

                else:

                    realized_move_pct = np.nan
                    outcome_time = None
                    minutes_to_outcome = np.nan

                outcome_rows.append({

                    "signal_number":
                        signal_number + 1,

                    "signal_time":
                        signal_time,

                    "entry_time":
                        entry_time,

                    "entry_price":
                        entry_price,

                    "horizon":
                        horizon_name,

                    "horizon_minutes":
                        horizon_minutes,

                    "target_pct":
                        target_level * 100,

                    "stop_pct":
                        stop_level * 100,

                    "target_label":
                        target_label,

                    "stop_label":
                        stop_label,

                    "outcome":
                        outcome,

                    "ambiguous":
                        ambiguous,

                    "target_hit":
                        target_touch,

                    "stop_hit":
                        stop_touch,

                    "target_first_time":
                        target_time,

                    "stop_first_time":
                        stop_time,

                    "outcome_time":
                        outcome_time,

                    "minutes_to_target":
                        target_minutes,

                    "minutes_to_stop":
                        stop_minutes,

                    "minutes_to_outcome":
                        minutes_to_outcome,

                    "raw_trade_move_pct":
                        realized_move_pct,

                    "mfe_pct":
                        analysis["mfe_pct"],

                    "mae_pct":
                        analysis["mae_pct"],

                    "gap_blocked":
                        analysis["gap_blocked"],

                    "cluster_id":
                        signal["cluster_id"],

                    "cluster_size":
                        signal["cluster_size"],
                })

        # ----------------------------------------------------
        # LONG-FORM INDIVIDUAL LEVEL DATA
        # ----------------------------------------------------

        for level in TARGET_LEVELS:

            label = pct_label(level)

            level_rows.append({

                "signal_number":
                    signal_number + 1,

                "signal_time":
                    signal_time,

                "entry_time":
                    entry_time,

                "entry_price":
                    entry_price,

                "horizon":
                    horizon_name,

                "horizon_minutes":
                    horizon_minutes,

                "side":
                    "FAVORABLE",

                "level_pct":
                    level * 100,

                "level_label":
                    label,

                "hit":
                    analysis[
                        "target_first_touch"
                    ][label],

                "first_hit_time":
                    analysis[
                        "target_first_time"
                    ][label],

                "minutes_to_hit":
                    analysis[
                        "target_first_minutes"
                    ][label],

                "cluster_id":
                    signal["cluster_id"],

                "cluster_size":
                    signal["cluster_size"],
            })

        for level in STOP_LEVELS:

            label = pct_label(level)

            level_rows.append({

                "signal_number":
                    signal_number + 1,

                "signal_time":
                    signal_time,

                "entry_time":
                    entry_time,

                "entry_price":
                    entry_price,

                "horizon":
                    horizon_name,

                "horizon_minutes":
                    horizon_minutes,

                "side":
                    "ADVERSE",

                "level_pct":
                    level * 100,

                "level_label":
                    label,

                "hit":
                    analysis[
                        "stop_first_touch"
                    ][label],

                "first_hit_time":
                    analysis[
                        "stop_first_time"
                    ][label],

                "minutes_to_hit":
                    analysis[
                        "stop_first_minutes"
                    ][label],

                "cluster_id":
                    signal["cluster_id"],

                "cluster_size":
                    signal["cluster_size"],
            })

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (signal_number + 1) % 20 == 0:

        print(
            f"Processed "
            f"{signal_number + 1}/"
            f"{len(signals)} signals..."
        )


# ============================================================
# DATAFRAMES
# ============================================================

horizon_df = pd.DataFrame(
    horizon_rows
)

outcome_df = pd.DataFrame(
    outcome_rows
)

level_df = pd.DataFrame(
    level_rows
)


# ============================================================
# SAVE RAW DETAILED OUTPUTS
# ============================================================

horizon_file = (
    OUTPUT_DIR
    / "signal_horizon_metrics.csv"
)

outcome_file = (
    OUTPUT_DIR
    / "signal_target_stop_outcomes.csv"
)

level_file = (
    OUTPUT_DIR
    / "signal_level_first_touch.csv"
)

horizon_df.to_csv(
    horizon_file,
    index=False
)

outcome_df.to_csv(
    outcome_file,
    index=False
)

level_df.to_csv(
    level_file,
    index=False
)


# ============================================================
# SUMMARY 1:
# TARGET × STOP MATRIX
# ============================================================

matrix_rows = []

for horizon_name in HORIZONS:

    subset = outcome_df[
        outcome_df["horizon"]
        == horizon_name
    ]

    for target_level in TARGET_LEVELS:

        for stop_level in STOP_LEVELS:

            pair = subset[
                (
                    subset["target_pct"]
                    == target_level * 100
                )
                &
                (
                    subset["stop_pct"]
                    == stop_level * 100
                )
            ]

            n = len(pair)

            target_first = (
                pair["outcome"]
                == "TARGET_FIRST"
            ).sum()

            stop_first = (
                pair["outcome"]
                == "STOP_FIRST"
            ).sum()

            ambiguous = (
                pair["outcome"]
                == "AMBIGUOUS"
            ).sum()

            neither = (
                pair["outcome"]
                == "NEITHER"
            ).sum()

            gap_blocked = (
                pair["outcome"]
                == "GAP_BLOCKED"
            ).sum()

            matrix_rows.append({

                "horizon":
                    horizon_name,

                "target_pct":
                    target_level * 100,

                "stop_pct":
                    stop_level * 100,

                "signals":
                    n,

                "target_first":
                    target_first,

                "stop_first":
                    stop_first,

                "ambiguous":
                    ambiguous,

                "neither":
                    neither,

                "gap_blocked":
                    gap_blocked,

                "target_first_rate_pct":
                    (
                        100 * target_first / n
                        if n else np.nan
                    ),

                "stop_first_rate_pct":
                    (
                        100 * stop_first / n
                        if n else np.nan
                    ),

                "ambiguous_rate_pct":
                    (
                        100 * ambiguous / n
                        if n else np.nan
                    ),

                "neither_rate_pct":
                    (
                        100 * neither / n
                        if n else np.nan
                    ),

                "target_before_stop_among_resolved_pct":
                    (
                        100 * target_first
                        / (
                            target_first
                            + stop_first
                        )
                        if (
                            target_first
                            + stop_first
                        )
                        else np.nan
                    ),
            })


matrix_df = pd.DataFrame(
    matrix_rows
)

matrix_file = (
    OUTPUT_DIR
    / "target_stop_matrix.csv"
)

matrix_df.to_csv(
    matrix_file,
    index=False
)


# ============================================================
# SUMMARY 2:
# HORIZON SUMMARY
# ============================================================

summary_rows = []

for horizon_name in HORIZONS:

    subset = horizon_df[
        horizon_df["horizon"]
        == horizon_name
    ]

    summary_rows.append({

        "horizon":
            horizon_name,

        "signals":
            len(subset),

        "completed":
            (
                subset["status"]
                == "COMPLETED"
            ).sum(),

        "ambiguous":
            (
                subset["status"]
                == "AMBIGUOUS"
            ).sum(),

        "no_threshold_hit":
            (
                subset["status"]
                == "NO_THRESHOLD_HIT"
            ).sum(),

        "gap_blocked":
            (
                subset["status"]
                == "GAP_BLOCKED"
            ).sum(),

        "no_entry_data":
            (
                subset["status"]
                == "NO_ENTRY_DATA"
            ).sum(),

        "median_mfe_pct":
            subset["mfe_pct"].median(),

        "mean_mfe_pct":
            subset["mfe_pct"].mean(),

        "median_mae_pct":
            subset["mae_pct"].median(),

        "mean_mae_pct":
            subset["mae_pct"].mean(),

        "median_time_to_mfe_minutes":
            subset[
                "time_to_mfe_minutes"
            ].median(),

        "median_time_to_mae_minutes":
            subset[
                "time_to_mae_minutes"
            ].median(),
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_file = (
    OUTPUT_DIR
    / "horizon_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# SUMMARY 3:
# YEARLY PERFORMANCE MATRIX
# ============================================================

outcome_df["year"] = (
    pd.to_datetime(
        outcome_df["signal_time"],
        utc=True
    ).dt.year
)

yearly_rows = []

for year in sorted(
    outcome_df["year"].unique()
):

    for horizon_name in HORIZONS:

        subset = outcome_df[
            (
                outcome_df["year"]
                == year
            )
            &
            (
                outcome_df["horizon"]
                == horizon_name
            )
        ]

        for target_level in TARGET_LEVELS:

            for stop_level in STOP_LEVELS:

                pair = subset[
                    (
                        subset["target_pct"]
                        == target_level * 100
                    )
                    &
                    (
                        subset["stop_pct"]
                        == stop_level * 100
                    )
                ]

                n = len(pair)

                target_first = (
                    pair["outcome"]
                    == "TARGET_FIRST"
                ).sum()

                stop_first = (
                    pair["outcome"]
                    == "STOP_FIRST"
                ).sum()

                yearly_rows.append({

                    "year":
                        year,

                    "horizon":
                        horizon_name,

                    "target_pct":
                        target_level * 100,

                    "stop_pct":
                        stop_level * 100,

                    "signals":
                        n,

                    "target_first":
                        target_first,

                    "stop_first":
                        stop_first,

                    "ambiguous":
                        (
                            pair["outcome"]
                            == "AMBIGUOUS"
                        ).sum(),

                    "neither":
                        (
                            pair["outcome"]
                            == "NEITHER"
                        ).sum(),

                    "gap_blocked":
                        (
                            pair["outcome"]
                            == "GAP_BLOCKED"
                        ).sum(),

                    "target_first_rate_pct":
                        (
                            100 * target_first / n
                            if n else np.nan
                        ),

                    "stop_first_rate_pct":
                        (
                            100 * stop_first / n
                            if n else np.nan
                        ),
                })


yearly_df = pd.DataFrame(
    yearly_rows
)

yearly_file = (
    OUTPUT_DIR
    / "yearly_target_stop_results.csv"
)

yearly_df.to_csv(
    yearly_file,
    index=False
)


# ============================================================
# SUMMARY 4:
# MONTHLY SIGNAL COUNTS
# ============================================================

signals["year"] = (
    signals["open_time"]
    .dt.year
)

signals["month"] = (
    signals["open_time"]
    .dt.month
)

monthly_signal_counts = (
    signals
    .groupby(
        ["year", "month"]
    )
    .size()
    .reset_index(
        name="signal_count"
    )
)

monthly_file = (
    OUTPUT_DIR
    / "monthly_signal_counts.csv"
)

monthly_signal_counts.to_csv(
    monthly_file,
    index=False
)


# ============================================================
# SUMMARY 5:
# CLUSTER SUMMARY
# ============================================================

cluster_summary = (
    signals
    .groupby("cluster_id")
    .agg(
        cluster_start=(
            "open_time",
            "min"
        ),
        cluster_end=(
            "open_time",
            "max"
        ),
        signal_count=(
            "open_time",
            "count"
        ),
        cluster_size=(
            "cluster_size",
            "first"
        ),
    )
    .reset_index()
)

cluster_summary["cluster_duration_minutes"] = (
    cluster_summary["cluster_end"]
    - cluster_summary["cluster_start"]
).dt.total_seconds() / 60

cluster_file = (
    OUTPUT_DIR
    / "signal_clusters.csv"
)

cluster_summary.to_csv(
    cluster_file,
    index=False
)


# ============================================================
# SUMMARY 6:
# SIGNAL-LEVEL RSI / CLUSTER DATA
# ============================================================

signal_metadata_file = (
    OUTPUT_DIR
    / "signal_metadata.csv"
)

signals.to_csv(
    signal_metadata_file,
    index=False
)


# ============================================================
# CONSOLE REPORT
# ============================================================

print()
print("=" * 72)
print("ANALYSIS COMPLETE")
print("=" * 72)

print()
print(
    f"Signal events: "
    f"{len(signals):,}"
)

print(
    f"4-hour clusters: "
    f"{signals['cluster_id'].nunique():,}"
)

print(
    f"Target × stop combinations per horizon: "
    f"{len(TARGET_LEVELS) * len(STOP_LEVELS)}"
)

print(
    f"Total detailed outcome rows: "
    f"{len(outcome_df):,}"
)


# ============================================================
# HORIZON SUMMARY
# ============================================================

print()
print("=" * 72)
print("HORIZON SUMMARY")
print("=" * 72)

print(
    summary_df.to_string(
        index=False
    )
)


# ============================================================
# ORIGINAL -2% VS +2% RESULT
# ============================================================

print()
print("=" * 72)
print("ORIGINAL STRATEGY: -2% TARGET VS +2% STOP")
print("=" * 72)

for horizon_name in HORIZONS:

    pair = outcome_df[
        (
            outcome_df["horizon"]
            == horizon_name
        )
        &
        (
            outcome_df["target_pct"]
            == -2.0
        )
        &
        (
            outcome_df["stop_pct"]
            == 2.0
        )
    ]

    n = len(pair)

    target_first = (
        pair["outcome"]
        == "TARGET_FIRST"
    ).sum()

    stop_first = (
        pair["outcome"]
        == "STOP_FIRST"
    ).sum()

    ambiguous = (
        pair["outcome"]
        == "AMBIGUOUS"
    ).sum()

    neither = (
        pair["outcome"]
        == "NEITHER"
    ).sum()

    gap_blocked = (
        pair["outcome"]
        == "GAP_BLOCKED"
    ).sum()

    print()
    print(f"{horizon_name}:")
    print(
        f"  -2% target first: "
        f"{target_first}/{n} "
        f"({100 * target_first / n:.1f}%)"
    )
    print(
        f"  +2% stop first:   "
        f"{stop_first}/{n} "
        f"({100 * stop_first / n:.1f}%)"
    )
    print(
        f"  Ambiguous:         "
        f"{ambiguous}"
    )
    print(
        f"  Neither:           "
        f"{neither}"
    )
    print(
        f"  Gap blocked:       "
        f"{gap_blocked}"
    )

    resolved = (
        target_first
        + stop_first
    )

    if resolved:

        print(
            f"  Target first among resolved: "
            f"{100 * target_first / resolved:.1f}%"
        )


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print("=" * 72)
print("OUTPUT FILES")
print("=" * 72)

for file in [
    horizon_file,
    outcome_file,
    level_file,
    matrix_file,
    summary_file,
    yearly_file,
    monthly_file,
    cluster_file,
    signal_metadata_file,
]:

    print(file)


print()
print("=" * 72)
print("DONE")
print("=" * 72)