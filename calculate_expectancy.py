import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = [-0.5, -1.0, -1.5, -2.0, -3.0, -5.0]
STOPS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
HORIZONS = {
    "24h": 24 * 60,
    "48h": 48 * 60,
    "72h": 72 * 60,
    "7d": 7 * 24 * 60,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Calculate target/stop profitability and expectancy for BTC short signals."
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="data/backtest_results")
    p.add_argument("--fee-bps", type=float, default=0.0,
                   help="Trading fee per side in basis points. Example: 4 = 0.04%%.")
    p.add_argument("--slippage-bps", type=float, default=0.0,
                   help="Adverse slippage per side in basis points. Example: 1 = 0.01%%.")
    p.add_argument("--initial-equity", type=float, default=10000.0)
    return p.parse_args()


def load_data(data_dir):
    data_dir = Path(data_dir)

    one = pd.read_csv(data_dir / "BTCUSDT_1m.csv")
    one["open_time"] = pd.to_datetime(one["open_time"], utc=True)
    for c in ["open", "high", "low", "close"]:
        one[c] = pd.to_numeric(one[c], errors="coerce")
    one = one.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    sig = pd.read_csv(data_dir / "BTCUSDT_signals.csv")
    sig["open_time"] = pd.to_datetime(sig["open_time"], utc=True)
    sig = sig.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    # A signal candle's RSI is known only after that 15m candle closes.
    sig["entry_time"] = sig["open_time"] + pd.Timedelta(minutes=15)

    return one, sig


def find_gaps(one_min):
    t = one_min["open_time"].array.asi8
    diffs = np.diff(t)
    one_min_ns = 60 * 1_000_000_000
    idx = np.flatnonzero(diffs > one_min_ns)
    gaps = []
    for i in idx:
        gaps.append(
            {
                "last_present": pd.Timestamp(t[i], tz="UTC"),
                "next_present": pd.Timestamp(t[i + 1], tz="UTC"),
            }
        )
    return gaps


def first_gap_in_window(gaps, start, end):
    # Return the first gap whose missing interval begins before the requested end.
    for g in gaps:
        if g["last_present"] < end and g["next_present"] > start:
            return g
    return None


def cost_adjusted_short_return(entry_px, exit_px, fee_bps, slippage_bps):
    """
    Approximate short return after adverse slippage on both sides and fees.
    Entry: sell slightly below market.
    Exit: buy slightly above market.
    """
    slip = slippage_bps / 10_000.0
    fee = fee_bps / 10_000.0

    entry_fill = entry_px * (1.0 - slip)
    exit_fill = exit_px * (1.0 + slip)

    gross = (entry_fill - exit_fill) / entry_fill
    # Approximate two notional fees.
    net = gross - 2.0 * fee
    return net * 100.0


def analyze_signal_window(
    one_min,
    entry_idx,
    horizon_minutes,
    target_pct,
    stop_pct,
    gaps,
    fee_bps,
    slippage_bps,
):
    entry_row = one_min.iloc[entry_idx]
    entry_time = entry_row["open_time"]
    entry_px = float(entry_row["open"])

    horizon_end = entry_time + pd.Timedelta(minutes=horizon_minutes)

    # Determine whether the requested horizon crosses a real data gap.
    gap = first_gap_in_window(gaps, entry_time, horizon_end)

    if gap is not None:
        scan_end = gap["last_present"]
        horizon_is_blocked = True
    else:
        scan_end = horizon_end
        horizon_is_blocked = False

    future = one_min.iloc[entry_idx:]
    future = future[future["open_time"] <= scan_end]

    # Do not include the entry candle in the hit test; stops/targets start
    # being evaluated after entry.
    if len(future) <= 1:
        return None

    future = future.iloc[1:]
    lows = future["low"].to_numpy(dtype=float)
    highs = future["high"].to_numpy(dtype=float)

    # For a short:
    # target is a price decrease; stop is a price increase.
    target_price = entry_px * (1.0 + target_pct / 100.0)
    stop_price = entry_px * (1.0 + stop_pct / 100.0)

    target_hits = np.flatnonzero(lows <= target_price)
    stop_hits = np.flatnonzero(highs >= stop_price)

    target_i = int(target_hits[0]) if len(target_hits) else None
    stop_i = int(stop_hits[0]) if len(stop_hits) else None

    if target_i is not None and stop_i is not None:
        if target_i < stop_i:
            exit_i = target_i
            outcome = "target"
            exit_price = target_price
            exit_time = future.iloc[exit_i]["open_time"]
        elif stop_i < target_i:
            exit_i = stop_i
            outcome = "stop"
            exit_price = stop_price
            exit_time = future.iloc[exit_i]["open_time"]
        else:
            # Same 1m candle touched both sides; OHLC cannot establish order.
            return {
                "status": "ambiguous",
                "exit_time": pd.NaT,
                "exit_price": np.nan,
                "gross_return_pct": np.nan,
                "net_return_pct": np.nan,
                "entry_time": entry_time,
                "entry_price": entry_px,
                "horizon_end": horizon_end,
            }
    elif target_i is not None:
        exit_i = target_i
        outcome = "target"
        exit_price = target_price
        exit_time = future.iloc[exit_i]["open_time"]
    elif stop_i is not None:
        exit_i = stop_i
        outcome = "stop"
        exit_price = stop_price
        exit_time = future.iloc[exit_i]["open_time"]
    else:
        if horizon_is_blocked:
            return {
                "status": "gap_blocked",
                "exit_time": pd.NaT,
                "exit_price": np.nan,
                "gross_return_pct": np.nan,
                "net_return_pct": np.nan,
                "entry_time": entry_time,
                "entry_price": entry_px,
                "horizon_end": horizon_end,
            }

        # Neither threshold was reached: close at the exact horizon endpoint.
        # The 1m candle OPEN at horizon_end is the next candle, so use the
        # previous candle's CLOSE to represent the market close of the horizon.
        endpoint = one_min[one_min["open_time"] == horizon_end]
        if len(endpoint):
            exit_row = endpoint.iloc[0]
            exit_price = float(exit_row["open"])
            exit_time = horizon_end
        else:
            # Defensive fallback for an unexpected missing exact timestamp.
            at_or_before = one_min[one_min["open_time"] < horizon_end]
            if len(at_or_before) == 0:
                return None
            exit_row = at_or_before.iloc[-1]
            exit_price = float(exit_row["close"])
            exit_time = exit_row["open_time"] + pd.Timedelta(minutes=1)

        outcome = "horizon_close"
        exit_i = None

    gross_return_pct = (entry_px - exit_price) / entry_px * 100.0
    net_return_pct = cost_adjusted_short_return(
        entry_px, exit_price, fee_bps, slippage_bps
    )

    return {
        "status": outcome,
        "exit_time": exit_time,
        "exit_price": float(exit_price),
        "gross_return_pct": float(gross_return_pct),
        "net_return_pct": float(net_return_pct),
        "entry_time": entry_time,
        "entry_price": entry_px,
        "horizon_end": horizon_end,
    }


def build_trade_results(one_min, signals, gaps, fee_bps, slippage_bps):
    times_ns = one_min["open_time"].array.asi8

    rows = []

    for sig_idx, sig in signals.iterrows():
        entry_time = sig["entry_time"]
        entry_ns = entry_time.value
        pos = np.searchsorted(times_ns, entry_ns)

        if pos >= len(one_min) or times_ns[pos] != entry_ns:
            for horizon_name in HORIZONS:
                for target in TARGETS:
                    for stop in STOPS:
                        rows.append({
                            "signal_time": sig["open_time"],
                            "entry_time": entry_time,
                            "signal_year": sig["open_time"].year,
                            "horizon": horizon_name,
                            "target_pct": target,
                            "stop_pct": stop,
                            "status": "no_entry_data",
                            "entry_price": np.nan,
                            "exit_price": np.nan,
                            "exit_time": pd.NaT,
                            "gross_return_pct": np.nan,
                            "net_return_pct": np.nan,
                        })
            continue

        for horizon_name, horizon_minutes in HORIZONS.items():
            for target in TARGETS:
                for stop in STOPS:
                    result = analyze_signal_window(
                        one_min,
                        pos,
                        horizon_minutes,
                        target,
                        stop,
                        gaps,
                        fee_bps,
                        slippage_bps,
                    )
                    if result is None:
                        status = "no_entry_data"
                        result = {
                            "status": status,
                            "entry_time": entry_time,
                            "entry_price": np.nan,
                            "exit_price": np.nan,
                            "exit_time": pd.NaT,
                            "gross_return_pct": np.nan,
                            "net_return_pct": np.nan,
                        }

                    rows.append({
                        "signal_time": sig["open_time"],
                        "entry_time": result["entry_time"],
                        "signal_year": sig["open_time"].year,
                        "horizon": horizon_name,
                        "target_pct": target,
                        "stop_pct": stop,
                        "status": result["status"],
                        "entry_price": result["entry_price"],
                        "exit_price": result["exit_price"],
                        "exit_time": result["exit_time"],
                        "gross_return_pct": result["gross_return_pct"],
                        "net_return_pct": result["net_return_pct"],
                    })

    return pd.DataFrame(rows)


def max_drawdown_from_returns(returns, initial_equity):
    if len(returns) == 0:
        return np.nan, np.nan

    equity = initial_equity * np.cumprod(1.0 + returns / 100.0)
    peaks = np.maximum.accumulate(np.r_[initial_equity, equity])[1:]
    dd = (equity / peaks - 1.0) * 100.0
    min_idx = int(np.argmin(dd))
    return float(dd[min_idx]), float(np.min(equity))


def summarize_group(g, initial_equity):
    valid = g[g["status"].isin(["target", "stop", "horizon_close"])].copy()
    valid["win"] = valid["net_return_pct"] > 0

    total = len(g)
    completed = len(valid)
    wins = int(valid["win"].sum()) if completed else 0
    losses = int((valid["net_return_pct"] < 0).sum()) if completed else 0

    gross = valid["gross_return_pct"].dropna()
    net = valid["net_return_pct"].dropna()

    gross_profit = gross[gross > 0].sum()
    gross_loss_abs = -gross[gross < 0].sum()
    net_profit = net[net > 0].sum()
    net_loss_abs = -net[net < 0].sum()

    max_dd, min_equity = max_drawdown_from_returns(net.to_numpy(), initial_equity)

    # Only target/stop first-touch outcomes.
    resolved = g[g["status"].isin(["target", "stop"])]
    target_first = int((resolved["status"] == "target").sum())
    stop_first = int((resolved["status"] == "stop").sum())

    return {
        "signals": total,
        "completed": completed,
        "no_entry_data": int((g["status"] == "no_entry_data").sum()),
        "gap_blocked": int((g["status"] == "gap_blocked").sum()),
        "ambiguous": int((g["status"] == "ambiguous").sum()),
        "horizon_close": int((g["status"] == "horizon_close").sum()),
        "target_first": target_first,
        "stop_first": stop_first,
        "target_first_rate_resolved_pct": (
            target_first / len(resolved) * 100.0 if len(resolved) else np.nan
        ),
        "win_rate_net_pct": wins / completed * 100.0 if completed else np.nan,
        "loss_rate_net_pct": losses / completed * 100.0 if completed else np.nan,
        "avg_gross_return_pct": gross.mean() if completed else np.nan,
        "avg_net_return_pct": net.mean() if completed else np.nan,
        "median_net_return_pct": net.median() if completed else np.nan,
        "total_gross_return_pct_sum": gross.sum() if completed else np.nan,
        "total_net_return_pct_sum": net.sum() if completed else np.nan,
        "gross_profit_pct": gross_profit,
        "gross_loss_pct": gross_loss_abs,
        "net_profit_pct": net_profit,
        "net_loss_pct": net_loss_abs,
        "profit_factor_gross": (
            gross_profit / gross_loss_abs if gross_loss_abs > 0 else np.inf
        ),
        "profit_factor_net": (
            net_profit / net_loss_abs if net_loss_abs > 0 else np.inf
        ),
        "max_drawdown_pct": max_dd,
        "ending_equity": (
            initial_equity * np.prod(1.0 + net.to_numpy() / 100.0)
            if completed else initial_equity
        ),
        "expectancy_pct": net.mean() if completed else np.nan,
    }


def build_matrix(trades, initial_equity):
    rows = []
    for (horizon, target, stop), g in trades.groupby(
        ["horizon", "target_pct", "stop_pct"], sort=False
    ):
        row = summarize_group(g, initial_equity)
        row.update({
            "horizon": horizon,
            "target_pct": target,
            "stop_pct": stop,
        })
        rows.append(row)

    matrix = pd.DataFrame(rows)
    return matrix[
        [
            "horizon", "target_pct", "stop_pct",
            "signals", "completed", "target_first", "stop_first",
            "horizon_close", "ambiguous", "gap_blocked", "no_entry_data",
            "target_first_rate_resolved_pct",
            "win_rate_net_pct", "avg_gross_return_pct", "avg_net_return_pct",
            "median_net_return_pct", "expectancy_pct",
            "total_gross_return_pct_sum", "total_net_return_pct_sum",
            "gross_profit_pct", "gross_loss_pct",
            "net_profit_pct", "net_loss_pct",
            "profit_factor_gross", "profit_factor_net",
            "max_drawdown_pct", "ending_equity",
        ]
    ]


def build_yearly(trades, initial_equity):
    rows = []
    eligible = trades[trades["status"].isin(["target", "stop", "horizon_close"])].copy()

    for (year, horizon, target, stop), g in eligible.groupby(
        ["signal_year", "horizon", "target_pct", "stop_pct"], sort=False
    ):
        row = summarize_group(g, initial_equity)
        row.update({
            "year": year,
            "horizon": horizon,
            "target_pct": target,
            "stop_pct": stop,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def build_non_overlapping_portfolio(trades, initial_equity):
    """
    For each parameter set, take signals chronologically and skip a new signal
    whenever the prior trade is still active. This avoids double-counting
    overlapping positions when calculating portfolio-style metrics.
    """
    rows = []

    eligible = trades[
        trades["status"].isin(["target", "stop", "horizon_close"])
    ].copy()

    for (horizon, target, stop), g in eligible.groupby(
        ["horizon", "target_pct", "stop_pct"], sort=False
    ):
        g = g.sort_values("entry_time")
        kept = []
        current_exit = None

        for _, r in g.iterrows():
            if current_exit is None or r["entry_time"] >= current_exit:
                kept.append(r)
                current_exit = r["exit_time"]

        kept = pd.DataFrame(kept)
        if len(kept) == 0:
            continue

        returns = kept["net_return_pct"].to_numpy(dtype=float)
        compounded = initial_equity * np.cumprod(1.0 + returns / 100.0)
        max_dd, _ = max_drawdown_from_returns(returns, initial_equity)

        wins = int((returns > 0).sum())
        losses = int((returns < 0).sum())
        gp = returns[returns > 0].sum()
        gl = -returns[returns < 0].sum()

        rows.append({
            "horizon": horizon,
            "target_pct": target,
            "stop_pct": stop,
            "trades_taken": len(kept),
            "win_rate_pct": wins / len(kept) * 100.0,
            "expectancy_pct": returns.mean(),
            "median_return_pct": np.median(returns),
            "profit_factor": gp / gl if gl > 0 else np.inf,
            "max_drawdown_pct": max_dd,
            "ending_equity": compounded[-1],
            "total_return_pct": (compounded[-1] / initial_equity - 1.0) * 100.0,
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    one_min, signals = load_data(args.data_dir)
    gaps = find_gaps(one_min)

    print(f"1m candles: {len(one_min):,}")
    print(f"Signal events: {len(signals):,}")
    print(f"1m gaps: {len(gaps)}")
    print(f"Fee per side: {args.fee_bps:.4f} bps")
    print(f"Slippage per side: {args.slippage_bps:.4f} bps")
    print()

    trades = build_trade_results(
        one_min,
        signals,
        gaps,
        args.fee_bps,
        args.slippage_bps,
    )

    matrix = build_matrix(trades, args.initial_equity)
    yearly = build_yearly(trades, args.initial_equity)
    non_overlap = build_non_overlapping_portfolio(trades, args.initial_equity)

    trades.to_csv(out_dir / "expectancy_trade_results.csv", index=False)
    matrix.to_csv(out_dir / "expectancy_matrix.csv", index=False)
    yearly.to_csv(out_dir / "expectancy_yearly.csv", index=False)
    non_overlap.to_csv(out_dir / "expectancy_non_overlapping.csv", index=False)

    # Rank candidates by expectancy, but keep this explicitly exploratory.
    ranked = (
        matrix[matrix["completed"] > 0]
        .sort_values(["expectancy_pct", "profit_factor_net"], ascending=False)
    )
    ranked.head(20).to_csv(out_dir / "expectancy_top20_exploratory.csv", index=False)

    print("WROTE:")
    print(out_dir / "expectancy_trade_results.csv")
    print(out_dir / "expectancy_matrix.csv")
    print(out_dir / "expectancy_yearly.csv")
    print(out_dir / "expectancy_non_overlapping.csv")
    print(out_dir / "expectancy_top20_exploratory.csv")
    print()

    print("TOP 20 EXPLORATORY COMBINATIONS BY NET EXPECTANCY")
    cols = [
        "horizon", "target_pct", "stop_pct", "completed",
        "win_rate_net_pct", "expectancy_pct",
        "profit_factor_net", "max_drawdown_pct", "ending_equity",
    ]
    print(ranked[cols].head(20).to_string(index=False))

    print()
    print("NOTE: This ranks the full 2021-2025 sample for exploration only.")
    print("Do NOT select a final strategy from this ranking without chronological")
    print("development/validation/test splits and costs appropriate to your exchange.")


if __name__ == "__main__":
    main()
