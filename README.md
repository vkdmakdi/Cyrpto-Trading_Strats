# BTCUSDT Multi-Timeframe RSI Shorting Study

A reproducible, point-in-time-safe backtest investigating whether simultaneous
multi-timeframe RSI overbought conditions identify short opportunities in
BTCUSDT.

## Status

**Research complete for the original hypothesis, on a gross-return basis.**
No robust, cost-adjusted, out-of-sample edge was established. Read
[Limitations](#limitations-read-this-before-trusting-any-number-above) before
citing any number in this document.

## Summary

Hypothesis: short BTCUSDT when RSI(14) > 70 simultaneously on 15m, 30m, 1h,
and 4h, targeting a −2% move.

Method: chronological development (2021–2023) → freeze a candidate rule →
validate (2024) → final test (2025), never re-tuning against 2024 or 2025
after freezing.

Result: the hypothesis is not supported. All 120 signal events occurred
inside a bullish EMA200 regime; absolute RSI level was a weak predictor of
outcome (AUC ≈ 0.45–0.59); an RSI-acceleration filter looked promising in
development (AUC ≈ 0.64) but did not hold up in validation or the final
test. Full narrative, including every intermediate result and the reasoning
behind each decision, is in [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

## Results at a glance

All figures gross (no fees, slippage, or funding) unless noted. Resolved WR
= win rate among trades that hit either the target or the stop; "neither"
trades are excluded from that percentage.

| Stage | Period | Signals | Resolved WR | Expectancy | Note |
|---|---|---:|---:|---:|---|
| Baseline (all 4 RSI > 70, −2%/+2%, 24h) | 2021–2025 | 120 | 36.8% | — | Full sample, not a valid strategy test on its own |
| Development | 2021–2023 | 70 | 38.6% | — | Used to discover candidate rules |
| Frozen strategy — validation | 2024 | 13 qualifying | 40.0% | −0.235% | Out-of-sample, rule unchanged since freeze |
| Frozen strategy — final test | 2025 | 5 qualifying | 33.3% | −0.218% | n too small to confirm or rule out an edge either way |

"Frozen strategy" = all four RSI > 70, plus `RSI30_delta_4 ≥ 4`, plus wait
for 15m RSI to cross back below 70, enter next candle, −2% target / +2%
stop, 24h max hold. See the research log for how this was derived.

## Repository layout

| Script | Reads | Writes | Purpose |
|---|---|---|---|
| `download_data.py` | Binance data dump | `data/BTCUSDT_1m.csv` | Pull 1m spot klines, 2021–2025 |
| `validate_data.py` | `data/BTCUSDT_1m.csv` | console report | Integrity check: gaps, duplicates, bad OHLC |
| `calculate_rsi.py` | `data/BTCUSDT_15m.csv` | `data/BTCUSDT_RSI.csv` | Gap-aware, point-in-time Wilder RSI on 15m/30m/1h/4h |
| `find_signals.py` | `data/BTCUSDT_RSI.csv` | `data/BTCUSDT_signals.csv` | De-duplicated 4-timeframe overbought signal events |
| `analyze_signals.py` | signals + 1m data | `backtest_results/*` | Multi-horizon, multi-target/stop first-touch grid, MFE/MAE, clustering |
| `calculate_expectancy.py` | signals + 1m data | `expectancy_*.csv` | Cost-adjusted expectancy, non-overlapping portfolio, drawdown |
| `build_features.py` | 15m data + signals | `signal_features.csv` | Point-in-time feature set (RSI slope, trend, ATR, volume, structure) |
| `build_strategy_dataset.py` | features + outcomes | `strategy_{development,validation,test}_*.csv` | Locks the 2021–2023 / 2024 / 2025 chronological split |
| `analyze_development.py` | development split only | `development_*.csv` | Exploratory feature associations, development-only |
| `analyze_features.py` | full feature set | `feature_vs_outcome_*.csv` | Univariate AUC/Cliff's-delta/correlation screen (exploratory) |
| `analyze_rsi30_acceleration_stability.py` | development split | `rsi30_acceleration_*.csv` | Pre-declared quartile/threshold test for the acceleration filter |
| `test_confirmation_strategies.py` / `test_exhaustion_components.py` | development split | `confirmation_*`, `exhaustion_*` | Development-only tests of entry-confirmation variants (B/C/D) |
| `validate_strategies_2024.py` | 2024 split | `strategy_validation_2024_*.csv` | Frozen-rule validation, unchanged after freeze |
| `test_strategy_2025.py` | 2025 split | `strategy_test_2025_*.csv` | Frozen-rule final test, unchanged after validation |

## Reproducing

```bash
pip install pandas numpy requests

python download_data.py                 # ~2.6M 1m candles; several GB, takes a while
python validate_data.py
python calculate_rsi.py
python find_signals.py
python analyze_signals.py
python calculate_expectancy.py --fee-bps 4.5 --slippage-bps 1   # see Limitations re: defaults
python build_features.py
python build_strategy_dataset.py
python analyze_development.py           # development split only — do not run on 2024/2025 for rule discovery
python analyze_rsi30_acceleration_stability.py
python test_confirmation_strategies.py
python test_exhaustion_components.py
python validate_strategies_2024.py      # run once, after the rule is frozen
python test_strategy_2025.py            # run once, after validation is complete
```

Each stage assumes the previous stage's output files already exist under
`data/` and `data/backtest_results/`. There is no orchestration script yet.

## Limitations (read this before trusting any number above)

- **Data is spot, not perpetual.** `download_data.py` pulls Binance *spot*
  BTCUSDT. The strategy shorts, which in practice means the perpetual
  contract — a different price series with its own basis and funding. Every
  ±2% first-touch result in this study was computed on a series you cannot
  directly trade this strategy on.
- **All reported figures are gross.** `calculate_expectancy.py` supports
  `--fee-bps` and `--slippage-bps`, but the defaults are 0, and none of the
  headline numbers above include them. `validate_strategies_2024.py` and
  `test_strategy_2025.py` have no cost parameter at all. Binance USDⓈ-M
  taker fees run ~4.5 bps/side; a −20 to −50 bps gross expectancy will not
  survive realistic round-trip costs.
- **Development and validation numbers use different metrics.** The
  development-stage scripts report mean mark-to-horizon-close return.
  `validate_strategies_2024.py` and `test_strategy_2025.py` report
  expectancy under a ±2% substitution (resolved trades forced to exactly
  ±2%). The apparent "decay" between development and validation is partly
  an artifact of comparing two different statistics, not fully a measured
  degradation — this needs to be recomputed on a common metric before
  drawing conclusions.
- **The 2025 final test is underpowered.** 5 qualifying signals, 3
  resolved. This sample cannot distinguish "no edge" from "small edge" in
  either direction — treat it as inconclusive, not as a failure.
- **Single asset.** The original research plan called for BTC, ETH, SOL,
  and other liquid pairs. Only BTCUSDT has been run. Cross-asset replication
  would be the cheapest available increase in effective sample size.
- **No benchmark/null model.** There is no measurement of what a
  randomly-timed short returns in the same bull-regime conditions, so the
  reported −0.2 to −0.5% expectancy cannot yet be separated from BTC's
  unconditional drift over the sample period.
- **Feature search covers several hundred comparisons on n≈70.** The
  RSI-acceleration filter's development-stage AUC (~0.64) is within the
  range expected from selection effects alone at that sample size and
  search width, which is consistent with (though does not prove) its
  failure to hold up out-of-sample.

## Open issues / not yet done

- `cluster_id` / `cluster_size` are computed for every signal but no
  reported statistic groups or bootstraps by cluster.
- `calculate_expectancy.py::build_non_overlapping_portfolio()` exists and
  writes `expectancy_non_overlapping.csv`; none of the headline numbers
  above are drawn from it.
- `development_feature_summary.csv` (in `build_strategy_dataset.py`)
  currently includes `mfe_pct` / `mae_pct` in its feature comparison —
  these are outcome-derived columns and should be excluded before anyone
  screens that file for feature ideas.
- Ambiguous same-bar target/stop touches are handled slightly differently
  between `analyze_signals.py` and the validation/test scripts (whether the
  entry bar itself is eligible for a same-minute hit).
- `test_strategy_2025.py` still contains a function named
  `load_2024_signals()` and a printed "2024" validation-rule message —
  functionally correct (date constants are right) but worth renaming for
  the record.

## Disclaimer

This is a research artifact, not trading advice. The strategies examined
here were not shown to be profitable after realistic costs on the data
tested.

## License

Free Use
