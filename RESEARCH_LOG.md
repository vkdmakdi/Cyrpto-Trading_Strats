# BTCUSDT Multi-Timeframe RSI Shorting Study — Research Log

Full research record, from raw data through the final 2025 test. See the
top-level [`README.md`](../README.md) for a summary, reproduction steps,
and the limitations that apply to every result below.

## Table of Contents

- [1. Original Hypothesis](#1-original-hypothesis)
- [2. Data We Assembled](#2-data-we-assembled)
- [3. RSI Construction](#3-rsi-construction)
- [4. Signal Generation](#4-signal-generation)
- [5. Why the Original Strategy Looked Suspicious](#5-why-the-original-strategy-looked-suspicious)
- [6. First Proper Outcome Test](#6-first-proper-outcome-test)
- [7. Original Strategy Results](#7-original-strategy-results)
- [8. MFE / MAE Finding](#8-mfe--mae-finding)
- [9. Full Target/Stop Grid](#9-full-targetstop-grid)
- [10. Signal Clustering](#10-signal-clustering)
- [11. Expectancy Testing](#11-expectancy-testing)
- [12. Feature Engineering](#12-feature-engineering)
- [13. First Feature-Analysis Result](#13-first-feature-analysis-result)
- [14. Chronological Data Split](#14-chronological-data-split)
- [15. Development Findings](#15-development-findings)
- [16. Confirmation Experiments](#16-confirmation-experiments)
- [17. RSI Acceleration Filter](#17-removing-the-12h-filter)
- [18. Frozen Candidate Strategy](#18-frozen-candidate-strategy)
- [19. 2024 Validation](#19-2024-validation)
- [20. 2025 Final Test](#20-2025-final-test)
- [21. What We Now Believe](#21-what-we-now-believe)
- [22. Things We Should Not Do](#22-things-we-should-not-do)
- [23. What We Should Do Next](#23-what-we-should-do-next)
- [24. Next Feature Research](#24-the-next-feature-research)
- [25. Considered and Set Aside: Dynamic Exits](#25-considered-and-set-aside-dynamic-exits)
- [26. Current Project State](#26-current-state-of-the-project)
- [27. The Most Important Lesson](#27-the-most-important-lesson-from-the-entire-exercise)

---

## 1. Original Hypothesis

The starting idea was:

> When BTCUSDT RSI(14) is overbought simultaneously on 15m, 30m, 1h, and 4h,
> short BTC expecting a reversal.

Initial assumption:

- 15m RSI > 70
- 30m RSI > 70
- 1h RSI > 70
- 4h RSI > 70
- Enter short
- Initial target idea ≈ −2%

We recognized early that this needed a proper backtest rather than judging
the setup visually.

---

## 2. Data We Assembled

We downloaded and validated BTCUSDT data from Binance for **January 1,
2021 through December 31, 2025**.

> **Note:** this is Binance *spot* BTCUSDT (`download_data.py` pulls from
> the spot klines endpoint). A short in practice means the perpetual
> contract, a related but distinct price series with its own basis and
> funding. See the README's Limitations section — this affects every
> result in this log.

### 15-minute data

- 175,226 candles
- Expected: 175,296
- Missing: 70 candles
- 7 genuine historical gaps
- No duplicates
- No missing OHLCV values
- No invalid highs/lows
- No non-positive prices

We **did not fill the gaps**. The seven gaps were explicitly identified
rather than silently interpolated.

### 1-minute data

Used for realistic target/stop first-touch testing.

- 2,628,367 candles
- 1,073 missing minutes
- Every missing interval corresponded exactly to the same seven underlying
  data gaps
- No duplicates
- No invalid OHLC values
- No non-positive prices

This gave us a clean 1m execution layer.

---

## 3. RSI Construction

We calculated Wilder-style RSI(14) for 15m, 30m, 1h, and 4h. The higher
timeframes were constructed from the 15m candles.

Important implementation decisions:

**No look-ahead.** A 15m observation only receives the RSI of a completed
higher-timeframe candle. A 15m candle does not get the eventual 1h RSI
before that 1h candle has closed.

**Gap-aware.** RSI calculations reset across real historical gaps. We did
not allow a missing-data interval to contaminate the next RSI calculation.

**Incomplete HTF candles.** Incomplete 30m/1h/4h bins were rejected — we
were deliberately conservative rather than inventing higher-timeframe
candles.

---

## 4. Signal Generation

The original signal was:

```text
15m RSI > 70
AND
30m RSI > 70
AND
1h RSI > 70
AND
4h RSI > 70
```

We then required a **new transition into that state**, rather than
repeatedly counting every subsequent candle.

Results:

- 478 candles had all four RSI values > 70
- Those produced **120 distinct signal events**

By year:

| Year | Signals |
|---|---:|
| 2021 | 15 |
| 2022 | 15 |
| 2023 | 40 |
| 2024 | 30 |
| 2025 | 20 |
| **Total** | **120** |

The RSI value corresponds to the **close of the signal candle**, so the
signal is only known after that 15m candle finishes. The correct entry
convention was:

> **signal timestamp + 15 minutes = next 15m candle open**

We avoided entering at the signal candle's open because that would
introduce look-ahead.

---

## 5. Why the Original Strategy Looked Suspicious

We identified several structural problems with the original idea.

**RSI > 70 doesn't mean "about to reverse."** It means momentum is strong
enough to be considered overbought by the indicator's conventional
interpretation. A market can remain overbought for a long time.

**All four timeframes are correlated.** 15m, 30m, 1h, and 4h RSI aren't
four independent signals — they are different views of the same underlying
price movement. Four timeframes overbought ≠ four independent
confirmations.

**Strong bull markets can repeatedly produce the signal.** This became
especially obvious once we built the feature layer. All **120/120 signals
occurred in our defined bullish EMA200 regime** (price above rising
EMA200). The original setup was overwhelmingly identifying strong bullish
momentum, not random overbought conditions.

---

## 6. First Proper Outcome Test

We tested the original strategy using 1-minute data.

```text
Target: BTC −2%
Stop:   BTC +2%
Horizon: 24h / 48h / 72h / 7d
```

The target and stop were determined by **which level was hit first**. We
did not ask "did BTC eventually fall 2%?" — that could happen after BTC
first went +8%. We asked which happened first, −2% or +2%. A single 1m
candle touching both levels was classified as **AMBIGUOUS**. Real data
gaps caused the test to stop rather than jump across the missing interval.

---

## 7. Original Strategy Results

For the −2% target / +2% stop (figures are gross — see README Limitations):

| Horizon | Target first | Stop first | Neither | Resolved win rate |
|---|---:|---:|---:|---:|
| 24h | 35 | 60 | 25 | 36.8% |
| 48h | 41 | 70 | 9 | 36.9% |
| 72h | 42 | 75 | 3 | 35.9% |
| 7d | 43 | 77 | 0 | 35.8% |

The result was remarkably consistent: roughly **36% target-first vs 64%
stop-first**. That is not an attractive 1:1 risk/reward short strategy.

---

## 8. MFE / MAE Finding

At 24h: median MFE ≈ **+1.49%**, median MAE ≈ **−2.09%**.
At 7d: median MFE ≈ **+2.72%**, median MAE ≈ **−6.33%**.

The typical trade experienced substantially more adverse movement than
favorable movement — consistent with these signals occurring during strong
upward momentum, rather than immediately before clean reversals.

---

## 9. Full Target/Stop Grid

We didn't only test −2/+2. We examined:

- Targets: −0.5%, −1%, −1.5%, −2%, −3%, −5%
- Stops: +0.5%, +1%, +1.5%, +2%, +3%, +5%, +10%

There was **no genuinely positive expectancy combination** across the full
2021–2025 sample, and this grid was run gross — before fees, slippage, or
funding. Some combinations looked superficially attractive because the
target was easy to hit, but the losses/tail behavior erased the advantage.

> Target-hit rate alone is not enough.

---

## 10. Signal Clustering

The 120 events weren't necessarily 120 independent market situations.

- 120 signals → 80 provisional 4-hour clusters
- 53 clusters of size 1, 17 of size 2, 8 of size 3, 1 of size 4, 1 of size 5

We cannot simply treat the 120 trades as 120 completely independent
observations — signals bunch during the same market episode, which matters
for statistical confidence. `cluster_id` / `cluster_size` are carried
through every downstream dataset, but no reported statistic here yet
groups or bootstraps by cluster (see README, Open Issues).

---

## 11. Expectancy Testing

We calculated actual trade P&L at the horizon for trades that didn't hit
target/stop — a trade that doesn't hit ±2% still has a real outcome at
24h, and ignoring that is a common mistake.

From the exploratory grid: −0.5/+0.5 was approximately flat; other
combinations remained negative; no tested combination produced a
convincing positive expectancy before costs.

---

## 12. Feature Engineering

We stopped asking "is overbought enough?" and started asking **"what
distinguishes the overbought signals that actually reverse from those that
keep going up?"**

We built a point-in-time-safe feature dataset:

- **RSI:** absolute levels, slopes, changes across several bars
- **Trend:** EMA20/50/200, distance from EMAs, EMA slopes
- **Volatility:** ATR, ATR%
- **Bollinger:** position within bands, width
- **Volume:** volume / 20-period average volume
- **Momentum:** 1h/4h/12h/24h return
- **Structure:** prior swing highs/lows, distance to those levels
- **Signal candle:** body, upper/lower wick, range

All features were checked for point-in-time availability.

---

## 13. First Feature-Analysis Result

For the original −2%/+2% setup on 24h, 95 signals were resolved: 35
target-first, 60 stop-first.

**`RSI30_delta_4` (RSI30 acceleration):** AUC ≈ 0.592, Spearman ≈ +0.154.
Stronger when later isolated inside the development period.

**Volume (`volume_ratio_20`):** modest positive separation.

**12h momentum (`return_12h_pct`):** successful shorts tended to have
*lower* recent momentum than failed shorts — consistent with strong
momentum simply continuing.

**Absolute RSI level:** weak (RSI30 level AUC ≈ 0.453, RSI1h level AUC ≈
0.454). "More overbought" did not mean "better short" — a major correction
to the original hypothesis.

---

## 14. Chronological Data Split

- **Development:** 2021–2023 — 70 signals, used to discover candidate
  features/rules
- **Validation:** 2024 — 30 signals, used to evaluate the frozen strategy
  without tuning it
- **Final test:** 2025 — 20 signals, untouched until the strategy was
  frozen

This separation is important: 2025 was not used to discover the rule.

---

## 15. Development Findings

Within 2021–2023: 70 signals — 35 stop-first, 22 target-first, 13 neither.
Baseline resolved win rate: **38.6%**.

The strongest development-only feature was `RSI30_delta_4`, AUC ≈ **0.644**
— materially stronger than the full-sample relationship. Quartile
breakdown:

| RSI30 acceleration | Resolved WR |
|---|---:|
| Q1 low | 25.0% |
| Q2 | 38.5% |
| Q3 | 57.1% |
| Q4 high | 62.5% |

That monotonic progression suggested a rapidly accelerating 30m RSI might
identify a different kind of overbought condition than a merely high RSI
value.

> **Caveat added on review:** this feature was one of ~43 numeric features
> screened, each tested against several statistics and quartile cuts —
> several hundred comparisons on n=70. An AUC of 0.644 as the *maximum*
> over a search that wide is within the range expected from chance alone.
> The out-of-sample degradation in Sections 19–20 is consistent with, though
> does not by itself prove, that this was a selection artifact rather than
> a real relationship.

---

## 16. Confirmation Experiments

We explored several ways to wait for actual reversal confirmation.

**Strategy B:** original signal → wait for 15m RSI to cross below 70 →
enter next candle. Development: 70 entries, 27 target, 32 stop, 11
neither, resolved WR ≈ **45.8%**. Slightly better than baseline, not
enough alone.

**Strategy C:** signal → RSI cross below 70 **and** break below the prior
4 completed 15m lows → enter next candle. Development: 64 entries, 23
target, 28 stop, 13 neither, resolved WR ≈ **45.1%**. Similar win rate,
somewhat better MFE/MAE. Still not obviously robust.

**Strategy D (initial):** RSI30 acceleration filter + low recent 12h
return + RSI cross. Only 5 trades qualified in this early version,
producing an 80% resolved win rate — we explicitly rejected this as too
few observations to trust.

---

## 17. Removing the 12h Filter

We isolated the RSI30 acceleration component. With the RSI cross
confirmation:

| Quartile | Trades | Resolved WR |
|---|---:|---:|
| Q1 | 18 | 25.0% |
| Q2 | 17 | 38.5% |
| Q3 | 17 | 57.1% |
| Q4 | 18 | 62.5% |

Then pre-declared thresholds:

| Threshold | Trades | Resolved WR | Mean raw short return |
|---|---:|---:|---:|
| ≥0 | 46 | 52.6% | +0.086% |
| ≥2 | 37 | 58.1% | +0.418% |
| **≥4** | **28** | **58.3%** | **+0.441%** |
| ≥6 | 17 | 60.0% | +0.396% |
| ≥8 | 14 | 58.3% | +0.951% |

We deliberately did **not** pick ≥8 despite its higher average — that
would have been chasing the best-looking historical number. We froze
**`RSI30_delta_4 ≥ 4`**, retaining a reasonable sample size while
preserving the monotonic relationship.

> Note: the figures in this section (`mean_raw_short_return_pct`) are
> mean mark-to-horizon-close return. Sections 19–20 report a different
> statistic (`expectancy_gross_pct`, with resolved trades substituted to
> exactly ±2%). These are not directly comparable — see the README's
> Limitations section before reading Section 17 vs. 19–20 as "decay."

---

## 18. Frozen Candidate Strategy

The strategy we locked was:

1. All four RSI values > 70.
2. `RSI30_delta_4 >= 4`.
3. Wait for 15m RSI to cross from ≥70 to <70.
4. Enter on the next 15m candle open.
5. Target = −2%. Stop = +2%. Maximum holding period = 24h.

This was the **only** rule carried forward into validation.

---

## 19. 2024 Validation

We did **not** change the rule after seeing 2024.

| Variant | Signals | Target | Stop | Neither | Resolved WR | Expectancy | Profit factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 30 | 9 | 18 | 3 | 33.3% | −0.543% | — |
| RSI cross only | 30 | 8 | 17 | 5 | 32.0% | −0.541% | — |
| RSI cross + swing break | 30 | 4 | 19 | 7 | 17.4% | −0.851% | — |
| **Frozen (accel + cross)** | 13 qualifying | 4 | 6 | 3 | 40.0% | −0.235% | 0.759 |

The frozen strategy was better than the baseline, but still not
profitable. Encouraging enough to justify the final test, but not enough
to declare an edge — and at n=13 qualifying signals, not enough to declare
its absence either.

---

## 20. 2025 Final Test

This was untouched before the test.

| Variant | Signals | Target | Stop | Neither | Resolved WR | Expectancy | Profit factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 20 | 4 | 7 | 9 | 36.4% | −0.389% | — |
| RSI cross | 20 | 3 | 7 | 10 | 30.0% | −0.454% | — |
| RSI cross + swing break | 20 | 1 | 4 | 15 | 20.0% | +0.111% | 1.238 |
| **Frozen (accel + cross)** | 5 qualifying | 1 | 2 | 2 | 33.3% | −0.218% | 0.728 |

Frozen strategy detail: mean raw short return +0.300%, median MFE +1.94%,
median MAE −1.52%.

**Reading this correctly:** with 5 qualifying signals and 3 resolved
trades, this test has essentially no statistical power. It cannot confirm
an edge and it cannot rule one out — it is inconclusive, not a failure.
The RSI-cross-plus-swing-break variant's flip from *worst* performer in
2024 validation (17.4% WR) to *best* in the 2025 sample (+0.111%,
PF 1.238, on 20 signals with 15 unresolved) is the more informative
result here, and it points toward noise rather than a real effect — see
[Section 25](#25-considered-and-set-aside-dynamic-exits).

---

## 21. What We Now Believe

**The original "all RSI > 70 → short" hypothesis is not supported.** It
loses more often than it wins under the tested 1:1 framework.

**Absolute RSI level isn't the useful information.** Being *more*
overbought doesn't make a short better.

**Momentum matters.** The signals happen in strong bullish environments.

**RSI trajectory looked more interesting than RSI level in development.**
`RSI30_delta_4` showed a strong development-period relationship.

**That relationship did not survive out-of-sample strongly enough.** 2024
improved on baseline; 2025's tiny qualifying sample neither confirmed nor
refuted it. We cannot promote it to a real trading edge on this evidence,
and the development-stage AUC is itself within the range expected from a
wide feature search on n=70 (Section 15).

**Fixed ±2% exits may be an overly simplistic representation.** A large
fraction of trades never touched either threshold — binary target/stop
statistics hide part of the underlying return distribution.

---

## 22. Things We Should Not Do

This is probably the most important section.

**Do not tune against 2025.** 2025 is now the final test set. Don't go
back and say "let's try ≥6 instead," or "use a 3% stop," or "RSI cross at
72." That contaminates the experiment.

**Do not pick the best historical threshold.** We explicitly avoided "≥8
had +0.95%, therefore use ≥8" — textbook overfitting.

**Do not optimize target/stop on the whole dataset.** Searching dozens of
TP/SL combinations and selecting the best one on 2021–2025 manufactures
false confidence.

**Do not treat RSI timeframes as independent confirmations.** They are
highly correlated observations of the same price movement.

**Do not interpret RSI > 70 as automatic reversal.** That was the core
conceptual mistake in the original idea.

**Do not judge a strategy solely by target-hit rate.** A strategy that
hits a target often can still lose money if stops are larger, losers are
larger, unresolved trades lose, or fees/slippage/funding matter.

**Do not ignore unresolved trades.** A trade that doesn't hit either
threshold still has a real return.

**Do not assume 120 signals = 120 independent experiments.** Signals
cluster; the effective sample size is lower.

**Do not add fees later and pretend they don't matter.** Everything above
is gross. Real crypto trading needs trading fees, slippage, funding, and
possibly bid/ask spread effects. A strategy with ~−0.2% gross expectancy
is particularly unlikely to survive realistic costs.

---

## 23. What We Should Do Next

We should stop trying to "fix" the original RSI strategy and change the
question.

**New research question:** rather than "can we find the right threshold to
short when RSI is overbought?", investigate **"what does BTC actually do
after a four-timeframe overbought event?"** — measure the subsequent
return distribution at 1h, 2h, 4h, 8h, 12h, 24h, 48h, 72h, 7d, for both
upside continuation and downside reversal, against a benchmark of
randomly-timed entries in the same regime (see README, Limitations: no
null model yet exists).

---

## 24. The Next Feature Research

Condition those outcomes on the variables already identified as
potentially useful:

- **Momentum:** 1h/4h/12h/24h return
- **RSI trajectory:** RSI30/1h/4h acceleration, RSI deceleration
- **Trend:** EMA20/50/200 distance, EMA slopes, trend strength
- **Volatility:** ATR%, Bollinger width
- **Volume:** volume / average volume
- **Structure:** distance from recent high/low, distance to breakout levels

The goal: what characteristics correspond to actual exhaustion, rather
than assuming overbought = exhaustion. Given Section 15's caveat, any new
feature search should pre-register a small, fixed candidate list rather
than screening dozens of features against a sample of this size.

---

## 25. Considered and Set Aside: Dynamic Exits

The 2025 results showed the swing-break confirmation variant with
positive horizon-close expectancy despite very few ±2% resolutions (see
Section 20). On its face this suggested a "dynamic exit" research
direction — trailing stops, volatility-scaled targets, time-based exits,
exit on RSI reversal, exit on return to EMA — instead of the fixed ±2%
framework.

**We're setting this aside rather than prioritizing it.** The same
variant was the *worst* performer in 2024 validation (17.4% resolved WR,
−0.851% expectancy) and the *best* in the 2025 sample. A worst-in-validation,
best-in-test flip on n=20 with 15 unresolved trades is the signature of
noise, not a discovered effect — and acting on it would mean tuning
against the final test set, which Section 22 explicitly rules out.

If dynamic exits are investigated later, it should be as a fresh
hypothesis researched from the development set forward, with the same
chronological discipline as the rest of this study — not as a
continuation of this particular observation.

---

## 26. Current State of the Project

**Confirmed:**
- Data pipeline: clean, validated, gap-aware, reproducible (on the caveat
  that it is spot, not perpetual, data — see README)
- RSI calculation: point-in-time safe, no look-ahead, higher-timeframe
  alignment handled correctly
- Original hypothesis: not supported
- Absolute RSI > 70: weak predictor of reversal
- Strong bullish regime: characteristic of these signals

**Interesting but unproven:**
- RSI30 acceleration: promising in development, not sufficiently robust
  out-of-sample, and its development-stage strength is itself consistent
  with a multiple-comparisons artifact

**Rejected / deprioritized:**
- RSI cross alone: too weak
- RSI cross + swing break: poor under the fixed-exit framework, and its
  2025 result specifically looks like noise (Section 25)
- Highly selective hand-picked filters: too prone to overfitting and tiny
  sample sizes

---

## 27. The Most Important Lesson From the Entire Exercise

We started with:

> "When everything is overbought, short."

After five years of data and multiple layers of testing, the evidence
instead points toward:

> "When everything is overbought, BTC is often displaying strong upward
> momentum. The interesting problem is identifying when that momentum is
> actually exhausting."

That's a much better research question. And we now have a reasonably
disciplined framework for answering it — development → freeze → validation
→ final test — without continuously moving the goalposts. That framework,
plus the point-in-time-safe data pipeline it's built on, is arguably more
valuable than the first strategy itself.
