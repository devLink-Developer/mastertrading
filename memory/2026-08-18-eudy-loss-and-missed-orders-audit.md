# Eudy loss and missed-orders audit — 2026-08-18

## Scope

- Eudy only (`account_alias=eudy`, compose project `trading_bot_eudy`).
- Audit cutoff: Eudy worker deploy `2026-08-18 00:19 UTC` (`eeec80d`).
- Ricardo/main remained read-only.

## Symptom investigated

The account had not opened trades after the last risk fix. The audit checked both:

1. whether Eudy had incurred new realized or unrealized losses;
2. whether actionable orders were generated but failed to reach BingX.

## Production evidence

- Closed operations since cutoff: `0`.
- Orders created since cutoff: `0`.
- Open positions: `0` in Django and `0` in BingX.
- Equity/free balance: `10.195 USDT / 10.195 USDT`, unchanged from the previous audit.
- Execution cycles: `746` with `orders_placed=0`; `0` cycles with a non-zero placement count.
- Critical/error/traceback log matches: `0`.
- Risk events: `0`; circuit breaker enabled but not tripped.

## Root cause hypothesis and confirmation

**Hypothesis:** orders were not lost in execution; the allocator never produced an actionable direction.

Confirmed from the Eudy signal database:

- Total signals since cutoff: `9,451`.
- Allocator output: `5,944 alloc_flat`, `0 alloc_long`, `0 alloc_short`.
- Therefore `_attempt_entry_open()` never received a directional allocator signal and no exchange order was expected.
- Active allocator modules: `trend`, `carry`.
- Inactive modules: `meanrev`, `grid`, `smc`.
- Threshold: `0.20`; maximum observed aligned flat score was `0.151269` (ADA short).
- Meta allocator metrics correctly identify weak recent edge:
  - trend PF `0.1275`, expectancy `-0.1562%`;
  - carry PF `0.1401`, expectancy `-0.1637%`.

This is a protective no-trade state driven by negative recent evidence, not an execution outage.

## Counterfactual missed-order analysis

### Trend + carry candidates

- `151` sub-threshold signal points grouped into `8` distinct episodes.
- Only ADA/DOGE shorts reached `|net_score| >= 0.12`; none reached `0.20`.
- Among episodes with enough subsequent market data, ADA shorts deteriorated by roughly `-0.40%` to `-0.58%` directionally.
- DOGE shorts produced only about `+0.07%` to `+0.10%` gross, approximately consumed by round-trip fees and far below the configured `0.8%` TP.
- Lowering the allocator threshold would not have produced a material positive result in this sample.

### SMC signals excluded by its feature flag

- `23` SMC signals grouped into `8` episodes.
- Directional 180-minute returns summed to approximately `-2.755%` gross across the episodes.
- Six episodes were negative; the two positive episodes were small (`+0.181%` and `+0.117%` gross).
- Enabling SMC or allowing it solo would have worsened this observed window.

### Operational anomalies

- `48` ADA ticker resolutions were rejected because spread was wider than the dynamic cap (`~28.8–34.9 bps`).
- At those times the allocator signal was still `flat`; no actionable order was lost.
- One transient BingX balance error (`code 100410`) caused one fail-closed cycle and recovered automatically. It did not overlap an actionable allocator signal.

## Fix decision

No code or configuration change was applied. There is no confirmed loss or missed actionable order to fix. Relaxing the score threshold, spread guard, module-count requirement, or enabling SMC would have increased exposure to candidates whose observed counterfactual performance was negative or fee-level.

## Verification

- Re-queried OperationReport, Order, Position, BingX positions and live balance.
- Cross-checked allocator signals, module flags, meta weights, logs and 1-minute candles.
- Ricardo was not deployed, restarted, or mutated.

## Status

`DONE` — no new losses; no execution defect; no unsafe change justified by evidence.
