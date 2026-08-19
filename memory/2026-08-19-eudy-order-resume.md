# Eudy order-resume investigation — 2026-08-19

## Scope

- Eudy only (`EUDY_RECOVERY_ENABLED=true`, compose project `trading_bot_eudy`).
- Ricardo/main remains behaviorally unchanged and must use the base compose.
- Baseline branch: `codex/eudy-profit-recovery` at `e875567`.

## Production evidence used

The latest production audit, through 2026-08-18 12:46 UTC, showed:

- 0 new closed trades, 0 orders and 0 open positions in DB or BingX.
- 746 execution cycles with `orders_placed=0` and no critical execution errors.
- 5,944 `alloc_flat` signals; 0 `alloc_long`; 0 `alloc_short`.
- Only `trend` and `carry` were active.
- Maximum aligned score: `0.151269`, below the `0.20` threshold.
- Meta weights assigned only 27.9% combined weight to the two active modules; 72.1% remained assigned to inactive modules.
- Recent trend/carry metrics were negative, so lowering the global threshold or disabling guards was rejected.

Fresh server access was unavailable in the managed offline sandbox because the SSH key and network were not exposed to its user. No production state is claimed after the audit cutoff until post-deploy verification runs.

## Root cause

Two Eudy-only controls formed a self-locking loop:

1. Meta-allocator performance weights were used both for signal scoring and risk allocation. With three modules disabled, recent penalties made even aligned `trend + carry` signals structurally unable to reach the actionable threshold. No new samples could then enter the edge guard.
2. When an Eudy context was still eligible for reduced-risk exploration, exchange `min_qty` produced a high multiplier relative to the tiny target budget. The Eudy integrity guard rejected it before the existing absolute account-risk cap was evaluated.

## Fix

1. In the Eudy recovery stack, pre-meta weights now determine signal direction/threshold while meta risk budgets continue to determine position risk. Non-Eudy stacks keep meta weights for both, preserving Ricardo's behavior.
2. The absolute `min_qty` cap is evaluated before the Eudy exploration integrity check. Exploration can pass a high relative multiplier only when actual loss at the stop is within the absolute cap.
3. `docker-compose.eudy.yml` explicitly enables the cap at 0.30% of equity for Eudy `web` and `worker` only.
4. Existing cohort brakes remain intact: negative early samples, failed PF/expectancy and over-cap risk still block entries.

## Regression coverage

- Eudy aligned `trend + carry` reproduces the production weights and becomes `alloc_short` above 0.20.
- The test confirms the risk-budget mix still comes from the meta overlay.
- With `EUDY_RECOVERY_ENABLED=false`, the same inputs remain `alloc_flat` and report `meta_overlay`, proving the Ricardo/non-Eudy path is unchanged.
- High `min_qty` relative risk remains blocked unless the independently calculated absolute cap allows it.

## Verification

- `execution.tests_eudy_recovery`: 14 passed.
- `signals.tests_meta_allocator`: 8 passed.
- Existing absolute-cap helper tests: 3 passed.
- Python compilation: passed.
- `python manage.py check`: passed.
- Eudy compose structure and values: passed.
- `git diff --check`: passed.

Known pre-existing failure: `signals.tests.AllocatorRuntimeThresholdCycleTest` expects a flat score of 0.19 but the existing trend/carry alignment boost produces an actionable score. It fails identically on the untouched baseline and is unrelated to this change.

## Deploy checklist

Use only:

```bash
docker compose -p trading_bot_eudy -f docker-compose.eudy.yml --env-file .env.eudy up -d --build web worker beat
```

Post-deploy verification must confirm:

- Eudy containers are `Up`.
- Runtime cap resolves to `true` and `0.003` (including DB-first overrides).
- New allocator payloads report `score_weight_source=pre_meta_eudy_recovery`.
- Directional signals reach execution when their cohort guard allows them.
- Actual stop risk remains at or below 0.30% equity.
- Ricardo containers and compose project were not restarted.
