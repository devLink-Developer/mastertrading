# Eudy post-recovery loss investigation — 2026-08-17

## Scope

- Account/stack: Eudy only (`account_alias=eudy`, compose project `trading_bot_eudy`).
- Audit window: since recovery deploy `2026-08-14 19:39 UTC`.
- Ricardo/main stack was read-only and was not restarted or modified.

## Production evidence

- Closed trades: 2, both `ADAUSDT sell`.
- Net PnL: `-0.0188004 USDT`; fees: `0.0032004 USDT`.
- Win: `+0.0062514 USDT` (`+0.2905%`), closed by `flat_signal_timeout`.
- Loss: `-0.0250518 USDT` (`-1.1923%`), closed by exchange stop.
- No open Eudy positions in the database or BingX before/after deployment.
- Current recovery cohort (4 trades): PF `0.634`, expectancy `-0.171%` per trade.

## Root cause

1. The Eudy guard correctly selected exploratory risk at `0.5x`, but BingX `min_qty` forced actual stop risk to `9.46x` and `10.25x` the reduced target.
2. `MIN_QTY_RISK_ABSOLUTE_CAP` then overrode the dynamic min-quantity block because absolute account risk remained below its generic cap. This erased the intended exploration reduction.
3. The edge classifier returned `explore` without calculating or acting on PF/expectancy until the full 12-trade sample, even though the first 4 outcomes were already negative.

## Fix

- Added an early exploration brake at 4 observations (or the configured full sample when lower).
- The brake blocks a cohort when early PF is below `1.0` or expectancy is below the configured floor; it cannot promote to full risk early.
- Added Eudy-only exploration risk-integrity enforcement. While status is `explore`/`telemetry_error`, actual min-quantity risk cannot exceed the existing allowlist watch boundary (`2.0x`).
- The new integrity check runs before either absolute-cap override.
- Non-Eudy accounts and already validated Eudy cohorts retain existing behavior.

## Verification

- Local pure regression tests: `7/7` passed.
- Eudy container integration suite: `11/11` passed with an isolated test database.
- Production decision for the losing context:
  - `allowed=False`
  - `status=exploration_brake`
  - `pf=0.634`
  - `expectancy=-0.171%`
- Production risk-integrity check at the observed `xRisk=10.25`: blocked (`max=2.00`).
- Ricardo check: explicit bypass; main worker remained `Up 10 days` and was not recreated.
- First Eudy execution cycle after deploy completed successfully with `orders_placed=0` and no application error.

## Deployment

- Commit: `eeec80d` (`fix(execution): stop unsafe Eudy exploration`).
- Server repo: `/opt/trading_bot_eudy`.
- Command scope: rebuilt/recreated only `trading_bot_eudy-worker` with `docker-compose.eudy.yml` and `.env.eudy`.

## Limitation

No trading system can guarantee zero future losses. This fix removes two repeatable risk-control failures; valid, capped trades may still lose due to market movement, slippage, gaps, or model error.
