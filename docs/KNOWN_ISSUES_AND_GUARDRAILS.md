# Known Issues And Guardrails

Last update: 2026-03-01

## 1) Known issues to keep in mind

Backtest vs live is not fully isomorphic
- Live runs in tighter cadence with async interactions and exchange behavior.
- Backtest bar-based simulation can misrepresent intrabar flips and fills.

Exchange close classification complexity
- Some exchange closures are inferred from price/pnl patterns.
- Keep `close_sub_reason` logic stable for analytics continuity.

Market regime drift
- Directional edge can shift by day/session/regime.
- Global parameter changes can overfit quickly.

Quantity/min notional edge cases
- Exchange minimum lot and precision can reject otherwise valid sizing.
- Any sizing change must be validated against adapter constraints.

## 2) Recent critical fixes (already applied)
- Duplicate close report race mitigation in sync path.
- HTF killer candle ordering bug fix:
  - `execution/tasks.py` switched `order_by("-open_time")` to `order_by("-ts")`.
  - Deployed to main and eudy stacks.

## 3) Guardrails for safe changes
- Change only one behavior class per deploy:
  - signal logic OR sizing OR exits OR session policy.
- Prefer feature flags over hard behavior replacement.
- Keep deterministic fallback on optional model failures.
- No multi-knob leverage/risk increases in one patch.

## 4) Minimum validation before deploy
1. Static check for touched files.
2. Smoke test: containers up and worker alive.
3. Runtime shell check: DB queries and open position read path.
4. Log scan for new exceptions.

## 5) Minimum validation after deploy
1. Confirm both stacks are running (`main`, `eudy`).
2. Confirm no repeated task crashes in worker logs.
3. Confirm entries still happen under valid conditions.
4. Review first closed trades with reasons and sub-reasons.

## 6) Strategy tuning cautions
- Weekend or session assumptions must be validated with your own data first.
- Do not lower allocator threshold globally as first reaction to no-trade periods.
- If market is strongly directional, review directional penalties before widening risk.

## 7) Sensitive data policy
- Never commit:
  - `.env`
  - API keys/secrets
  - DB dumps/backups
  - private credential files
- If leaked, rotate immediately and clean history if needed.

## 8) Change log discipline
- Every prod-impacting patch should record:
  - objective
  - files touched
  - expected risk
  - validation steps
  - rollback commit

