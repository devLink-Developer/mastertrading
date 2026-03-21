---
name: mastertrading
description: Work effectively in the MasterTrading repository. Use when the task is inside the mastertrading repo or affects trading logic, signal modules, allocator behavior, BingX live/demo stacks, risk guards, backtests, Telegram reporting, or production deploy workflows documented in AGENTS.md and repo docs.
---

# MasterTrading

## Overview

Use this skill when working inside the `mastertrading` repo so we do not re-discover the project's trading rules, deploy constraints, and analysis workflow each time.

Treat the repo `AGENTS.md` as the primary source of truth. Use the references in this skill to get oriented quickly, then open repo docs only where needed.

## Default Workflow

1. Read `AGENTS.md` first.
2. For broad orientation, read `docs/AI_AUDIT_PROJECT_MAP.md`.
3. For trading logic, exits, sizing, session policy, or stack comparisons, read [references/trading-playbook.md](references/trading-playbook.md).
4. For deploys and stack operations, read [references/runtime-ops.md](references/runtime-ops.md).
5. For a short rollout sequence, read [references/deploy-checklist.md](references/deploy-checklist.md).
6. For code navigation and recurring touchpoints, read [references/repo-map.md](references/repo-map.md).
7. Prefer DB-first runtime overrides for supported strategy toggles; keep secrets, URLs, ports, and credentials in env.
8. Before changing trading behavior, define the smallest useful validation: targeted tests, backtest, replay, or live/demo comparison.

## Workflow Decision Tree

### Trading logic changes

- Touch the smallest surface possible first.
- Distinguish whether the issue is:
  - bad entry
  - bad exit
  - exchange sizing / min-qty distortion
  - stale or mismatched context
- Check `mfe_r`, `mae_r`, close reason, and session before changing exits.
- Avoid symbol-specific conclusions from a short recent slice alone.

### Backtests and analysis

- Prefer the repo backtest engine and existing scripts.
- 1m backtests are usually more faithful than 5m bars when signal timing matters.
- Align live gates in backtests before trusting optimization output.
- If a result looks good only on a short window, verify a broader out-of-sample window before deploying.

### Runtime config changes

- Strategy toggles belong in DB-first runtime overrides when available.
- Infra/bootstrap settings belong in `.env`.
- Compare both env and DB/runtime overrides before explaining why stacks differ.

### Deploys

- Main stack and `eudy` use different compose workflows.
- Do not overwrite a dirty server worktree blindly.
- Deploy only the services you need when possible.

## MasterTrading Guardrails

- Never deploy `eudy` with the base compose file.
- Never assume a losing trade was a bad exit before checking whether it had meaningful favorable excursion.
- Watch for exchange `min_qty` / notional floors that inflate real risk beyond target risk.
- Orphan `reduceOnly` orders are operational garbage, not fresh directional intent.
- If stacks diverge, compare:
  - runtime overrides
  - feature flags
  - live vs demo mode
  - candle parity on the relevant timeframe
  - module diagnostics for the exact timestamp

## References

- [references/repo-map.md](references/repo-map.md): architecture, key files, code navigation
- [references/runtime-ops.md](references/runtime-ops.md): deploy rules, stack differences, operational pitfalls
- [references/deploy-checklist.md](references/deploy-checklist.md): short prod rollout checklist
- [references/trading-playbook.md](references/trading-playbook.md): analysis workflow, risk/exit lessons, backtest cautions
