# LLM Context Pack Index

Last update: 2026-03-01

Purpose
- Fast bootstrap for an LLM/copilot to understand how this project works.
- Keep this file short and stable. Use it as the first read.

Recommended read order
1. `docs/ARCHITECTURE.md`
2. `docs/TRADING_RULES.md`
3. `docs/ENV_REFERENCE.md`
4. `docs/OPERATIONS_RUNBOOK.md`
5. `docs/KNOWN_ISSUES_AND_GUARDRAILS.md`
6. `docs/API_ADMIN_CONFIG.md` (LLM API setup and token budget flow)
7. `agents.md` (historical context and decisions log)

Code map (high value files)
- `execution/tasks.py`: live execution loop, entries, exits, sync, risk guards.
- `signals/tasks.py`: signal engine and SMC flow.
- `signals/allocator.py`: multi-module score aggregation and dynamic weights.
- `signals/sessions.py`: session classification and session-based score/risk.
- `signals/modules/trend.py`: trend module.
- `signals/modules/meanrev.py`: mean reversion module.
- `signals/modules/carry.py`: funding carry module.
- `config/settings.py`: env parsing and all runtime flags.
- `execution/models.py`: `Order`, `Position`, `OperationReport`, `BalanceSnapshot`.
- `marketdata/models.py`: `Candle`, `FundingRate`, `OrderBookSnapshot`.
- `backtest/engine.py`: walk-forward backtest engine.
- `risk/management/commands/perf_dashboard.py`: performance summary by module/symbol.
- `risk/management/commands/monte_carlo.py`: risk of ruin and drawdown simulations.

Runtime flow (one-line)
- Marketdata -> signals -> allocator -> execution -> sync -> operation report -> notifications.

Current production topology
- Main stack: `/opt/trading_bot` (port `8008`).
- Eudy stack: same repo path, compose project `trading_bot_eudy` (port `8010`).
- Camping chatbot is a separate project in `/opt/chatbot` (port `8006`), not part of this trading runtime.

Safety notes for AI-generated patches
- Do not change risk defaults and execution gates in one patch without backtest comparison.
- Any change touching `execution/tasks.py` must include a quick smoke verification.
- Keep secrets out of git. Use `.env` on server/local only.
- Prefer feature flags in `config/settings.py` for new behavior.
