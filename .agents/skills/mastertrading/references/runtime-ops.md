# Runtime Ops

## Main stack vs eudy

- Main stack (`rortigoza`): use the base repo compose and `.env`
- `eudy`: use `docker-compose.eudy.yml` with `.env.eudy`
- Do not deploy `eudy` with the base compose file.

## Deploy rules

1. Check whether the server worktree is dirty before pulling.
2. If tracked files changed on the server, inspect or stash them; do not overwrite blindly.
3. Deploy only the services you need when possible, usually `web`, `worker`, `beat`.
4. After deploy, verify:
   - containers are `Up`
   - runtime settings are what you expect
   - relevant tests or smoke commands pass

## Config rules

- DB-first runtime overrides are preferred for strategy toggles already supported that way.
- `.env` remains the source of truth for secrets, URLs, ports, queue/bootstrap, and exchange account wiring.
- When stacks differ, compare both env and DB/runtime overrides before diagnosing logic differences.

## Common operational pitfalls

- Orphan `reduceOnly` orders can remain after a close if cleanup is missing or regressed.
- `timestamp is invalid` from BingX points to clock drift or missing re-sync handling.
- Different 1d history depth between stacks can produce different MTF regime context.
- `min_qty` / notional constraints can make a trade unsafe even if the signal is valid.

## Quick prod checks

- Open positions vs open orders
- `free` vs `equity` balance
- recent `RiskEvent` kinds
- `OperationReport` close reasons and leverage
- `min_qty_risk_report` for tradable/watch/blocked state by account size
