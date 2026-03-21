# Trading Playbook

## How to judge a bad trade

Separate three cases:
1. bad entry: the trade never developed; low `mfe_r`, meaningful `mae_r`
2. bad exit: the trade developed and then gave back too much
3. bad sizing: the exchange minimum forced much more real risk than the model intended

Do not blame exits before checking `mfe_r`, `mae_r`, session context, and close reason.

## Backtest cautions

- 1m backtests are usually more faithful than 5m for this project when signal timing matters.
- Keep live gates aligned in backtests before using optimization results.
- For symbol-specific changes, require broader coverage and enough data depth.

## Session/context lessons

- `ny_open` behaves like a distinct sub-session and needs stricter handling.
- BTC-led and MTF regime context matter for alt filtering.
- A trade can be strong in score yet still be structurally weak if session/context is transition-heavy.

## Exits and timing

- A fast loss is not necessarily a bad exit if the trade never had favorable excursion.
- A quick manual winner later in the session can still be a different-quality setup than an earlier loser in the same symbol/direction.
- Use session development and reachability ideas when evaluating whether TP was realistic.

## Min-qty risk discipline

- The model sizes by intended risk, but exchange minimums can break that.
- Use `min_qty_risk_report` to classify symbols as:
  - `tradable`
  - `watch`
  - `blocked`
- A symbol can be valid logically and still be invalid operationally for a given account size.

## Practical debugging sequence

1. Check if the stack was live or demo.
2. Compare runtime overrides and feature flags.
3. Inspect the entry signal and active modules.
4. Inspect `OperationReport` and close reason.
5. Check if exchange constraints or stale orphan orders distorted the situation.
6. Only then decide whether to change entry logic, exits, or risk rules.
