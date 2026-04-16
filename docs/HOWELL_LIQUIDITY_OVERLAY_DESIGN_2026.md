# Howell-Style Liquidity Overlay Design

## Objective

Design a slow macro overlay inspired by Michael Howell / CrossBorder Capital ideas:

- global liquidity matters more than most top-down narratives
- crypto and high-beta risk assets tend to perform best when liquidity is expanding
- financial stress / collateral stress changes the quality of that liquidity

For `mastertrading`, this should be a **regime and risk-budget layer**, not a direct entry trigger.

The goal is not to predict every market turn. The goal is to:

- reduce long exposure when macro liquidity rolls over
- avoid adding alt beta in stress regimes
- allow higher conviction on BTC/ETH trend setups when liquidity expansion and market structure align

## What We Keep From Howell

The useful part of Howell's framework for this repo is:

1. Liquidity is a slow state variable.
2. The most useful output is regime classification, not candle-level timing.
3. BTC and crypto beta are strongly affected by liquidity conditions.
4. Financial conditions and collateral stress can override "headline" liquidity.
5. China / global policy liquidity can matter, but we should start with public, auditable proxies.

## What We Do Not Import Blindly

We should **not** directly encode:

- the 5-year / 65-month cycle as a trading rule
- proprietary CrossBorder indexes we cannot reproduce
- macro calls as hard directional entries on `5m`
- broad "risk-on" claims without validating against our own BTC / ETH / alt behavior

## Repo Fit

This repo already has good places to plug a macro regime overlay:

- `signals/multi_strategy.py`
  - already computes BTC MTF lead state and recommended bias
- `signals/allocator.py`
  - already supports direction-aware multipliers, risk budgets, and context-based score adjustments
- `execution/tasks.py`
  - already applies macro/high-impact and multiple risk multipliers before sizing
- `signals/runtime_overrides.py`
  - already supports DB-first runtime config for toggles and thresholds

This means we do **not** need a separate architecture. We need one more slow regime source.

## Design Principle

The overlay should be:

- slow: refresh every `6h` or `24h`, not every minute
- additive: modulate existing allocator and execution logic
- auditable: use public sources and persist snapshots
- bounded: affect risk and score multipliers, not rewrite core signal logic

## Proposed Module

New file:

- `signals/howell_liquidity.py`

Optional persistence:

- add `MacroLiquiditySnapshot` model under `signals/models.py`

Recommended persisted fields:

- `asof`
- `regime`
- `confidence`
- `composite_score`
- `composite_momentum`
- `fed_net_liquidity_z`
- `financial_conditions_z`
- `dollar_z`
- `stablecoin_growth_z`
- `crypto_market_z`
- `btc_etf_flow_z`
- `details_json`

Why persist:

- allows audit and backtest joins
- avoids Redis-only black box behavior
- lets us compare `main` vs `eudy`

## Public Proxy Stack

### Phase 1: auditable, easy, good-enough

Use only public, stable sources.

1. Fed net liquidity proxy

- `WALCL` from FRED
- `WTREGEN` from FRED
- `RRPONTSYD` from FRED

Core proxy:

```text
fed_net_liquidity = WALCL - WTREGEN - RRPONTSYD
```

This is not "global liquidity", but it is a useful public proxy for USD system liquidity.

2. Financial conditions / collateral stress

- `NFCI` or `ANFCI` from Chicago Fed / FRED
- optional `NFCIRISK`

Interpretation:

- easing financial conditions -> supportive
- tightening / risk spike -> hostile

3. Dollar pressure

- broad trade-weighted USD proxy from FRED, e.g. `DTWEXBGS`

Interpretation:

- stronger USD often tightens global liquidity for risk assets

4. Crypto-native liquidity

- stablecoin market cap growth via DefiLlama stablecoin API
- optional CoinGecko global crypto market cap / dominance

Interpretation:

- stablecoin expansion = better crypto plumbing
- contraction = weaker crypto-specific liquidity

5. BTC spot ETF flow

- optional SoSoValue API

Interpretation:

- useful BTC-specific demand proxy
- should remain optional because it is newer and narrower than system liquidity

### Phase 2: optional upgrades

- China credit / liquidity proxy
- HY spreads / credit spread proxy
- UST collateral stress / Treasury volatility proxy
- QQQ / Nasdaq beta overlay

## Composite Score

We want a bounded score, not a fragile single indicator.

Proposed normalized components:

```text
L1 = zscore(4w ROC of fed_net_liquidity)
L2 = -zscore(ANFCI)                  # lower ANFCI = easier conditions
L3 = -zscore(4w ROC of broad USD)    # falling USD = easier global liquidity
L4 = zscore(4w ROC of stablecoin mcap)
L5 = zscore(4w BTC ETF net flows)    # optional, default weight 0 if unavailable
```

Composite:

```text
liquidity_score =
    0.35 * L1 +
    0.25 * L2 +
    0.15 * L3 +
    0.15 * L4 +
    0.10 * L5
```

Also compute:

```text
liquidity_momentum = ema_5(liquidity_score) - ema_20(liquidity_score)
```

This gives level + slope.

## Regime States

Use only four states. More than that becomes false precision.

### `expanding`

Conditions:

- `liquidity_score >= +0.50`
- `liquidity_momentum > 0`

Meaning:

- macro supportive
- favor BTC/ETH longs
- allow alt longs if local regime also supports

### `late_expanding`

Conditions:

- `liquidity_score >= +0.50`
- `liquidity_momentum <= 0`

Meaning:

- liquidity still supportive, but impulse fading
- still risk-on, but reduce chasing and cap alt beta

### `rollover`

Conditions:

- `-0.50 < liquidity_score < +0.50`
- `liquidity_momentum < 0`

Meaning:

- risk support deteriorating
- keep BTC long only if local trend is strong
- reduce alt risk
- meanrev/grid become relatively more useful than blind trend continuation

### `stress`

Conditions:

- `liquidity_score <= -0.50`
- or `ANFCI / NFCIRISK` spike above stress threshold

Meaning:

- hard reduce risk
- block weak alt longs
- prefer flat over forcing shorts unless local trend is strong

## Integration Rules

### 1. Risk budget layer

Best first integration point.

Add helper in `signals/howell_liquidity.py`:

```python
def liquidity_risk_mult(symbol: str) -> float:
    ...
```

Default behavior:

- `expanding`
  - BTC/ETH: `1.10`
  - alts: `1.00`
- `late_expanding`
  - BTC/ETH: `1.00`
  - alts: `0.90`
- `rollover`
  - BTC/ETH: `0.85`
  - alts: `0.70`
- `stress`
  - BTC/ETH: `0.65`
  - alts: `0.40`

Apply in:

- `execution/tasks.py`
  - multiply `effective_risk_pct` after existing regime/session multipliers

This is the safest first rollout.

### 2. Direction score overlay

Second integration point.

Expose:

```python
def liquidity_direction_mult(symbol: str, direction: str, module: str) -> float:
    ...
```

Suggested rules:

- `expanding`
  - boost `trend long`
  - damp `carry short`
- `late_expanding`
  - neutral BTC longs, damp alt longs slightly
- `rollover`
  - damp `trend long`
  - allow `meanrev short` / `carry short` only if local structure agrees
- `stress`
  - strong damp on `long` for alts
  - no boost to shorts unless local trend is strong

Apply in:

- `signals/allocator.py`
  - near existing direction/context multiplier logic

### 3. Hard blocks

Use sparingly.

Only one hard block is recommended in Phase 1:

- block **alt longs** in `stress` unless:
  - `trend_context.is_strong == True`
  - `btc_lead_state` is not bearish

Do **not** hard block BTC longs in Phase 1.

## Symbol Buckets

The overlay should not treat all instruments equally.

### Base bucket

- `BTCUSDT`
- `ETHUSDT`

### Beta alt bucket

- `SOLUSDT`
- `XRPUSDT`
- `DOGEUSDT`
- `ADAUSDT`
- `LINKUSDT`
- `ENAUSDT`

Reason:

- Howell-style liquidity effects should hit beta alts harder than BTC

## Runtime Settings

Suggested env/settings names:

- `HOWELL_LIQUIDITY_ENABLED`
- `HOWELL_LIQUIDITY_REFRESH_HOURS`
- `HOWELL_LIQUIDITY_CACHE_TTL_HOURS`
- `HOWELL_LIQUIDITY_USE_ETF_FLOWS`
- `HOWELL_LIQUIDITY_USE_STABLECOIN_GROWTH`
- `HOWELL_LIQUIDITY_EXPANDING_BTC_RISK_MULT`
- `HOWELL_LIQUIDITY_EXPANDING_ALT_RISK_MULT`
- `HOWELL_LIQUIDITY_LATE_EXPANDING_ALT_RISK_MULT`
- `HOWELL_LIQUIDITY_ROLLOVER_BTC_RISK_MULT`
- `HOWELL_LIQUIDITY_ROLLOVER_ALT_RISK_MULT`
- `HOWELL_LIQUIDITY_STRESS_BTC_RISK_MULT`
- `HOWELL_LIQUIDITY_STRESS_ALT_RISK_MULT`
- `HOWELL_LIQUIDITY_STRESS_ALT_LONG_BLOCK_ENABLED`
- `HOWELL_LIQUIDITY_STRESS_ANFCI_THRESHOLD`

DB-first overrides should be used for:

- enable/disable
- risk multipliers
- alt-long block toggle

Keep external API URLs / keys in `.env`.

## Celery / Refresh

New task:

- `signals.tasks.run_howell_liquidity_refresh`

Cadence:

- every `6h` is enough
- daily is also acceptable for a first version

Why not every minute:

- this is not intraday signal data
- higher cadence adds noise and API dependency without real edge

## Backtest / Research Workflow

This must be validated as a **macro overlay**, not as a standalone strategy.

### Phase A: historical feature build

Build a daily dataframe with:

- liquidity proxies
- derived regime
- next `1d`, `3d`, `7d`, `14d` BTC / ETH / alt-basket returns

Questions:

- do BTC/ETH forward returns improve in `expanding` vs `rollover/stress`?
- do alt returns deteriorate more than BTC in `stress`?
- does trend module perform materially better in `expanding`?

### Phase B: join to trade history

Join regime to `OperationReport.closed_at` and `opened_at`.

Measure by regime:

- WR
- expectancy
- PF
- max DD
- trade frequency
- by module
- by symbol bucket

### Phase C: shadow rollout

First live deployment should be **shadow-only**:

- compute regime
- log it in allocator reasons / diagnostics
- do not alter risk or signals yet

### Phase D: bounded risk rollout

Only after shadow confirmation:

- activate risk multiplier only
- no hard blocks
- no score boosts

### Phase E: score/block rollout

Only if Phase D helps:

- add alt-long stress block
- add trend/carry direction multipliers

## Minimum Validation Criteria

Do not ship live behavior unless we improve at least two of these:

- expectancy
- PF
- max drawdown
- alt-bucket loss containment in stress periods

And do not accept a change that:

- improves PF by killing too many trades
- improves BTC but materially worsens ETH + alt basket
- duplicates what HMM / MTF regime already captures with no incremental value

## Why This Might Help This Repo

Current repo strengths:

- strong local structure logic
- rich execution/risk controls
- decent MTF context

Current likely gap:

- slow top-down liquidity regime is only partially represented through:
  - BTC MTF lead state
  - HMM
  - macro high-impact windows

Howell-style overlay could add:

- a slower macro filter for BTC/alt beta
- better risk budgeting when the broad liquidity tide changes
- a principled reason to separate BTC/ETH from alt behavior

## Why This Could Fail

Main failure modes:

1. Proxy mismatch

- public proxy != CrossBorder proprietary global liquidity index

2. Lag

- daily macro overlays may be too slow for crypto inflections

3. Redundancy

- HMM + BTC MTF + session logic may already capture most useful effect

4. Overfitting

- too many macro knobs can become narrative-fitting rather than robust signal

## Recommended Rollout Sequence

1. Implement `signals/howell_liquidity.py` with public proxies only.
2. Persist daily snapshots.
3. Add shadow diagnostics to allocator reasons.
4. Audit `OperationReport` by regime.
5. If useful, add only risk multipliers.
6. Only later add directional dampeners / stress blocks.

## Practical Recommendation

If we build this, the **first live version should be risk-only**.

That means:

- no direct entries from liquidity regime
- no global long/short switches
- no full block on BTC
- yes to reducing alt beta in `rollover/stress`
- yes to slightly boosting BTC/ETH trend risk in `expanding`

That is the cleanest way to import Howell's framework into `mastertrading` without turning the bot into a macro discretionary system.

## External References

- CrossBorder Capital / Michael Howell background:
  - https://www.crossbordercapital.com/about.html
  - https://www.crossbordercapital.com/people.html
- Howell framework and cycle references:
  - https://crossbordercapital.com/Docs/Understanding_Liquidity.pdf
  - https://link.springer.com/book/10.1007/978-3-030-39288-8
- Public proxy sources:
  - https://fred.stlouisfed.org/series/WALCL
  - https://fred.stlouisfed.org/series/RRPONTSYD
  - https://fred.stlouisfed.org/series/WTREGEN
  - https://fred.stlouisfed.org/series/ANFCI
  - https://fred.stlouisfed.org/series/NFCI
  - https://defillama.com/docs/api
  - https://docs.coingecko.com/
  - https://sosovalue.gitbook.io/soso-value-api-doc/history-document/v1-soso-value-bitcoin-spot-etf-api-doc
