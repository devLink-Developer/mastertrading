# API Config In Admin + Token Optimization

Last update: 2026-03-01

## What was added
- `core.ApiProviderConfig`
- `core.ApiContextFile`
- `core.ApiTokenUsageLog`
- `core.AiFeedbackEvent`
- `core.ExchangeCredential.ai_enabled` (bool per trading account)
- `core.ExchangeCredential.ai_provider_config` (optional FK)
- Runtime helpers: `core/api_runtime.py`
- Runtime feedback stream: `core/ai_feedback.py`
- Command: `python manage.py preview_api_context`
- Command: `python manage.py rebuild_ai_feedback_stream`

## Admin tables

`ApiProviderConfig`
- One row per provider/model profile.
- Stores encrypted `api_key`.
- Includes token budgets:
  - `max_input_tokens`
  - `max_output_tokens`
- Supports `active` and `is_default`.

`ApiContextFile`
- Links files to a provider config.
- Controls:
  - `priority`
  - `max_tokens` per file
  - `trim_mode` (`head`/`tail`)
  - `required` vs optional

`ApiTokenUsageLog`
- Stores prompt/completion/total tokens and metadata for auditing and cost tracking.

`AiFeedbackEvent`
- Structured runtime feedback for AI decisions and execution errors.
- Stored in PostgreSQL and mirrored to compact JSONL stream.
- Used to improve decisions over time with real production outcomes.

`ExchangeCredential` (updated)
- `ai_enabled`: enable/disable AI gate for that account (`bingx`, `kucoin`, etc).
- `ai_provider_config`: optional direct profile binding for the account.

## Token optimization flow
1. Resolve active API config from DB.
2. Read enabled context files in priority order.
3. Apply per-file token cap and trim strategy.
4. Fit all context into:  
   `max_input_tokens - reserve_output_tokens - user_prompt_tokens`
5. Return optimized context text + file-level token report.

## Entry gate integration
- Execution uses account context from active `ExchangeCredential`.
- If `ai_enabled=false` -> no AI call.
- If `ai_enabled=true`:
  1. Resolve config from `ai_provider_config`, else fallback by owner/global default.
  2. Build optimized context from `ApiContextFile`.
  3. Call `/v1/responses` (or custom `base_url`) and expect JSON:
     - `allow` (bool)
     - `risk_mult` (0..1)
     - `reason` (short text)
  4. If `allow=false`, entry is blocked.
  5. If `risk_mult<1`, position risk is reduced.
  6. Token usage is logged in `ApiTokenUsageLog`.
  7. Decision + runtime error feedback is logged to:
     - `AiFeedbackEvent` table
     - `AI_FEEDBACK_JSONL_PATH` compact stream (append-only JSONL)

## Prompt/response format (token-efficient)
- Prompt payload uses compact JSON keys (`sym, st, dir, sc, atr, spr, sl, ses, sig`).
- Signal subpayload uses compact keys (`ns, mr, rb, er, rg, se`), where `mr` rows are arrays `[module,dir,confidence,raw_score]`.
- Response contract remains strict JSON only:
  - `allow` (bool)
  - `risk_mult` (0..1)
  - `reason` (short text)
- Runtime supports TOON context files (`*.toon.md` or files containing `FORMAT: TOON`) and compacts them automatically before token trim.
- Effective context path: compact JSON prompt + compact TOON context + compact JSONL feedback stream.

## Efficient file for AI (always-readable stream)
- Default path: `tmp/ai/feedback_stream.jsonl`
- Format: compact JSONL keys (`t, ev, lv, acc, svc, sym, st, ok, rm, r, lat, fp, p`)
- Write mode: append-only, auto-trim by max bytes.
- Read mode: tail by token budget (used automatically by AI gate).

## Rebuild feedback stream from PostgreSQL
```bash
python manage.py rebuild_ai_feedback_stream --hours 168 --limit 5000
python manage.py rebuild_ai_feedback_stream --hours 72 --limit 2000 --write tmp/ai/feedback_stream.jsonl
```

## Preview command
Examples:
```bash
python manage.py preview_api_context --alias gpt-main --prompt-text "analiza ETH"
python manage.py preview_api_context --alias gpt-main --prompt-file tmp/prompt.txt --reserve-output 800 --write tmp/ctx.txt
```

Output includes:
- user tokens
- context tokens
- total prompt tokens
- budget available
- per-file include/skip reason

## Security notes
- `api_key` is encrypted at rest (`EncryptedCredentialField`).
- Context file paths are restricted to `BASE_DIR` to block path traversal.
- Keep API secrets out of git and `.md` docs.

## Optional env flags
- `AI_ENTRY_GATE_ENABLED=true|false` (legacy fallback; prefer DB row in `StrategyConfig` with `version=runtime_cfg_v1`)
- `AI_ENTRY_GATE_FAIL_OPEN=true|false`
- `AI_ENTRY_GATE_ONLY_ALLOCATOR=true|false`
- `AI_ENTRY_GATE_DEFAULT_PROVIDER=openai`
- `AI_ENTRY_GATE_MAX_OUTPUT_TOKENS=96`
- `AI_ENTRY_GATE_NOTIFY_ERRORS=true|false`
- `AI_EXIT_GATE_ENABLED=true|false` (legacy fallback; prefer DB row in `StrategyConfig` with `version=runtime_cfg_v1`)
- `AI_EXIT_GATE_ONLY_ALLOCATOR=true|false`
- `AI_EXIT_GATE_DEFAULT_PROVIDER=openai`
- `AI_EXIT_GATE_MAX_OUTPUT_TOKENS=96`
- `AI_EXIT_GATE_NOTIFY_ERRORS=true|false`
- `AI_EXIT_GATE_NEAR_TP_RATIO=0.88`
- `AI_EXIT_GATE_MIN_R=0.8`
- `AI_EXIT_GATE_MIN_RECHECK_SECONDS=45`
- `CONFIDENCE_LEVERAGE_BOOST_ENABLED=true|false`
- `CONFIDENCE_LEVERAGE_ONLY_ALLOCATOR=true|false`
- `CONFIDENCE_LEVERAGE_SCORE_THRESHOLD=0.90`
- `CONFIDENCE_LEVERAGE_ML_PROB_THRESHOLD=0.70`
- `CONFIDENCE_LEVERAGE_REQUIRE_BOTH=false`

## DB-first runtime overrides
- Table: `StrategyConfig`
- `version=runtime_cfg_v1`
- `name=<SETTING_KEY>`
- `enabled=true` means the override row is active
- `params_json={"value": ...}` stores the actual value

Examples:
- `name=AI_ENTRY_GATE_ENABLED`, `params_json={"value": false}`
- `name=BTC_LEAD_FILTER_ENABLED`, `params_json={"value": true}`
- `name=REGIME_BULL_SHORT_RETRACE_MIN_ALLOWED_MODULES`, `params_json={"value": 2}`
- `name=REGIME_BULL_SHORT_RETRACE_ALLOWED_MODULES`, `params_json={"value": ["meanrev","smc"]}`
- `CONFIDENCE_LEVERAGE_MULT=2.00`
- `CONFIDENCE_LEVERAGE_MAX=10.0`
- `AI_FEEDBACK_CONTEXT_MAX_TOKENS=700`
- `AI_FEEDBACK_JSONL_ENABLED=true|false`
- `AI_FEEDBACK_JSONL_PATH=tmp/ai/feedback_stream.jsonl`
- `AI_FEEDBACK_JSONL_MAX_BYTES=2000000`
- `AI_FEEDBACK_JSONL_TRIM_KEEP_RATIO=0.70`

Per service env sync support:
- `BINGX_AI_ENABLED=true|false`
- `BINGX_AI_PROVIDER_CONFIG_ALIAS=gpt-main`
- `KUCOIN_AI_ENABLED=true|false`
- `BINANCE_AI_ENABLED=true|false`
