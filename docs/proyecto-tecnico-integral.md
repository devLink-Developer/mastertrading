# MasterTrading - Documento Tecnico Integral

Fecha de corte del analisis: 2026-02-13 (UTC)  
Fuente: codigo del repo + estado runtime real (Docker/Postgres/Redis/Celery) + reportes en `reports/`.

## 1) Objetivo del documento

Este documento consolida en un solo lugar:

- aspectos tecnicos (arquitectura, servicios, modulos, datos, configuracion),
- aspectos funcionales (como opera el bot en produccion),
- historial de cambios (sin Git local, usando migraciones + trazas de backtests/reportes),
- historial de transacciones (datos reales en DB),
- metodos del modelo matematico (senales, sizing, TP/SL, riesgo, backtest),
- oportunidades de mejora para diversificar estrategias (incluyendo opcion de scalping).

## 2) Resumen ejecutivo

- Stack principal: Django + DRF + Celery + Redis + Postgres + ccxt + pandas/numpy.
- Exchange runtime activo al corte: `bingx` en `demo` (VST), tomado desde tabla `core_exchangecredential` (no desde `.env`).
- Politicas activas al corte:
  - `SESSION_POLICY_ENABLED=true`
  - `SIGNAL_DIRECTION_MODE=long_only`
  - bloqueos por simbolo: `BTCUSDT`, `ADAUSDT`, `XRPUSDT` en `disabled`
  - `SIGNAL_COOLDOWN_MINUTES=60`
- Estado salud:
  - `GET /health` OK
  - `GET /metrics` OK
  - `python manage.py check` OK
  - tests: 12/12 OK (`--keepdb --noinput`)
- Estado operativo real (DB):
  - senales: 1903
  - ordenes: 35 (25 filled, 10 rejected)
  - operaciones cerradas: 34
  - PnL acumulado de operaciones: `-2294.06727`
  - eventos de riesgo: 928 (predomina `daily_dd_limit`)
- Hallazgo operativo clave:
  - el PnL negativo esta concentrado en BTC (11 operaciones, `-2254.84`).
  - muchos eventos `daily_dd_limit`/`weekly_dd_limit` muestran valores extremos (incluyendo `dd ~ -1.0`), indicando posible inconsistencia de baseline de equity entre ciclos.

## 3) Arquitectura tecnica

### 3.1 Stack y dependencias

Dependencias base (`requirements.txt`):

- `Django`, `djangorestframework`, `celery[redis]`, `redis`, `psycopg2-binary`
- `ccxt`, `pandas`, `numpy`, `tenacity`
- `python-telegram-bot`, `httpx`
- `prometheus-client`, `gunicorn`, `whitenoise`

### 3.2 Servicios Docker Compose

Archivo: `docker-compose.yml`

- `web`: Django + gunicorn en puerto `8008`
- `worker`: Celery queue `trading,celery`
- `worker-data`: Celery queue `marketdata`
- `beat`: scheduler de tareas periodicas
- `market-data`: loop de ingesta (comando `marketdata_loop`, hoy stub)
- `telegram-bot`: bot interactivo Telegram
- `postgres`: DB principal (host `5434`)
- `redis`: broker/result backend + estado de riesgo/locks (host `6381`)

Nota tecnica: Compose muestra warning por `version` obsoleto.

### 3.3 Ruteo y scheduling Celery

Archivo: `config/settings.py`

Rutas:

- `signals.tasks.run_signal_engine` -> `trading`
- `execution.tasks.execute_orders` -> `trading`
- `marketdata.tasks.fetch_ohlcv_and_funding` -> `marketdata`
- `marketdata.tasks.fetch_instrument_data` -> `marketdata`

Beat:

- `run_signal_engine`: cada minuto
- `execute_orders`: cada minuto
- `fetch_ohlcv_and_funding`: intervalo configurable (`MARKETDATA_POLL_INTERVAL`)

### 3.4 Seleccion de exchange en runtime

Flujo (`adapters/credentials.py`, `core/exchange_runtime.py`):

1. busca exchange activo en tabla `ExchangeCredential` (`active=true`),
2. si no hay DB/tabla, cae a `.env` (`EXCHANGE` + variables por servicio),
3. construye contexto runtime (`service`, `sandbox`, `env`, `balance_assets`, `risk_namespace`),
4. recarga adapter automaticamente si cambia firma de credenciales.

Impacto: permite cambiar servicio/sandbox sin reiniciar procesos.

### 3.5 Flujo E2E de datos

1. `marketdata` trae OHLCV/funding y guarda en DB (`Candle`, `FundingRate`).
2. `signals` consume velas y funding, detecta setup SMC, persiste `Signal`.
3. `execution` consume ultima senal por instrumento, sincroniza posiciones exchange, gestiona salidas (TP/SL/trailing/flip), y abre nuevas entradas si pasa todos los gates.
4. `risk` registra `RiskEvent` y notifica Telegram.
5. `backtest` reutiliza logica de deteccion y reglas de riesgo en simulacion walk-forward.

## 4) Aspectos funcionales del bot

### 4.1 Universo e ingestion

- Instrumentos seed: BTC, ETH, SOL, XRP, DOGE, ADA, LINK (`seed_instruments`).
- Timeframes activos por defecto: `1m, 5m, 15m, 1h, 4h, 1d`.
- Ingestion usa `bulk_create(update_conflicts=True)` para upsert eficiente.
- Hay comando de backfill paginado (`marketdata backfill`) para periodos largos (ej: anos de 1h/4h).

### 4.2 Motor de senales (SMC + filtros)

Archivo principal: `signals/tasks.py`

Logica base:

- estructura HTF via swings + confirmacion EMA20/EMA50 (`_trend_from_swings`)
- en LTF exige confluencia:
  - sweep de liquidez + CHoCH (no BOS solo)
  - confirmacion de vela
  - HTF alineado (gate duro)
  - filtro funding no adverso
- scoring ponderado (0..1) con componentes:
  - HTF, structure break, sweep, confirmacion, FVG, OB, funding, CHoCH bonus
- confluencia EMA multiperiodo opcional (`EMA_CONFLUENCE_*`):
  - bonus o penalty de score
  - hard block opcional
- politica de sesiones:
  - determina sesion UTC (`overlap`, `london`, `ny`, `dead`, `asia`)
  - aplica `min_score` por sesion
  - puede bloquear emision de senales en `dead zone`
- direccion:
  - `both`, `long_only`, `short_only`, `disabled`
  - override por simbolo (`PER_INSTRUMENT_DIRECTION`)
- deduplicacion temporal (`SIGNAL_DEDUP_SECONDS`).

### 4.3 Motor de ejecucion

Archivo principal: `execution/tasks.py`

Entrada:

- descarta si trading deshabilitado o paper mode,
- sincroniza posiciones exchange -> tabla `Position`,
- chequea balance/equity y leverage efectivo,
- chequea DD diario/semanal + circuit breaker + data staleness,
- aplica TTL de senal, gate de score (global o por sesion), cooldown por simbolo,
- aplica exposure cap por instrumento,
- sizing por riesgo con ATR/SL + multiplicador de sesion + overrides por simbolo,
- valida margen y cap de leverage pre-trade,
- envia market order, recalcula TP/SL con fill real, coloca stop de proteccion.

Gestion de posicion abierta:

- reconciliacion de SL en exchange (si falta o quedo demasiado ajustado),
- breakeven stop (por multiple R),
- trailing stop basado en HWM,
- parcial de posicion por `PARTIAL_CLOSE_AT_R`,
- cierre por TP/SL/flip.

Registro:

- `Order`, `OperationReport`, `BalanceSnapshot`, `RiskEvent`,
- notificaciones Telegram de apertura/cierre/riesgo/error.

### 4.4 Riesgo

Controles implementados:

- DD diario y semanal (Redis key-value por `risk_namespace`),
- max leverage efectivo,
- max errores consecutivos por simbolo,
- stale data gate por instrumento,
- circuit breaker singleton editable en Admin:
  - daily DD,
  - total DD desde peak,
  - max consecutive losses,
  - cooldown post-trigger.

### 4.5 API y observabilidad

Endpoints:

- `/health`, `/metrics`
- `/instruments` (+ enable/disable)
- `/signals`, `/positions`, `/orders`, `/risk`
- `/config/strategy` (+ toggle)

Admin:

- control de credenciales por exchange,
- acciones para sincronizar/cerrar posiciones,
- configuracion de `CircuitBreakerConfig` y `RegimeFilterConfig`.

Telegram:

- notificaciones push (`risk/notifications.py`),
- bot interactivo con menu (`risk/telegram_bot.py`).

## 5) Modelo matematico y metodos cuantitativos

### 5.1 Indicadores y filtros

ATR (% sobre precio):

- `TR = max(high-low, |high-prev_close|, |low-prev_close|)`
- `ATR = mean(TR ultimos N)`
- `atr_pct = ATR / close`

ADX (Wilder):

- calcula `+DM`, `-DM`, `TR`, suavizado Wilder,
- deriva `+DI`, `-DI`, `DX`, luego ADX.

Funding filter:

- bloquea long si funding actual > umbral extremo,
- bloquea short si funding actual < -umbral extremo.

### 5.2 Score de senal

Score base:

- suma ponderada de condiciones booleanas:
  - htf_trend_aligned 0.20
  - structure_break 0.20
  - liquidity_sweep 0.20
  - confirmation_candle 0.05
  - fvg_aligned 0.10
  - order_block 0.10
  - funding_ok 0.10
  - choch_bonus 0.05

Score final:

- aplica bonus/penalty EMA si confluencia habilitada,
- gate final contra `min_score` (global o por sesion).

### 5.3 TP/SL y sizing

TP/SL:

- `tp_pct = max(TAKE_PROFIT_PCT, atr_pct * ATR_MULT_TP)`
- `sl_pct = max(STOP_LOSS_PCT, atr_pct * ATR_MULT_SL, MIN_SL_PCT)`

Precios:

- long: `tp = entry*(1+tp_pct)`, `sl = entry*(1-sl_pct)`
- short: `tp = entry*(1-tp_pct)`, `sl = entry*(1+sl_pct)`

Sizing por riesgo:

- `risk_amount = equity * risk_pct`
- `qty = risk_amount / (stop_distance_abs * contract_size)`
- luego aplica `session_risk_multiplier`, limites de lote, caps de margen/exposicion/leverage.

### 5.4 PnL y metrica de operacion

Operacion cerrada:

- `pnl_abs = (exit - entry) * qty * contract_size * direction_sign`
- `pnl_pct = ((exit-entry)/entry) * direction_sign`
- `notional = |qty * entry * contract_size|`
- `margin_used = notional / leverage`

### 5.5 Trailing y breakeven

Trailing:

- guarda HWM (`max favorable excursion`) en Redis,
- activa trailing al superar `TRAILING_STOP_ACTIVATION_R`,
- lock dinamico de ganancia: `trail_sl = entry * (1 +/- max_fav_pct * lock_in_pct)`.

Breakeven:

- si `max_fav / sl_pct >= BREAKEVEN_STOP_AT_R`, mueve SL a entry (+offset opcional).

### 5.6 Modelo de fees y metricas en backtest

Backtest (`backtest/engine.py`):

- fee taker default: 4 bps por lado,
- aplica mismas reglas de score/sesion/direccion/riesgo,
- metricas:
  - PnL total, PF, Sharpe aprox, DD max abs/%,
  - win rate, expectancy,
  - contadores de bloqueos por sesion.

## 6) Modelo de datos y estado de base

### 6.1 Entidades principales

- `core`: `Instrument`, `ExchangeCredential`, `AuditLog`
- `marketdata`: `Candle`, `FundingRate`, `OrderBookSnapshot`
- `signals`: `StrategyConfig`, `Signal`
- `execution`: `Order`, `TradeFill`, `Position`, `OperationReport`, `BalanceSnapshot`
- `risk`: `RiskEvent`, `CircuitBreakerConfig`, `RegimeFilterConfig`
- `backtest`: `BacktestRun`, `BacktestTrade`

### 6.2 Indices y optimizacion de query

Indices relevantes:

- vela unica por `(instrument, timeframe, ts)`
- funding unico por `(instrument, ts)`
- senales por `(instrument, ts)`
- ordenes por `(instrument, status, opened_at)`
- positions por `(instrument, is_open)`
- operation reports por `(instrument, closed_at)`

Mejoras de eficiencia ya aplicadas en loop de ejecucion:

- preload de instrumentos habilitados en una sola query,
- agregacion de `latest_1m_ts` y `latest_signal_id` via `Subquery`,
- carga en bloque de senales (`in_bulk`) para evitar N+1.

### 6.3 Snapshot de volumen de datos (Postgres, 2026-02-13 UTC)

Cardinalidad aproximada:

- `marketdata_candle`: 393,528
- `marketdata_fundingrate`: 47,882
- `signals_signal`: 1,903
- `execution_order`: 35
- `execution_operationreport`: 34
- `execution_balancesnapshot`: 4,830
- `risk_riskevent`: 928
- `backtest_backtestrun`: 65
- `backtest_backtesttrade`: 22,785

Tamano de tablas (aprox):

- `marketdata_candle`: 78 MB
- `marketdata_fundingrate`: 5.5 MB
- `backtest_backtesttrade`: 4.2 MB

## 7) Historial de cambios (sin repositorio Git local)

## 7.1 Limite de trazabilidad

No hay metadata `.git` en este workspace, por lo que el historial se reconstruye por:

- migraciones Django,
- nombres/resultados de `BacktestRun`,
- reportes en `reports/`,
- estado runtime y codigo actual.

### 7.2 Evolucion de esquema (migraciones)

Linea base:

- `core 0001`: `Instrument`, `AuditLog`
- `signals 0001`: `StrategyConfig`, `Signal`
- `marketdata 0001`: `Candle`, `FundingRate`, `OrderBookSnapshot`
- `execution 0001`: `Order`, `TradeFill`, `Position`
- `risk 0001`: `RiskEvent`
- `backtest 0001`: `BacktestRun`, `BacktestTrade`

Evolucion posterior destacada:

- `core 0003`: agrega `ExchangeCredential`
- `core 0002`: indice `core_instr_enabled_symbol_idx`
- `execution 0002..0008`:
  - agrega `OperationReport`, `BalanceSnapshot`,
  - agrega campos de notional/margin/fee/leverage/opened/closed,
  - agrega indices por instrumento/fecha/estado.
- `risk 0002`: agrega `CircuitBreakerConfig` y `RegimeFilterConfig`
- `marketdata 0002`: ajustes de fields/indexing en relaciones.

### 7.3 Evolucion funcional reciente (observada en codigo y runs)

Bloques funcionales ya integrados:

- politica de sesiones (`signals/sessions.py`) en senal, ejecucion y backtest,
- confluencia EMA configurable (bonus/penalty/hard-block),
- direccion global y por simbolo (`signals/direction_policy.py`),
- runtime exchange por DB (sin reinicio),
- soporte BingX demo/live en adapter y contexto balance multi-asset,
- trailing + breakeven + parcial + SL exchange reconciliation,
- comandos operativos para setear exchange activo y sandbox.

Versionado empirico (por nombres de runs):

- runs tempranos (`v6`, `v7`, `v8`...) con DD muy alto y PF bajo,
- iteraciones de mejora hacia `v18`..`v23` con SL/EMA/session tuning,
- runs recientes con PF > 1.16 y PnL positivo en ventana evaluada.

## 8) Historial transaccional (datos reales)

Snapshot operativo consultado en DB (2026-02-13 UTC):

- total operaciones cerradas: 34
- wins/losses: 14 / 20 (WR 41.18%)
- PnL neto total: `-2294.06727`
- ordenes:
  - 25 filled
  - 10 rejected
- posiciones abiertas al corte: 0

### 8.1 Breakdown por instrumento (30 dias)

| Simbolo | Ops | Wins | WR | PnL |
|---|---:|---:|---:|---:|
| BTCUSDT | 11 | 2 | 18.18% | -2254.84047 |
| ETHUSDT | 10 | 6 | 60.00% | -35.59240 |
| SOLUSDT | 5 | 2 | 40.00% | -3.08420 |
| ADAUSDT | 2 | 2 | 100.00% | +0.54420 |
| LINKUSDT | 2 | 1 | 50.00% | +0.07420 |
| XRPUSDT | 2 | 1 | 50.00% | -0.62460 |
| DOGEUSDT | 2 | 0 | 0.00% | -0.54400 |

### 8.2 Breakdown por razon de cierre

| Reason | Cantidad | PnL |
|---|---:|---:|
| exchange_close | 19 | +11.53703 |
| sl | 9 | -2359.79570 |
| tp | 3 | +28.17640 |
| signal_flip | 3 | +26.01500 |

Lectura: el daño principal viene de cierres por `sl`, especialmente en BTC.

### 8.3 Jornadas de mayor impacto

| Fecha UTC | Operaciones | Wins | PnL diario |
|---|---:|---:|---:|
| 2026-02-09 | 11 | 3 | -1812.80300 |
| 2026-02-10 | 2 | 0 | -492.88500 |
| 2026-02-11 | 11 | 6 | -0.22837 |
| 2026-02-12 | 8 | 4 | +0.34910 |
| 2026-02-13 | 2 | 1 | +11.50000 |

### 8.4 Mayores perdidas individuales

Top pérdidas (todas BTC long por `sl`):

- 2026-02-09 08:32 UTC: `-581.1`
- 2026-02-10 21:29 UTC: `-489.0`
- 2026-02-09 10:17 UTC: `-448.2`

## 9) Historial de backtests y validacion

### 9.1 Reportes en `reports/`

Archivos clave:

- `v15_baseline.json`
- `v17_dead_zone_only.json`
- `v17_dead_zone_score.json`
- `v17_full_session.json`
- `session_matrix_summary.md/json`
- `wf_smoke.json`
- `ema_tuning_quick_20260101_20260122.json`

Matriz de sesiones (2026-01-01 -> 2026-01-08, segun `session_matrix_summary.md`):

| Run | PnL | PF | Trades | Max DD% | WinRate% |
|---|---:|---:|---:|---:|---:|
| v15_baseline | +43.5534 | 1.306 | 34 | 3.79 | 47.06 |
| v17_dead_zone_only | +43.0963 | 1.315 | 32 | 3.86 | 43.75 |
| v17_dead_zone_score | +43.0963 | 1.315 | 32 | 3.86 | 43.75 |
| v17_full_session | +43.0963 | 1.315 | 32 | 3.86 | 43.75 |

### 9.2 Backtest runs persistidos en DB

- total runs: 65
- total trades simulados: 22,785

Mejor PnL historico observado:

- run `#41` `combo_session_strict_scores`: PnL `+443.5157`, PF `1.13`, DD `31.05%`, 530 trades.

Mejor PF (considerando >=100 trades):

- run `#37` `session_on_strict_no_overlap`: PF `1.241`, PnL `+250.6343`, DD `12.23%`, 189 trades.

Run reciente relevante:

- run `#65` `v23_sl1.5_block_ny_overlap`: PnL `+174.4483`, PF `1.189`, DD `13.76%`, 211 trades.

## 10) Configuracion efectiva al corte

Valores runtime relevantes (web container, 2026-02-13 UTC):

- `MODE=live`
- `TRADING_ENABLED=True`
- `EXCHANGE` en settings: `kucoin` (fallback)
- exchange runtime efectivo: `bingx demo` (desde DB)
- `MIN_SIGNAL_SCORE=0.80`
- `EXECUTION_MIN_SIGNAL_SCORE=0.82`
- `TAKE_PROFIT_PCT=0.015`
- `STOP_LOSS_PCT=0.007`
- `ATR_MULT_TP=3.0`
- `ATR_MULT_SL=1.5`
- `MIN_SL_PCT=0.008`
- `SIGNAL_COOLDOWN_MINUTES=60`
- `SESSION_POLICY_ENABLED=True`
- `SESSION_DEAD_ZONE_BLOCK=True`
- `EMA_CONFLUENCE_ENABLED=False`
- `SIGNAL_DIRECTION_MODE=long_only`
- `PER_INSTRUMENT_DIRECTION={'BTCUSDT':'disabled','ADAUSDT':'disabled','XRPUSDT':'disabled'}`

Estado de credenciales en tabla:

- `bingx`: active=true, sandbox=true, key_set=true
- `kucoin`: active=false, key_set=true
- `binance`: active=false, key_set=false

## 11) Riesgos tecnicos y oportunidades de mejora

### 11.1 Riesgos detectados

1. Eventos de riesgo excesivos:
- `daily_dd_limit` aparece 793 veces de 928 eventos.
- se observan valores extremos (`dd ~ -1.0`) y repeticion por ciclo.
- impacto: ruido operativo, alert fatigue, y posible bloqueo de entradas valido/no valido.

2. Inconsistencia de severidad en `RiskEvent`:
- modelo define `info/warn/critical`,
- el codigo emite tambien `high` y `medium`.
- impacto: taxonomia inconsistente para filtros/reportes.

3. Cobertura de tests limitada:
- solo 12 tests, mayormente de modelo/helper.
- poca cobertura de escenarios integrados de ejecucion/riesgo/exchange adapters.

4. Historial sin Git en workspace:
- limita auditoria formal de cambios.
- hoy se depende de migraciones + nomenclatura de backtests.

### 11.2 Mejoras de eficiencia (priorizadas)

P0 (alto impacto, bajo/medio esfuerzo):

- normalizar y throttlear `RiskEvent` (agregado por ventana y dedup por key tecnica),
- corregir baseline DD para evitar falsos `-100%`,
- consolidar severidades a enum canonico (`info/warn/critical`).

P1 (impacto medio):

- reducir carga de polling en condiciones de mercado quieto (adaptive poll),
- mover logs de muy alta frecuencia a nivel debug y/o muestreo,
- evaluar concurrencia dinamica Celery por host (hoy fija en `--concurrency=4` en ambos workers).

P2 (impacto estrategico):

- caching de datos de mercado por ciclo en memoria worker para evitar fetch repetido,
- monitoreo de latencia por tarea (`run_signal_engine`, `execute_orders`, `fetch_instrument_data`) con percentiles.

## 12) Diversificacion de metodos (para mayor rentabilidad controlada)

### 12.1 Estado actual del edge

El sistema actual esta optimizado para:

- SMC con confirmaciones fuertes (sweep+CHoCH+HTF+funding),
- marcos `5m` entrada y `1h/4h` contexto,
- gestion de riesgo estricta.

No esta especializado en scalping puro.

### 12.2 Si se quiere agregar scalping

Requisitos minimos antes de implementar:

1. Dataset intradia mas profundo y homogeneo:
- cobertura consistente de 1m (hoy varios simbolos solo desde 2026-02-09).

2. Modelo de costo realista:
- fees + slippage + spread dinamico por hora/sesion.

3. Reglas de salida mas rapidas:
- take-profit parcial/temprano, time-stop, y trailing mas agresivo para micro-trend.

4. Framework de validacion:
- walk-forward por regimen,
- OOS separado,
- metricas por sesion y por simbolo,
- control de DD por estrategia.

### 12.3 Estrategias candidatas de diversificacion

A. `SMC Swing` (actual, mantener): captura movimientos mas amplios con confirmacion estructural.

B. `Scalp Pullback 1m/5m` (nuevo):
- entradas de continuidad en micro tendencia,
- objetivo corto (0.2%-0.6%), stop ajustado, salida temporal.

C. `Mean Reversion Session-bound` (nuevo):
- operar extremos en ventanas de baja expansion,
- bloqueado en eventos macro/funding extremo.

Recomendacion: ejecutar en modo portfolio por estrategia con limites de riesgo independientes y sin mezclar resultados.

## 13) Plan sugerido de siguiente iteracion

1. Higiene operativa inmediata:
- corregir DD baseline y spam de eventos.

2. Medicion estructurada:
- reporte diario automatico con:
  - conversion signal->order,
  - WR/PnL por simbolo/sesion/reason,
  - DD real por namespace.

3. Diversificacion controlada:
- implementar primer prototipo `Scalp Pullback` solo en `ETHUSDT` demo,
- backtest + paper paralelo 2-4 semanas,
- pasar a live solo si mejora PF sin aumentar DD materialmente.

---

## Anexo A - Comandos utiles de operacion

- activar exchange:
  - `python manage.py set_active_exchange --service bingx`
- sandbox on/off:
  - `python manage.py set_exchange_sandbox --service bingx --on`
  - `python manage.py set_exchange_sandbox --service bingx --off`
- sync credenciales env -> DB:
  - `python manage.py sync_exchange_credentials`
- backtest:
  - `python manage.py backtest --start YYYY-MM-DD --end YYYY-MM-DD --symbols BTCUSDT,ETHUSDT`
- walk-forward optimizer:
  - `python manage.py optimize_walkforward --start ... --end ...`
- analisis de perdedoras con MFE:
  - `python scripts/analyze_losers_mfe.py --days 30 --timeframe 1m`

## Anexo B - Scripts utilitarios relevantes

- `scripts/startup_check.py`: chequeo integral post arranque
- `scripts/analyze_losers_mfe.py`: detecta perdedoras que estuvieron en verde
- `scripts/cleanup_vscode_codex.ps1`: limpieza de caches/cookies VS Code + Codex

