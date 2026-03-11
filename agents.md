# MasterTrading — Aprendizajes del Agente

> Archivo acumulativo de conocimiento adquirido sobre el proyecto.
> Se actualiza en cada sesión de trabajo para mantener continuidad entre conversaciones.
> Última actualización: 2026-02-21 (sesión 4)

---

## ÍNDICE

1. [Arquitectura del Proyecto](#1-arquitectura-del-proyecto)
2. [Pipeline de Señales](#2-pipeline-de-señales)
3. [Modelo Matemático](#3-modelo-matemático)
4. [Gestión de Riesgo](#4-gestión-de-riesgo)
5. [Rendimiento Observado](#5-rendimiento-observado)
6. [Decisiones Tácticas y Lecciones](#6-decisiones-tácticas-y-lecciones)
7. [Directrices del Supervisor](#7-directrices-del-supervisor)
8. [Estado Actual de Producción](#8-estado-actual-de-producción)
9. [Deuda Técnica y Brechas](#9-deuda-técnica-y-brechas)
10. [Plan de Acción Validado](#10-plan-de-acción-validado)
11. [Runbook Proyecto Camping](#11-runbook-proyecto-camping)

---

## 1. ARQUITECTURA DEL PROYECTO

### Stack
- Django 5.0 + DRF + Celery + Redis 7 + PostgreSQL 15
- Docker Compose: web, worker (trading), worker-data (marketdata/ml), beat, market-data, telegram-bot
- Exchange: KuCoin Futures vía CCXT ≥4.1, leverage=5, cross margin, USDT-margined
- ML: scikit-learn (Logistic Regression), actualmente DESHABILITADO

### Apps Django
| App | Responsabilidad |
|---|---|
| core | Instrument, ExchangeCredential, TimeStampedModel |
| marketdata | Candle, FundingRate, OrderBookSnapshot |
| signals | 4 módulos (trend, meanrev, carry, smc) + allocator + sessions |
| execution | Órdenes, posiciones, trailing, breakeven, OperationReport |
| risk | RiskEvent, CircuitBreaker, drawdown, notificaciones |
| backtest | Motor walk-forward, SimPosition |
| api | ViewSets DRF |
| adapters | Wrappers exchange (KuCoin, BingX) con retry |

### Instrumentos activos
BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, LINKUSDT

### Timeframes
- LTF = 5 minutos (señales) / 1 minuto (backtest alta fidelidad)
- HTF = 1 hora (contexto)
- HTF secundario = 4h (SMC dual-TF)

### Flujo principal
```
Market Data → Candle/FundingRate → DB
→ run_signal_engine (cada 60s) → 4 módulos → allocator → Signal → DB
→ execute_orders (cada 30-60s) → validaciones → sizing → orden + SL → trailing/breakeven
→ _sync_positions → cierre detectado → OperationReport → Telegram
```

---

## 2. PIPELINE DE SEÑALES

### Módulo Trend (signals/modules/trend.py)
- Filtro: ADX(14) LTF ≥ 20
- Dirección: EMA20 vs EMA50 en HTF + precio relativo
- Guards: impulse bar (body > 2.2× avg), bounce (> 0.50% desde extremo)
- Score: 0.50 + min(0.35, ema_gap×35) + min(0.15, (ADX-20)/100)

### Módulo Mean Reversion (signals/modules/meanrev.py)
- Filtro: ADX(14) HTF ≤ 18 (solo mercados sin tendencia)
- Entrada: Z-score de desviación (close - EMA20) ≥ 1.2
- Guards: impulse bar contra-dirección, bounce
- Score: |z| / max(2.5, z_entry + 0.8)

### Módulo Carry (signals/modules/carry.py)
- Filtro: funding rate extremo (threshold = 0.001 × 1.8)
- Filtro vol: ATR% < 2%
- Score: |funding|/threshold - vol_penalty + mr_hint

### Módulo SMC (signals/tasks.py — Smart Money Concepts)
- 14 gates secuenciales: datos, sesión, HTF trend, dual-TF, CHoCH, sweep, confluencia, EMA50, ADX, confirmación, impulse, funding, EMA confluence
- Confluencia obligatoria: sweep + CHoCH en misma dirección
- Score ponderado: htf_trend(0.20) + structure_break(0.20) + sweep(0.20) + candle(0.05) + fvg(0.10) + ob(0.10) + funding(0.10) + choch_bonus(0.05) + adx_strong(0.05)
- Ajustes: EMA confluence ±0.06/0.10, short penalty configurable

### Allocator (signals/allocator.py)
- Pesos: trend=0.30, meanrev=0.20, carry=0.15, smc=0.35
- net_score = Σ(peso × mult_confluencia × confidence × signo_dirección)
- Threshold: 0.20
- Mínimo módulos activos: 2
- SMC confluence boost: 1.25× con confluencia, 0.85× sin ella

---

## 3. MODELO MATEMÁTICO

### Position Sizing (fixed fractional)
```
qty = (risk_pct × equity) / (SL_distance × entry_price × contract_size)
```
Default: risk_pct = 0.003 (0.30%), BTC override = 0.0015 (0.15%)

### Volatility-Adjusted Risk
- ATR ≤ 0.8%: full risk
- 0.8% < ATR < 1.5%: interpolación lineal (1.0× → 0.6×)
- ATR ≥ 1.5%: 0.6× risk

### TP/SL dinámicos (ATR-based)
- SL = max(SL_base, ATR% × 1.5, MIN_SL_PCT=1.2%)
- TP = max(TP_base, ATR% × 1.8) × TP_mult

### Trailing Stop (3 capas)
1. Breakeven: SL → entry a 1.0R
2. Partial close: 50% a 1.0R (una vez)
3. Trailing: activa a 2.5R, lock 50% del HWM

### Daily Trade Throttling (ADX-adaptive)
- ADX < 20: máx 3 trades
- 20 ≤ ADX ≤ 25: máx 6
- ADX > 25: máx 10

### Kelly Criterion (observado)
- WR=48.5%, payoff=1.057 → f* ≈ -0.002 (edge marginalmente negativo)
- Confirmado por supervisor: Kelly no aplicable directo; usar fractional Kelly capado

---

## 4. GESTIÓN DE RIESGO

### Capas de protección
| Nivel | Mecanismo |
|---|---|
| Position | SL stop-market en exchange, breakeven, partial close, trailing |
| Position | Stale cleanup (>12h near breakeven) |
| Account | Risk per trade 0.30%, max eff leverage 3.0×, min equity 5 USDT |
| Account | Daily drawdown limit, weekly DD limit, circuit breaker |
| Market | ADX regime gate (per-instrument), daily trade throttle, session policy |
| Market | Signal flip min age (5 min), macro high-impact filter |

### Sesiones (UTC)
| Sesión | Horario | Score min | Risk mult |
|---|---|---|---|
| overlap | 12-14 | 0.55 | 1.00 |
| london | 06-14 | 0.56 | 1.00 |
| ny | 14-20 | 0.58 | 0.90 |
| dead | 20-23 | 0.80 | 0.00 (bloqueada) |
| asia | 23-06 | 0.62 | 0.70 |

---

## 5. RENDIMIENTO OBSERVADO (14 días, ~Feb 6-20 2026)

| Métrica | Valor |
|---|---|
| Total trades | 274 |
| Win rate | 48.5% |
| PnL total | +1.51% |
| Profit factor | 0.73 |
| Avg win / Avg loss | +0.333% / -0.315% |
| Payoff ratio | 1.057 |
| Avg duration | 44.6 min |
| Max consecutive losses | 7 |

### Por dirección
- Longs: 133 trades, 50.4% WR, -0.58% PnL
- Shorts: 141 trades, 46.8% WR, +2.09% PnL

### Por razón de cierre
- signal_flip: 170 trades, +4.31% (principal mecanismo)
- exchange_close: 73 trades, -17.90% (**mayor fuente de pérdidas**)
- tp: 19 trades, +20.08%
- sl: 9 trades, -5.78%

### Por instrumento
- Mejor: ETHUSDT (+8.55%)
- Peor: BTCUSDT (-6.90%)

### Tests
- 69 total: 68 pass, 1 fail (risk.tests_report_controls — pre-existente)

### Datos en DB
- 496K candles, 91K signals (14d), 1,743 risk events (0 critical)

---

## 6. DECISIONES TÁCTICAS Y LECCIONES

### 2026-02-20: Crisis 7/8 trades perdedores

**Problema:** Mercado choppy (BTC rango -0.28% en 14h). ADX medido en 5m pasaba el filtro (25-35) mientras ADX 1h estaba en 13-20.

**Cambios implementados (todos GO del supervisor):**
1. ALLOCATOR_MIN_MODULES_ACTIVE=2 — evita entradas de 1 solo módulo
2. MAX_DAILY_TRADES_LOW_ADX=3 — throttle en choppy
3. MARKET_REGIME_ADX_MIN=17 — per-instrument ADX gate (originalmente BTC-only global, cambiado a per-instrument en sesión 4)
4. SIGNAL_FLIP_MIN_AGE_ENABLED=true, 5 min — evita flips rápidos (mediana observada: 1 min)
5. PER_INSTRUMENT_RISK={"BTCUSDT": 0.0015} — BTC mitad de riesgo

### 2026-02-20 (sesión 2): Análisis exchange_close + backtest allocator

**exchange_close clasificado (73 trades, -$4,428):**
- 42 exchange_stop (SL en exchange): PnL -$5,153, median -0.67% → cluster en -0.6% a -0.8% = exactamente SL_PCT
- 14 exchange_tp_limit: PnL +$641, median +0.51%
- 17 near_breakeven: PnL +$84, median +0.06%
- **Causa raíz**: race condition en _sync_positions — el SL de KuCoin se llena, sync detecta posición desaparecida antes de que execute_orders procese el fill → se loguea como exchange_close en vez de sl
- **Fix aplicado**: _classify_exchange_close ahora tiene fallback por PnL-pattern (layer 4). close_sub_reason retroactivamente poblado en 73 OperationReports.
- **Impacto real**: 42+9=51 SL reales (no 9), total SL losses = ~$7,513

**Backtest con allocator integrado (4 módulos):**
- Motor reescrito: trend+meanrev+carry+smc → signal_cache → allocator → signal_flip
- Resultado: 89 trades, WR 46%, PnL -3.52%, PF 0.66, 1 signal_flip
- **Limitación estructural**: backtest en 5m bars NO puede replicar cadencia 60s de live
  - Live: módulos corren cada 60s independientemente, señales persisten en DB 130s
  - Backtest: todos los módulos corren simultáneamente cada 300s
  - Meanrev: bounce_guard (0.50%) bloquea 92% de señales a resolución 5m
  - En live, meanrev emite 3,716 señales (60s cadence catches brief windows below threshold)

**Módulos en vivo (14d signal counts):**
- trend: 13,605 | carry: 13,279 | meanrev: 3,716 | smc: 2,054
- allocator: 8,378 long + 5,844 short + 46,603 flat

**perf_dashboard (live 274 trades):**
- Mejor módulo contributivo: meanrev (+$1.88, WR 50%)
- Peor módulo contributivo: trend (-$12.62, WR 41%)
- Trend es el módulo dominante en más trades (133) pero con peor performance

**Monte Carlo (10K paths × 274 trades):**
- Risk of ruin (20% DD): 0.33%
- Median max DD: -7.59%, Mean max DD: -8.28% ±3.40%
- Median return: +1.31%, Mean return: +1.63% ±8.19%
- p5 max DD: -14.86%, p1 max DD: -18.16%

### Lección: exchange_close es la mayor pérdida
- 73 trades, -17.90% PnL
- Clasificado: 42 son SL de exchange (race condition con sync), 14 son TP, 17 son breakeven
- close_sub_reason ahora se popula automáticamente con fallback PnL-based

### Lección: signal_flip age importa
- Mediana de edad entre flips: 1 minuto → demasiado rápido
- Gate de 5 minutos activado para estabilizar

### 2026-02-21 (sesión 4): Edge improvements basados en datos 7d

**Análisis 7d (88 trades, -$1,058):**
- exchange_close: 33 trades, -$2,663 (24 exchange_stop = SL hits, -$3,040)
- signal_flip: 42 trades, +$243
- tp: 13 trades, +$1,362
- Longs: 48 trades, WR=40%, -$1,386 | Shorts: 40 trades, WR=57%, +$328
- Top SL losers: SOL longs (-$725), LINK longs (-$537), ADA longs (-$496), BTC longs (-$300)

**Cambios implementados:**
1. MODULE_TREND_HTF_ADX_MIN=18 — trend module requiere ADX(14) en 1h >= 18 (evita trends falsos en ranging HTF)
2. ALLOCATOR_LONG_SCORE_PENALTY=0.85 — net_score × 0.85 para longs (penaliza dirección débil)
3. PER_INSTRUMENT_RISK expandido: SOL/LINK/ADA = 0.002 (reducido de 0.003)
4. HMM regime activo + GARCH blended vol activo (ambos activados esta sesión)
5. MARKET_REGIME_ADX_MIN gate cambiado de BTC-only global a per-instrument — cada instrumento se evalúa con su propio ADX 1h

---

## 7. DIRECTRICES DEL SUPERVISOR

### Correcciones importantes
1. **Carry SÍ está activo** — 12K señales en 14 días (no son pocas)
2. **SMC shorts NO están eliminados** — 1,362 emitidos
3. **BTC long_only sería PEOR** — datos del supervisor lo confirman
4. **ADX gate 18 → 17** — basado en análisis de datos del supervisor
5. **Kelly/Optimal F**: concepto válido pero peligroso aplicar directo. Fractional Kelly capado.
6. **BTC es activo base** — no apagar; mejor ajustar edge rolling

### Criterio de éxito para cambios
Mejora OOS en 3 ejes simultáneos:
1. Expectancy > baseline
2. Max drawdown ≤ baseline
3. Trades/day dentro de banda objetivo (sin dejar bot parado)

### Formato de trabajo
- Cada cambio: objetivo, archivos tocados, riesgo, plan de prueba
- Supervisor responde: riesgos críticos, ajustes, pruebas mínimas, go/no-go
- Antes de cambiar lógica de señales/ejecución: definir test/backtest mínimo

---

## 8. ESTADO ACTUAL DE PRODUCCIÓN

### Configuración clave (.env)
```
MODE=live
TRADING_ENABLED=true
KUCOIN_LEVERAGE=5
RISK_PER_TRADE_PCT=0.003
MAX_EFF_LEVERAGE=3.0
STOP_LOSS_PCT=0.007
TAKE_PROFIT_PCT=0.008
ATR_MULT_SL=1.5
ATR_MULT_TP=1.8
MIN_SL_PCT=0.012
ALLOCATOR_MIN_MODULES_ACTIVE=2
MAX_DAILY_TRADES_LOW_ADX=3
MARKET_REGIME_ADX_MIN=17.0
SIGNAL_FLIP_MIN_AGE_ENABLED=true
SIGNAL_FLIP_MIN_AGE_MINUTES=5
PER_INSTRUMENT_RISK={"BTCUSDT": 0.0015}
ALLOCATOR_MODULE_WEIGHTS={"trend":0.30,"meanrev":0.20,"carry":0.15,"smc":0.35}
ALLOCATOR_DYNAMIC_WEIGHTS_ENABLED=true
ALLOCATOR_DYNAMIC_WINDOW_DAYS=7
SESSION_SCORE_MIN={"asia":0.62,"london":0.56,"ny":0.58,"overlap":0.55,"dead":0.80}
SESSION_RISK_MULTIPLIER={"asia":0.70,"london":1.0,"ny":0.90,"overlap":1.0,"dead":0.0}
```

### Archivos clave (mapa rápido)
| Archivo | Líneas | Función |
|---|---|---|
| execution/tasks.py | ~3465 | Core: execute_orders, sizing, trailing, sync |
| signals/tasks.py | ~1106 | SMC detection (_detect_signal), signal engine |
| signals/allocator.py | ~300 | Weighted module combination + Bayesian dynamic weights |
| signals/sessions.py | 70 | Session definitions + score/risk multipliers |
| signals/modules/trend.py | 118 | Trend following module |
| signals/modules/meanrev.py | 120 | Mean reversion module |
| signals/modules/carry.py | 65 | Funding rate carry module |
| signals/modules/common.py | 331 | ADX, ATR, impulse, bounce helpers |
| signals/tests_dynamic_weights.py | ~200 | 13 tests for Bayesian weights |
| signals/management/commands/show_dynamic_weights.py | ~110 | Dynamic weight dashboard |
| signals/regime.py | ~270 | HMM 2-state regime detection (GaussianHMM) |
| signals/garch.py | ~265 | GARCH(1,1) volatility forecasting (arch library) |
| signals/tests_garch.py | ~170 | 15 tests for GARCH module |
| signals/management/commands/fit_regime.py | ~70 | Fit HMM + display regime state |
| signals/tests_regime.py | ~190 | 17 tests for HMM regime |
| config/settings.py | ~750 | All env vars parsed |
| adapters/kucoin.py | 165+ | CCXT wrapper with retry |
| execution/ml_entry_filter.py | 306 | ML filter (38 features, disabled) |
| backtest/engine.py | ~895 | Walk-forward backtest engine (4 módulos + allocator, soporta 1m/5m) |
| risk/management/commands/monte_carlo.py | ~140 | Monte Carlo + risk of ruin |
| risk/management/commands/perf_dashboard.py | ~200 | Performance dashboard por módulo/símbolo |
| risk/models.py | 153 | RiskEvent, CircuitBreakerConfig |
| execution/models.py | 152 | Order, Position, OperationReport |

---

## 9. DEUDA TÉCNICA Y BRECHAS

### No implementado
1. ~~Monte Carlo ni cálculo formal de riesgo de ruina~~ → IMPLEMENTADO (monte_carlo command)
2. Purged CV/embargo ni White's Reality Check
3. ~~HMM/Markov/GARCH productivos (solo filtros ADX/ATR)~~ → HMM IMPLEMENTADO (signals/regime.py)
4. Microestructura: OrderBookSnapshot existe pero sin uso operativo (OI/CVD/order imbalance)
5. ~~Backtest y live no son 100% isomorfos~~ → 1m bars implementado, signal_flip gap reducido 35×

### Limitaciones conocidas del backtest
- Signal_flip: 35 en backtest 1m vs 170 en live (gap reducido; 1 en 5m vs 170 era el anterior)
- En 5m: Meanrev bounce_guard bloquea 92% (en 1m funciona mejor, similar a live 60s)
- Carry: fire rate OK (~20-40%) pero overlap con otros módulos es bajo dentro del TTL
- 1m backtest tarda ~20 min para 10 días × 7 instrumentos (14K bars)
- Partial close implementado: 26 partials en backtest, PnL mejoró -3.14% → -2.72%

### Backtest 1m vs 5m vs Live (Feb 11-21)
| Métrica | 1m+partial | 1m base | 5m bars | LIVE |
|---------|------------|---------|---------|------|
| Trades | 289 | 263 | 90 | 274 |
| WR | 61.9% | 58.2% | 45.6% | 48.5% |
| PnL | -2.72% | -3.14% | -3.90% | +1.51% |
| PF | 0.886 | 0.87 | 0.64 | 1.04 |
| Signal flips | 35 | 35 | 1 | 170 |
| TP exits | 136 | 136 | 40 | 19 |
| SL exits | 87 | 87 | 47 | 9 |
| Partial close | 26 | 0 | 0 | ~many |

### Test failure conocido
- risk.tests_report_controls — 1 test falla (pre-existente)

---

## 10. PLAN DE ACCIÓN VALIDADO (por supervisor)

### Fase 1 (inmediata, 1-2 semanas) — Alto ROI / baja complejidad ✅ COMPLETADA
1. ✅ Tablero de métricas objetivo → perf_dashboard command
2. ✅ Monte Carlo de secuencias → monte_carlo command (risk of ruin 0.33%)
3. ✅ Cálculo de riesgo de ruina → integrado en monte_carlo
4. ✅ Risk tiers por instrumento → _volatility_adjusted_risk() reescrito
5. ✅ exchange_close clasificado → _classify_exchange_close con PnL fallback
6. Bot activo — monitorear trades/día

### Fase 2 (2-4 semanas) — Mejora estructural de edge
1. ✅ Pesos dinámicos del allocator por performance rolling (Bayesian beta-binomial)
   - Activado en producción 2026-02-21
   - 7d rolling, Beta(2,2), clamp [0.5×, 2.0×], min 10 trades
   - Efecto observado: trend 0.30→0.27 (↓), carry 0.15→0.18 (↑)
2. ~~Sizing por tiers de riesgo~~ → IMPLEMENTADO (INSTRUMENT_RISK_TIERS)
3. ✅ Alinear backtest con live → 1m bars implementado
   - Signal flips: 1 (5m) → 35 (1m) — gap reducido 35×
   - Trade count: 90 (5m) → 263 (1m) — cercano a live (274)
   - WR: 45.6% (5m) → 58.2% (1m)

### Fase 3 (4-8 semanas) — Modelado cuantitativo avanzado
1. ✅ Regime model HMM 2 estados (signals/regime.py)
   - GaussianHMM en features 1h: log-return, realised vol (24h), ADX(14)
   - Labels: state con mayor vol media → choppy (risk_mult=0.7), otro → trending (risk_mult=1.0)
   - Integrado en allocator cycle (multi_strategy.py): regime_risk_mult × session_risk_mult
   - Celery task `run_regime_detection` cada 6h, cached en Redis
   - 17 tests pasan, Management command `fit_regime`
   - Resultados reales (Feb 21): todos los instrumentos en "trending" (conf>0.99), BTC ADX=25
   - Activado en producción 2026-02-21 (HMM_REGIME_ENABLED=true)
2. ✅ GARCH(1,1) volatility forecast (signals/garch.py)
   - arch library, GARCH(1,1) zero-mean + Student-t en 1h log-returns
   - Rescale ×100 para estabilidad numérica, forecast 1-step-ahead
   - Blend: vol = 0.6×GARCH + 0.4×ATR (GARCH_BLEND_WEIGHT=0.6)
   - Integrado en execute_orders: reemplaza ATR puro para sizing
   - Celery task `run_garch_forecast` cada 6h (:15), cached en Redis (TTL 12h)
   - 15 tests pasan, Management command `fit_garch`
   - Resultados reales (Feb 21): persistence 0.84-0.96, BTC cond_vol=0.40%, ETH=0.50%
   - Activado en producción 2026-02-21 (GARCH_ENABLED=true)
3. Purged CV + embargo para modelos supervisados

### Fase 4 (experimental)
1. Meta-labeling / triple barrier en sandbox
2. RL solo sandbox, sin paso a live hasta validación OOS dura

**GO:** Fase 1 + Fase 2 | **GO condicionado:** HMM/GARCH | **NO-GO:** RL en producción

---

## 11. RUNBOOK PROYECTO CAMPING

### Objetivo
Dejar un procedimiento operativo simple y repetible para:
1. Entrar al server
2. Subir cambios al remoto (push)
3. Bajar cambios en server (pull)
4. Deploy con Docker

### Datos operativos actuales (2026-02-26)
- Repo remoto: `https://github.com/devLink-Developer/chatbot_camping.git`
- Ruta en server: `/opt/chatbot`
- Host prod: `200.58.107.187`
- Puerto SSH: `5344`
- Usuario SSH: `rortigoza`
- Puerto app: `8006`

### A. Push desde local (repo camping)
```bash
cd C:\Users\rortigoza\Documents\Proyectos\chatbot_camping
git checkout main
git pull --ff-only origin main
git status
git add -A
git commit -m "tu mensaje"
git push origin main
```

### B. Entrar al server
Linux/macOS:
```bash
ssh -p 5344 rortigoza@200.58.107.187
```

Windows (PuTTY/plink):
```powershell
& "C:\Program Files\PuTTY\plink.exe" -ssh -P 5344 -l rortigoza 200.58.107.187
```

### C. Pull en server (repo camping)
```bash
cd /opt/chatbot
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
```

### D. Deploy en server
```bash
cd /opt/chatbot
docker compose up -d --build
docker compose ps
docker compose logs --tail=120 chatbot
```

### E. Verificación rápida post deploy
1. Contenedor `aca_lujan_chatbot` en estado `Up`
2. Responde admin: `https://chatbot-api.devlink.com.ar:8006/admin`
3. Logs sin errores críticos en `docker compose logs --tail=120 chatbot`

### F. Nota de operación
- `entrypoint.sh` ya ejecuta automáticamente:
1. `python manage.py migrate --noinput`
2. `python scripts/importar_datos.py`
3. `python manage.py collectstatic --noinput`
- Por eso el deploy estándar es `docker compose up -d --build` y no hace falta correr esos comandos a mano cada vez.

---

## 12. AI Prompt Token Efficiency (2026-03-01)

### Applied changes
- execution/ai_entry_gate.py now uses compact candidate payload keys:
  - top-level: sym, st, dir, sc, atr, spr, sl, ses, sig
  - signal: ns, mr, rb, er, rg, se (mr rows: [module,dir,confidence,raw_score])
- System/user prompt text reduced to minimize fixed token overhead per call.
- Empty placeholders were removed (no "(no extra context)" or "(no recent feedback)").

### New conservative defaults
- AI_ENTRY_GATE_MAX_OUTPUT_TOKENS: 96 (was 180)
- AI_FEEDBACK_CONTEXT_MAX_TOKENS: 700 (was 900)
- Updated in config/settings.py, .env.example and docs/API_ADMIN_CONFIG.md.

### "toon" format status
- No runtime format named "toon" exists in this project.
- The active efficient format is compact JSON + compact JSONL stream (tmp/ai/feedback_stream.jsonl).

### 2026-03-01 (TOON integration)

- New docs detected and reviewed:
  - `docs/TOON_FORMAT_SPECIFICATION_2026.md`
  - `docs/AI_TOON_MASTERTRADING_CONTEXT_2026.toon.md`
- Runtime updated to support TOON context natively:
  - `core/api_runtime.py` detects TOON files (`*.toon.md` or `FORMAT: TOON`) and compacts them before token trim.
  - Compaction keeps deterministic rule lines and removes separator/narrative noise.
- Tests added/updated:
  - `core/tests_api_runtime.py` now validates TOON compaction + auto-compaction in `build_optimized_context`.
- Docs synced:
  - `docs/API_ADMIN_CONFIG.md` now states TOON support is active.
  - `docs/LLM_INDEX.md` now includes TOON spec/context in recommended read order.


### 2026-03-01 (P3 bounded rollout prep)

- Added `signals/meta_allocator.py` (bounded overlay):
  - Uses recent module-attributed returns to compute expectancy/stdev/PF/loss-cluster/correlation penalties.
  - Supports optional `single winner` mode and bucket-capped risk budgets.
  - Integrated in `signals/multi_strategy.py` behind `META_ALLOCATOR_ENABLED`.
- Extended Monte Carlo command (`risk/management/commands/monte_carlo.py`):
  - Optional `--regime-aware` Markov transitions.
  - Stress profiles `none|balanced|bear` + explicit stress overrides.
  - JSON output now includes stress params and regime summary.
- Added nightly automation task `risk.tasks.run_nightly_monte_carlo` and scheduler flags (`MONTE_CARLO_NIGHTLY_*`).
- Added TOON validator:
  - `core/toon_validator.py`
  - `python manage.py validate_toon_context --glob "docs/*.toon.md" --strict`
- Added tests:
  - `signals/tests_meta_allocator.py`
  - `core/tests_toon_validator.py`


### 2026-03-01 (P4 bucket isolation rollout)

- Implemented optional P4 controls in meta allocator:
  - Per-module max drawdown tracking from attributed returns (`max_dd_pct`).
  - Per-module daily PnL tracking (`today_pnl_pct`).
  - Progressive throttle at 50%/75% of bucket caps + hard freeze at 100%.
  - Sample-size guard for low-N modules (`META_ALLOCATOR_P4_MIN_SAMPLE`).
- Added strict no-cross-subsidy budget mode:
  - `META_ALLOCATOR_P4_STRICT_BUCKET_ISOLATION_ENABLED=true` keeps unallocated risk unassigned.
  - `META_ALLOCATOR_P4_MAX_TOTAL_RISK_BUDGET` caps total allocated risk budget.
  - `ALLOCATOR_BUDGET_MIX_MIN_MULT` made configurable (set `0.0` for strict mode).
- Runtime diagnostics expanded in allocator signals:
  - `meta_allocator.p4_enabled`
  - `meta_allocator.p4_strict_bucket_isolation`
  - `meta_allocator.risk_budget_total`
- Files touched:
  - `signals/meta_allocator.py`
  - `signals/allocator.py`
  - `signals/multi_strategy.py`
  - `config/settings.py`
  - `.env.example`
  - docs: `docs/ENV_REFERENCE.md`, `docs/CALIBRATION_CANONICAL.md`, `docs/LLM_INDEX.md`
- Tests added:
  - P4 freeze behavior and strict budget isolation in `signals/tests_meta_allocator.py`
  - Budget floor toggle in `signals/tests.py`

### 2026-03-01 (AI audit single-file map)

- Added `docs/AI_AUDIT_PROJECT_MAP.md` as canonical one-file map for AI audits.
- Includes:
  - Full repository domain map by app and critical file paths.
  - Runtime dataflow, model map, feature-flag surfaces, Celery/schedule map.
  - Production deploy topology and audit checklist.
  - Machine-readable YAML quick index for LLM parsers.
- Updated `docs/LLM_INDEX.md` read order to start with this file.

---

## 13. Grid Module (2026-03-03)

### Objetivo
- Agregar un mÃ³dulo de tipo grid/reversiÃ³n para mercados de rango con volatilidad utilizable.

### ImplementaciÃ³n
- Nuevo detector: `signals/modules/grid.py`
  - Entrada en extremos de rango con z-score/Bollinger.
  - Gating por ADX HTF (rango), ATR% (volatilidad mÃ­n/mÃ¡x), gap EMA20/EMA50.
  - Gate opcional por rÃ©gimen HMM (`choppy`) con fail-open configurable.
  - Bloqueo anti-knife-catch por vela impulso (opcional).
- IntegraciÃ³n:
  - `signals/multi_strategy.py`: soporte `grid` en `run_module_engine`, `_active_modules` y query del allocator.
  - `signals/tasks.py`: nueva tarea `run_grid_engine`.
  - `signals/feature_flags.py`: nueva flag runtime `feature_mod_grid`.
  - `config/settings.py`: nuevos env vars `MODULE_GRID_*` + scheduler/route Celery para `run_grid_engine`.
  - `signals/allocator.py`: `MODULE_ORDER` extendido con `grid` y defaults backward-safe.

### Defaults operativos
- `MODULE_GRID_ENABLED=false` (no altera producciÃ³n hasta activarlo).
- Cuando se activa sin pesos custom:
  - allocator usa peso/risk-budget inicial de grid conservador (0.15) en fallback interno.

### Tests
- `signals/tests_module_filters.py`:
  - emisiÃ³n de `grid short` en extremo superior de rango.
  - bloqueo por gate de rÃ©gimen.
- `signals/tests.py`:
  - `_active_modules` incluye `grid` cuando su flag estÃ¡ activa.
- `signals/tests_dynamic_weights.py`:
  - adaptado para `MODULE_ORDER` extendido (incluye `grid`).

### 2026-03-04 update: grid solo BTC/ETH
- Nuevo env var: `MODULE_GRID_ALLOWED_SYMBOLS`.
- Default operativo en código y `.env.example`: `BTCUSDT,ETHUSDT`.
- `signals/modules/grid.py` ahora bloquea emisión cuando el símbolo no está en el allowlist.
- Cobertura agregada en `signals/tests_module_filters.py` para garantizar que `XRPUSDT` no emita grid si el allowlist es `BTCUSDT,ETHUSDT`.

### 2026-03-04 update: ADX regime gate adaptativo por símbolo/sesión
- Nuevo env var: `MARKET_REGIME_ADX_MIN_BY_CONTEXT` (JSON).
- Precedencia de override:
  1. `SYMBOL:session`
  2. `SYMBOL:*`
  3. `*:session`
  4. `*:*`
  5. fallback `MARKET_REGIME_ADX_MIN`
- Integrado en `execution/tasks.py` para bloqueo de entradas por régimen.
- Logging de bloqueo ahora muestra el umbral efectivo por símbolo.

### 2026-03-04 update: strong-trend solo adaptativo (allocator)
- Nuevo env var: `ALLOCATOR_STRONG_TREND_ADX_MIN_BY_CONTEXT` (JSON).
- Permite bajar solo el umbral ADX para habilitar `min_modules=1` en contexto puntual (ej. `BTCUSDT:london`) sin relajar globalmente todo el bot.
- Integrado en `signals/allocator.py` con fallback al valor global `ALLOCATOR_STRONG_TREND_ADX_MIN`.

---

## 14. Runtime Overrides DB-First (2026-03-10)

- Se reutiliza `signals.StrategyConfig` para overrides runtime sin crear otra tabla.
- Convención:
  - `version=runtime_cfg_v1`
  - `name=<SETTING_KEY>`
  - `enabled=true` significa que la override está activa
  - `params_json={"value": ...}` guarda el valor real
- Helper nuevo: `signals/runtime_overrides.py`
  - cachea en Redis por 30s
  - fallback seguro a `settings.*` si DB/Redis falla
- Integraciones activas:
  - `AI_ENTRY_GATE_ENABLED`
  - `AI_EXIT_GATE_ENABLED`
  - `BTC_LEAD_FILTER_ENABLED`
  - `REGIME_BULL_SHORT_BLOCK_ENABLED`
  - `REGIME_BULL_SHORT_RETRACE_MIN_ALLOWED_MODULES`
  - `REGIME_BULL_SHORT_RETRACE_ALLOWED_MODULES`
- Política operativa:
  - secretos, URLs, puertos, credenciales y bootstrap infra siguen en `.env`
  - flags/toggles de estrategia deben preferir DB para evitar duplicación entre stacks


### Deploy multi-stack: regla obligatoria
- `rortigoza` y `eudy` NO usan el mismo compose operativo.
- Stack principal (`rortigoza`):
  - usar `docker compose ...` con `docker-compose.yml` y `.env`
- Stack `eudy`:
  - usar `docker compose -p trading_bot_eudy -f docker-compose.eudy.yml --env-file .env.eudy ...`
- No desplegar `eudy` con `docker-compose.yml` base.
  - Motivo: el compose base publica `5434:5432` y `6381:6379`
  - Si se usa para `eudy`, intenta recrear `postgres/redis` con los mismos puertos del stack principal
  - Resultado: conflicto de puertos y fallo de despliegue, no de logica
- Regla practica:
  - para cambios de `eudy`, siempre usar solo `docker-compose.eudy.yml`
  - para cambios del stack principal, usar solo `docker-compose.yml`

### 2026-03-10 update: data_stale por transicion
- `execution/tasks.py` ya no debe emitir `RiskEvent(data_stale)` en cada ventana fija mientras un simbolo siga stale.
- Politica correcta:
  - emitir solo en transicion `fresh -> stale`
  - limpiar el estado cuando el simbolo vuelve a estar fresco
  - incluir en Telegram `Symbol`, `Latest 1m` y `Age`
- Motivo:
  - evita spam operativo cuando un simbolo queda atrasado varios minutos
  - mantiene la senal de incidente real sin ocultar el problema de ingestion

### 2026-03-10 update: carry no puede monopolizar el allocator
- Se corrigio un hueco del meta allocator:
  - `META_ALLOCATOR_WEIGHT_CAP` ahora se aplica tambien post-normalizacion
  - antes podia quedar un modulo en ~90%+ aunque el cap fuese menor, si los demas colapsaban
- Guardrail nuevo en `signals/allocator.py`:
  - `ALLOCATOR_CARRY_CONTRA_TREND_MAX_EFFECTIVE_WEIGHT`
  - si `carry` va contra la direccion del `trend`, su peso efectivo queda capado (default `0.20`)
- Objetivo:
  - evitar shorts/longs carry-dominados fragiles
  - bloquear casos donde `carry` solo intenta imponerse contra un `trend` moderado/opuesto

### 2026-03-10 update: NY open y sesgo suave por dia de semana
- `signals/sessions.py` ahora modela `ny_open` como sub-sesion separada:
  - `ny_open`: 13:30-14:00 UTC
  - `overlap`: 12:00-13:30 UTC
  - `london`: 06:00-12:00 UTC
  - `ny`: 14:00-20:00 UTC
- Motivo:
  - la apertura cash de Wall Street tiene mas volatilidad y mas probabilidad de fake move temprano
  - no se trata como "NY normal", sino como una ventana con score/risk mas estrictos
- Defaults operativos:
  - `SESSION_SCORE_MIN["ny_open"] = 0.68`
  - `SESSION_RISK_MULTIPLIER["ny_open"] = 0.65`
  - `ALLOCATOR_STRONG_TREND_SOLO_DISABLED_SESSIONS=ny_open`
- Sesgo semanal suave:
  - `WEEKDAY_CONTEXT_ENABLED=true`
  - lunes/viernes: pequeno endurecimiento (`score +0.01`, `risk x0.95`)
  - sabado/domingo: endurecimiento mayor (`score +0.03`, `risk x0.85`)
- Politica:
  - usarlo como prior suave por sesion/dia
  - no como regla dura tipo "lunes siempre long"

### 2026-03-10 update: motor microvolatilidad separado
- Se agrego un motor directo `microvol` para capturar expansiones rapidas en `1m`, separado del allocator principal.
- Alcance actual:
  - solo `BTCUSDT` y `ETHUSDT`
  - sesiones por default: `london`, `overlap`, `ny_open`, `ny`
  - entrada por breakout + impulso + volumen + sesgo HTF
- Principios operativos:
  - no se mezcla como peso dentro del allocator
  - usa riesgo reducido (`MODULE_MICROVOL_RISK_MULT`)
  - usa cooldown corto propio y timeout maximo de hold (`MODULE_MICROVOL_MAX_HOLD_MINUTES`)
  - perfil de salida mas agresivo: TP mas corto, breakeven y trailing antes
- Rollout recomendado:
  - activar primero solo en `demo`
  - no activar en `live` sin observar algunos dias de ejecucion y rechazos

### 2026-03-11 update: trend permite pullback chico sobre EMA20 HTF
- Se detecto que `trend` quedaba en `0` en Asia aunque hubiese estructura alcista/bajista, porque exigia:
  - `ema20 > ema50` y `last >= ema20` para long
  - `ema20 < ema50` y `last <= ema20` para short
- Eso bloqueaba continuaciones validas cuando el precio estaba apenas debajo/encima de `EMA20` en un pullback sano.
- Se agrego `MODULE_TREND_EMA20_PULLBACK_TOLERANCE_PCT` (default `0.003` = `0.30%`).
- Nueva logica:
  - long: `last >= ema20 * (1 - tol)`
  - short: `last <= ema20 * (1 + tol)`
- Politica:
  - es una relajacion chica y controlada
  - no elimina el gate de ADX ni los guards de impulso/rebote
  - sirve para recuperar emisiones de `trend` en continuaciones con pullback, sin convertirlo en mean reversion

### 2026-03-10 update: cierre heuristico por progreso a TP
- Se agrego un evaluador de salida temprana `tp_progress_exit` en `execution/tasks.py`.
- Objetivo:
  - capturar ganancias cuando el trade ya recorrio buena parte del TP pero pierde continuidad antes de completar el ultimo tramo
  - evitar casos donde llega al 70-90% del objetivo y despues devuelve todo
- Variables usadas:
  - `progress = pnl_gate / tp_pct`
  - `giveback_ratio` contra el HWM (`trail:max_fav`)
  - desalineacion de la señal actual (`signal_mismatch`)
  - sesgo MTF/BTC en contra (`bias_opposed`)
  - envejecimiento relativo de `microvol`
- Defaults operativos:
  - `TP_PROGRESS_EARLY_EXIT_ENABLED=true`
  - `TP_PROGRESS_EARLY_EXIT_MIN_PROGRESS=0.70`
  - `TP_PROGRESS_EARLY_EXIT_MIN_R=0.8`
  - `TP_PROGRESS_EARLY_EXIT_MAX_GIVEBACK_RATIO=0.25`
  - `TP_PROGRESS_EARLY_EXIT_FORCE_PROGRESS=0.90`
  - `TP_PROGRESS_EARLY_EXIT_FORCE_GIVEBACK_RATIO=0.18`
  - `TP_PROGRESS_EARLY_EXIT_CLOSE_SCORE=2`
  - `TP_PROGRESS_EARLY_EXIT_MICROVOL_AGE_RATIO=0.50`
- Politica:
  - no reemplaza el TP normal ni el trailing
  - actua antes del `AI_EXIT_GATE`
  - registra `reason=tp_progress_exit` y `close_sub_reason` con la causa heuristica

### 2026-03-11 update: backtest alineado con el gate real de ejecucion
- Se detecto una brecha entre `backtest` y `live`:
  - el engine de backtest podia abrir operaciones que en runtime real hubiesen sido filtradas por `EXECUTION_MIN_SIGNAL_SCORE` / `SESSION_SCORE_MIN`
  - eso hacia que cualquier optimizacion sobre TP/SL/score quedara contaminada
- Fix aplicado en `backtest/engine.py`:
  - helper `_execution_min_signal_score(...)`
  - helper `_passes_execution_score_gate(...)`
  - antes de abrir posicion, el backtest ahora exige el mismo piso de score que usa ejecucion real
- Test agregado:
  - `backtest/tests_engine_execution_score.py`
- Motivo:
  - no optimizar sobre un motor mas permisivo que produccion
  - evitar elegir combinaciones de parametros que solo "funcionan" porque el backtest estaba dejando pasar entradas invalidas

### 2026-03-11 update: calibracion reciente de salidas y decision operativa
- Se corrio una busqueda reducida en datos locales `2026-02-12 -> 2026-02-23` con reportes:
  - `reports/combo_search_recent_20260212_20260223.json`
  - `reports/context_search_recent_20260212_20260223.json`
- Mejor combinacion reciente probada:
  - `ATR_MULT_TP=1.6`
  - `ATR_MULT_SL=1.5`
  - `MIN_SIGNAL_SCORE=0.45`
  - `trailing_stop=on`
- Resultado de esa combinacion en la ventana reciente:
  - `PnL +0.3629`
  - `PF 1.013`
  - `DD max 1.10%`
  - `59 trades`
  - `WR 64.41%`
- Comparacion contra el set similar con `TP=1.8`:
  - `TP=1.8 / SL=1.5 / score=0.45` dio `PnL -1.4431` y `PF 0.943`
- Lectura:
  - bajar `TP` de `1.8` a `1.6` mejora la captura de ganancias en el tramo reciente
  - `SL=1.5` fue mejor que apretarlo a `1.3`
  - subir `MIN_SIGNAL_SCORE` a `0.50` no aporto mejora visible en esa ventana
  - apagar trailing empeoro el resultado
- Limite importante:
  - esta mejora NO vuelve al sistema robustamente rentable por si sola
  - en la ventana previa `2026-02-01 -> 2026-02-12`, el mismo set siguio negativo
  - conclusion: TP/SL ayudan, pero el edge sigue dependiendo mas de contexto y calidad de entrada que de salidas solamente

### 2026-03-11 update: cambios aplicados en `rortigoza` demo y que NO conviene dejar
- Cambio aplicado en `/opt/trading_bot/.env` del stack principal:
  - `ATR_MULT_TP=1.6`
  - `ATR_MULT_SL=1.5`
  - `MIN_SIGNAL_SCORE=0.45`
- Motivo:
  - es el mejor set reciente de los probados y mantiene el bot dentro de un perfil conservador
- Se revisaron tambien dos relajaciones contextuales que se habian usado para destrabar entradas:
  - `MARKET_REGIME_ADX_MIN_BY_CONTEXT`
  - `ALLOCATOR_CARRY_CONTRA_TREND_MAX_EFFECTIVE_WEIGHT=0.18`
- Decision:
  - NO dejarlas activas en demo como configuracion base
  - se revirtieron a:
    - `MARKET_REGIME_ADX_MIN=17.0`
    - `MARKET_REGIME_ADX_MIN_BY_CONTEXT={}` (sin override)
    - `ALLOCATOR_CARRY_CONTRA_TREND_MAX_EFFECTIVE_WEIGHT=0.20`
- Motivo:
  - el backtest reciente mostro que bajar el gate ADX a `16.0` empeora PnL (`-0.3040`) aunque suba la cantidad de trades
  - capear `carry` en `0.18` no mejoro nada frente a `0.20`
  - regla operativa: no dejar en runtime relajaciones de entrada que no demostraron mejora en backtest reciente

### 2026-03-11 update: segunda brecha critica entre backtest y live
- Se detecto otra desalineacion importante:
  - el backtest hacia un `continue` si `len(module_signals) < ALLOCATOR_MIN_MODULES_ACTIVE`
  - eso impedia que el allocator evaluara `strong trend solo`
  - en live, ese caso SI debe poder abrir cuando `trend` es fuerte y la sesion no esta bloqueada
- Fix aplicado en `backtest/engine.py`:
  - ya no corta antes por `min_modules`
  - ahora delega la decision al allocator via `_resolve_backtest_symbol_allocation(...)`
  - el helper pasa `symbol` y `session_name`, para que funcionen igual que en live:
    - `ALLOCATOR_STRONG_TREND_ADX_MIN_BY_CONTEXT`
    - `ALLOCATOR_STRONG_TREND_SOLO_DISABLED_SESSIONS`
- Tests agregados en `backtest/tests_engine_execution_score.py`:
  - valida que el bridge de backtest pase `symbol/session`
  - valida que `ny_open` siga bloqueando `strong trend solo` aunque el trend sea fuerte
- Implicancia:
  - las optimizaciones de entradas hechas antes de este fix subestimaban de forma fuerte el potencial del setup actual
  - cualquier calibracion nueva de `allocator/session/regime` debe leerse solo despues de esta correccion

### 2026-03-11 update: recalibracion de entradas con el backtest corregido
- Tras corregir la brecha de `strong trend solo`, el mismo setup base mejoro mucho en la ventana reciente `2026-02-12 -> 2026-02-23`:
  - `ATR_MULT_TP=1.6`
  - `ATR_MULT_SL=1.5`
  - `MIN_SIGNAL_SCORE=0.45`
  - `ALLOCATOR_STRONG_TREND_ADX_MIN=25`
  - `ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN=0.80`
  - `SESSION_SCORE_MIN={"asia":0.62,"london":0.56,"ny":0.58,"overlap":0.55,"dead":0.80}`
- Resultado reciente con el backtest ya alineado:
  - `PnL +26.4239`
  - `PF 1.914`
  - `DD max 0.60%`
  - `114 trades`
  - `WR 75.44%`
- Matriz probada en `reports/entry_context_search_recent_20260212_20260223.json`:
  - bajar `ALLOCATOR_STRONG_TREND_ADX_MIN` a `23` o `21` NO mejoro el baseline
  - bajar `ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN` a `0.78` tampoco mejoro
  - aflojar `SESSION_SCORE_MIN` para `ny/asia` tampoco mejoro
- Conclusiones operativas:
  - no conviene seguir relajando entradas por sesion/régimen a ciegas
  - el baseline actual de `strong trend solo` (`ADX 25 / confidence 0.80`) es el mejor de los probados en la ventana reciente
  - el problema de edge ya no parece estar en "falta abrir mas", sino en periodos puntuales/simbolos puntuales

### 2026-03-11 update: donde se pierde el periodo malo con el setup corregido
- Se comparo el mismo setup base en dos ventanas:
  - `prior`: `2026-02-01 -> 2026-02-12`
  - `recent`: `2026-02-12 -> 2026-02-23`
- Resultado:
  - `prior`: `PnL -11.5011`, `PF 0.783`, `129 trades`
  - `recent`: `PnL +26.4239`, `PF 1.914`, `114 trades`
- Breakdown util del periodo malo (`prior`):
  - por direccion:
    - `short`: `121 trades`, `PnL -7.9244`
    - `long`: `8 trades`, `PnL -3.5767`
  - por simbolo:
    - `ETHUSDT`: `PnL -9.1356`
    - `SOLUSDT`: `PnL -7.6096`
    - `BTCUSDT`: `PnL +3.0491`
- Lectura:
  - el drenaje principal del tramo malo no fue "todo el bot", sino sobre todo `ETH` y `SOL`
  - eso apunta a calibracion por simbolo/modulo/contexto, no a relajar globalmente el sistema

### 2026-03-11 update: penalizacion contextual por `symbol:session:direction`
- Se agrego una nueva superficie general en el allocator:
  - `ALLOCATOR_DIRECTION_SCORE_MULT_BY_CONTEXT`
  - formato JSON:
    - `{"ETHUSDT:london:short":0.70,"SOLUSDT:london:short":0.70}`
  - semantica:
    - multiplicador sobre el `net_score` del allocator solo cuando coincide `symbol + session + direction`
    - sirve para penalizar contextos malos sin tocar todo el sistema
- Motivo:
  - en el periodo malo `2026-02-01 -> 2026-02-12`, el leak estaba concentrado en:
    - `ETHUSDT` shorts
    - `SOLUSDT` shorts
    - especialmente en `london`
  - no conviene responder a eso relajando o endureciendo reglas globales
  - conviene un control fino y generalizable por contexto
- Evidencia guardada:
  - `reports/eth_sol_prior_window_breakdown_20260201_20260212.json`
  - `reports/eth_sol_direction_penalty_search_20260201_20260223.json`
- Resultado del barrido:
  - baseline `sin penalizacion`:
    - ventana `prior`: `PnL -11.5011`, `PF 0.783`, `DD 1.68%`
    - tramo completo `2026-02-01 -> 2026-02-23`: `PnL +15.9688`, `PF 1.201`, `DD 1.68%`
  - `ETH/SOL + london + short = 0.80`:
    - mejora minima, insuficiente
  - `ETH/SOL + london + short = 0.70`:
    - ventana `prior`: `PnL -5.8224`, `PF 0.881`, `DD 1.32%`
    - tramo completo `2026-02-01 -> 2026-02-23`: `PnL +21.8112`, `PF 1.289`, `DD 1.32%`
    - ventana `recent`: sin deterioro observable frente al baseline (`PnL +26.4239`, `PF 1.914`)
- Lectura operativa:
  - la penalizacion contextual `0.70` para `ETHUSDT:london:short` y `SOLUSDT:london:short`
    mejora claramente el tramo malo
    y no degrada el tramo bueno en las ventanas probadas
  - eso la convierte en un candidato razonable para rollout controlado, primero en `demo`
