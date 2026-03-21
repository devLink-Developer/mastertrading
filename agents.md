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

### 2026-03-19 update: limpiar `reduceOnly` huerfanas sin posicion
- Se confirmo en prod que BingX puede dejar ordenes abiertas `reduceOnly` despues de que una posicion ya se cerro.
- Caso real auditado:
  - `BTCUSDT`, `DOGEUSDT`, `LINKUSDT`, `XRPUSDT`
  - sin posiciones abiertas en DB ni exchange
  - balance live `free == equity`, `notional = 0`, o sea no estaban reteniendo margen
  - pero las ordenes seguian visibles en la UI como `open orders`
- Causa raiz:
  - `execution/tasks.py::_sync_positions()` marcaba la posicion cerrada
  - pero no existia una limpieza general de ordenes `reduceOnly` huerfanas
- Fix aplicado:
  - helper `_cleanup_orphan_reduce_only_orders()`
  - corre por simbolo en `_sync_positions()` para instrumentos sin posicion abierta en exchange
  - cancela solo ordenes `reduceOnly`
  - throttled por Redis para no golpear API en cada ciclo
- Settings:
  - `ORPHAN_REDUCE_ONLY_CLEANUP_ENABLED=true`
  - `ORPHAN_REDUCE_ONLY_CLEANUP_INTERVAL_SECONDS=600`
- Politica operativa:
  - esto no toca ordenes de entrada
  - solo barre stops/close orders colgados cuando ya no hay nada que reducir

### 2026-03-19 update: allowlist dinamico por tamano de cuenta + reporte diario
- Se agrego una clasificacion dinamica por `xRisk` (riesgo real forzado por `min_qty` vs riesgo objetivo):
  - `tradable`
  - `watch`
  - `blocked`
- Defaults:
  - `MIN_QTY_DYNAMIC_ALLOWLIST_ENABLED=true`
  - `MIN_QTY_DYNAMIC_ALLOWLIST_WATCH_MULTIPLIER=2.0`
  - `MIN_QTY_DYNAMIC_ALLOWLIST_BLOCK_MULTIPLIER=3.0`
- Runtime:
  - `execution/tasks.py` ahora clasifica cada intento de entrada por `xRisk`
  - `blocked` => no abre
  - `watch` => deja traza en logs, pero no bloquea
- Reporte:
  - comando `python manage.py min_qty_risk_report --days 7`
  - ahora muestra `state`, `long_xRisk`, `short_xRisk` y `worst_xRisk`
  - tarea Celery nueva `risk.tasks.send_min_qty_risk_report`
  - envio diario por Telegram con resumen `blocked/watch/tradable`
- Objetivo:
  - convertir el analisis de `min_qty` en una decision operativa reutilizable
  - evitar que cuentas chicas/medianas sigan intentando simbolos estructuralmente desalineados con su equity

### 2026-03-21 update: auditoria de perdidas recientes + mejor trazabilidad de `exchange_close`
- Auditoria de los ultimos 5 dias:
  - la mayoria de las perdidas fueron reales
  - el grupo que mas valor tiene para replay dirigido es `uptrend_short_kill` en shorts
  - aflojar stops globalmente empeora el agregado reciente; no es la direccion correcta
- Se detecto un bug de diagnostico, no de PnL:
  - algunos `exchange_close` terminaban como `close_sub_reason=unknown`
  - causa raiz: `_classify_exchange_close()` podia quedar ciego cuando el cierre venia de un stop ya movido por breakeven/trailing y el exchange no devolvia el fill de forma clara
- Fix aplicado en codigo:
  - `execution/tasks.py` ahora persiste el ultimo `protective stop price` efectivo por posicion en Redis
  - `_sync_positions()` reutiliza ese `stop_hint` al clasificar un `exchange_close`
  - con eso, un cierre por stop protector ya no deberia caer tan facil en `unknown`
- Herramienta nueva:
  - `python manage.py audit_recent_losses --days 5 --post-minutes 60`
  - re-juega las perdidas recientes con velas `1m`
  - resume `bug_candidate`, `timing_candidate`, `real_loss_stop`, `real_loss_stop_late_recovery`, etc.
- Politica derivada:
  - no tocar SL global por intuicion
  - usar replay dirigido para familias de cierre protectivo (`uptrend_short_kill`, `downtrend_long_kill`) antes de cambiar timing

### 2026-03-21 update: replay `stop-aware` para `uptrend_short_kill`
- Se agrego `python manage.py audit_uptrend_short_kill_variants`.
- Objetivo:
  - evaluar variantes de gracia (`15/30/45/60m`) despues de `uptrend_short_kill`
  - pero contando si en el medio el precio hubiera tocado el stop duro reconstruido
- Leccion importante:
  - el replay ingenuo de "esperar 60m y ver si recupera" puede ser demasiado optimista
  - antes de recomendar mas paciencia, hay que validar si la posicion seguia viva sin pegar el stop
- Politica:
  - usar esta herramienta antes de relajar `uptrend_short_kill`
  - si una variante mejora solo porque ignora un stop intermedio, no es una mejora real

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

### 2026-03-16 update: divergencia MTF entre stacks por historico `1d`
- Se detecto una diferencia real entre `rortigoza` y `eudy` en `MTF regime BTC context`:
  - main: `monthly=bear_confirmed`
  - eudy: `monthly=transition`
- La causa raiz no era config ni codigo:
  - ambos stacks tenian la misma logica MTF
  - el problema era que cada stack usa su propia base Postgres
  - `signals/regime_mtf.py` calcula mensual/semanal desde `latest_candles(..., \"1d\")`
  - `eudy` tenia mucho menos historico `1d` cargado que el stack principal
- Verificacion:
  - antes del fix, `BTCUSDT 1d`:
    - main: `436` candles
    - eudy: `222` candles
  - al truncar el lookback del main a `222`, su snapshot pasaba a `transition`, igual que `eudy`
- Correccion operativa aplicada en prod:
  - backfill `1d` desde `2024-09-30` para todos los instrumentos activos en ambos stacks
  - comando:
    - main: `docker compose exec -T web python manage.py backfill --start 2024-09-30 --timeframes 1d --limit 500`
    - eudy: `docker compose -p trading_bot_eudy -f docker-compose.eudy.yml --env-file .env.eudy exec -T web python manage.py backfill --start 2024-09-30 --timeframes 1d --limit 500`
- Estado despues del fix:
  - ambos stacks quedaron con `534` velas `1d` de `BTCUSDT`
  - mismo digest de la serie y mismo snapshot:
    - `monthly=bear_confirmed`
    - `weekly=transition`
    - `daily=transition`
- Leccion:
  - cuando dos stacks comparten codigo pero no comparten DB, hay que auditar tambien cobertura historica y no solo `.env`/flags
  - cualquier rollout de `regime_mtf`, `weekday context` o reglas dependientes de `1d` debe validar paridad de velas `1d` entre stacks

### 2026-03-16 update: sizing, capital por operacion y leverage real en prod

- El bot no usa un monto fijo por trade.
- Usa `risk sizing` por stop:
  - `qty = (risk_pct * equity) / stop_distance_abs`
- Primero decide cuanto esta dispuesto a perder si pega el SL y despues deriva el tamaño.

#### Runtime actual observado en prod

| Parametro | Valor |
|---|---|
| `RISK_PER_TRADE_PCT` | `0.003` |
| `PER_INSTRUMENT_RISK` | `{"BTCUSDT":0.0015,"SOLUSDT":0.002,"LINKUSDT":0.002,"ADAUSDT":0.002}` |
| `MAX_EFF_LEVERAGE` | `5.0` |
| `CONFIDENCE_LEVERAGE_BOOST_ENABLED` | `true` |
| `CONFIDENCE_LEVERAGE_ONLY_ALLOCATOR` | `true` |
| `CONFIDENCE_LEVERAGE_ALLOW_MICROVOL` | `true` |
| `CONFIDENCE_LEVERAGE_SCORE_THRESHOLD` | `0.90` |
| `CONFIDENCE_LEVERAGE_MICROVOL_SCORE_THRESHOLD` | `0.60` |
| `CONFIDENCE_LEVERAGE_MULT` | `1.5` |
| `CONFIDENCE_LEVERAGE_MAX` | `10.0` |

#### Riesgo nominal por trade con equity observado

Snapshot usado:
- `rortigoza`: equity `57.9251 USDT`
- `eudy`: equity `10.7528 USDT`

| Stack | Regla general | BTC | SOL/LINK/ADA |
|---|---|---|---|
| `rortigoza` | `0.1738 USDT` | `0.0869 USDT` | `0.1159 USDT` |
| `eudy` | `0.0323 USDT` | `0.0161 USDT` | `0.0215 USDT` |

Nota:
- esos valores son el riesgo al stop, no el notional total de la posicion
- el notional final depende de SL efectivo, ATR, caps de exposicion y minimos del exchange

#### Ejemplo real reciente de capital usado

| Stack | Simbolo | Notional | Margen usado | Leverage |
|---|---|---|---|---|
| `rortigoza` | `BTCUSDT` | `7.4722 USDT` | `1.4944 USDT` | `5.0x` |
| `eudy` | `BTCUSDT` | `7.5696 USDT` | `1.5139 USDT` | `5.0x` |

#### Escalado automatico de leverage

- Si la señal supera umbrales de conviccion, el bot puede subir leverage automaticamente.
- Se aplica a:
  - `allocator` cuando `sig_score >= 0.90`
  - `microvol` cuando `sig_score >= 0.60`
- Desde 2026-03-16 se deja configurado para poder llegar a `10x` por conviccion sin tocar el cap efectivo total de cuenta:
  - `CONFIDENCE_LEVERAGE_MULT=2.0`
  - `CONFIDENCE_LEVERAGE_MAX=10.0`
  - `MAX_EFF_LEVERAGE` se mantiene aparte como freno de exposicion total

#### Evidencia real de uso del auto-escalado

Ultimos 7 dias:

| Stack | Reports | Trades con leverage `> 5x` | Max observado |
|---|---|---|---|
| `rortigoza` | `38` | `15` | `6.5x` |
| `eudy` | `26` | `9` | `7.5x` |

Conclusion operativa:
- si, el leverage ya esta escalando automaticamente en prod
- no sube de forma indiscriminada: sigue condicionado por score, caps de exposicion, margen libre y riesgo por stop

### 2026-03-16 update: hipotesis NY open + TP mas corto
- Se evaluaron dos hipotesis practicas para el trade perdedor reciente de `ETHUSDT` en `ny_open`:
  1. penalizar longs de `ETH` en `ny_open`
  2. usar `ATR_MULT_TP` mas corto como proxy de toma de ganancias mas realista
- Script reusable:
  - `scripts/evaluate_ny_open_hypotheses.py`
  - ahora soporta filtros `--windows` y `--configs`
  - la ventana `full` usa realmente el `--start/--end` solicitado
- Reportes generados:
  - `reports/ny_open_hypotheses_btc_eth_20260201_20260315.json`
  - `reports/ny_open_hypotheses_btc_eth_htf1h_20260201_20260315.json`
  - `reports/ny_open_hypotheses_btc_eth_recentlatest_20260310_20260317.json`
- Hallazgo clave:
  - el trade real de `ETH` del 2026-03-16 **no fue un caso de TP lejano**
  - en live tuvo `mfe_r` muy bajo y termino en `exchange_stop`
  - o sea: no estuvo "encaminado" y luego devolvio; fue una entrada que no desarrollo
- Resultado de backtest:
  - el proxy `ATR_MULT_TP=1.4` no mostro mejora robusta
  - en algunas ventanas viejas mejora apenas el total, pero empeora la ventana reciente
  - por eso **no conviene acortar TP globalmente** todavia
- Limitacion importante:
  - el backtest actual no reprodujo trades `BTC/ETH long` en `ny_open` en las ventanas auditadas
  - entonces la hipotesis de endurecer `ETH:ny_open:long` no quedo validada ni invalidada por backtest; simplemente no hubo muestra util en el motor
- Politica operativa derivada:
  - no cambiar TP global por intuicion
  - para casos `ny_open`, priorizar replay/auditoria de señales live y mejoras del motor de backtest (cadencia 60s / tp_progress_exit) antes de activar filtros nuevos

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

### 2026-03-11 update: dashboard operativo de captura y hotspots
- Se amplio `risk/management/commands/perf_dashboard.py` para responder tres preguntas operativas:
  1. donde gana/pierde por simbolo, sesion, estrategia y regimen
  2. donde deja ganancias sobre la mesa (`mfe_capture_ratio`, `giveback`)
  3. donde se concentran los stops reales (`sl`, `exchange_stop`, `trailing_stop`)
- Nuevos breakdowns:
  - `by_session`
  - `by_weekday`
  - `by_strategy`
  - `by_reason_detail` (`reason:close_sub_reason`)
  - `by_recommended_bias`
  - `by_btc_lead_state`
  - `by_mtf_snapshot` (`monthly|weekly|daily`)
  - `by_symbol_session`
- Nuevas metricas de bucket:
  - `avg_mfe_r`
  - `avg_mae_r`
  - `mfe_capture_avg / p50 / p75`
  - `giveback_avg`
  - `capture_samples`
- Se agregaron dos vistas de analisis rapido:
  - `capture_hotspots`
    - buckets `symbol|session` con peor captura relativa
    - sirven para detectar donde el bot devuelve mucho despues de ir bien
  - `stop_clusters`
    - buckets `symbol|side|session` con cluster de salidas tipo stop
    - sirven para detectar contextos de entrada estructuralmente malos
- Regla de interpretacion:
  - si baja `mfe_capture_avg` pero sube `avg_mfe_r`, el problema suele ser de salida
  - si `avg_mfe_r` ya es pobre y encima hay `stop_clusters`, el problema suele ser de entrada/contexto

### 2026-03-12 update: robustez por simbolo en ventana larga antes de tocar reglas finas
- Se definio una regla operativa explicita:
  - no activar ni desactivar overrides por simbolo solo porque una ventana reciente sale bien o mal
  - antes hay que mirar un backtest mas largo por simbolo y revisar calidad de datos
- Se agrego `scripts/evaluate_symbol_robustness.py` para comparar configuraciones fijas por simbolo usando todo el rango disponible dentro de una ventana pedida.
- Configuraciones comparadas en la corrida inicial:
  - `baseline_tp18_sl15_score045`
  - `candidate_tp16_sl15_score045`
- Rango pedido:
  - `2026-01-01 -> 2026-02-23`
- Hallazgos:
  - `BTCUSDT` (`53d`, cobertura `1.00`):
    - baseline levemente mejor (`+5.042` vs `+5.013`)
    - conclusion: `TP 1.6` no aporta mejora clara en BTC
  - `ETHUSDT` (`53d`, cobertura `1.00`):
    - candidato claramente mejor (`+12.313` vs `+5.587`, PF `1.164` vs `1.071`, DD menor)
  - `SOLUSDT` (`53d`, cobertura `0.994`):
    - candidato claramente mejor (`+23.633` vs `+19.764`, PF `1.492` vs `1.403`, DD menor)
  - `ADAUSDT` (`13.1d`, cobertura `0.974`):
    - candidato mejor, pero la muestra es demasiado corta para sacar una regla estructural
  - `DOGEUSDT` (`23d`, cobertura `0.773`):
    - baseline mejor
  - `LINKUSDT` (`43d`, cobertura `0.414`):
    - candidato menos malo, pero la cobertura es demasiado pobre
  - `XRPUSDT` (`53d`, cobertura `0.336`):
    - baseline mejor, pero la cobertura es demasiado pobre para confiar en el resultado
- Lectura operativa:
  - `TP 1.6` si muestra evidencia seria en `ETH` y `SOL`
  - `BTC` queda practicamente neutro
  - en `DOGE/LINK/XRP` no conviene sacar conclusiones fuertes porque el historial 5m esta incompleto
  - por eso no corresponde meter reglas finas por simbolo todavia basadas en esos tres
- Archivo de evidencia:
  - `reports/symbol_robustness_20260101_20260223.json`

### 2026-03-12 update: robustez por simbolo rerun con cobertura completa
- Despues de corregir el backfill minute-level y completar `ADA/DOGE/LINK/XRP` en `5m`, se repitio la misma comparacion:
  - `baseline_tp18_sl15_score045`
  - `candidate_tp16_sl15_score045`
  - rango `2026-01-01 -> 2026-02-23`
- Archivo de evidencia corregido:
  - `reports/symbol_robustness_20260101_20260223_fullcov.json`
- Resultado final con cobertura completa:
  - `BTCUSDT`:
    - baseline apenas mejor (`+5.042` vs `+5.013`)
    - lectura: `TP 1.6` es neutro en BTC, no mejora clara
  - `ETHUSDT`:
    - candidato mejor (`+12.313` vs `+5.587`, PF `1.164` vs `1.071`, DD menor)
  - `SOLUSDT`:
    - candidato mejor (`+23.633` vs `+19.764`, PF `1.492` vs `1.403`, DD menor)
  - `ADAUSDT`:
    - candidato mejor (`+18.864` vs `+14.542`, PF `1.273` vs `1.204`, DD menor)
  - `DOGEUSDT`:
    - baseline levemente mejor (`+6.619` vs `+6.527`)
    - lectura: `TP 1.6` no agrega ventaja clara
  - `LINKUSDT`:
    - candidato menos malo (`-0.412` vs `-1.768`, PF `0.993` vs `0.970`)
  - `XRPUSDT`:
    - candidato mucho mejor (`+42.809` vs `+29.291`, PF `1.569` vs `1.360`)
- Lectura operativa corregida:
  - una vez resuelta la calidad de datos, `TP 1.6` deja de verse como mejora solo en `ETH/SOL`
  - la mejora pasa a ser amplia en `ETH`, `SOL`, `ADA`, `LINK` y `XRP`
  - `BTC` y `DOGE` quedan practicamente neutros / levemente mejor con baseline
  - conclusion: el cambio `TP 1.8 -> 1.6` si tiene soporte bastante mas general que lo que parecia con la muestra incompleta
  - aun asi, eso no justifica overrides por simbolo automaticamente; justifica primero considerar `TP 1.6` como candidato global y despues validar en `demo/live`

### 2026-03-12 update: limite real de backfill BingX
- Se corrigio `backtest/management/commands/backfill_candles.py`:
  - `1m` y `5m` ahora usan `limit=1440`, no `1500`
- Motivo:
  - BingX rechaza requests minute-level con `limit > 1440`
  - error observado:
    - `code 109400: limit must be less than or equal to 1440`
- Leccion:
  - si falta historico en simbolos BingX, primero revisar cobertura real de velas antes de interpretar un backtest
  - un resultado por simbolo con cobertura `0.33-0.77` no sirve para justificar un override estructural
  - el comando tambien dejo de usar `fetch_ticker` para resolver simbolos y ahora reintenta el mismo chunk cuando BingX devuelve `109429`
  - politica correcta:
    - nunca avanzar el cursor por un `rate limit`
    - solo avanzar cuando realmente se persistio ohlcv del chunk actual

### 2026-03-12 update: microvol no debe deformarse para forzar frecuencia
- Se audito `microvol` en prod/demo (`rortigoza`) con diagnostico bar-by-bar sobre `BTCUSDT` y `ETHUSDT`.
- Hallazgo:
  - en `overlap` de `2026-03-12 12:07-12:08 UTC`, `microvol` no emitia por un bug de logica de HTF similar al que antes tenia `trend`:
    - `ema20_htf > ema50_htf`
    - pero `last_htf` quedaba apenas debajo de `ema20_htf`
    - el detector exigia `last_htf >= ema20_htf` para long y devolvia `None`
- Fix aplicado:
  - `signals/modules/microvol.py` ahora usa `MODULE_MICROVOL_HTF_PULLBACK_TOLERANCE_PCT`
  - logica nueva:
    - long: `last_htf >= ema20_htf * (1 - tol)`
    - short: `last_htf <= ema20_htf * (1 + tol)`
  - default: `0.003` (`0.30%`)
- Leccion importante:
  - aun corrigiendo ese gate, el mismo `overlap` seguia sin ser setup valido de `microvol`
  - despues del HTF, se caia por:
    - `ATR 1m` demasiado bajo
    - y, si se relajaba mas ATR, por falta de `impulse bar`
  - conclusion:
    - no conviene seguir bajando `ATR_MIN` o `IMPULSE_MIN_BODY_PCT` solo para fabricar mas trades
    - eso convertiria `microvol` en otro detector de continuacion floja, no en un motor de expansion rapida

### 2026-03-12 update: macro high-impact no debe bloquear ciegamente microvol
- Se observo una senal real:
  - `mod_microvol_long`
  - `ETHUSDT`
  - `2026-03-12 14:16 UTC`
  - `score=0.6392`
- Esa senal no termino en orden porque `execution/tasks.py` la bloqueo con:
  - `Macro high-impact window blocked entry on ETHUSDT`
- Decision:
  - agregar excepcion controlada al filtro macro solo para `microvol`
  - limitada por:
    - `MACRO_HIGH_IMPACT_ALLOW_MICROVOL`
    - `MACRO_HIGH_IMPACT_ALLOW_MICROVOL_SYMBOLS`
- Politica operativa:
  - si la excepcion esta desactivada, el comportamiento sigue igual
  - si esta activada, solo `microvol` en simbolos permitidos puede pasar la ventana macro
  - el riesgo sigue reducido por:
    - `MODULE_MICROVOL_RISK_MULT`
    - y el multiplicador macro existente `MACRO_HIGH_IMPACT_RISK_MULTIPLIER`
- Criterio:
  - permitir solo `BTCUSDT` y `ETHUSDT`
  - no abrir la excepcion al allocator completo ni a alts secundarias

### 2026-03-12 update: strong trend solo en NY para BTC/ETH no queda validado
- Se reviso el caso de `ETH` en NY donde aparecia `trend long` sin segunda confirmacion del allocator.
- Diagnostico:
  - no fue un bug del modulo `trend`
  - la segunda confirmacion faltaba porque `carry/meanrev/smc` no emitian en la misma direccion
  - bajo `ALLOCATOR_MIN_MODULES_ACTIVE=2`, el `alloc_flat` fue el comportamiento correcto
- Script reusable agregado:
  - `scripts/evaluate_ny_strong_trend_solo.py`
  - compara baseline vs overrides contextuales de `ALLOCATOR_STRONG_TREND_ADX_MIN_BY_CONTEXT` para `BTCUSDT` y `ETHUSDT` en `ny`
  - desglosa `overall`, `ny` y `ny_open`
- Evidencia guardada:
  - `reports/ny_strong_trend_solo_20260201_20260223.json`
- Configs evaluadas:
  - `baseline_25_25`
  - `btc18_eth19_ny`
  - `btc17_eth18_ny`
  - `btc16_eth17_ny`
  - chequeos adicionales puntuales:
    - `eth19_only`
    - `eth19_only_conf90`
    - `btc18_eth19_conf90`
- Resultado:
  - bajar solo el ADX contextual en `ny` agrega trades, pero empeora la ventana reciente y el agregado de `BTC+ETH`
  - `BTC` sale mejor sin relajar la regla
  - `ETH` mejora en una ventana previa, pero no sostiene la mejora en la ventana reciente
  - incluso subiendo `ALLOCATOR_STRONG_TREND_CONFIDENCE_MIN` a `0.90`, la mejora no queda estable OOS
- Decision operativa:
  - NO activar por ahora una regla global de `strong trend solo` para `BTC/ETH` en `ny`
  - la ausencia de segunda confirmacion debe tratarse como falta real de confluencia, no como permiso automatico para abrir
- Si se retoma esta linea, la proxima version defendible no deberia ser solo `ADX` mas bajo:
  - deberia exigir condiciones mas finas, por ejemplo ausencia de `carry` opuesto, mejor follow-through o filtro de microestructura
  - y validarse OOS separando `BTC` y `ETH`
  - se prototipo un guardrail opcional `ALLOCATOR_STRONG_TREND_SOLO_REQUIRES_NO_OPPOSING_CARRY`
  - aun con `ETHUSDT:ny=19`, `confidence>=0.90` y veto a `carry` opuesto, la mejora siguio sin sostenerse en la ventana reciente
  - conclusion: tampoco corresponde activarlo todavia; queda solo como herramienta opcional para futuros experimentos

### 2026-03-13 update: OperationReport debe atribuir al signal de entrada, no al state de salida
- Se detecto un bug de atribucion en `execution/tasks.py` dentro de `_manage_open_position`.
- Sintoma operativo:
  - cierres ganadores recientes aparecian ligados a `signal.strategy=alloc_flat`
  - eso rompia la lectura de si el trade habia nacido como `microvol` o `alloc`
- Causa raiz:
  - varios caminos de cierre seguian llamando `_log_operation(...)` con:
    - `signal_id=str(sig.id)`
    - `correlation_id=f"{sig.id}-{inst.symbol}"`
  - `sig` en esa etapa representa el signal del ciclo de salida, no necesariamente el signal que abrio la posicion
  - como el allocator emite `alloc_flat` con frecuencia, los reports quedaban contaminados con el estado mas reciente
- Fix aplicado:
  - helper nuevo `execution.tasks._position_origin_refs(...)`
  - resuelve de forma estable:
    - `signal_id` = signal de origen de la posicion cuando existe
    - `correlation_id` = root correlation de la posicion, no el signal del ciclo actual
  - se aplico a todos los cierres de `_manage_open_position`:
    - `trailing_stop`
    - `microvol_timeout`
    - `stale_cleanup`
    - `uptrend_short_kill`
    - `downtrend_long_kill`
    - `tp_progress_exit`
    - `ai_tp_early_exit`
    - `tp/sl`
    - `signal_flip`
- Tests agregados:
  - helper puro: preferencia por entry signal + root correlation
  - integracion: un TP con `origin_signal=mod_microvol_long` y `sig actual=alloc_flat` debe loguear el origin, no el flat
- Politica:
  - cuando se analicen cierres por estrategia, confiar en `OperationReport` solo despues de este fix
  - antes del fix, la atribucion historica reciente puede mezclar estrategia de entrada con estado de salida

### 2026-03-13 update: eudy plano no implica fallo
- En la revision de las ultimas 12h:
  - `rortigoza` tuvo cierres positivos y actividad normal
  - `eudy` quedo plano, sin ordenes nuevas ni `RiskEvent`
- Los logs muestran que `eudy` no estaba caido:
  - `trend module emitted=6`
  - `carry module emitted=3`
  - `allocator emitted=7`
  - `execute_orders` corria normal
- La diferencia visible frente al stack principal fue:
  - `microvol:disabled_flag` en `eudy`
  - en `rortigoza`, `microvol` si corre aunque muchas veces emita `0`
- Para los simbolos con score suficiente parcial, el cuello de botella real fue:
  - `Signal score too low for execution ... (session=london)`
- Conclusiones operativas:
  - `eudy` plano en esa ventana no fue un crash ni un bloqueo de DD
  - fue combinacion de:
    - `microvol` desactivado en live
    - y setups del allocator que no superaron el `score gate` de sesion

### 2026-03-16 update: BingX `timestamp is invalid` en `fetch_balance`
- Se observaron warnings recurrentes en el stack principal:
  - `Balance check failed (manage-only): bingx {"code":109400,"msg":"timestamp is invalid","data":{}}`
- Frecuencia observada:
  - 3 ocurrencias en 24h
  - no tumbaban el worker, pero generaban ruido operativo y podian impedir el balance check puntual
- Causa probable:
  - drift entre reloj local y reloj del exchange
  - el adapter de BingX no estaba forzando `adjustForTimeDifference` ni `recvWindow`
  - tampoco tenia re-sync explicito al detectar `109400`
- Fix aplicado en `adapters/bingx.py`:
  - `recvWindow=10000`
  - `options.adjustForTimeDifference=True`
  - `_sync_clock()` usando `load_time_difference()`
  - `fetch_balance()` ahora:
    - detecta `timestamp is invalid`
    - sincroniza reloj
    - reintenta una vez
- Politica:
  - el fix es acotado a BingX y a `fetch_balance`
  - no se debe capturar de forma genérica cualquier exception y reintentar a ciegas
  - si reaparece el 109400 en otros endpoints autenticados, extender el mismo patron endpoint por endpoint

### 2026-03-16 update: diagnostico deterministico para `microvol`
- Se detecto una divergencia real entre stacks:
  - `eudy` emitio un `mod_microvol_long` en BTC
  - `rortigoza` no emitio nada en la misma ventana
  - la comparacion previa ya habia descartado:
    - diferencia de candles crudas
    - diferencia de flags/thresholds de `microvol`
- Para no volver a diagnosticar esto a ciegas, `signals/modules/microvol.py` ahora expone `explain(...)`:
  - devuelve `stage` de rechazo/aceptacion
  - y los valores finales mas importantes del detector
    - `adx_htf`
    - `atr_pct`
    - `body_pct`
    - `breakout_pct`
    - `volume_ratio`
    - `confidence`
    - `close_location`
- `signals/multi_strategy.py` puede loguear ese resumen cuando:
  - `MODULE_MICROVOL_DEBUG_ENABLED=true`
  - y opcionalmente el simbolo esta en `MODULE_MICROVOL_DEBUG_SYMBOLS`
- Politica:
  - mantener debug apagado por default
  - activarlo solo al perseguir divergencias entre stacks o gaps raros de emision
  - usarlo preferentemente para `BTCUSDT`/`ETHUSDT`, no para todo el universo

### 2026-03-16 update: leverage sube por conviccion, no globalmente
- Se reforzo la politica de leverage controlado:
  - `MAX_EFF_LEVERAGE` default sube moderadamente a `2.5`
  - no se vuelve a abrir la canilla global a todos los trades
- `execution/tasks.py` ahora permite `confidence leverage boost` tambien en `microvol`:
  - `CONFIDENCE_LEVERAGE_BOOST_ENABLED=true` por default
  - `CONFIDENCE_LEVERAGE_ONLY_ALLOCATOR=true` sigue vigente
  - pero con `CONFIDENCE_LEVERAGE_ALLOW_MICROVOL=true`, `microvol` puede recibir boost sin abrir otros modulos
- Umbrales:
  - allocator sigue usando `CONFIDENCE_LEVERAGE_SCORE_THRESHOLD=0.90`
  - microvol usa su propio umbral:
    - `CONFIDENCE_LEVERAGE_MICROVOL_SCORE_THRESHOLD=0.60`
- Razon:
  - `microvol` y `allocator` no puntuan en la misma escala
  - exigir `0.90` a `microvol` lo dejaba practicamente sin boost
- Politica:
  - preferir subida de leverage por conviccion del trade
  - no subir leverage indiscriminadamente en todos los simbolos/sesiones

### 2026-03-19 update: `SOLUSDT` puede sobre-riesgar por minimo de lote
- Caso auditado en prod (`rortigoza`, Asia):
  - `alloc_short` abierto `2026-03-19 01:54 UTC`
  - cerrado `2026-03-19 03:20 UTC`
  - `reason=uptrend_short_kill`
  - perdida realizada `-0.671 USDT`
- Hallazgos:
  - no fue cierre por `SL` del exchange
  - el hard stop seguia mas arriba (`91.3229`)
  - despues del cierre, el precio todavia hizo un maximo peor (`91.36`), por lo que el `kill` probablemente redujo dano frente al stop duro
  - el precio recien volvio debajo de la entry unos `46.9` minutos despues del cierre
- Causa estructural:
  - `SOLUSDT` en BingX usa `lot_size=1`
  - en cuentas chicas/medianas eso puede obligar a abrir `1 SOL` aunque el sizing por riesgo pida bastante menos
  - el riesgo realizado puede quedar muy por encima del `risk_budget_pct` presupuestado
- Comparacion con `eudy`:
  - vio la misma senal `alloc_short`
  - no abrio porque el minimo de `1 SOL` excedia el `MAX_EFF_LEVERAGE=5.0` de la cuenta
  - log real: `Pre-trade leverage cap ... max_new=54.74 min_qty=1.0; skipping`
- Politica derivada:
  - no juzgar este patron solo como problema de timing/salida
  - revisar siempre `lot_size`, notional minimo y riesgo realizado vs riesgo objetivo
  - futuro hardening candidato: saltar entradas cuando `min_qty` implique un riesgo al stop muy por encima del presupuesto

### 2026-03-19 update: guardrail activo contra `min_qty` que rompe el riesgo
- Se implemento hardening en `execution/tasks.py`:
  - despues de calcular `qty` final, el bot compara:
    - `target_risk_amount = equity * effective_risk_pct`
    - `actual_stop_risk_amount = qty * stop_distance_pct * entry_price * contract_size`
  - si el `min_qty` del exchange forzo una posicion y el riesgo real supera varias veces el presupuesto, la entrada se bloquea
- Nuevos settings:
  - `MIN_QTY_RISK_GUARD_ENABLED=true`
  - `MIN_QTY_RISK_MULTIPLIER_MAX=3.0`
- Intencion:
  - evitar casos como `SOLUSDT` donde el minimo de `1` contrato puede transformar un trade de riesgo pequeno en una perdida de ~1% del capital
  - mantener tolerancia a pequenas desviaciones normales, pero frenar overshoot grosero de riesgo

### 2026-03-21 update: `ny_open buy` se audita mejor con replay de trades reales
- Para la hipotesis de endurecer `buy` en `ny_open`, el backtest clasico no es suficiente por si solo:
  - los reportes `reports/ny_open_hypotheses*.json` mostraron que el motor no estaba reproduciendo casi nada de `BTC/ETH ny_open buy`
  - eso hace que una penalizacion simple en backtest pueda dar `0 cambios` aunque live ya haya mostrado el patron
- Se agrego comando de replay sobre `OperationReport` reales:
  - `python manage.py audit_ny_open_buy_context --days 30`
  - `python manage.py audit_ny_open_buy_context --days 30 --symbol ETHUSDT`
- Archivo:
  - `risk/management/commands/audit_ny_open_buy_context.py`
- Variantes evaluadas por el comando:
  - `block_ny_open_buy_balanced_transition`
  - `block_ny_open_buy_balanced`
  - `block_ny_open_buy_weak_long_context`
- Politica derivada:
  - para `ny_open`, usar primero replay sobre trades reales de prod
  - solo si el replay y la muestra reciente convergen, pasar a cambio de runtime
  - no bloquear `ETH` por simbolo sin evidencia estable; preferir gates contextuales por `session + lead_state + recommended_bias`

### 2026-03-21 update: gate live para `ny_open long` en contexto debil
- Se agrego un gate de entrada chico y reversible en `execution/tasks.py`:
  - helper `_ny_open_weak_long_precheck()`
  - corre dentro del pipeline de apertura antes del ML gate
- Intencion:
  - bloquear `long` en `ny_open` cuando el contexto de BTC siga demasiado ambiguo para una apertura agresiva
  - evitar repetir entradas tipo `ETH ny_open buy` que en live mostraron poco desarrollo favorable
- Politica por default:
  - solo aplica en `session=ny_open`
  - solo a `signal_direction=long`
  - `microvol` queda exento
- Superficie configurable:
  - `NY_OPEN_WEAK_LONG_BLOCK_ENABLED`
  - `NY_OPEN_WEAK_LONG_BLOCK_LEAD_STATES`
  - `NY_OPEN_WEAK_LONG_BLOCK_RECOMMENDED_BIASES`
- Configuracion inicial recomendada:
  - `NY_OPEN_WEAK_LONG_BLOCK_ENABLED=true`
  - `NY_OPEN_WEAK_LONG_BLOCK_LEAD_STATES=transition`
  - `NY_OPEN_WEAK_LONG_BLOCK_RECOMMENDED_BIASES=balanced`
- Criterio operativo:
  - empezar por `transition + balanced`
  - no ampliar a `tactical_long` ni a `bear_weak` sin nueva evidencia

### 2026-03-21 update: auditoria de estructura de sesion para `ny_open/london buy`
- Se agrego comando:
  - `python manage.py audit_session_structure_gate --days 14 --sessions ny_open,london --side buy`
- Archivo:
  - `risk/management/commands/audit_session_structure_gate.py`
- Mide por trade real:
  - `session_progress`
  - `tp_extension_pct`
  - `tp_extension_vs_session_range`
  - barrera de `previous-day high/low`
  - follow-through de los primeros `2m`
- Hallazgo en prod:
  - `rortigoza`: el trade malo `ETHUSDT ny_open buy` de `2026-03-16` aparecio como caso claro de chase:
    - `session_progress=0.9008`
    - `tp_extension_pct=0.5304%`
    - `tp_extension_vs_session_range=0.4800`
    - `recommended_bias=balanced`, `btc_lead_state=transition`
  - `eudy`: el `ETHUSDT ny_open buy` perdedor del mismo dia tambien mostro estructura fragil:
    - `session_progress=0.7624`
    - `tp_extension_vs_session_range=0.3384`
    - `recommended_bias=balanced`, `btc_lead_state=transition`
- Lectura operativa:
  - el gate live `balanced + transition` ya deberia capturar el caso `ETH ny_open buy` observado en ambos stacks
  - la capa de `session structure` todavia conviene usarla como auditoria/replay, no como regla live adicional, hasta juntar mas muestra
