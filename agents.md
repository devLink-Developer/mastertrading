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
