# AUDITORÍA INTEGRAL — MasterTrading

**Auditor:** Perfil Supervisor / Economista + Ing. Comercial con experiencia en mercados
**Fecha:** 2026-02-21
**Alcance:** 15 archivos core, ~9,500 líneas de código productivo
**Archivos auditados:** execution/tasks.py, signals/tasks.py, signals/allocator.py, signals/sessions.py, signals/modules/{trend,meanrev,carry,common}.py, signals/{regime,garch,multi_strategy}.py, risk/{models,tasks,report_controls,notifications}.py, execution/models.py, backtest/engine.py, config/{settings,celery}.py, docker-compose.yml, adapters/kucoin.py, core/models.py

---

## ÍNDICE

1. [Hallazgos Críticos (Riesgo Financiero Directo)](#i-hallazgos-críticos-riesgo-financiero-directo)
2. [Hallazgos Medios (Degradan Calidad/Precisión)](#ii-hallazgos-medios-degradan-calidadprecisión)
3. [Hallazgos de Seguridad](#iii-hallazgos-de-seguridad)
4. [Hallazgos de Arquitectura / Deuda Técnica](#iv-hallazgos-de-arquitectura--deuda-técnica)
5. [Validación Matemática](#v-validación-matemática)
6. [Calificación por Área](#vi-calificación-por-área)
7. [Plan de Acción Recomendado](#vii-plan-de-acción-recomendado-priorizado)

---

## I. HALLAZGOS CRÍTICOS (RIESGO FINANCIERO DIRECTO)

### 1. `create_order` NO tiene retry — adapters/kucoin.py L126

La función más importante del sistema — colocar una orden en el exchange — es la **única** que no tiene el decorador `@_retry_exchange`. Todas las demás funciones (`fetch_positions`, `cancel_order`, `fetch_open_orders`, etc.) sí lo tienen.

**Impacto:** Un error de red transitorio durante el envío de una orden:
- Si ocurre **antes** de que KuCoin reciba la orden → orden perdida, sin posición, sin SL
- Si ocurre **después** de que KuCoin procesa la orden pero antes de recibir confirmación → **posición fantasma** abierta en exchange sin registro en DB, sin SL, sin trailing

**Severidad:** **CRÍTICO**. Es el single point of failure más peligroso del sistema.

---

### 2. `_volatility_adjusted_risk` anula el risk budget del allocator — execution/tasks.py L842

Cuando el allocator calcula un `risk_budget_pct` ajustado por confianza (ej. señal débil = 0.0005), el flujo en `execute_orders` (L3232-L3255) primero asigna ese budget, y luego llama a `_volatility_adjusted_risk()`. Si el instrumento está en `PER_INSTRUMENT_RISK`, la función **ignora el budget y devuelve un valor fijo**:

```python
# L842-843
if symbol in per_inst:
    return float(per_inst[symbol])  # IGNORA base_risk
```

**Impacto para BTCUSDT** (`PER_INSTRUMENT_RISK=0.0015`):
- Señal de baja confianza (risk_budget=0.0005) → se **triplica** a 0.0015
- Señal de alta confianza (risk_budget=0.0025) → se **reduce** a 0.0015

**Efecto:** Toda la calibración del allocator (Bayesian weights, risk budget, confidence) queda anulada para BTC, SOL, LINK y ADA. El sistema toma riesgo uniforme sin importar la calidad de la señal.

**Severidad:** **CRÍTICO**. Undermines the entire signal-quality → position-sizing pipeline.

---

### 3. Redis expuesto sin contraseña — docker-compose.yml L127

```yaml
redis:
  image: redis:7
  ports:
    - "6381:6379"
  command: redis-server --appendonly yes
```

Sin `--requirepass`. Puerto mapeado al host. Cualquier proceso o persona en la red puede:
- `FLUSHALL` → borra locks, caches HMM/GARCH, estado de trailing/partial close
- `SET` → falsificar el estado de positions, inyectar señales falsas
- `KEYS *` → leer toda la configuración de runtime

**Severidad:** **CRÍTICO** si la máquina está expuesta a cualquier red (incluso LAN).

---

### 4. Celery sin `acks_late` ni `task_time_limit` — config/celery.py

El archivo de configuración tiene 17 líneas. No configura:
- `task_acks_late`: si un worker muere a mitad de `execute_orders`, la tarea ya fue reconocida y **se pierde**
- `task_time_limit`: un API call colgado al exchange bloquea el worker **indefinidamente**
- `task_reject_on_worker_lost`: tareas en crash no se re-encolan
- Dead letter queue: tareas fallidas desaparecen sin traza

**Impacto:** Un ciclo perdido de `execute_orders` (cada 30-60s) puede significar no cerrar una posición que va contra el mercado, no mover un trailing stop, o no detectar una liquidación.

**Severidad:** **ALTO**

---

### 5. Signal engine sin retry — signals/tasks.py L950

`run_signal_engine` es un `@shared_task` sin `autoretry_for`, `max_retries` ni `acks_late`. Si la DB está temporalmente inaccesible o la query falla, el ciclo de señales se pierde silenciosamente. Con cadencia de 60s, perder 2-3 ciclos puede significar entrar tarde a un movimiento o no detectar un cambio de régimen.

**Severidad:** **ALTO**

---

## II. HALLAZGOS MEDIOS (DEGRADAN CALIDAD/PRECISIÓN)

### 6. `_classify_exchange_close` — thresholds no proporcionales al SL real

execution/tasks.py L1612: El fallback PnL-based usa ±0.25% como frontera entre "exchange_stop" y "exchange_tp_limit", pero el `MIN_SL_PCT` real es 1.2%. Una pérdida del 0.3% se clasifica como SL cuando realmente es un cierre near-breakeven. Contamina las métricas de performance y los dynamic weights del allocator.

---

### 7. Partial close trunca posiciones fraccionarias

execution/tasks.py L1290: `int(abs(current_qty))` — para posiciones como 1.5 contratos, `int(1.5) = 1`, y el guard `if total_abs >= 2` bloquea el partial close. Muchos instrumentos en KuCoin Futures tienen sizing fraccionario.

---

### 8. `ALLOCATOR_MIN_MODULES_ACTIVE` no se valida dentro del allocator

signals/allocator.py: `resolve_symbol_allocation()` calcula `active_module_count` pero **no lo usa como gate**. La validación se hace en el signal engine o en execute_orders. Si se usa el allocator desde otro contexto (backtest, testing), el gate está ausente.

---

### 9. Session defaults extremadamente restrictivos vs producción

signals/sessions.py L15-L26: Si la variable de entorno `SESSION_SCORE_MIN` falta, los defaults del código son ~0.18 puntos más altos que producción:

| Session | Code Default `score_min` | Production `.env` | Delta |
|---|---|---|---|
| overlap | 0.73 | 0.55 | +0.18 |
| london | 0.75 | 0.56 | +0.19 |
| ny | 0.75 | 0.58 | +0.17 |
| asia | 0.80 | 0.62 | +0.18 |
| dead | 1.00 | 0.80 | +0.20 |

El bot produciría muchos menos trades sin error visible.

---

### 10. `_reconcile_sl` y `_check_trailing_stop` compiten en el mismo ciclo

Ambas funciones manipulan el SL en exchange dentro del mismo ciclo de `execute_orders`. Reconcile puede colocar un SL nuevo que trailing inmediatamente reemplaza → 2 cancel+create por tick. Desperdicia rate limit y puede causar errores de "order not found".

---

### 11. Backtest: SL se chequea antes que TP — sesgo pesimista

backtest/engine.py L311-L336: En barras donde el rango cubre tanto SL como TP, siempre gana el SL. En realidad depende de la trayectoria intra-barra. Lo estándar es decidir por cercanía al open. Esto explica parte de la brecha backtest (-0.15%) vs live (+1.51%).

---

### 12. HMM: labels de estado inestables entre refits

signals/regime.py L136-L161: El estado "choppy" se asigna al estado HMM con mayor volatilidad media. Pero no hay persistencia de labels entre refits — el estado 0 puede ser "choppy" en un fit y "trending" en el siguiente, causando oscilaciones en el `risk_mult` sin cambio real de régimen.

---

### 13. GARCH: sin validación de persistence > 1 ni floor de razonabilidad

signals/garch.py: Si α + β > 1 (IGARCH), la volatilidad condicional **explota**. No hay warning ni cap. Además, `blended_vol` no tiene un floor — un GARCH cond_vol de 0.01% con ATR de 0.8% produce un blend de 0.33%, estrechando SL/TP peligrosamente.

---

### 14. `LIVE_GRADUAL_ENABLED` excluye SMC primero

signals/multi_strategy.py L109-L112: `modules[:cap]` trunca por orden de declaración (trend, meanrev, carry, smc). Con `cap=2`, SMC (peso 0.35, el más alto) es el primero en excluirse. Debería ordenar por importancia.

---

## III. HALLAZGOS DE SEGURIDAD

### 15. API keys en texto plano en la DB

core/models.py L47-L49: `api_key`, `api_secret`, `api_passphrase` son `CharField` sin encriptación. Cualquiera con acceso a Django Admin, a un pg_dump, o a un backup puede leer credenciales que dan acceso a fondos reales en KuCoin.

---

### 16. `SECRET_KEY` por defecto = "changeme-in-prod"

config/settings.py L13: Sin validación de que se cambió. Si `.env` no se carga, Django arranca con un secret conocido → session hijacking, CSRF bypass.

---

### 17. `ALLOWED_HOSTS = "*"`

Permite host header injection. En producción con un proxy reverso esto es menos grave, pero sin proxy es un vector de ataque.

---

### 18. API pública con `IsAuthenticatedOrReadOnly`

config/settings.py L675-L679: Cualquier persona que llegue al puerto 8008 puede leer posiciones, señales, balances, sin autenticación.

---

### 19. PostgreSQL expuesto en puerto 5434

Puerto mapeado al host sin restricción. Combinado con credenciales por defecto ("mastertrading"/"mastertrading"), da acceso total a la DB.

---

## IV. HALLAZGOS DE ARQUITECTURA / DEUDA TÉCNICA

### 20. `execute_orders()` = 1,300 líneas en una sola función

execution/tasks.py L2184-L3513: Esta función maneja sync, balances, circuit breaker, loop por instrumento, trailing, SL, TP, signal flip, gates de entrada, sizing, colocación de órdenes, y post-trade logging. Es prácticamente intesteable como unidad. Cualquier cambio tiene alto riesgo de side effects.

---

### 21. `_volatility_adjusted_risk` duplicada en backtest

backtest/engine.py L77-L100 copia la función de execution/tasks.py L825. Si una se actualiza y la otra no, el backtest diverge del live. Ya ha sucedido.

---

### 22. No hay dead letter queue ni alerting de tareas fallidas

Celery sin DLQ significa que si `execute_orders` o `run_signal_engine` fallan, no hay registro, no hay alerta, no hay retry. Para un sistema de trading 24/7, esto es inaceptable.

---

### 23. 15+ constantes hardcoded en execution/tasks.py

| Constante | Valor | Línea |
|---|---|---|
| ATR low vol | 0.008 (0.8%) | L857 |
| ATR high vol | 0.015 (1.5%) | L858 |
| Min ATR scale | 0.6 (60%) | L859 |
| Min move pct (SL update) | 0.0002 (0.02%) | L1327 |
| PnL classify threshold | 0.0025 (0.25%) | L1605 |
| Redis partial/HWM TTL | 172800s (2 days) | L1302 |
| SL too-tight tolerance | 80% | L1196 |
| SL too-wide tolerance | 200% | L1199 |
| Dedup window (sync) | 3 min | L1774 |

Todos deberían ser settings configurables.

---

## V. VALIDACIÓN MATEMÁTICA

### Lo que está BIEN ✅

| Componente | Fórmula | Verificación |
|---|---|---|
| Position sizing | `(R × E) / (SL × P × C)` | Correcto, estándar fixed-fractional |
| ADX Wilder | 14 períodos, smoothing exponencial | Implementación fiel al original |
| Z-score mean reversion | `(close - EMA20 - μ) / σ` sobre 60 barras | Matemáticamente sólido |
| SMC score ponderado | 9 condiciones, pesos suman ~1.0 | Caps correctos, min(1.0, ...) |
| Trailing stop R-multiple | HWM tracking + lock-in 50% | Lógica correcta |
| Volatility ATR ramp | Interpolación lineal 0.8%→1.5% → 1.0x→0.6x | Sound engineering |
| Trend score | `0.50 + min(0.35, gap×35) + min(0.15, (ADX-20)/100)` | Rango [0.50, 1.00] correcto |
| Carry score | `|funding|/threshold - vol_penalty + mr_hint` | Correcto con floor 0.05 |

### Lo que necesita atención ⚠️

| Componente | Problema | Impacto |
|---|---|---|
| Sharpe ratio (backtest) | Usa √(bars_per_day) en vez de √252 sobre retornos diarios | Sobreestima Sharpe |
| Kelly Criterion | f* = -0.002 (edge negativo), pero el sistema opera igual | No hay implementación concreta de fractional Kelly |
| Fees en TP/SL check | PnL calculation ignora fees (~0.06-0.12% round-trip) | Puede causar TP prematuro |
| ATR% (common.py) | Usa SMA en vez de Wilder smoothing | ~1% diferencia, consistente internamente |
| Bounce detection | Usa solo closes, no highs/lows | Subestima magnitud de rebotes con wicks |

---

## VI. CALIFICACIÓN POR ÁREA

| Área | Nota | Comentario |
|---|---|---|
| **Modelo de señales** | **B+** | 4 módulos bien diseñados, math correcta, guards razonables |
| **Allocator** | **B** | Bayesian weights es sofisticado, pero el bug de PER_INSTRUMENT_RISK lo neutraliza |
| **Ejecución** | **C** | Core funcional pero función monolítica, race conditions, sin retry en create_order |
| **Gestión de riesgo** | **C+** | Múltiples capas en papel, pero circuit breaker es pasivo, drawdown checks son in-process |
| **Backtest** | **B-** | Motor completo con paridad mejorada, pero sesgo SL-antes-TP y código duplicado |
| **Infraestructura** | **D+** | Docker funcional pero Redis sin auth, Celery sin protecciones, puertos expuestos |
| **Seguridad** | **D** | API keys plaintext, hosts *, secret key por defecto, API pública |
| **Mantenibilidad** | **C-** | execute_orders 1300 líneas, código duplicado, constantes hardcoded |

---

## VII. PLAN DE ACCIÓN RECOMENDADO (PRIORIZADO)

### Fase Inmediata (esta semana) — Riesgo financiero directo

| # | Acción | Archivo | Esfuerzo |
|---|---|---|---|
| 1 | Agregar `@_retry_exchange` a `create_order` + idempotency check post-retry | adapters/kucoin.py | 1h |
| 2 | Fix `_volatility_adjusted_risk`: usar `min(per_inst, base_risk)` para allocator signals | execution/tasks.py | 1h |
| 3 | Redis auth: agregar `--requirepass` + actualizar `REDIS_URL` | docker-compose.yml, .env | 30m |
| 4 | Celery: `task_acks_late=True`, `task_time_limit=300`, `task_soft_time_limit=240` | config/celery.py | 30m |
| 5 | Signal engine: `autoretry_for=(Exception,)`, `max_retries=3`, `retry_backoff=True` | signals/tasks.py | 30m |

### Fase Corto Plazo (2 semanas) — Calidad operativa

| # | Acción | Archivo | Esfuerzo |
|---|---|---|---|
| 6 | Fix partial close para posiciones fraccionarias (float en vez de int) | execution/tasks.py | 1h |
| 7 | `_classify_exchange_close` proporcional al SL real del trade | execution/tasks.py | 2h |
| 8 | Floor en `blended_vol` + validar GARCH persistence ≤ 1 | signals/garch.py | 1h |
| 9 | Persistir labels HMM entre refits (comparar vols, no indices) | signals/regime.py | 2h |
| 10 | Mover 15+ hardcoded a settings con defaults sanos | execution/tasks.py, config/settings.py | 3h |

### Fase Medio Plazo (4 semanas) — Arquitectura

| # | Acción | Archivo | Esfuerzo |
|---|---|---|---|
| 11 | Descomponer `execute_orders` en ~5 funciones testables | execution/tasks.py | 8h |
| 12 | Extraer `_volatility_adjusted_risk` a módulo compartido | execution/tasks.py, backtest/engine.py | 2h |
| 13 | Encriptar API keys en DB (django-encrypted-model-fields) | core/models.py | 3h |
| 14 | Restringir `ALLOWED_HOSTS`, cambiar DRF a `IsAuthenticated` | config/settings.py | 1h |
| 15 | Agregar DLQ y alerting a Celery | config/celery.py | 4h |

---

## LÍNEA DE FONDO

El sistema tiene una base cuantitativa sólida (señales, sizing, allocator son **B+**). El riesgo real está en la **infraestructura de ejecución** — la ausencia de retry en `create_order`, la falta de protecciones de Celery, y los puertos expuestos. Antes de escalar capital, los puntos 1-5 son **obligatorios**.

---

*Documento generado como parte de la auditoría de código del proyecto MasterTrading.*
*Última actualización: 2026-02-21*
