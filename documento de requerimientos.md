# Requerimientos (Funcionales y Técnicos) — Bot de Trading de Futuros Cripto (SMC) en Docker + Python + Django

> Documento de alcance para construir un **bot de trading de futuros/perpetuos** en **criptomonedas top 10**, usando señales inspiradas en **Smart Money Concepts (SMC)** + filtros cuantitativos (microestructura, derivados, estacionalidad, correlación macro) y un stack **Docker + Python + Django**.  
> **Disclaimer:** esto es especificación técnica/funcional, no asesoramiento financiero.

---

## 1) Objetivo del sistema

Construir un sistema que:
- Obtenga datos de mercado en tiempo real e históricos (spot + futuros/perps) para el **top 10** de criptos por capitalización (y/o un top 10 configurable por liquidez en derivados).
- Calcule señales basadas en SMC (estructura, BOS/CHoCH, zonas de liquidez, sweeps, FVG, order blocks) **cuantificadas**.
- Ejecute órdenes en exchanges (futuros/perps) con controles de riesgo estrictos.
- Permita backtesting, paper trading y modo live.
- Ofrezca observabilidad, auditoría y configurabilidad vía API/UI (Django Admin o panel).

---

## 2) Alcance funcional (FR)

### FR-1 Gestión de instrumentos (Top 10)
- El sistema debe soportar una lista de instrumentos objetivo: `{BTC, ETH, BNB, XRP, SOL, TRX, DOGE, BCH, ...}` y permitir excluir stablecoins si aparecen en top cap.
- Debe permitir dos modos:
  - **Top por market cap**
  - **Top por liquidez en derivados** (volumen, OI, spreads, profundidad)
- Debe soportar múltiples símbolos por exchange (ej.: `BTCUSDT`, `ETHUSDT`, `BTC-PERP`).

**Criterios de aceptación**
- El bot opera solo en símbolos habilitados y valida que existan en el exchange.
- Permite configurar whitelist/blacklist.

---

### FR-2 Ingesta de datos (Market Data)
- Debe consumir:
  - **OHLCV** multi-timeframe (1m, 5m, 15m, 1h, 4h, 1d)
  - **Order book (L2)**: bids/asks y profundidad por niveles
  - **Trades (tape)**: prints, volumen, agresión (si el exchange lo provee)
  - **Derivados**: funding rate (actual e histórico), mark price, index price, open interest (si disponible), tasas de liquidación (si hay feed)
- Debe soportar:
  - **WebSocket** (tiempo real)
  - **REST** (snapshots y fallback)
- Debe normalizar timestamps a UTC.

**Criterios de aceptación**
- Latencia controlada (p.ej. < 1s para streams WS en condiciones normales).
- Mecanismo de reconexión y re-suscripción automático.
- Persistencia confiable de velas y snapshots.

---

### FR-3 Motor de señales (SMC + cuant)
El sistema debe detectar y computar, al menos:

#### FR-3.1 Market Structure
- Swing highs/lows por timeframe.
- **BOS** (break of structure) y **CHoCH** (change of character).
- Tendencia/rango por HTF (ej. 4H/1D).

#### FR-3.2 Liquidez y “sweeps”
- Identificar clusters de liquidez (máximos/mínimos cercanos, equal highs/lows).
- Detectar **liquidity sweep**: ruptura + rechazo (reversión rápida).

#### FR-3.3 Imbalance / FVG
- Detectar gaps de valor justo (FVG) por reglas definidas (3 velas, desplazamiento y vacío relativo).
- Medir tamaño del gap y “fill ratio”.

#### FR-3.4 Order Blocks (OB)
- Detectar OB “último bloque contrario antes del impulso” con validaciones:
  - desplazamiento fuerte posterior
  - mitigación posterior (retest)
  - filtro de contexto (HTF)

#### FR-3.5 Filtros de derivados
- Funding:
  - extremos (percentiles) y reversión hacia media
  - evitar entrar cuando el carry es adverso extremo (configurable)
- Mark vs last:
  - señales se basan en precio (last/mid), pero **riesgo** se evalúa en **mark**.

#### FR-3.6 Microestructura / ejecución
- Spread, profundidad al X% (0.1%, 0.5%, 1%).
- Order book imbalance.
- Señal de “libro fino” (thin book) para limitar trading.

#### FR-3.7 Estacionalidad
- Matriz (día x hora) por activo:
  - volatilidad realizada
  - spread medio
  - profundidad
- Reglas tipo:
  - reducir tamaño en ventanas históricamente ilíquidas
  - reglas específicas para fin de semana

#### FR-3.8 Correlación macro / risk-on risk-off
- Calcular correlación rodante de BTC/ETH con:
  - índices tradicionales (p.ej. SPX/NQ) o proxies disponibles
- Detectar regímenes:
  - correlación alta (crypto sigue bolsa)
  - descorrelación (crypto independiente)

**Criterios de aceptación**
- Todas las señales deben poder habilitarse/deshabilitarse por config.
- La señal final debe incluir un “explain payload” (qué condiciones se cumplieron).

---

### FR-4 Estrategias: Long y Short (plantillas configurables)

> Nota: se describen como **algoritmos** (reglas), no como promesa de rentabilidad.

#### FR-4.1 Estrategia Long (Sweep + CHoCH + Mitigación)
Condiciones mínimas:
1. Contexto HTF: rango o downtrend extendido.
2. Sweep de low relevante (LTF) y rechazo.
3. CHoCH/BOS alcista en LTF.
4. Entrada en mitigación:
   - retest a OB o FVG con confirmación (cierre/volumen/estructura)
5. Filtros:
   - funding no extremo en contra
   - liquidez mínima (spread y profundidad OK)
   - evitar ventanas de noticias (si módulo macro activo)

Stops/TP:
- Stop debajo del swing del sweep + buffer por ATR/volatilidad.
- TP por:
  - siguiente pool de liquidez (prev highs)
  - R múltiplos (1R/2R/3R) o trailing.

#### FR-4.2 Estrategia Short (Sweep high + BOS bajista)
Condiciones mínimas:
1. Contexto HTF: distribución o tendencia bajista.
2. Sweep de high y giro bajista.
3. BOS bajista.
4. Entrada en retest a OB de oferta o fill parcial de FVG.
5. Filtros:
   - funding no extremadamente negativo (crowded shorts)
   - liquidez mínima
   - control fin de semana

**Criterios de aceptación**
- El motor debe poder correr múltiples estrategias en paralelo con “conflict resolver”.

---

### FR-5 Gestión de órdenes y ejecución
- Soportar:
  - market, limit, stop-market, stop-limit (según exchange)
  - reduce-only
  - post-only (si aplica)
- Gestión de slippage:
  - estimación previa (book depth)
  - abortar o degradar tamaño si slippage esperado excede umbral
- Reintentos idempotentes:
  - evitar duplicar órdenes ante timeouts

**Criterios de aceptación**
- Cada orden debe quedar registrada con correlation_id.
- Debe ser posible reconstruir el estado de una posición con logs + DB.

---

### FR-6 Gestión de riesgo
Mínimos obligatorios:
- **Max leverage efectivo**: `sum(nocional)/equity`.
- **Riesgo por trade**: % de equity (p.ej. 0.25%–1% configurable).
- **Límites diarios/semanales** (max drawdown / max loss).
- **Kill-switch** automático por:
  - desconexión datos
  - latencia alta
  - errores de orden repetidos
  - equity cercano a margin maintenance
- Control de exposición:
  - por instrumento
  - por lado (net long/net short)
  - por correlación (cluster risk)

**Criterios de aceptación**
- El bot no debe abrir nuevas posiciones si se activa un guardrail.
- Debe cerrar o reducir posiciones según reglas de emergencia.

---

### FR-7 Modos de operación
- **Backtesting** (histórico reproducible)
- **Paper trading** (simulado con datos reales)
- **Live trading**
- Replay:
  - reproducir una sesión con mismos datos para debugging

**Criterios de aceptación**
- Misma lógica de señales en backtest/paper/live (cambio solo en “broker adapter”).

---

### FR-8 Configuración y control (Django)
- Panel (admin o UI) para:
  - activar instrumentos
  - setear parámetros (risk, filtros, timeframes)
  - activar/desactivar estrategias
  - ver posiciones, PnL, logs, alertas
- API REST para:
  - estado del bot
  - métricas y señales
  - órdenes/posiciones (solo lectura o controlado por permisos)

**Criterios de aceptación**
- Roles: admin / operador / solo lectura.
- Auditoría: cambios de config versionados.

---

### FR-9 Observabilidad y alertas
- Métricas:
  - latencia WS
  - reconexiones
  - tasa de errores de órdenes
  - PnL, drawdown, exposición
- Logs estructurados (JSON):
  - señal generada
  - decisión
  - orden enviada
  - fill
- Alertas:
  - Telegram/Slack/Email (configurable)
  - eventos críticos (kill-switch, margin risk, caídas de feed)

---

### FR-10 Cumplimiento y seguridad operativa
- Gestión segura de credenciales (API keys):
  - nunca en código
  - via secrets (env + vault si aplica)
- Rate limits y políticas del exchange:
  - respetar límites y backoff
- Cumplimiento jurisdiccional:
  - el bot debe permitir “modo sin trading” para análisis si el usuario no puede operar derivados.

---

## 3) Alcance técnico (TR / NFR)

### NFR-1 Arquitectura (alto nivel)
Componentes recomendados:
1. **Django API + Admin** (control, configs, reporting)
2. **Worker(s) de trading** (motor de señales + ejecución)
3. **Market Data Service** (WS/REST, normalización, cache)
4. **Scheduler** (calendario macro, funding times, tareas)
5. **DB** (PostgreSQL)
6. **Cache/cola** (Redis)
7. **Message broker opcional** (RabbitMQ/Kafka si escala)
8. **Observabilidad** (Prometheus + Grafana / OpenTelemetry)

---

### NFR-2 Contenerización (Docker)
Debe existir:
- `docker-compose.yml` con servicios:
  - `web` (Django)
  - `worker` (Celery/RQ/management command)
  - `postgres`
  - `redis`
  - (opcional) `nginx`
- Imágenes reproducibles:
  - `Dockerfile` multi-stage
- Variables por entorno:
  - `.env` (no commiteado)
  - `.env.example`

---

### NFR-3 Stack Python/Django (sugerido)
- Python 3.11+
- Django + Django REST Framework
- Celery + Redis (o RQ)
- Librerías:
  - `ccxt` (si se usa unificación de exchanges) o SDK oficial
  - `pandas`, `numpy`
  - `pydantic` para schemas de eventos
  - `sqlalchemy` opcional (si se separa del ORM)
  - `httpx` / `websockets`
  - `tenacity` (retries)
  - `prometheus-client` o OpenTelemetry

---

### NFR-4 Datos y almacenamiento
Tablas mínimas:
- `Instrument` (símbolos, exchange, activo)
- `Candle` (OHLCV por timeframe)
- `OrderBookSnapshot` (opcional, si se persiste)
- `FundingRate` (histórico)
- `Signal` (output del motor)
- `StrategyConfig` (versionado)
- `Order` (estado, exchange_id, correlation_id)
- `TradeFill`
- `Position` (net, avg price, unrealized/realized)
- `RiskEvent` (kill-switch, drawdown, etc.)
- `AuditLog` (cambios config)

Política de retención:
- ticks / orderbook: corto plazo o almacenamiento frío
- velas/funding: largo plazo
- compresión/particionado por fecha

---

### NFR-5 Confiabilidad y consistencia
- Reintentos con backoff exponencial.
- Idempotencia en creación de órdenes.
- Recuperación tras reinicio:
  - reconstruir estado desde exchange + DB
- Tolerancia a fallos:
  - si WS cae, usar REST temporalmente
  - si datos incompletos, deshabilitar trading (“data unhealthy”)

---

### NFR-6 Rendimiento
- Señales en timeframes pequeños requieren eficiencia:
  - cache en Redis para features
  - procesamiento incremental (no recalcular todo)
- Metas iniciales:
  - 10 instrumentos, 1m–15m, sin HFT
  - ciclo de decisión < 1s (best-effort)

---

### NFR-7 Seguridad
- Secrets en Docker secrets / Vault (ideal).
- API con JWT/OAuth2, rate limiting.
- Permisos estrictos para endpoints de “trading”.
- Registro de acciones (auditoría).

---

## 4) Especificación matemática mínima del bot (para implementación)

### 4.1 Variables de estado
Por instrumento `i`:
- `M_i(t)`: mark price
- `P_i(t)`: last/mid price
- `q_i(t)`: posición (positivo long, negativo short)
- `E(t)`: equity (balance + PnL - fees - funding)
- `N_i(t) = |q_i(t)| * M_i(t)`: nocional
- `L_eff(t) = sum_i N_i(t) / E(t)`: leverage efectivo

### 4.2 Riesgo por trade
- Riesgo nominal por operación:
  - `R = r * E(t)` donde `r` es % riesgo (configurable)
- Tamaño por stop distance:
  - `size = R / stop_distance`
- `stop_distance` puede basarse en:
  - swing + buffer ATR: `stop = swing ± k * ATR`
  - volatilidad intradía estimada

### 4.3 Funding como costo esperado
- Para funding rate `f_i` en el próximo settlement:
  - `FundingCost ≈ N_i(t) * f_i`
- El motor debe incluir el funding esperado en el “edge”:
  - `EV_net = EV_signal - fees - expected_slippage - expected_funding`

### 4.4 Control por liquidez
- Profundidad al 1%: `Depth_1%`
- Umbral mínimo:
  - si `Depth_1% < D_min` o `spread > S_max` ⇒ no operar o reducir size

---

## 5) Integraciones externas (requerimientos)

### EX-1 Exchange Adapter
Debe soportar al menos un exchange de futuros (configurable):
- Autenticación API Key/Secret.
- Endpoints:
  - market data (WS/REST)
  - create/cancel order
  - fetch open orders / positions / balances
  - funding / mark price

### EX-2 Data providers opcionales
- Precios agregados / OI / derivados (si se requiere redundancia).
- Calendario macroeconómico (para ventanas de riesgo):
  - guardar eventos (CPI, NFP, FOMC, etc.) y activar reglas.

---

## 6) Entregables del proyecto

1. Repositorio con:
   - `docker-compose.yml`, `Dockerfile`s
   - proyecto Django
   - workers y adapters
2. Documentación:
   - README (setup, configuración, seguridad)
   - runbooks (operación, incidentes)
3. Suite de testing:
   - unit tests (señales, riesgo)
   - integration tests (exchange sandbox si existe)
4. Backtesting:
   - scripts y dataset mínimo
   - reporte de performance (Sharpe, DD, winrate, etc.)
5. Observabilidad:
   - métricas base + dashboards (si aplica)

---

## 7) Criterios de “MVP” (primera versión)
- 3 instrumentos (BTC/ETH/SOL) en 1 exchange.
- Timeframes: 5m + 1h.
- 1 estrategia long + 1 short (sweep+CHoCH) con filtros básicos:
  - funding, spread, fin de semana.
- Paper trading completo.
- UI mínima en Django Admin.
- Kill-switch y límites diarios.

---

## 8) Riesgos y consideraciones
- Riesgo de liquidación por apalancamiento y gaps de liquidez (especialmente fines de semana).
- Cambios en APIs / rate limits de exchanges.
- Necesidad de datos confiables (mark/index/funding).
- SMC es interpretativo: requiere definición **determinista** para automatizarlo.
- Evitar sobre-optimización (walk-forward, out-of-sample).

---
