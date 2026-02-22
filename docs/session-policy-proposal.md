# Session Policy — Análisis y Plan de Implementación

## 1. Análisis del concepto vs. el estado actual del bot

### Lo que ya tenemos (v15 en producción)

| Control | Valor actual | Efecto |
|---------|-------------|--------|
| MIN_SCORE | 0.75 (hardcoded en `_detect_signal()`) | Filtra señales débiles |
| Cooldown | 24h BTC/SOL, 36h ETH | Máximo ~1 trade/día por instrumento |
| MAX_SPREAD_BPS | 20 bps | Ya rechaza spreads anchos |
| Risk per-instrument | BTC 1%, ETH 0.8%, SOL 0.8% | Sizing asimétrico |
| Funding filter | Block si \|rate\| > 0.1% | Evita extremos de crowding |
| Signal TTL | 5 min | No ejecuta señales viejas |
| CHoCH-only + sweep AND gate | Hard gate | Solo señales de alta convicción |

### Impacto real con $25 de equity

Con $25 y contratos mínimos de 1 en KuCoin Futures:
- **BTC**: 1 contrato = 0.001 BTC ≈ $100 notional → ~4x leverage
- **ETH**: 1 contrato = 0.01 ETH ≈ $30 notional → ~1.2x leverage  
- **SOL**: 1 contrato = 0.1 SOL ≈ $20 notional → ~0.8x leverage

**Conclusión**: El `risk_multiplier` por sesión (ej: Asia × 0.6) NO tiene efecto práctico porque ya operamos el mínimo (1 contrato). Solo tendría sentido cuando el equity crezca a >$100 donde el sizing pueda variar entre 1-3+ contratos.

### Frecuencia de trades actual

Con v15 en backtest (Jun 2025 — Feb 2026):
- **535 trades en ~245 días ≈ 2.2 trades/día** (sumando los 3 instrumentos)
- Cooldown de 24-36h ya limita a **máximo 1 trade/día por instrumento**

Agregar filtro de sesiones reduciría aún más el trade count. Con dead zone (17-20h, 3 horas) + Asia más estricta (20-03h, 7 horas), estamos bloqueando ~10 horas = **42% del día**.

---

## 2. Evaluación honesta de cada componente

### 2.1 Dead Zone Block (17:00–20:00 UTC-3) → **ALTO VALOR, IMPLEMENTAR**

- **Costo**: Pierde ~15% de trades potenciales (3h de 20h operativas)
- **Beneficio**: Evita la peor franja de liquidez en crypto
- **Riesgo de sobreajuste**: MUY BAJO — es una regla estructural del mercado, no data-mined
- **Implementación**: Trivial, 5 líneas en `run_signal_engine()`

### 2.2 Score mínimo por sesión → **MEDIO VALOR, VALIDAR ANTES**

- Asia 0.80-0.85, Londres/NY 0.75, Overlap 0.70-0.75
- **Problema**: Con el AND gate actual (CHoCH + sweep + HTF + confirm), los scores típicos ya son 0.75-0.90. No hay mucho rango para discriminar.
- **Riesgo de sobreajuste**: MEDIO — los umbrales exactos necesitan validación OOS
- **Recomendación**: Solo implementar si los backtests separados por sesión muestran diferencia significativa en PF

### 2.3 Risk multiplier por sesión → **BAJO VALOR HOY (equity insuficiente)**

- Con $25, siempre operamos 1 contrato → el multiplicador no cambia nada
- **Cuándo activar**: Cuando equity > $100-150 y el sizing pueda variar
- **Implementación**: Preparar la infraestructura pero desactivar (multiplicadores = 1.0)

### 2.4 Spread/depth guardrails por sesión → **MEDIO-ALTO VALOR**

- Ya tenemos `MAX_SPREAD_BPS=20` global
- Hacer más estricto en Asia/dead zone tiene sentido (ej: 15 bps)
- **Orderbook depth**: Requiere `fetch_order_book()` → carga extra de API, no prioritario
- **Recomendación**: Spread por sesión sí; depth no (complejidad vs beneficio)

### 2.5 Setup type weights por sesión → **NO IMPLEMENTAR**

- Agregar pesos por tipo de setup (range vs continuation) por sesión suma complejidad significativa
- El AND gate actual ya es muy selectivo
- Riesgo de sobreajuste: ALTO
- Mejor mantener la regla simple: CHoCH + sweep siempre

---

## 3. Plan de implementación recomendado (fases)

### Fase 1: Dead Zone + Score mínimo (MVP)

**Cambios mínimos, máximo impacto.**

#### 3.1 Modelo de datos

Opción A — **Settings/env puro** (recomendada para empezar):
```python
# config/settings.py
SESSION_POLICY_ENABLED = os.getenv("SESSION_POLICY_ENABLED", "false").lower() == "true"
SESSION_DEAD_ZONE_BLOCK = os.getenv("SESSION_DEAD_ZONE_BLOCK", "true").lower() == "true"

# Formato JSON: {"asia": 0.82, "london": 0.75, "ny": 0.75, "overlap": 0.72, "dead": 1.0}
_SESSION_SCORE_MIN_RAW = os.getenv("SESSION_SCORE_MIN", '{}')
SESSION_SCORE_MIN = json.loads(_SESSION_SCORE_MIN_RAW) if _SESSION_SCORE_MIN_RAW != '{}' else {}

# Risk multipliers (preparados pero todos 1.0 hasta que equity lo justifique)
_SESSION_RISK_MULT_RAW = os.getenv("SESSION_RISK_MULTIPLIER", '{}')
SESSION_RISK_MULTIPLIER = json.loads(_SESSION_RISK_MULT_RAW) if _SESSION_RISK_MULT_RAW != '{}' else {}
```

Opción B — **Modelo Django** (para poder cambiar desde admin sin restart):
```python
# risk/models.py
class SessionPolicy(models.Model):
    session_name = models.CharField(max_length=20, unique=True)  # asia, london, ny, overlap, dead
    start_hour_utc = models.IntegerField()
    end_hour_utc = models.IntegerField()
    score_min = models.FloatField(default=0.75)
    risk_multiplier = models.FloatField(default=1.0)
    max_spread_bps = models.FloatField(default=20.0)
    enabled = models.BooleanField(default=True)
    block_trading = models.BooleanField(default=False)  # dead zone
```

**Recomendación**: Opción A para empezar (simple, sin migración). Migrar a Opción B si funciona bien.

#### 3.2 Helper de sesión

```python
# signals/sessions.py (nuevo archivo)
from datetime import datetime, timezone

# Sesiones definidas en UTC (Buenos Aires = UTC-3)
# Asia:    23:00–06:00 UTC  (20:00–03:00 UTC-3)
# London:  06:00–14:00 UTC  (03:00–11:00 UTC-3)
# NY:      12:00–20:00 UTC  (09:00–17:00 UTC-3)
# Overlap: 12:00–14:00 UTC  (09:00–11:00 UTC-3)
# Dead:    20:00–23:00 UTC  (17:00–20:00 UTC-3)

SESSIONS = {
    "overlap": (12, 14),  # check first (subset of london+ny)
    "london":  (6, 14),
    "ny":      (14, 20),  # 14-20 after removing overlap
    "dead":    (20, 23),
    "asia":    (23, 6),   # wraps midnight
}

DEFAULT_SCORE_MIN = {
    "overlap": 0.72,
    "london":  0.75,
    "ny":      0.75,
    "asia":    0.82,
    "dead":    1.00,  # effectively blocks (score max is ~0.95)
}

DEFAULT_RISK_MULT = {
    "overlap": 1.0,
    "london":  1.0,
    "ny":      1.0,
    "asia":    0.7,
    "dead":    0.0,  # no trade
}

def get_current_session(utc_hour: int = None) -> str:
    if utc_hour is None:
        utc_hour = datetime.now(timezone.utc).hour
    for name, (start, end) in SESSIONS.items():
        if start < end:
            if start <= utc_hour < end:
                return name
        else:  # wraps midnight (asia)
            if utc_hour >= start or utc_hour < end:
                return name
    return "dead"  # fallback

def get_session_score_min(session: str, overrides: dict = None) -> float:
    if overrides and session in overrides:
        return float(overrides[session])
    return DEFAULT_SCORE_MIN.get(session, 0.75)

def get_session_risk_mult(session: str, overrides: dict = None) -> float:
    if overrides and session in overrides:
        return float(overrides[session])
    return DEFAULT_RISK_MULT.get(session, 1.0)
```

#### 3.3 Integración en `signals/tasks.py`

Punto de inyección en `_detect_signal()` — reemplazar el `MIN_SCORE = 0.75` hardcoded:

```python
# Antes:
MIN_SCORE = 0.75
if score < MIN_SCORE:
    ...

# Después:
from signals.sessions import get_current_session, get_session_score_min
session = get_current_session()
min_score = get_session_score_min(session, getattr(settings, 'SESSION_SCORE_MIN', {}))
explain["session"] = session
explain["session_score_min"] = min_score
if score < min_score:
    explain["result"] = f"score_too_low_for_{session} ({score} < {min_score})"
    return False, "", explain, score
```

#### 3.4 Integración en `execution/tasks.py`

Punto de inyección en sección 3h (open new position), después del cooldown check:

```python
# Aplicar risk_multiplier de sesión al sizing
from signals.sessions import get_current_session, get_session_risk_mult
session = get_current_session()
session_risk_mult = get_session_risk_mult(session, getattr(settings, 'SESSION_RISK_MULTIPLIER', {}))

if session_risk_mult <= 0:
    logger.info("Session %s blocks trading (risk_mult=0)", session)
    continue

# En el risk-based sizing:
qty = _risk_based_qty(equity_usdt, last_price, stop_dist, ...) * session_risk_mult
```

#### 3.5 Integración en backtest engine

Necesario para **validar** el impacto antes de activar en live:

```python
# En el loop del backtest, al evaluar cada señal:
session = get_current_session(bar_ts.hour)  # bar_ts ya es UTC
min_score = get_session_score_min(session, session_overrides)
if score < min_score:
    skip_reasons["session_filter"] += 1
    continue
```

---

### Fase 2: Spread por sesión (posterior)

```python
DEFAULT_MAX_SPREAD = {
    "overlap": 25,   # más permisivo, mejor liquidez
    "london":  20,
    "ny":      20,
    "asia":    15,   # más estricto
    "dead":    10,   # muy estricto (o bloqueado)
}
```

Integrar en `_spread_ok()` pasando el session como parámetro.

### Fase 3: Risk multiplier real (cuando equity > $100)

Activar los multiplicadores de sizing que ya estarán preparados en la infra.

---

## 4. Validación antes de activar

### Backtest necesario (OBLIGATORIO antes de activar)

Correr estos 4 backtests y comparar:

```
# 1. Baseline (v15 actual, sin sesiones)
backtest --name v15_baseline

# 2. Solo dead zone block
backtest --name v17_dead_zone_only

# 3. Dead zone + score por sesión (propuesta conservadora)
backtest --name v17_dead_zone_score

# 4. Todo activado (dead zone + score + risk mult)
backtest --name v17_full_session
```

**Criterios de aceptación:**
- PF debe mantenerse ≥ 1.05 (actualmente 1.088)
- Trade count no debe caer más de 30% (de 535 a ≥ 375)
- Los 3 instrumentos deben seguir siendo positivos
- OOS (Nov-Feb) no debe degradar vs baseline

### Métricas por sesión a recolectar

Antes de implementar, agregar logging para recopilar data en live:

```python
# En run_signal_engine, al emitir señal:
logger.info("Signal emitted session=%s: %s %s score=%.3f", session, inst.symbol, direction, score)

# En execute_orders, al abrir:
logger.info("Trade opened session=%s: %s %s", session, inst.symbol, side)
```

Después de 2-4 semanas de data live, analizar PF por sesión para decidir los umbrales reales.

---

## 5. Valores iniciales recomendados (conservadores)

### .env para Fase 1

```env
# Session policy
SESSION_POLICY_ENABLED=true
SESSION_DEAD_ZONE_BLOCK=true

# Score mínimo por sesión (conservador)
SESSION_SCORE_MIN={"asia": 0.80, "london": 0.75, "ny": 0.75, "overlap": 0.73, "dead": 1.00}

# Risk multipliers (todos 1.0 con equity actual de $25)
SESSION_RISK_MULTIPLIER={"asia": 1.0, "london": 1.0, "ny": 1.0, "overlap": 1.0, "dead": 0.0}
```

### Diferencias clave vs la propuesta original

| Parámetro | Propuesta original | Mi recomendación | Razón |
|-----------|-------------------|-------------------|-------|
| Asia score_min | 0.80–0.85 | **0.80** | Con AND gate ya muy selectivo, 0.85 podría eliminar demasiados trades válidos |
| Overlap score_min | 0.70–0.75 | **0.73** | Bajar mucho riesga degradar PF por señales mediocres |
| Dead zone | score_min=0.85 | **Block total (risk_mult=0)** | Más simple y robusto; 3h sin operar no duele con cooldown de 24h |
| Risk mult Asia | 0.6–0.8 | **1.0** (hoy) | Con $25 no aplica; activar cuando equity > $100 |
| Depth check | Incluido | **No implementar** | Requiere `fetch_order_book`, carga API, complejidad vs beneficio bajo |
| Setup weights | Incluido | **No implementar** | Riesgo alto de sobreajuste; AND gate actual ya es selectivo |

---

## 6. Resumen ejecutivo

| Prioridad | Componente | Impacto esperado | Esfuerzo | Riesgo sobreajuste |
|-----------|------------|-------------------|----------|---------------------|
| **P0** | Dead zone block | +PF (elimina peores trades) | Bajo | Muy bajo |
| **P1** | Session logging (sin filtrar) | Data para decisiones | Mínimo | Ninguno |
| **P2** | Score mínimo por sesión | +PF moderado | Medio | Medio |
| **P3** | Spread por sesión | +PF menor | Bajo | Bajo |
| **P4** | Risk multiplier | +Risk management | Ya preparado | Bajo (cuando aplique) |
| **Skip** | Setup weights por sesión | Incierto | Alto | Alto |
| **Skip** | Orderbook depth | Incierto | Alto | Bajo |

**Recomendación**: Implementar P0 + P1 primero. Recopilar 2-4 semanas de data con logging de sesiones. Luego decidir P2 basándose en evidencia real, no en optimización de backtest.

---
---

# EMA 200 como Filtro de Contexto — Análisis y Plan de Implementación

## 7. Por qué EMA200 encaja con nuestro sistema SMC

La EMA200 es uno de los filtros de contexto más robustos que existen:
- **No es data-mined** — es un estándar de la industria con décadas de uso
- **Complementa SMC** — evita operar CHoCH + sweep "contra la marea" en tendencias claras
- **Ataca nuestro problema OOS-1** — el período Nov-Jan (PF 0.94) fue chop/rango donde operar contra la tendencia macro destruye edge

### Estado actual de nuestro HTF gate

```python
# signals/tasks.py — _detect_signal()
# Hoy: HTF trend determinado por swing highs/lows en 4h
htf_trend = _trend_from_swings(df_htf, period=3)  # "bull", "bear", "range"

# Hard gate: direction no puede ir CONTRA htf_trend
if direction == "long" and htf_trend == "bear": → BLOCK
if direction == "short" and htf_trend == "bull": → BLOCK

# Scoring: htf_trend_aligned tiene peso 0.20
```

**Limitación**: `_trend_from_swings()` con period=3 es reactivo y ruidoso. Cambia rápido en rangos. La EMA200 es más estable y complementaria.

---

## 8. Tres niveles de integración

### Nivel A — Soft Score Weight (RECOMENDADO como MVP)

**No bloquea nada.** Suma o resta al score existente.

```python
# Calcular EMA200 en 1H
ema200 = df_htf["close"].ewm(span=200, adjust=False).mean().iloc[-1]
last_close_htf = df_htf["close"].iloc[-1]
ema_bias_ok = (
    (direction == "long" and last_close_htf > ema200) or
    (direction == "short" and last_close_htf < ema200)
)
```

**Ajuste al scoring engine actual:**

```python
# Pesos actuales (suman 1.00):
weights = {
    "htf_trend_aligned":   0.20,
    "structure_break":     0.20,
    "liquidity_sweep":     0.20,
    "confirmation_candle": 0.05,
    "fvg_aligned":         0.10,
    "order_block":         0.10,
    "funding_ok":          0.10,
    "choch_bonus":         0.05,
}

# Propuesta: agregar ema200_aligned como AJUSTE (no peso)
# Si alineado:   score += 0.07  (bonus)
# Si contra:     score -= 0.12  (penalización asimétrica)
```

**Por qué asimétrico (-0.12 vs +0.07)**:
- Operar CONTRA la EMA200 es más peligroso que operar A FAVOR es beneficioso
- Un trade contra tendencia macro con score 0.78 bajaría a 0.66 → no pasa el MIN_SCORE 0.75
- Un trade a favor con score 0.73 subiría a 0.80 → ahora sí pasa

**Impacto estimado en el scoring**:

| Escenario | Score base | Con EMA200 | Pasa MIN_SCORE? |
|-----------|-----------|-----------|-----------------|
| Confluencia completa + a favor EMA | 0.90 | **0.97** | Sí |
| Confluencia completa + contra EMA | 0.90 | **0.78** | Sí (pero raspa) |
| Sin FVG ni OB + a favor EMA | 0.75 | **0.82** | Sí (cómodo) |
| Sin FVG ni OB + contra EMA | 0.75 | **0.63** | **NO** — filtrado |
| Solo CHoCH+sweep+HTF + a favor | 0.70 | **0.77** | Sí (rescatado) |
| Solo CHoCH+sweep+HTF + contra | 0.70 | **0.58** | **NO** — filtrado |

**Esto es exactamente lo que queremos**: los trades mediocres contra tendencia mueren, los buenos a favor se refuerzan.

### Nivel B — Hard Filter (más agresivo, NO recomendado como primera opción)

```python
# Solo buscar LONG si close > EMA200
# Solo buscar SHORT si close < EMA200
if direction == "long" and last_close_htf < ema200:
    return False, "", explain, 0.0
```

**Problema**: Mata reversiones SMC legítimas (CHoCH en fondo/techo). Un sweep_low + CHoCH_bull cuando precio está DEBAJO de EMA200 es precisamente una señal de reversión — que es la esencia de SMC.

**Cuándo usarlo**: Solo si los backtests muestran que el Nivel A no es suficiente y el PF de trades contra-EMA es consistentemente < 0.8.

### Nivel C — Régimen por pendiente de EMA200 (avanzado, Fase 2+)

```python
# Pendiente de EMA200 como indicador de régimen
ema200_values = df_htf["close"].ewm(span=200, adjust=False).mean()
ema_now = ema200_values.iloc[-1]
ema_prev = ema200_values.iloc[-10]  # 10 velas atrás (10h en 1H, ~2 días en 4H)
ema_slope = (ema_now - ema_prev) / ema_prev

# Clasificación de régimen
if abs(ema_slope) < 0.005:      # < 0.5% en 10 velas
    regime = "range"
elif ema_slope > 0.005:
    regime = "trend_up"
else:
    regime = "trend_down"
```

**Acciones por régimen:**

| Régimen | Regla | Efecto |
|---------|-------|--------|
| `trend_up` | Solo longs (o shorts penalizados -0.15) | Sigue la tendencia macro |
| `trend_down` | Solo shorts (o longs penalizados -0.15) | Sigue la tendencia macro |
| `range` | Ambos lados, pero score_min sube a 0.82 | Más selectivo en chop |

**Valor**: Este nivel ataca directamente el problema OOS-1 (Nov-Jan rango). Pero añade un parámetro más (slope threshold = 0.005) que necesita validación.

---

## 9. Implementación concreta — Nivel A (MVP)

### 9.1 Cambios en `signals/tasks.py`

#### En `_detect_signal()` — calcular EMA200:

```python
def _detect_signal(df_ltf, df_htf, funding_rates):
    # ... código existente hasta después de determinar direction ...

    # NEW: EMA200 bias check on HTF
    ema200_bias = None
    if len(df_htf) >= 200:
        ema200 = df_htf["close"].ewm(span=200, adjust=False).mean().iloc[-1]
        last_htf_close = df_htf["close"].iloc[-1]
        ema200_bias = "above" if last_htf_close > ema200 else "below"
        explain["ema200"] = {
            "value": round(float(ema200), 2),
            "htf_close": round(float(last_htf_close), 2),
            "bias": ema200_bias,
        }
    else:
        explain["ema200"] = "insufficient_data"
```

#### En `_compute_score()` — agregar como ajuste:

```python
def _compute_score(conditions: dict) -> float:
    weights = {
        "htf_trend_aligned":   0.20,
        "structure_break":     0.20,
        "liquidity_sweep":     0.20,
        "confirmation_candle": 0.05,
        "fvg_aligned":         0.10,
        "order_block":         0.10,
        "funding_ok":          0.10,
        "choch_bonus":         0.05,
    }

    score = 0.0
    for key, weight in weights.items():
        if conditions.get(key, False):
            score += weight

    # EMA200 adjustment (asymmetric: penalty > bonus)
    ema200_alignment = conditions.get("ema200_aligned")  # True/False/None
    if ema200_alignment is True:
        score += 0.07
    elif ema200_alignment is False:
        score -= 0.12

    return round(max(min(score, 1.0), 0.0), 3)
```

#### En conditions dict — agregar la condición:

```python
    # Después de las conditions existentes:
    if ema200_bias is not None:
        conditions["ema200_aligned"] = (
            (direction == "long" and ema200_bias == "above") or
            (direction == "short" and ema200_bias == "below")
        )
    else:
        conditions["ema200_aligned"] = None  # no data, no adjustment
```

### 9.2 Cambios en `backtest/engine.py`

Mismo cálculo de EMA200 en el loop del backtest para poder validar:

```python
# En el loop por barra, después de calcular df_htf:
ema200_bias = None
if len(df_htf) >= 200:
    ema200 = df_htf["close"].ewm(span=200, adjust=False).mean().iloc[-1]
    ema200_bias = "above" if df_htf["close"].iloc[-1] > ema200 else "below"

# Pasar al scoring:
conditions["ema200_aligned"] = (
    (direction == "long" and ema200_bias == "above") or
    (direction == "short" and ema200_bias == "below")
) if ema200_bias else None
```

### 9.3 Consideración importante: datos suficientes

EMA200 en 1H requiere **200 velas de 1H = 8.3 días** de datos previos.

En 4H requiere **200 × 4h = 33 días** de datos previos.

- Nuestro backtest empieza Jun 2025 con datos desde antes → OK
- En live, `fetch_ohlcv` carga 200 velas → OK
- El `_latest_candles(inst, "1h", lookback=240)` ya carga 240 → OK para EMA200

---

## 10. Combinación EMA200 + Session Policy

Si se implementan ambos features, la penalización por contra-EMA puede variar por sesión:

```python
# Penalización EMA200 diferenciada por sesión
EMA_COUNTER_PENALTY = {
    "asia":    -0.15,   # Más castigo: sin follow-through contra tendencia
    "london":  -0.12,   # Estándar
    "ny":      -0.10,   # Más liquidez, reversiones más viables
    "overlap": -0.07,   # Máximo follow-through, reversiones más legítimas
    "dead":    -0.20,   # Máximo castigo (o directamente no operar)
}

EMA_ALIGNED_BONUS = {
    "asia":    +0.05,
    "london":  +0.07,
    "ny":      +0.07,
    "overlap": +0.08,
    "dead":    +0.03,
}
```

**Mi recomendación**: NO combinar ambos en la primera versión. Implementar EMA200 primero (solo), medir, y luego considerar la interacción con sesiones. Demasiados ajustes simultáneos hacen imposible saber qué mejoró qué.

---

## 11. Settings/env para EMA200

```env
# EMA200 filter
EMA200_ENABLED=true
EMA200_TIMEFRAME=1h          # HTF para calcular EMA200
EMA200_ALIGNED_BONUS=0.07    # Score bonus cuando alineado
EMA200_COUNTER_PENALTY=0.12  # Score penalty cuando contra (sin signo negativo)
EMA200_HARD_FILTER=false     # Si true, bloquea trades contra EMA200 (Nivel B)
```

```python
# config/settings.py
EMA200_ENABLED = os.getenv("EMA200_ENABLED", "false").lower() == "true"
EMA200_TIMEFRAME = os.getenv("EMA200_TIMEFRAME", "1h")
EMA200_ALIGNED_BONUS = float(os.getenv("EMA200_ALIGNED_BONUS", "0.07"))
EMA200_COUNTER_PENALTY = float(os.getenv("EMA200_COUNTER_PENALTY", "0.12"))
EMA200_HARD_FILTER = os.getenv("EMA200_HARD_FILTER", "false").lower() == "true"
```

---

## 12. Validación — Backtests necesarios

### Tests obligatorios antes de activar

```bash
# 1. Baseline v15 (sin EMA200)
backtest --name v15_baseline

# 2. EMA200 Nivel A (score weight, bonus=0.07, penalty=0.12)
backtest --name v18_ema200_soft

# 3. EMA200 Nivel B (hard filter, bloquea contra-EMA)
backtest --name v18_ema200_hard

# 4. EMA200 + Dead zone (combinado)
backtest --name v18_ema200_deadzone
```

### Métricas clave a comparar

| Métrica | v15 baseline | v18 ema soft | v18 ema hard | Objetivo |
|---------|-------------|-------------|-------------|----------|
| PF | 1.088 | ≥ 1.10 | ? | Subir |
| Trades | 535 | ≥ 400 | ? | No caer >25% |
| Max DD | ? | ≤ baseline | ? | Bajar o mantener |
| PF OOS-1 (Nov-Jan) | 0.94 | > 1.0 | ? | **Clave**: mejorar el peor período |
| Sharpe | 2.07 | ≥ 2.0 | ? | Mantener |
| Los 3 instrumentos positivos | Sí | Sí | ? | Obligatorio |

### Señales de que funciona

- PF sube especialmente en OOS-1 (rango/chop)
- DD máximo baja
- Trade count no cae más de 25%
- No se eliminan las mejores reversiones SMC

### Señales de que NO funciona

- PF cae (los trades filtrados eran buenos)
- Trade count cae >30% (demasiado restrictivo con el AND gate actual)
- Los instrumentos más volátiles (SOL) se vuelven negativos

---

## 13. Orden de implementación global actualizado

| Prioridad | Feature | Impacto | Esfuerzo | Riesgo sobreajuste |
|-----------|---------|---------|----------|---------------------|
| **P0** | Dead zone block | Alto | Bajo | Muy bajo |
| **P0** | EMA200 Nivel A (score weight) | Alto | Medio | **Bajo** (estándar de mercado) |
| **P1** | Session + EMA200 logging | Data | Mínimo | Ninguno |
| **P2** | Score mínimo por sesión | Medio | Medio | Medio |
| **P3** | EMA200 Nivel C (régimen slope) | Medio-Alto | Medio | Medio |
| **P4** | EMA200 penalty por sesión | Medio | Bajo | Medio |
| **P5** | Risk multiplier por sesión | Bajo (hoy) | Ya preparado | Bajo |
| **Skip** | EMA200 Nivel B (hard filter) | Incierto | Bajo | Medio |
| **Skip** | Setup weights por sesión | Incierto | Alto | Alto |
| **Skip** | Orderbook depth | Incierto | Alto | Bajo |

**Plan recomendado**:
1. Implementar EMA200 Nivel A en backtest engine → validar con los 4 tests
2. Si PF mejora: implementar en live (`signals/tasks.py`)
3. Simultáneamente: agregar dead zone block + logging de sesiones
4. Después de 2-4 semanas de data: evaluar P2-P4 basándose en evidencia
