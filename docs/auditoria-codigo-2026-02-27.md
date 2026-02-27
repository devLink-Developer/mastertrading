# AUDITORÍA INTEGRAL — MasterTrading (Actualización)

**Auditor:** Perfil Supervisor / Analista Cuantitativo
**Fecha:** 2026-02-27
**Alcance:** Revisión de hallazgos anteriores, análisis de gestión de riesgo, y búsqueda de nuevas oportunidades de edge.

---

## ÍNDICE

1. [Estado de Hallazgos Anteriores (2026-02-21)](#1-estado-de-hallazgos-anteriores-2026-02-21)
2. [Nuevos Hallazgos y Correcciones Implementadas](#2-nuevos-hallazgos-y-correcciones-implementadas)
3. [Evaluación de Modelos Cuantitativos (Fase 3)](#3-evaluación-de-modelos-cuantitativos-fase-3)
4. [Oportunidades de Mejora (Edge & Ganancias)](#4-oportunidades-de-mejora-edge--ganancias)
5. [Plan de Acción Recomendado](#5-plan-de-acción-recomendado)

---

## 1. ESTADO DE HALLAZGOS ANTERIORES (2026-02-21)

Se verificó la resolución de los hallazgos críticos reportados en la auditoría del 21 de febrero:

*   ✅ **`create_order` sin retry:** Solucionado. La función en `adapters/kucoin.py` ahora utiliza `_create_order_with_retry` con el decorador `@_retry_exchange`, mitigando el riesgo de órdenes perdidas o posiciones fantasma.
*   ✅ **Redis expuesto:** Solucionado. El archivo `docker-compose.yml` ahora requiere contraseña (`--requirepass "$${REDIS_PASSWORD}"`).
*   ✅ **Celery sin `acks_late`:** Solucionado. `config/celery.py` ahora incluye `task_acks_late=True`, `task_time_limit` y `task_reject_on_worker_lost=True`, asegurando que las tareas fallidas se reencolen.
*   ✅ **Signal engine sin retry:** Solucionado. `run_signal_engine` en `signals/tasks.py` ahora tiene el decorador `@shared_task(autoretry_for=(Exception,), retry_backoff=True, max_retries=3, acks_late=True)`.

---

## 2. NUEVOS HALLAZGOS Y CORRECCIONES IMPLEMENTADAS

Durante esta auditoría se detectaron y corrigieron inmediatamente dos problemas importantes que afectaban la gestión de riesgo y la protección del capital.

### 2.1. Bug Crítico en `volatility_adjusted_risk` (Riesgo no escalado)
*   **Archivo:** `execution/risk_policy.py`
*   **Problema:** La función `volatility_adjusted_risk` tenía un *early return* si el símbolo estaba configurado en `PER_INSTRUMENT_RISK`. Esto significaba que para los activos principales (BTC, SOL, LINK, ADA), **el riesgo NUNCA se reducía por volatilidad**. El sistema ignoraba por completo el escalado de ATR para estos activos, exponiendo la cuenta a pérdidas desproporcionadas durante picos de volatilidad.
*   **Solución Implementada:** Se corrigió la lógica para que asigne el riesgo base del instrumento (`effective_base`) y luego **continúe** hacia la lógica de escalado por ATR. Ahora todos los instrumentos respetan la reducción de riesgo en alta volatilidad.

### 2.2. Brecha de Seguridad: Falta de `Downtrend Long Killer`
*   **Archivo:** `execution/tasks.py`
*   **Problema:** El sistema contaba con un `UPTREND_SHORT_KILLER_ENABLED` que cerraba posiciones en corto si la tendencia HTF (4h) se volvía alcista. Sin embargo, **no existía la protección simétrica para los longs**. Sabiendo que los longs han tenido un peor rendimiento reciente (Longs: -0.58% PnL vs Shorts: +2.09% PnL), esta era una brecha importante que dejaba posiciones largas expuestas a caídas macro.
*   **Solución Implementada:** Se añadió el bloque `DOWNTREND_LONG_KILLER_ENABLED` en `_manage_open_position`. Ahora, si hay un long abierto y la tendencia macro (4h) cambia a bajista (`bear`), el sistema cerrará la posición inmediatamente a mercado para proteger el capital.

---

## 3. EVALUACIÓN DE MODELOS CUANTITATIVOS (FASE 3)

Se revisó la integración de los nuevos modelos avanzados implementados recientemente. Su estado es excelente y matemáticamente sólido:

*   **GARCH(1,1) Volatility Forecast:** La mezcla de volatilidad (`blended_vol`) entre ATR (backward-looking) y GARCH (forward-looking) está correctamente implementada en `signals/garch.py` y se inyecta de forma segura en el cálculo de TP/SL y *position sizing* en `execution/tasks.py`.
*   **HMM Regime Detection:** La detección de régimen (Trending vs Choppy) mediante `GaussianHMM` en `signals/regime.py` está bien acoplada al *allocator* y actúa como un multiplicador de riesgo efectivo, reduciendo la exposición en mercados laterales.
*   **Pesos Dinámicos (Bayesianos):** El modelo Beta-Binomial en `signals/allocator.py` es robusto. Utiliza un prior Beta(2,2) (débilmente informativo, centrado en 0.5) y ajusta los pesos de los módulos basándose en su *win rate* reciente, penalizando a los módulos con bajo rendimiento y premiando a los exitosos.

---

## 4. OPORTUNIDADES DE MEJORA (EDGE & GANANCIAS)

Basado en el análisis del código y el rendimiento histórico (Longs sufriendo, Shorts ganando, alta tasa de cierres por exchange), se proponen las siguientes mejoras para aumentar el *edge*:

### 4.1. Asimetría en Take Profit (TP)
*   **Contexto:** Actualmente, el cálculo de TP (`_compute_tp_sl_prices`) es simétrico para longs y shorts. Sin embargo, los mercados cripto tienden a caer más rápido de lo que suben.
*   **Propuesta:** Implementar un multiplicador de TP asimétrico. Permitir que los shorts tengan un TP ligeramente más amplio (ej. `ATR_MULT_TP_SHORT = 2.0`) y los longs un TP más conservador (`ATR_MULT_TP_LONG = 1.6`) para asegurar ganancias más rápido en subidas lentas.

### 4.2. Refinamiento del Módulo SMC (Smart Money Concepts)
*   **Contexto:** El módulo SMC tiene muchos *gates* (14 en total). Aunque esto filtra ruido, puede estar bloqueando entradas válidas de alta calidad.
*   **Propuesta:** 
    *   Revisar el impacto del `SMC_ADX_MIN` (actualmente 18.0). SMC a menudo funciona bien en rangos amplios donde el ADX es bajo. Considerar relajar este filtro si hay confluencia de FVG y Order Block.
    *   Evaluar la reactivación de los shorts en SMC (si estaban penalizados), dado que los shorts tienen mejor *win rate* general en el sistema.

### 4.3. Optimización del Trailing Stop
*   **Contexto:** El trailing stop actual se activa a 2.5R (`TRAILING_STOP_ACTIVATION_R`).
*   **Propuesta:** Implementar un trailing stop dinámico basado en la volatilidad (GARCH/ATR) en lugar de un R fijo. En regímenes de alta volatilidad, activar el trailing antes (ej. 1.5R) para proteger ganancias rápidas que tienden a revertirse bruscamente.

### 4.4. Reactivación del Filtro de Machine Learning (ML)
*   **Contexto:** El filtro de entrada ML (`execution/ml_entry_filter.py`) está actualmente deshabilitado.
*   **Propuesta:** Reentrenar el modelo de Regresión Logística con los datos de las últimas semanas (incluyendo las nuevas features de GARCH y HMM) y evaluar su rendimiento en *shadow mode* (solo logueando predicciones sin bloquear trades) para ver si mejora el *win rate* base del 48.5%.

---

## 5. PLAN DE ACCIÓN RECOMENDADO

1.  **Inmediato:**
    *   Añadir `DOWNTREND_LONG_KILLER_ENABLED=true` al archivo `.env` de producción para activar la nueva protección de longs.
    *   Desplegar los cambios realizados en `execution/risk_policy.py` y `execution/tasks.py`.
2.  **Corto Plazo (1 semana):**
    *   Implementar la asimetría de Take Profit (TP) en `_compute_tp_sl_prices`.
    *   Analizar en backtest el impacto de relajar el filtro ADX en el módulo SMC.
3.  **Mediano Plazo (2-3 semanas):**
    *   Reentrenar y evaluar el modelo de Machine Learning en *shadow mode*.
    *   Desarrollar el trailing stop dinámico basado en volatilidad.