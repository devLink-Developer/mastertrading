# Skills de Trading Priorizadas (Aplicacion en MasterTrading)

Base tomada de tu ranking en `skills.sh` (query `trading`) y del estado actual del bot.

## Top 3 recomendadas

1. `backtesting-trading-strategies`
2. `risk-management`
3. `market-regimes`

## Por que estas 3

- `backtesting-trading-strategies`: mejora validacion y evita overfitting antes de pasar cambios a demo/live.
- `risk-management`: reduce drawdowns y reentradas malas con filtros de ejecucion y control de exposicion.
- `market-regimes`: evita operar igual en compresion, tendencia y lateralidad.

## Traduccion concreta al repo

1. Backtesting
- Mantener matriz de pruebas por ventana (baseline vs cambio) con `backtest/management/commands/optimize_walkforward.py`.
- Regla operativa: no promover un cambio si empeora `profit factor` o `max drawdown` en walk-forward.

2. Risk management
- Cambio aplicado: filtro de volumen en ejecucion para bloquear entradas con baja participacion.
- Archivos:
  - `execution/tasks.py` (`_volume_activity_ratio`, `_volume_gate_allowed`, gate en `execute_orders`)
  - `config/settings.py` (nuevas variables `ENTRY_VOLUME_FILTER_*`)
  - `.env.example` (configuracion de ejemplo)
  - `execution/tests_tasks.py` (tests del gate de volumen)

3. Market regimes
- Continuar usando `risk.RegimeFilterConfig` (`atr_pct`/`adx`) como gate previo.
- Siguiente paso sugerido: ajustar `ENTRY_VOLUME_FILTER_MIN_RATIO` por sesion/regimen (asia/london/ny).

4. Aprendizajes SMC intradia (aplicados)
- No perseguir velas de desplazamiento: anti-chase en `signals/tasks.py` para `smc_long/smc_short`.
- Trend no entra tarde tras impulsos: filtro de impulso + distancia EMA20 en `signals/modules/trend.py`.
- MeanReversion evita "cazar cuchillos": bloqueo tras desplazamientos fuertes en `signals/modules/meanrev.py`.
- Carry evita extremos de volatilidad: tope ATR en `signals/modules/carry.py`.
- Ejecucion protege beneficio en alta volatilidad: `VOL_FAST_EXIT_*` en `execution/tasks.py`.

## Config sugerida inicial (demo)

```env
ENTRY_VOLUME_FILTER_ENABLED=true
ENTRY_VOLUME_FILTER_TIMEFRAME=5m
ENTRY_VOLUME_FILTER_LOOKBACK=48
ENTRY_VOLUME_FILTER_MIN_RATIO=0.75
ENTRY_VOLUME_FILTER_MIN_RATIO_BY_SESSION={"asia": 0.65, "london": 0.85, "ny": 0.90, "overlap": 0.95, "dead": 1.20}
ENTRY_VOLUME_FILTER_FAIL_OPEN=true
```

Notas:
- `0.75` permite operar cuando volumen actual >= 75% de su mediana reciente.
- `FAIL_OPEN=true` evita bloquear por falta de datos; en live estricto puedes pasar a `false`.
