# Feature Flags de Estrategias (runtime)

Este proyecto permite prender/apagar estrategias **sin reiniciar servicios** usando `StrategyConfig` en DB.

## Flags disponibles

- `feature_multi_strategy`: habilita todo el pipeline multi-modulo.
- `feature_mod_trend`: habilita modulo `trend`.
- `feature_mod_meanrev`: habilita modulo `meanrev`.
- `feature_mod_carry`: habilita modulo `carry`.
- `feature_allocator`: habilita allocator y emision `alloc_*`.

Version de flags: `feature_flags_v1`.

## Efecto operativo

- Si `feature_multi_strategy=false`, no corren modulos ni allocator.
- Si `feature_allocator=false`, `execution` vuelve a consumir la ultima senal general (modo legacy).
- Si ambos (`feature_multi_strategy=true` y `feature_allocator=true`), `execution` consume solo `alloc_*`.
- Senal `alloc_flat`: no abre posiciones nuevas; solo gestiona posiciones abiertas.

## API para control rapido

Base: `/api/config/strategy/`

- `GET /api/config/strategy/features/`
  - devuelve defaults, estado resuelto y filas en DB.
- `POST /api/config/strategy/set_feature/`
  - body JSON: `{"name":"feature_mod_meanrev","enabled":false}`
- `POST /api/config/strategy/toggle_feature/`
  - body JSON: `{"name":"feature_mod_meanrev"}`

Tambien sigue disponible el toggle por ID:

- `POST /api/config/strategy/{id}/toggle/`

## Control desde Telegram

En el bot interactivo:

1. `/start`
2. Boton `Estrategias`
3. Toca el modulo para toggle ON/OFF (`Multi`, `Trend`, `MeanRev`, `Carry`, `Allocator`)
4. Boton `Refrescar` para releer estado

Los cambios aplican en runtime y afectan los ciclos siguientes de Celery.

## Reporte automatico cada 15 minutos

Se envia por Telegram desde Celery Beat con:

- `PERFORMANCE_REPORT_ENABLED=true`
- `PERFORMANCE_REPORT_WINDOW_MINUTES=15`
- `PERFORMANCE_REPORT_BEAT_MINUTES=15`

Task: `risk.tasks.send_performance_report`.
Incluye senales por modulo, alloc long/short/flat, ordenes, operaciones cerradas, PnL y estado de cuenta.

## Nota de fallback

Si una flag no existe en DB, el sistema usa el default de `.env` y crea la fila automaticamente cuando se resuelve por primera vez.
