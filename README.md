# Mastertrading Bot (SMC)

MVP de bot de trading de futuros/perps con Django, Celery, Redis y PostgreSQL. Expone API en `8008`, base de datos en `5434` y Redis en `6381` cuando se corre con Docker Compose.

## Requisitos rápidos
- Docker / Docker Compose
- Puertos libres: 8008 (web), 5434 (PostgreSQL host), 6381 (Redis host)

## Configuración (local con Docker + Postgres)
1. Copia `.env.example` a `.env` (ya trae `USE_SQLITE=false`).
2. Levanta servicios:
   ```powershell
   docker compose up --build -d
   ```
3. Migrar y seed:
   ```powershell
   docker compose exec web python manage.py migrate
   docker compose exec web python manage.py seed_instruments
   docker compose exec web python manage.py createsuperuser
   ```
Atajo: `.\scripts\run_local.ps1 -Build` hace build, levanta servicios y corre migrate+seed.

## Configuración opcional (tests rápidos con SQLite)
- Para tests locales sin Postgres: exporta `USE_SQLITE=true` antes de correr `manage.py test`.
5. Seed de instrumentos base (BTC/ETH/SOL):
   ```bash
   docker compose exec web python manage.py seed_instruments
   ```

## Servicios
- `web`: Django + DRF + gunicorn (`/health`, `/metrics`, API REST).
- `worker`: Celery worker para señales/ejecución.
- `beat`: Celery beat para scheduling (funding, tareas de limpieza).
- `market-data`: comando stub para ingesta WS/REST.
- `postgres`: imagen oficial 15 (puerto host 5434).
- `redis`: imagen oficial 7 (puerto host 6381).

## Credenciales / exchanges
- `EXCHANGE=kucoin` por defecto. Si usas Binance cambia a `binance`.
- KuCoin Futures: define `KUCOIN_API_KEY`, `KUCOIN_API_SECRET`, `KUCOIN_API_PASSPHRASE`; `KUCOIN_SANDBOX=true` para sandbox.
- Binance Futures: `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET=true`.
- En runtime usa la tabla `core_exchangecredential` (Admin) para cada servicio (`active` + `sandbox`).
- Toggle rapido sin reinicio:
  - `python manage.py set_active_exchange --service bingx`
  - `python manage.py set_exchange_sandbox --service bingx --on`  (demo/VST)
  - `python manage.py set_exchange_sandbox --service bingx --off` (live/USDT)

## Celery
- Tareas periódicas incluidas:
  - `signals.tasks.run_signal_engine` (cada minuto, heurística sweep+CHoCH y funding extremo)
  - `execution.tasks.execute_orders` (cada minuto, crea órdenes placeholder según señales)
  - `marketdata.tasks.fetch_ohlcv_and_funding` (polling REST; ajusta `MARKETDATA_POLL_INTERVAL`)

## Celery
- Tareas periódicas incluidas:
  - `signals.tasks.run_signal_engine` (cada minuto, heurística sweep+CHoCH y funding extremo)
  - `execution.tasks.execute_orders` (cada minuto, crea órdenes placeholder según señales)
  - Agrega `marketdata.tasks.fetch_ohlcv_and_funding` al beat si quieres polling REST (o usa WS).

## Apps Django
- `core`: instrumentos y auditoría.
- `marketdata`: velas, funding y snapshots de libro.
- `signals`: configuración de estrategias y señales.
- `execution`: órdenes, fills, posiciones.
- `risk`: eventos de riesgo/kill-switch.
- `api`: serializers y endpoints DRF.

## Endpoints REST (base path `/`)
- `GET /health`, `GET /metrics`
- `GET /instruments`, `POST /instruments/{id}/enable|disable`
- `GET /signals`, `GET /positions`, `GET /orders`, `GET /risk`
- `GET/POST /config/strategy/`
- `POST /config/strategy/{id}/toggle`

## Pruebas
```bash
set USE_SQLITE=true  # opcional para entorno local sin Postgres
python manage.py test
```

## Notas
- El adaptador de exchange y el motor de señales están stub; sirven de armazón para conectar Binance Futures testnet y lógica SMC.
- Los logs se emiten en JSON a stdout para facilitar ingesta en observabilidad.
