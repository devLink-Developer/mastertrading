# Corrección del registro de cierres — 14/09/2026 UTC

Estado: **subida y desplegada en Main DEMO y Eudy LIVE**, con autorización del usuario. Se aplicó `execution.0012` en ambas bases PostgreSQL y se verificaron servicios, consultas y configuración. Esta corrección mejora el registro y las decisiones que consumen PnL; no demuestra una estrategia rentable.

## Problema y cambio

Los cierres voluntarios conservan la respuesta del exchange, consultan el ID exacto cuando hace falta y guardan cada orden de cierre. La conciliación de posiciones desaparecidas busca identidad, dirección, cantidad y fecha compatibles en historial acotado. Las entradas se verifican por ID mediante el adaptador de la cuenta activa; una respuesta histórica almacenada no basta para demostrar la cuenta.

El resumen combina parciales y cierre final, suma ambas comisiones una sola vez y contempla ampliaciones vinculadas a la entrada raíz. La identidad por cuenta y correlación evita mezclar operaciones cercanas. Un ACK incompleto produce `pending`: PnL, salida y comisión desconocidos quedan NULL. Los reintentos mantienen evidencia aunque la posición local ya se haya cerrado; los cierres confirmados antiguos no bloquean la cola.

Ante `no position`, se conserva el contexto hasta la próxima sincronización para recuperar el stop que se adelantó al bot. Un giro con cierre pendiente espera otro snapshot antes de permitir una entrada opuesta. El parcial aceptado se marca antes de la escritura contable para no repetirlo cuando esa escritura falla; el registro durable también evita repetirlo tras perder Redis.

Pesos dinámicos, entrenamiento, estadísticas y controles de resultados omiten pendientes. Aprendizaje y controles operativos separan demo/live. Los avisos distinguen conciliación pendiente de PnL de ejecución con comisiones incluidas y funding excluido, sin inventar equity posterior.

## Ejemplo real reproducido sin red

Se cargaron dos órdenes BTC de Main DEMO exportadas en modo lectura y se ejecutó el código corregido en SQLite en memoria:

| Dato | Resultado |
|---|---:|
| Cantidad short | 0,0835 BTC |
| Promedio ejecutado de entrada | 76.999,8 |
| Promedio ejecutado de salida | 76.875,4 |
| Ambas comisiones | 6,424291 VST |
| PnL calculado con esos promedios | **+3,963109 VST** |
| Reporte productivo anterior #832 | +10,59308555 VST |
| Neto del ledger asociado en la auditoría | +3,96505942 VST |

El desvío de 0,00195042 VST respecto del ledger es compatible con el redondeo de los promedios publicados; no se recuperaron fills individuales para demostrar esa atribución. `confirmed` significa evidencia de ejecución verificada, **no conciliación exacta del saldo**. El ledger sigue siendo la fuente del resultado económico total. Evidencia: `output/profit_fix_20260913/replay_main_btc.json`; exportación original: `tmp/profit_fix/order_evidence_main.txt`.

## Validación

Con las dependencias de `requirements.txt` instaladas en un entorno Python de pruebas:

```powershell
python scripts/test_close_accounting.py
```

**310 pruebas pasan; 0 intentos de red.** El runner ignora `.env`, usa SQLite en memoria, deshabilita trading/Celery real/Telegram y bloquea sockets. Incluye 34 pruebas del resolvedor, 34 del registro durable, 28 de integración y regresiones de ejecución/consumidores. La migración `execution.0012` se aplica durante las pruebas. Log local: `tmp/profit_fix_env/release_validation.log`.

Casos cubiertos: ticker diferente del fill, fee ausente, rebate, duplicados, parciales, ampliaciones, namespace, fechas inválidas, fallo al guardar reportes, reintento tras eliminar la posición, más de 50 cierres antiguos, carrera contra stops y ausencia de reentrenamiento antes del commit.

## Límites y paso a producción

- El cálculo no atribuye funding por operación ni reconstruye fills individuales desde promedios redondeados. Es PnL de ejecución antes de funding.
- Los reportes históricos permanecen `legacy`, sin reescritura. No recupera automáticamente cierres antiguos que nunca tuvieron registro durable.
- Datos ambiguos, fees ausentes, órdenes parcialmente ejecutadas y luego canceladas o historial insuficiente quedan pendientes; no se convierten en cero ni en ganancia.
- Un fallo simultáneo de DB y Redis después del ACK no tiene recuperación atómica garantizada. La conciliación no envía órdenes.
- La migración se probó en SQLite y se aplicó y verificó en ambos PostgreSQL productivos. Revertir a código anterior con reportes NULL requiere compatibilidad; no revertir ciegamente la migración.

Se hizo backup completo, se detuvieron los consumidores antes de migrar y luego se reiniciaron. Main usa `/opt/trading_bot` y compose base; Eudy `/opt/trading_bot_eudy`, `docker-compose.eudy.yml` y `.env.eudy`. Se preservaron los archivos previamente modificados del servidor. No se activó una estrategia nueva ni se modificó el presupuesto de riesgo o apalancamiento. La próxima operación natural permitirá comparar el nuevo registro con el ledger.

## Despliegue verificado

| Stack | Commit de código | Finalización UTC | Pruebas en la imagen del servidor |
|---|---|---|---|
| Main DEMO | `b5eb99786e683881bf545f79586ab9236d770f2f` | 2026-09-14 01:22:51 | 310, una omitida |
| Eudy LIVE | `6ed70d867ecb34cbc7ff5a9bfa6987630d4343f2` | 2026-09-14 01:26:47 | 315, una omitida |

La prueba omitida requiere la exportación privada del exchange, que no se publicó. Los contenedores de prueba no tuvieron red. Además pasaron 35 pruebas locales específicas de Eudy. Dos fallos de una suite ampliada de señales fueron reproducidos en el commit Eudy anterior; no pertenecen a esta corrección.

Verificaciones posteriores: HTTP 200 en ambos servicios y en la URL pública de Main; respuestas `pong` de los workers de trading y datos; esquema nullable e índice único válidos; consulta PostgreSQL de reintentos correcta; señales y velas nuevamente actualizadas. Los 824 reportes históricos de Main y 332 de Eudy permanecen `legacy`. Sin cambios de saldo, posiciones u órdenes abiertas al comprobar las cuentas. Sin errores detectados en los logs revisados posteriores al reinicio.

Se conservaron `docker-compose.eudy.yml`, el calibrador agregado a `risk/tasks.py` y los archivos no versionados de Main; Eudy conserva su configuración y restricciones específicas. `.env` y `.env.eudy` coinciden por hash con sus copias previas. No se reconstruyeron imágenes: las dependencias no cambiaron y los servicios montan el código del checkout.

Respaldos y logs del servidor: `/opt/trading_deploy_backups/20260914-close-accounting`, propietario root y permisos 0700. Dumps completos de aproximadamente 289 MiB y 259 MiB, con catálogo leído y SHA-256 calculado. La restauración después de actividad nueva del exchange requiere conciliación; no es un rollback automático. Evidencia local adicional: `tmp/profit_fix/verify_release_result.txt`, `state_after.txt` y resultados de ambos despliegues.
