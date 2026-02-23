
## 2026-02-22 - Limpieza fuerte de caches Brave/Spotify/Teams
- Objetivo: liberar espacio en C sin tocar datos de proyecto ni usar `wsl --shutdown`.
- Acciones ejecutadas:
  - cierre de procesos: Brave, Spotify, Teams
  - borrado de caches:
    - Brave `Default/Cache`, `Default/Code Cache`, `Default/Service Worker`
    - Spotify UWP `LocalCache`
    - MSTeams UWP `LocalCache`
- Espacio recuperado:
  - C: de ~84.02 GB libres a ~138.92 GB libres (aprox +54.9 GB)
- Estado final:
  - targets de cache eliminados
  - `C:\Users\rortigoza\AppData\Local\Packages` quedó en ~0.04 GB
