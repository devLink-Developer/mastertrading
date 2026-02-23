#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-backups}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
FILE_NAME="${1:-mastertrading_${TIMESTAMP}.dump}"
OUTPUT_PATH="${OUTPUT_DIR}/${FILE_NAME}"

mkdir -p "${OUTPUT_DIR}"

docker compose exec -T postgres sh -lc 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "${OUTPUT_PATH}"

SIZE_BYTES="$(wc -c < "${OUTPUT_PATH}")"
echo "Backup created: ${OUTPUT_PATH} (${SIZE_BYTES} bytes)"
