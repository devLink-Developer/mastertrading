#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <backup_file.dump>" >&2
  exit 1
fi

BACKUP_FILE="$1"

if [[ ! -f "${BACKUP_FILE}" ]]; then
  echo "Backup file not found: ${BACKUP_FILE}" >&2
  exit 1
fi

docker compose exec -T postgres sh -lc 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists --no-owner --no-privileges' < "${BACKUP_FILE}"

echo "Restore completed from: ${BACKUP_FILE}"
