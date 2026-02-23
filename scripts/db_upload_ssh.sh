#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <backup_file.dump> <user@host> [remote_dir] [port]" >&2
  exit 1
fi

BACKUP_FILE="$1"
REMOTE_HOST="$2"
REMOTE_DIR="${3:-/var/backups/mastertrading}"
SSH_PORT="${4:-22}"

if [[ ! -f "${BACKUP_FILE}" ]]; then
  echo "Backup file not found: ${BACKUP_FILE}" >&2
  exit 1
fi

ssh -p "${SSH_PORT}" "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}'"
scp -P "${SSH_PORT}" "${BACKUP_FILE}" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Backup uploaded to ${REMOTE_HOST}:${REMOTE_DIR}/"
