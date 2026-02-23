# DB Backup / Restore

This project versions backup/restore scripts in Git, but ignores the generated backup files.

## Files versioned in Git

- `scripts/db_dump.sh`
- `scripts/db_restore.sh`
- `scripts/db_upload_ssh.sh`
- `scripts/db_dump.ps1`
- `scripts/db_restore.ps1`
- `scripts/db_upload_ssh.ps1`

## Files ignored by Git

Backups in `backups/` (`.dump`, `.sql`, `.gz`, `.tar`) are ignored by `.gitignore`.

## Linux / macOS usage

```bash
chmod +x scripts/db_dump.sh scripts/db_restore.sh scripts/db_upload_ssh.sh
./scripts/db_dump.sh
./scripts/db_upload_ssh.sh backups/mastertrading_YYYYMMDD_HHMMSS.dump user@your-server /var/backups/mastertrading 22
./scripts/db_restore.sh backups/mastertrading_YYYYMMDD_HHMMSS.dump
```

## Windows PowerShell usage

```powershell
.\scripts\db_dump.ps1
.\scripts\db_upload_ssh.ps1 -File .\backups\mastertrading_YYYYMMDD_HHMMSS.dump -RemoteHost user@your-server -RemoteDir /var/backups/mastertrading -Port 22
.\scripts\db_restore.ps1 -File .\backups\mastertrading_YYYYMMDD_HHMMSS.dump
```

## Notes

- Scripts use the running `postgres` service from `docker compose`.
- `db_restore` runs `pg_restore --clean --if-exists`, so it replaces existing objects in the target DB.
