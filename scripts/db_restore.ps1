[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$File
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not (Test-Path $File)) {
    throw "Backup file not found: $File"
}

$backupPath = (Resolve-Path $File).Path
$restoreCmd = "docker compose exec -T postgres sh -lc ""pg_restore -U `"`$POSTGRES_USER`" -d `"`$POSTGRES_DB`" --clean --if-exists --no-owner --no-privileges"" < ""$backupPath"""

cmd /c $restoreCmd | Out-Null

if ($LASTEXITCODE -ne 0) {
    throw "pg_restore failed with exit code $LASTEXITCODE"
}

Write-Host "Restore completed from: $backupPath"
