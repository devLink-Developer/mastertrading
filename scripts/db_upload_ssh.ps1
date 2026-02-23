[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$File,
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,
    [string]$RemoteDir = "/var/backups/mastertrading",
    [int]$Port = 22
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $File)) {
    throw "Backup file not found: $File"
}

$backupPath = (Resolve-Path $File).Path

ssh -p $Port $RemoteHost "mkdir -p '$RemoteDir'"
if ($LASTEXITCODE -ne 0) {
    throw "SSH mkdir failed with exit code $LASTEXITCODE"
}

scp -P $Port $backupPath "${RemoteHost}:${RemoteDir}/"
if ($LASTEXITCODE -ne 0) {
    throw "SCP upload failed with exit code $LASTEXITCODE"
}

Write-Host "Backup uploaded to ${RemoteHost}:${RemoteDir}/"
