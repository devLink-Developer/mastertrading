[CmdletBinding()]
param(
    [string]$OutputDir = "backups",
    [string]$FileName = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if ([string]::IsNullOrWhiteSpace($FileName)) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $FileName = "mastertrading_$timestamp.dump"
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$outputPath = Join-Path $OutputDir $FileName
$dumpCmd = "docker compose exec -T postgres sh -lc ""pg_dump -U `"`$POSTGRES_USER`" -d `"`$POSTGRES_DB`" -Fc"" > ""$outputPath"""

cmd /c $dumpCmd | Out-Null

if ($LASTEXITCODE -ne 0) {
    throw "pg_dump failed with exit code $LASTEXITCODE"
}

$sizeBytes = (Get-Item $outputPath).Length
Write-Host "Backup created: $outputPath ($sizeBytes bytes)"
