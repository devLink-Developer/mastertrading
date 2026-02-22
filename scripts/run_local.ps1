Param(
    [switch]$Build
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path $MyInvocation.MyCommand.Path -Parent) | Out-Null
Set-Location ..

if ($Build) {
    docker compose build
}

docker compose up -d postgres redis
docker compose up -d web worker beat market-data

docker compose exec web python manage.py migrate
docker compose exec web python manage.py seed_instruments

Write-Host "Servicios arriba. API en http://localhost:8008"
Write-Host "Log web: docker compose logs -f web"
