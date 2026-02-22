Param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [string]$Language = "Spanish",
    [string]$Model = "medium",
    [string]$OutputDir = "tmp/transcripts",

    [switch]$InstallDeps,
    [switch]$KeepAudio
)

$ErrorActionPreference = "Stop"

function Test-CommandExists {
    Param([Parameter(Mandatory = $true)][string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-Dependencies {
    if ($InstallDeps) {
        if (Test-CommandExists "choco") {
            if (-not (Test-CommandExists "yt-dlp")) {
                choco install yt-dlp -y | Out-Host
            }
            if (-not (Test-CommandExists "ffmpeg")) {
                choco install ffmpeg -y | Out-Host
            }
        }

        if (-not (Test-CommandExists "python")) {
            throw "Python no esta disponible en PATH. Instala Python 3.10+ y reintenta."
        }
        python -m pip install -U openai-whisper | Out-Host
    }

    if (-not (Test-CommandExists "yt-dlp")) {
        throw "Falta yt-dlp. Instala con: choco install yt-dlp -y"
    }
    if (-not (Test-CommandExists "ffmpeg")) {
        throw "Falta ffmpeg. Instala con: choco install ffmpeg -y"
    }
    if (-not (Test-CommandExists "python")) {
        throw "Falta Python en PATH."
    }
}

Ensure-Dependencies

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$targetDir = Resolve-Path -Path "." | ForEach-Object { Join-Path $_ $OutputDir }
New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

$videoId = (yt-dlp --get-id $Url 2>$null | Select-Object -First 1)
if ([string]::IsNullOrWhiteSpace($videoId)) {
    $videoId = "youtube_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$audioTemplate = Join-Path $targetDir "$videoId.%(ext)s"
$audioPath = Join-Path $targetDir "$videoId.mp3"

Write-Host "Descargando audio..."
yt-dlp -x --audio-format mp3 --audio-quality 0 --output $audioTemplate $Url | Out-Host

if (-not (Test-Path $audioPath)) {
    throw "No se encontro el audio esperado: $audioPath"
}

Write-Host "Transcribiendo con Whisper..."
python -m whisper $audioPath --language $Language --task transcribe --model $Model --output_format all --output_dir $targetDir --fp16 False | Out-Host

if (-not $KeepAudio) {
    Remove-Item -Path $audioPath -Force -ErrorAction SilentlyContinue
}

Write-Host "Listo. Archivos generados en: $targetDir"
