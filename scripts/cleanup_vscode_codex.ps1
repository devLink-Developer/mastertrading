$ErrorActionPreference = "SilentlyContinue"

function Get-Bytes([string]$Path) {
  if (-not (Test-Path $Path)) { return [int64]0 }
  $sum = (Get-ChildItem $Path -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
  if ($null -eq $sum) { return [int64]0 }
  return [int64]$sum
}

function Remove-ChildrenSafe([string]$DirPath) {
  if (-not (Test-Path $DirPath)) { return [PSCustomObject]@{ removed = 0; failed = 0 } }
  $removed = 0
  $failed = 0
  foreach ($item in (Get-ChildItem $DirPath -Force -ErrorAction SilentlyContinue)) {
    try {
      Remove-Item $item.FullName -Recurse -Force -ErrorAction Stop
      $removed++
    } catch {
      $failed++
    }
  }
  return [PSCustomObject]@{ removed = $removed; failed = $failed }
}

function Mb([int64]$Bytes) {
  return [math]::Round(($Bytes / 1MB), 2)
}

$vsBase = Join-Path $env:APPDATA "Code"
$codexBase = Join-Path $env:USERPROFILE ".codex"

$vsCacheDirs = @(
  (Join-Path $vsBase "Cache"),
  (Join-Path $vsBase "CachedData"),
  (Join-Path $vsBase "Code Cache"),
  (Join-Path $vsBase "GPUCache"),
  (Join-Path $vsBase "Service Worker\CacheStorage"),
  (Join-Path $vsBase "Service Worker\ScriptCache"),
  (Join-Path $vsBase "Session Storage"),
  (Join-Path $vsBase "shared_proto_db"),
  (Join-Path $vsBase "WebStorage")
)

$trackedPaths = @()
$trackedPaths += $vsCacheDirs
$trackedPaths += (Join-Path $vsBase "logs")
$trackedPaths += (Join-Path $vsBase "Network\Cookies")
$trackedPaths += (Join-Path $vsBase "Network\Cookies-journal")
$trackedPaths += (Join-Path $codexBase "tmp")
$trackedPaths += (Join-Path $codexBase "sessions")

$before = @{}
foreach ($p in $trackedPaths) {
  if (Test-Path $p) {
    if ((Get-Item $p).PSIsContainer) { $before[$p] = Get-Bytes $p }
    else { $before[$p] = (Get-Item $p).Length }
  } else {
    $before[$p] = [int64]0
  }
}

$actions = @()

# VS Code cache/cookies
foreach ($dir in $vsCacheDirs) {
  if (Test-Path $dir) {
    $res = Remove-ChildrenSafe $dir
    $actions += "VS cache cleaned: $dir removed=$($res.removed) failed=$($res.failed)"
  }
}

$logsDir = Join-Path $vsBase "logs"
if (Test-Path $logsDir) {
  $oldLogDirs = Get-ChildItem $logsDir -Directory -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) }
  $oldCount = ($oldLogDirs | Measure-Object).Count
  foreach ($d in $oldLogDirs) {
    try { Remove-Item $d.FullName -Recurse -Force -ErrorAction Stop } catch {}
  }
  $actions += "VS old log dirs removed: $oldCount"
}

foreach ($cookieFile in @((Join-Path $vsBase "Network\Cookies"), (Join-Path $vsBase "Network\Cookies-journal"))) {
  if (Test-Path $cookieFile) {
    try {
      Remove-Item $cookieFile -Force -ErrorAction Stop
      $actions += "VS cookie file removed: $cookieFile"
    } catch {
      $actions += "VS cookie file locked: $cookieFile"
    }
  }
}

# Codex cleanup (do NOT touch auth/config/skills)
$codexTmp = Join-Path $codexBase "tmp"
if (Test-Path $codexTmp) {
  $res = Remove-ChildrenSafe $codexTmp
  $actions += "Codex tmp cleaned: removed=$($res.removed) failed=$($res.failed)"
}

$codexSessions = Join-Path $codexBase "sessions"
$deletedOldSessionFiles = 0
if (Test-Path $codexSessions) {
  $oldFiles = Get-ChildItem $codexSessions -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) }
  foreach ($f in $oldFiles) {
    try {
      Remove-Item $f.FullName -Force -ErrorAction Stop
      $deletedOldSessionFiles++
    } catch {}
  }
  # prune empty folders
  Get-ChildItem $codexSessions -Recurse -Directory -ErrorAction SilentlyContinue |
    Sort-Object FullName -Descending |
    ForEach-Object {
      if (-not (Get-ChildItem $_.FullName -Force -ErrorAction SilentlyContinue)) {
        try { Remove-Item $_.FullName -Force -ErrorAction Stop } catch {}
      }
    }
  $actions += "Codex old session files deleted (>30d): $deletedOldSessionFiles"
}

$after = @{}
foreach ($p in $trackedPaths) {
  if (Test-Path $p) {
    if ((Get-Item $p).PSIsContainer) { $after[$p] = Get-Bytes $p }
    else { $after[$p] = (Get-Item $p).Length }
  } else {
    $after[$p] = [int64]0
  }
}

$totalBefore = [int64]0
$totalAfter = [int64]0
Write-Output "=== CLEANUP SUMMARY ==="
foreach ($p in $trackedPaths) {
  $b = [int64]$before[$p]
  $a = [int64]$after[$p]
  $totalBefore += $b
  $totalAfter += $a
  $freed = $b - $a
  Write-Output ("{0}`tBefore={1} MB`tAfter={2} MB`tFreed={3} MB" -f $p, (Mb $b), (Mb $a), (Mb $freed))
}
Write-Output ("TOTAL_FREED_MB={0}" -f (Mb ($totalBefore - $totalAfter)))
Write-Output ("CODEX_OLD_SESSION_FILES_DELETED={0}" -f $deletedOldSessionFiles)
Write-Output "=== ACTIONS ==="
foreach ($a in $actions) { Write-Output $a }

