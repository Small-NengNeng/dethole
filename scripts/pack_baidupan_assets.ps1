# Pack work_dirs for Baidu Pan upload (weights/logs; dataset is on Baidu Pan separately).
$ErrorActionPreference = 'Stop'

$OutDir = 'E:\4070project\mmyolo\baidupan'
$MmyoloRoot = Split-Path $PSScriptRoot -Parent

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$SourceDir = Join-Path $MmyoloRoot 'work_dirs'
$ArchivePath = Join-Path $OutDir 'mmyolo_work_dirs.tar.gz'

if (-not (Test-Path $SourceDir)) {
    Write-Warning "Skip (not found): $SourceDir"
    exit 1
}

Write-Host "Packing $SourceDir -> $ArchivePath"
Push-Location $MmyoloRoot
try {
    tar -czf $ArchivePath work_dirs
} finally {
    Pop-Location
}

$mb = [math]::Round((Get-Item $ArchivePath).Length / 1MB, 2)
Write-Host "Done: $ArchivePath ($mb MB)"
