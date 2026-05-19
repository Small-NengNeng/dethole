# Pack large assets for Baidu Pan upload (not tracked by git).
$ErrorActionPreference = 'Stop'

$OutDir = 'E:\4070project\mmyolo\baidupan'
$MmyoloRoot = Split-Path $PSScriptRoot -Parent
$HoledetOak = 'E:\4070project\holedet\oak'

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Pack-TarGz {
    param(
        [string]$SourceDir,
        [string]$ArchivePath,
        [string]$TarPathInArchive
    )
    if (-not (Test-Path $SourceDir)) {
        Write-Warning "Skip (not found): $SourceDir"
        return
    }
    Write-Host "Packing $SourceDir -> $ArchivePath"
    $parent = Split-Path $SourceDir -Parent
    $name = Split-Path $SourceDir -Leaf
    Push-Location $parent
    try {
        tar -czf $ArchivePath $TarPathInArchive
    } finally {
        Pop-Location
    }
    $mb = [math]::Round((Get-Item $ArchivePath).Length / 1MB, 2)
    Write-Host "Done: $ArchivePath ($mb MB)"
}

Pack-TarGz -SourceDir (Join-Path $MmyoloRoot 'work_dirs') `
    -ArchivePath (Join-Path $OutDir 'mmyolo_work_dirs.tar.gz') `
    -TarPathInArchive 'work_dirs'

Pack-TarGz -SourceDir (Join-Path $HoledetOak 'oak_datasetv1') `
    -ArchivePath (Join-Path $OutDir 'oak_datasetv1.tar.gz') `
    -TarPathInArchive 'oak_datasetv1'

Pack-TarGz -SourceDir (Join-Path $HoledetOak 'oak_datasetv2') `
    -ArchivePath (Join-Path $OutDir 'oak_datasetv2.tar.gz') `
    -TarPathInArchive 'oak_datasetv2'

Write-Host 'All archives written to' $OutDir
