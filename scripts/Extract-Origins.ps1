# Extract-Origins.ps1
# Extracts all zips in an origin root into a single merged tree, using the folder
# structure stored inside the zips (a scanning service typically ships 01/..06/ or
# <job>/03/, <job>/05/ ...).
#
# Resumable: a file that already exists at the correct size is skipped, so you
# can stop this at any time and re-run it. Nothing is ever overwritten or deleted.
#
# Run in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\Extract-Origins.ps1 `
#       -Src "$env:USERPROFILE\Pictures\family-archive\scans-2026-07" `
#       -Dst "$env:USERPROFILE\Pictures\family-archive\scans-2026-07-extracted" -Expected 284
#
# The extracted tree is a SIBLING of the zips, never a child of them: origin roots sit
# side by side at the top of the archive folder.

param(
    [Parameter(Mandatory = $true)][string]$Src,     # folder holding the delivered zips
    [Parameter(Mandatory = $true)][string]$Dst,     # <Src>-extracted, beside it
    [int]$Expected = 0          # 0 = do not check the final count
)

# --------------------------------------------------------------------------

Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force -Path $Dst | Out-Null

$zips = Get-ChildItem -Path $Src -Filter *.zip -File | Sort-Object Name
if (-not $zips) { Write-Error "No .zip files found in $Src"; exit 1 }

$totalBytes = ($zips | Measure-Object -Property Length -Sum).Sum
Write-Host ("{0} zip file(s), {1:N1} GB compressed" -f $zips.Count, ($totalBytes/1GB)) -ForegroundColor Cyan
Write-Host "Source     : $Src"
Write-Host "Destination: $Dst`n" -ForegroundColor Cyan

$written = 0; $skipped = 0; $failed = 0
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$zi = 0

foreach ($z in $zips) {
    $zi++
    try { $arc = [System.IO.Compression.ZipFile]::OpenRead($z.FullName) }
    catch { Write-Warning "Cannot open $($z.Name): $($_.Exception.Message)"; $failed++; continue }

    try {
        foreach ($e in $arc.Entries) {
            if ([string]::IsNullOrEmpty($e.Name)) { continue }   # directory entry

            $tgt = Join-Path $Dst $e.FullName

            # Resume: skip if already extracted at the expected size
            if (Test-Path -LiteralPath $tgt) {
                $existing = Get-Item -LiteralPath $tgt
                if ($existing.Length -eq $e.Length) { $skipped++; continue }
            }

            $dir = Split-Path -Parent $tgt
            if (-not (Test-Path -LiteralPath $dir)) {
                New-Item -ItemType Directory -Force -Path $dir | Out-Null
            }

            try {
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, $tgt, $true)
                # preserve the timestamp recorded in the archive
                (Get-Item -LiteralPath $tgt).LastWriteTime = $e.LastWriteTime.DateTime
                $written++
            } catch {
                Write-Warning "Failed: $($e.FullName) -- $($_.Exception.Message)"
                $failed++
            }
        }
    } finally { $arc.Dispose() }

    $pct = [int](100 * $zi / $zips.Count)
    Write-Progress -Activity 'Extracting' -Status "$($z.Name)  [$zi/$($zips.Count)]  written=$written skipped=$skipped" -PercentComplete $pct
    Write-Host ("[{0,2}/{1}] {2,-45} written={3,5} skipped={4,5} {5:mm\:ss}" -f `
        $zi, $zips.Count, $z.Name, $written, $skipped, $sw.Elapsed)
}

Write-Progress -Activity 'Extracting' -Completed
$sw.Stop()

$finalCount = (Get-ChildItem -Path $Dst -File -Recurse).Count
$finalSize  = (Get-ChildItem -Path $Dst -File -Recurse | Measure-Object -Property Length -Sum).Sum

Write-Host "`n================ SUMMARY ================" -ForegroundColor Green
Write-Host ("Elapsed        : {0:hh\:mm\:ss}" -f $sw.Elapsed)
Write-Host "Files written  : $written"
Write-Host "Files skipped  : $skipped   (already present at correct size)"
Write-Host "Failures       : $failed"
Write-Host "Files on disk  : $finalCount"
Write-Host ("Size on disk   : {0:N1} GB" -f ($finalSize/1GB))
if ($Expected -gt 0) {
    Write-Host "Expected       : $Expected files"
    if ($finalCount -eq $Expected -and $failed -eq 0) {
        Write-Host "RESULT: complete and matches expected count." -ForegroundColor Green
    } else {
        Write-Host "RESULT: count differs from expected - re-run to resume, or check warnings above." -ForegroundColor Yellow
    }
} elseif ($failed -eq 0) {
    Write-Host "RESULT: complete, no failures." -ForegroundColor Green
}
