# Rescue-DiscFiles.ps1
#
# Pulls named files off a failing optical disc, one at a time, verifying each and
# resuming where the last attempt died.
#
# For a disc that drops out of the drive shortly after a bulk copy starts and needs the tray opened and closed before it is readable again, so robocopy
# over the whole tree never finishes. This copies a targeted list instead, in small
# independent steps, so that every attempt keeps whatever it managed to read.
#
# THE FAILURE THIS IS BUILT AROUND
#
# A dying disc does not always fail loudly. It can return a SHORT READ that Copy-Item
# reports as success, leaving a file that opens, shows the top of the image, and is
# quietly truncated: e.g. a JPEG cut at exactly 262144 bytes - a clean block boundary -
# with 1.2 MB of zeros after it, unnoticed until it is hashed against a good copy.
#
# So every file is verified after it lands: a complete JPEG ends with the EOI marker
# FFD9, and anything that does not is DELETED and left in the wanted list for the next
# pass. A file that survives this script is complete or it is not there at all. Never
# relax that check to "the copy succeeded" - the whole point is that the copy lies.
#
# USAGE - run this on the machine with the disc drive. In a remote desktop session,
# redirect your local drive (Local Resources -> More -> Drives), so \\tsclient\C is your own machine.
#
#   powershell -ExecutionPolicy Bypass -File \\tsclient\C\...\Rescue-DiscFiles.ps1 `
#       -Src 'D:\Pictures' `
#       -Dst '\\tsclient\C\Users\<you>\Documents\disc-pick' `
#       -WantList '\\tsclient\C\Users\<you>\Documents\want.txt'
#
# Re-run it after every disc reload. It re-reads what is already in -Dst and asks the
# drive only for what is still missing, so ten interrupted passes add up to one copy.
# Nothing is ever overwritten or deleted on the disc; -Dst only gains files.

param(
    [Parameter(Mandatory = $true)] [string]$Src,
    [Parameter(Mandatory = $true)] [string]$Dst,
    [Parameter(Mandatory = $true)] [string]$WantList,

    # Consecutive failures that mean "the disc has gone away again" rather than "this
    # one file is bad". Hitting this stops the pass immediately: once the drive has
    # dropped the disc, every further read fails too, and grinding through 100 doomed
    # reads only heats the drive and wastes the session.
    [int]$AbortAfter = 5
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $WantList)) { Write-Error "want list not found: $WantList"; exit 1 }
New-Item -ItemType Directory -Force -Path $Dst | Out-Null

# A complete JPEG ends FFD9. Trailing NULs are tolerated before it: some CD filesystems
# pad the last sector, and that padding is not damage - a zero tail alone must never be
# read as truncation. (Learned the hard way on a VideoCD rip whose 2336 trailing NULs
# were one CD-XA sector of legitimate padding.)
function Test-JpegComplete {
    param([string]$Path)
    try {
        $len = (Get-Item $Path).Length
        if ($len -lt 4) { return $false }
        $fs = [System.IO.File]::OpenRead($Path)
        try {
            $n = [Math]::Min(4096, $len)
            $fs.Seek(-$n, [System.IO.SeekOrigin]::End) | Out-Null
            $buf = New-Object byte[] $n
            $null = $fs.Read($buf, 0, $n)
        } finally { $fs.Dispose() }
        $i = $buf.Length - 1
        while ($i -ge 0 -and $buf[$i] -eq 0) { $i-- }          # skip sector padding
        return ($i -ge 1 -and $buf[$i] -eq 0xD9 -and $buf[$i - 1] -eq 0xFF)
    } catch { return $false }
}

$want = Get-Content -LiteralPath $WantList -Encoding UTF8 |
        ForEach-Object { $_.Trim() } | Where-Object { $_ }

# Resume: a file already in -Dst counts as done only if it verifies. One that is present
# but truncated is removed here, so a bad read from an earlier pass is retried rather
# than inherited.
$todo = @()
$already = 0
foreach ($name in $want) {
    $d = Join-Path $Dst $name
    if (Test-Path -LiteralPath $d) {
        if (Test-JpegComplete $d) { $already++; continue }
        Write-Host "  discarding truncated file from an earlier pass: $name" -ForegroundColor DarkYellow
        Remove-Item -LiteralPath $d -Force
    }
    $todo += $name
}

Write-Host ""
Write-Host "wanted    $($want.Count)" -ForegroundColor Cyan
Write-Host "verified  $already (already here)" -ForegroundColor Cyan
Write-Host "to fetch  $($todo.Count)" -ForegroundColor Cyan
Write-Host "source    $Src"
Write-Host "dest      $Dst"
Write-Host ""

if ($todo.Count -eq 0) {
    Write-Host "Nothing left to fetch. All wanted files are present and verified." -ForegroundColor Green
    exit 0
}

$ok = 0; $bad = 0; $missing = 0; $streak = 0
$log = Join-Path $Dst ("rescue-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))

foreach ($name in $todo) {
    $s = Join-Path $Src $name
    $d = Join-Path $Dst $name

    if (-not (Test-Path -LiteralPath $s)) {
        # Not on this disc at all. Not a read failure, so it must not count toward the
        # abort streak - otherwise a disc that simply lacks five wanted files looks
        # identical to a disc that has dropped.
        "MISSING $name" | Tee-Object -FilePath $log -Append | Write-Host -ForegroundColor DarkGray
        $missing++
        continue
    }

    try {
        Copy-Item -LiteralPath $s -Destination $d -Force -ErrorAction Stop
        if (Test-JpegComplete $d) {
            $sz = (Get-Item $d).Length
            "OK      $name  $sz bytes" | Tee-Object -FilePath $log -Append |
                Write-Host -ForegroundColor Green
            $ok++; $streak = 0
        } else {
            # The copy "succeeded" and the file is short. This is the silent failure.
            Remove-Item -LiteralPath $d -Force -ErrorAction SilentlyContinue
            "TRUNCATED $name  (removed, will retry next pass)" |
                Tee-Object -FilePath $log -Append | Write-Host -ForegroundColor Yellow
            $bad++; $streak++
        }
    } catch {
        Remove-Item -LiteralPath $d -Force -ErrorAction SilentlyContinue
        "FAIL    $name  $($_.Exception.Message)" | Tee-Object -FilePath $log -Append |
            Write-Host -ForegroundColor Red
        $bad++; $streak++
    }

    if ($streak -ge $AbortAfter) {
        Write-Host ""
        Write-Host "$AbortAfter reads failed in a row - the drive has probably dropped the disc." -ForegroundColor Yellow
        Write-Host "Open the tray, close it, wait for the disc to spin up, then run this again." -ForegroundColor Yellow
        Write-Host "Everything already fetched is kept; the next pass resumes from here." -ForegroundColor Yellow
        break
    }
}

$done = $already + $ok
Write-Host ""
Write-Host ("this pass: {0} fetched, {1} failed, {2} not on this disc" -f $ok, $bad, $missing)
Write-Host ("overall:   {0} of {1} verified present, {2} still wanted" -f $done, $want.Count, ($want.Count - $done)) -ForegroundColor Cyan
Write-Host "log:       $log"
if ($done -lt $want.Count) {
    Write-Host ""
    Write-Host "Re-run after reloading the disc to continue." -ForegroundColor Cyan
}
