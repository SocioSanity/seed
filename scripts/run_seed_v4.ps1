# Seed v4 launcher with configurable paths.
# Usage (from the repo root, or from this folder):
#   powershell -ExecutionPolicy Bypass -File scripts/run_seed_v4.ps1
#   powershell -File scripts/run_seed_v4.ps1 -Out runs\myrun -Minutes 60 -Device cuda -Mix "path\to\text.txt"
#
# Environment overrides: SEED_PYTHON, SEED_OUT, SEED_DEVICE, SEED_MINUTES
# Stop cleanly: create the file <Out>\STOP (it saves, then exits). Rerun with the same -Out to resume.

param(
    [string]$Python = $(if ($env:SEED_PYTHON) { $env:SEED_PYTHON } else { "python" }),
    [string]$Out = $(if ($env:SEED_OUT) { $env:SEED_OUT } else { "runs\run1" }),
    [string]$Device = $(if ($env:SEED_DEVICE) { $env:SEED_DEVICE } else { "cuda" }),
    [double]$Minutes = $(if ($env:SEED_MINUTES) { [double]$env:SEED_MINUTES } else { 60 }),
    [string]$Mix = "",
    [string]$Data = "",
    [string]$PartsDir = "parts",
    [string]$TokenizerDir = "tokenizers\seed-bpe4096-v1",
    [string]$SocketDir = "socket\seed-socket-v1",
    [int]$VramCapMB = 5000,
    [int]$MemCapMB = 2048,
    [switch]$Resume = $true
)

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$outPath = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $RepoRoot $Out }
New-Item -ItemType Directory -Force $outPath | Out-Null

$argList = @(
    "seed_v4.py",
    "--out", $outPath,
    "--device", $Device,
    "--minutes", "$Minutes",
    "--vram-cap-mb", "$VramCapMB",
    "--mem-cap-mb", "$MemCapMB",
    "--parts-dir", $PartsDir,
    "--tokenizer-dir", $TokenizerDir,
    "--socket-dir", $SocketDir
)
if ($Resume) { $argList += "--resume" }
if ($Data)   { $argList += @("--data", $Data) }
if ($Mix)    { $argList += @("--mix", $Mix) }

$stdout = Join-Path $outPath "stdout.log"
$stderr = Join-Path $outPath "stderr.log"
$p = Start-Process $Python -ArgumentList $argList -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Set-Content (Join-Path $outPath "pid.txt") $p.Id
"started pid $($p.Id)  out=$outPath  device=$Device"
