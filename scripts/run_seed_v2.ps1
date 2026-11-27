# Seed v2.0 launcher (150k tokenizer, factorized embeddings).
# Usage (from the repo root):
#   powershell -ExecutionPolicy Bypass -File scripts/run_seed_v2.ps1
#   powershell -File scripts/run_seed_v2.ps1 -Out runs2-eng1 -MixJson v2\mix.json -Device cuda
#
# Stop cleanly: create the file <Out>\STOP (it saves, then exits). Rerun with the same -Out to resume.

param(
    [string]$Python = $(if ($env:SEED_PYTHON) { $env:SEED_PYTHON } else { "python" }),
    [string]$Out = $(if ($env:SEED_OUT) { $env:SEED_OUT } else { "runs\v2-eng1" }),
    [string]$Device = $(if ($env:SEED_DEVICE) { $env:SEED_DEVICE } else { "cuda" }),
    [string]$MixJson = "v2\mix.example.json",
    [string]$Tokenizer = "tokenizers\seed-bpe150k-v2\tokenizer.json",
    [int]$VramCapMB = 4800,
    [int]$MemCapMB = 2048,
    [double]$TargetCe = 1.0,
    [switch]$Resume = $true
)

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$outPath = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $RepoRoot $Out }
New-Item -ItemType Directory -Force $outPath | Out-Null

$argList = @(
    "v2/seed_v2.py",
    "--out", $outPath,
    "--device", $Device,
    "--tokenizer", $Tokenizer,
    "--mix-json", $MixJson,
    "--d", "512",
    "--e-dim", "128",
    "--target-ce", "$TargetCe",
    "--vram-cap-mb", "$VramCapMB",
    "--mem-cap-mb", "$MemCapMB"
)
if ($Resume) { $argList += "--resume" }

$stdout = Join-Path $outPath "stdout.log"
$stderr = Join-Path $outPath "stderr.log"
$p = Start-Process $Python -ArgumentList $argList -WorkingDirectory $RepoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Set-Content (Join-Path $outPath "pid.txt") $p.Id
"started pid $($p.Id)  out=$outPath  device=$Device  tokenizer=$Tokenizer"
