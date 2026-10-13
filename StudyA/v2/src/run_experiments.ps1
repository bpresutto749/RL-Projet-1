# Full study: 6 configurations x len(Seeds) seeds.
# One process per configuration; inside it, all seeds are trained together on the GPU (main_vec.py).
# Benchmark (15k steps, all 30 runs at once): 274 s with GPU at ~96%, vs ~1075 s with one process per run.
# Writes v2\runs (TensorBoard) and v2\logs; uses the .venv at the repository root.
# Usage (from v2\):  .\src\run_experiments.ps1
#                    .\src\run_experiments.ps1 -Seeds 1,2,3,4,5,6,7,8 -TotalTimesteps 200000
param(
    [int[]]$Seeds = @(1, 2, 3, 4, 5),
    [int]$TotalTimesteps = 300000,
    [string]$Device = "cuda"
)

$configs = @(
    @{ name = "DDPG";          args = "--algo ddpg" },
    @{ name = "DDPG+LN";       args = "--algo ddpg --use-layernorm" },
    @{ name = "TD3";           args = "--algo td3" },
    @{ name = "TD3+LN";        args = "--algo td3 --use-layernorm" },
    @{ name = "TD3+LN+HL3";    args = "--algo td3 --use-layernorm --hidden-layers 3" },
    @{ name = "TD3+LN+UTD2";   args = "--algo td3 --use-layernorm --utd-ratio 2" }
)

$v2 = Split-Path -Parent $PSScriptRoot                                  # v2\
$python = Join-Path (Split-Path -Parent $v2) ".venv\Scripts\python.exe" # repository root
$logs = Join-Path $v2 "logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$seedArgs = $Seeds -join " "
$running = @()

foreach ($c in $configs) {
    $tag = $c.name -replace '\+', '_'
    Write-Host "$(Get-Date -Format HH:mm) start $($c.name), seeds $seedArgs"
    $argList = "src\main_vec.py $($c.args) --seeds $seedArgs --total-timesteps $TotalTimesteps --device $Device"
    $running += Start-Process -FilePath $python -ArgumentList $argList -WorkingDirectory $v2 `
        -NoNewWindow -PassThru -RedirectStandardOutput "$logs\$tag.out" -RedirectStandardError "$logs\$tag.err"
}
$running | Wait-Process
Write-Host "$(Get-Date -Format HH:mm) all runs finished (progress was logged in v2\logs\*.out)"
