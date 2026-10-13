#!/usr/bin/env bash
# Full study for macOS / Linux (same as run_experiments.ps1 on Windows):
# 6 configurations x N seeds. One process per configuration; inside it, all seeds are
# trained together (main_vec.py). The 6 processes run in parallel.
#
# Writes v2/runs (TensorBoard) and v2/logs; uses the .venv at the repository root.
# Usage (from v2/):  ./src/run_experiments.sh
#         ./src/run_experiments.sh --seeds "1 2 3 4 5 6 7 8" --steps 200000 --device cpu
#         ./src/run_experiments.sh --dry-run            (print the commands without running them)
# Options: --seeds "LIST"  (default "1 2 3 4 5")
#          --steps N       (default 300000)
#          --device D      auto | cuda | cpu (default auto: CUDA if available, else CPU;
#                          Apple-silicon GPUs (MPS) are not supported, macOS runs on CPU)
#          --threads N     torch CPU threads per process (default 2)
# The Python interpreter is ../.venv/bin/python (repository root), or $PYTHON if set.

set -euo pipefail

SEEDS="1 2 3 4 5"
STEPS=300000
DEVICE="auto"
THREADS=2
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --seeds)   SEEDS="$2"; shift 2 ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --device)  DEVICE="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,/^$/p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")/.."                     # v2/
PYTHON="${PYTHON:-../.venv/bin/python}"

# name | arguments  (plain indexed arrays: works with the bash 3.2 shipped with macOS)
NAMES=("DDPG" "DDPG+LN" "TD3" "TD3+LN" "TD3+LN+HL3" "TD3+LN+UTD2")
ARGS=("--algo ddpg"
      "--algo ddpg --use-layernorm"
      "--algo td3"
      "--algo td3 --use-layernorm"
      "--algo td3 --use-layernorm --hidden-layers 3"
      "--algo td3 --use-layernorm --utd-ratio 2")

mkdir -p logs
PIDS=()

# Ctrl-C stops every run
cleanup() {
    echo "$(date +%H:%M) interrupted, stopping runs"
    for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill "$pid" 2>/dev/null || true; done
    exit 130
}
trap cleanup INT TERM

for i in "${!NAMES[@]}"; do
    tag=$(echo "${NAMES[$i]}" | tr '+' '_')
    # word splitting of ARGS and SEEDS is intended
    cmd="$PYTHON src/main_vec.py ${ARGS[$i]} --seeds $SEEDS --total-timesteps $STEPS --device $DEVICE --threads $THREADS"
    echo "$(date +%H:%M) start ${NAMES[$i]}, seeds $SEEDS"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "    $cmd > logs/$tag.out 2> logs/$tag.err"
        continue
    fi
    $cmd > "logs/$tag.out" 2> "logs/$tag.err" &
    PIDS+=($!)
done

[ "$DRY_RUN" -eq 1 ] && exit 0

FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "$(date +%H:%M) ${NAMES[$i]} FAILED, see logs/$(echo "${NAMES[$i]}" | tr '+' '_').err" >&2
        FAILED=1
    fi
done
echo "$(date +%H:%M) all runs finished (progress was logged in logs/*.out)"
exit $FAILED
