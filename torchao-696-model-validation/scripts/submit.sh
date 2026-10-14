#!/usr/bin/env bash
#SBATCH --job-name=fd696-ablation
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=320G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
export TZ=UTC
ROOT=<REPO>/.claude/worktrees/open-pr-supplementary-experiments-c0eca3/artifacts/experiment-supplement-20261014-ablation
VENV=<REPO>/.venv-coverage-cpu-isolated
cd "$ROOT"
export PATH=$VENV/bin:/usr/local/cuda/bin:$PATH
exec "$VENV/bin/python" "$ROOT/ablation_matrix.py" "$@"
