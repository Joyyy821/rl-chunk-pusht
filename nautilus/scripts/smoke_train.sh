#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export PROJECT_DIR="${PROJECT_DIR:-/workspace/rl-chunk-pusht}"
export HOME="${HOME:-${PROJECT_DIR}/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_DIR}/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/cache/uv}"
export TMPDIR="${TMPDIR:-${PROJECT_DIR}/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/cache/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/cache/wandb}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export HWLOC_HIDE_ERRORS="${HWLOC_HIDE_ERRORS:-2}"
export PATH="${HOME}/.local/bin:${PATH}"

ENV_MODE="${ENV_MODE:-position_target}"
AGENT="${AGENT:-acfql}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
ACTION_CHUNKING="${ACTION_CHUNKING:-true}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/smoke}"
EXP_NAME="${EXP_NAME:-smoke-${ENV_MODE}-c${CHUNK_SIZE}-seed${SEED}}"

mkdir -p "${OUTPUT_DIR}" "${TMPDIR}" "${WANDB_DIR}" "${MPLCONFIGDIR}"
cd "${REPO_DIR}"

if ! command -v uv >/dev/null 2>&1 || [[ ! -d ".venv" ]]; then
  bash nautilus/scripts/setup_env.sh
fi

env -u LD_LIBRARY_PATH HWLOC_HIDE_ERRORS="${HWLOC_HIDE_ERRORS}" uv run python -m rl_chunk_pusht.train \
  --env-mode "${ENV_MODE}" \
  --agent "${AGENT}" \
  --chunk-size "${CHUNK_SIZE}" \
  --action-chunking "${ACTION_CHUNKING}" \
  --seed "${SEED}" \
  --online-steps "${ONLINE_STEPS:-200}" \
  --warmup-steps "${WARMUP_STEPS:-50}" \
  --replay-size "${REPLAY_SIZE:-10000}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --log-interval "${LOG_INTERVAL:-50}" \
  --eval-interval "${EVAL_INTERVAL:-0}" \
  --save-interval "${SAVE_INTERVAL:--1}" \
  --eval-episodes "${EVAL_EPISODES:-2}" \
  --video-episodes "${VIDEO_EPISODES:-0}" \
  --save-dir "${OUTPUT_DIR}" \
  --wandb-mode disabled \
  --wandb-group smoke \
  --exp-name "${EXP_NAME}"
