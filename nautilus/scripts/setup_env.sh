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

mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}" "${WANDB_DIR}"

if [[ ! -f "${HOME}/.bashrc" ]] || ! grep -q "rl-chunk-pusht Nautilus environment" "${HOME}/.bashrc"; then
  cat >> "${HOME}/.bashrc" <<'EOF'

# rl-chunk-pusht Nautilus environment
export PROJECT_DIR="${PROJECT_DIR:-/workspace/rl-chunk-pusht}"
export HOME="${HOME:-${PROJECT_DIR}/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_DIR}/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT_DIR}/cache/uv}"
export TMPDIR="${TMPDIR:-${PROJECT_DIR}/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/cache/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-${PROJECT_DIR}/cache/wandb}"
export HWLOC_HIDE_ERRORS="${HWLOC_HIDE_ERRORS:-2}"
export PATH="${PROJECT_DIR}/home/.local/bin:${PROJECT_DIR}/src/rl-chunk-pusht/.venv/bin:${PATH}"
EOF
fi

if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" == "0" ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3 \
    python3-pip \
    python3-venv
  rm -rf /var/lib/apt/lists/*
else
  echo "Skipping apt install because this container is not running as root or apt-get is unavailable."
fi

if ! command -v uv >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m pip install --user uv
  else
    echo "Could not install uv: curl, wget, and python3 are all unavailable." >&2
    exit 1
  fi
fi

export PATH="${HOME}/.local/bin:${PATH}"

cd "${REPO_DIR}"
uv python install 3.11
uv sync --group dev --extra cuda12

env -u LD_LIBRARY_PATH HWLOC_HIDE_ERRORS="${HWLOC_HIDE_ERRORS}" uv run python - <<'PY'
import jax
import gym_pusht

print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())
print("gym_pusht import: ok")
PY
