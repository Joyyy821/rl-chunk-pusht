#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export HOME="${HOME:-/workspace/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/cache/uv}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/workspace/cache/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-/workspace/cache/wandb}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export HWLOC_HIDE_ERRORS="${HWLOC_HIDE_ERRORS:-2}"
export PATH="${HOME}/.local/bin:${PATH}"

mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}" "${WANDB_DIR}"

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
