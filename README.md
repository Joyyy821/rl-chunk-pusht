# rl-chunk-pusht

Push-T reinforcement learning experiments for comparing action chunking across position-target and velocity-command action parameterizations.

## Goal

The motivating hypothesis (for now) is that position-level action data is more multi-modal than lower-level control commands, so flow policies with action chunking may help more for position targets than for velocity commands.

This repo tests that hypothesis in Push-T with Q-chunking-style RL:

- `position_target`: stock Push-T semantics, where action is an absolute 2D
  target position for the built-in PD controller.
- `velocity_kinematic`: direct 2D velocity command for the Push-T kinematic
  end-effector. This replaces the built-in target-position controller.
- `velocity_delta_target`: diagnostic only. It integrates velocity into a target
  position and still uses the built-in PD controller.

## Setup

Use `uv` from inside this repo:

```bash
uv sync --group dev
```

The default dependency set uses normal CPU JAX packages. For CUDA JAX, install
the matching JAX wheel for your machine with a repo extra:

```bash
uv sync --group dev --extra cuda12
```

Check the result:

```bash
uv run python -c "import jax; print(jax.devices())"
```

If this still prints only `CpuDevice`, first check that `nvidia-smi` works in
the same shell. CUDA JAX needs a working NVIDIA driver; installing Python
packages alone will not fix a missing driver. Also unset `LD_LIBRARY_PATH` when
using the pip-installed CUDA wheels:

```bash
env -u LD_LIBRARY_PATH uv run python -c "import jax; print(jax.devices())"
```

## Training

Pure online Q-chunking on the direct velocity env:

```bash
uv run python -m rl_chunk_pusht.train \
  --env-mode velocity_kinematic \
  --agent acfql \
  --chunk-size 5 \
  --action-chunking true
```

Position target version:

```bash
uv run python -m rl_chunk_pusht.train \
  --env-mode position_target \
  --agent acfql \
  --chunk-size 5 \
  --action-chunking true
```

No-chunk H-step baseline:

```bash
uv run python -m rl_chunk_pusht.train \
  --env-mode position_target \
  --agent acfql \
  --chunk-size 5 \
  --action-chunking false
```

One-step baseline:

```bash
uv run python -m rl_chunk_pusht.train \
  --env-mode position_target \
  --agent acfql \
  --chunk-size 1 \
  --action-chunking true
```

Logs, configs, CSVs, and checkpoints are written under `exp/`.

A note to myself: add `env -u LD_LIBRARY_PATH` before the `uv run python` command to unset `LD_LIBRARY_PATH` and avoid CUDA conflicts in JAX on my local machine.

## Visualization

Overlay multiple stochastic rollouts from the same reset state:

```bash
uv run python -m rl_chunk_pusht.visualize \
  --checkpoint exp/rl-chunk-pusht/debug/position_target/<run>/params_<step>.pkl \
  --output-dir figures/position_target_seed0 \
  --reset-seed 0 \
  --num-rollouts 16
```

Agent paths are drawn with stronger lines; block-center paths are drawn with
lighter lines.

## Experiment Matrix

Run the same seeds for:

- `acfql`, `chunk_size=1`, `action_chunking=true`
- `acfql`, `chunk_size=5`, `action_chunking=true`
- `acfql`, `chunk_size=5`, `action_chunking=false`
- optional sanity baseline: `acrlpd`, `chunk_size=1`

Repeat the matrix for `position_target` and `velocity_kinematic`.

## Attribution

The Q-chunking/FQL agent structure, replay sequence sampling, and Flax training utilities are adapted from the official `qc` implementation [here](https://github.com/ColinQiyangLi/qc) of "Reinforcement
Learning with Action Chunking."
The Push-T data-loading utilities are adapted
from the [CS 285 HW1 imitation-learning](https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw1) code, which adopts the Push-T 2D simulation environment supports from [gym-pusht](https://github.com/huggingface/gym-pusht/tree/main).
This repo keeps those pieces local instead of using a submodule so the Push-T-specific experiment remains easy to run and modify.
