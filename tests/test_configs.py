from __future__ import annotations

import tyro

from rl_chunk_pusht.configs import TrainConfig, normalize_bool_cli_args


def test_normalize_bool_cli_args_accepts_explicit_values():
    args = normalize_bool_cli_args(
        [
            "--env-mode",
            "position_target",
            "--chunk-size",
            "8",
            "--action-chunking",
            "true",
        ]
    )
    config = tyro.cli(TrainConfig, args=args)
    assert config.env_mode == "position_target"
    assert config.chunk_size == 8
    assert config.action_chunking is True


def test_normalize_bool_cli_args_accepts_equals_values():
    args = normalize_bool_cli_args(
        [
            "--action-chunking=false",
            "--layer-norm",
            "false",
            "--use-fourier-features=yes",
        ]
    )
    config = tyro.cli(TrainConfig, args=args)
    assert config.action_chunking is False
    assert config.layer_norm is False
    assert config.use_fourier_features is True
