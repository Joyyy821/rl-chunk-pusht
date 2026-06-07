"""Configuration helpers for Push-T Q-chunking experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


EnvMode = Literal["position_target", "velocity_kinematic", "velocity_delta_target"]
AgentName = Literal["acfql", "acrlpd"]
ActorType = Literal["best-of-n", "distill-ddpg"]

BOOLEAN_CLI_FIELDS = frozenset(
    {
        "action-chunking",
        "layer-norm",
        "use-fourier-features",
    }
)
TRUE_LITERALS = frozenset({"1", "true", "t", "yes", "y", "on"})
FALSE_LITERALS = frozenset({"0", "false", "f", "no", "n", "off"})


@dataclass
class TrainConfig:
    """Top-level training config parsed by tyro."""

    env_mode: EnvMode = "velocity_kinematic"
    agent: AgentName = "acfql"
    seed: int = 0

    # Action chunking. For the no-chunk H-step baseline, set chunk_size=H and
    # action_chunking=False.
    chunk_size: int = 5
    action_chunking: bool = True

    # Push-T environment.
    max_speed_px_s: float = 120.0
    episode_length: int = 300
    render_size: tuple[int, int] = (256, 256)

    # Online RL loop.
    online_steps: int = 1_000_000
    warmup_steps: int = 5_000
    replay_size: int = 1_000_000
    batch_size: int = 256
    utd_ratio: int = 1
    discount: float = 0.99

    # Optimization and model size.
    lr: float = 3e-4
    actor_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    value_hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    layer_norm: bool = True
    num_qs: int = 2
    tau: float = 0.005
    q_agg: Literal["mean", "min"] = "mean"

    # Flow actor settings used by ACFQL.
    actor_type: ActorType = "best-of-n"
    actor_num_samples: int = 32
    flow_steps: int = 10
    bc_flow_alpha: float = 100.0
    use_fourier_features: bool = False
    fourier_feature_dim: int = 64
    weight_decay: float = 0.0

    # SAC/RLPD settings used by ACRLPD.
    target_entropy: float | None = None
    target_entropy_multiplier: float = 0.5
    init_temp: float = 1.0
    bc_alpha: float = 0.0

    # Logging/evaluation.
    save_dir: Path = Path("exp")
    exp_name: str | None = None
    log_interval: int = 1_000
    eval_interval: int = 25_000
    save_interval: int = -1
    eval_episodes: int = 20
    video_episodes: int = 0
    video_frame_skip: int = 3
    wandb_project: str = "rl-chunk-pusht"
    wandb_group: str = "debug"
    wandb_mode: Literal["online", "offline", "disabled"] = "disabled"


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def parse_bool_literal(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in TRUE_LITERALS:
        return True
    if normalized in FALSE_LITERALS:
        return False
    return None


def normalize_bool_cli_args(args: list[str]) -> list[str]:
    """Accept explicit bool values while keeping Tyro flag pairs.

    Tyro's default bool UX is `--flag` or `--no-flag`. For experiment commands it
    is easy to type `--flag true`, so normalize those forms before invoking
    Tyro.
    """

    normalized_args: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if not arg.startswith("--"):
            normalized_args.append(arg)
            i += 1
            continue

        raw_name = arg[2:]
        if "=" in raw_name:
            name, value = raw_name.split("=", 1)
            bool_value = parse_bool_literal(value)
            if bool_value is not None:
                if name in BOOLEAN_CLI_FIELDS:
                    normalized_args.append(f"--{name}" if bool_value else f"--no-{name}")
                    i += 1
                    continue
                if name.startswith("no-") and name[3:] in BOOLEAN_CLI_FIELDS:
                    normalized_args.append(f"--no-{name[3:]}" if bool_value else f"--{name[3:]}")
                    i += 1
                    continue

        name = raw_name
        is_negative = name.startswith("no-")
        field_name = name[3:] if is_negative else name
        if field_name in BOOLEAN_CLI_FIELDS and i + 1 < len(args):
            bool_value = parse_bool_literal(args[i + 1])
            if bool_value is not None:
                if is_negative:
                    normalized_args.append(f"--no-{field_name}" if bool_value else f"--{field_name}")
                else:
                    normalized_args.append(f"--{field_name}" if bool_value else f"--no-{field_name}")
                i += 2
                continue

        normalized_args.append(arg)
        i += 1

    return normalized_args


def build_agent_config(config: TrainConfig) -> dict[str, Any]:
    """Build the mutable config dictionary expected by the ported agents."""

    base = dict(
        lr=config.lr,
        batch_size=config.batch_size,
        actor_hidden_dims=config.actor_hidden_dims,
        value_hidden_dims=config.value_hidden_dims,
        layer_norm=config.layer_norm,
        actor_layer_norm=False,
        discount=config.discount,
        tau=config.tau,
        q_agg=config.q_agg,
        num_qs=config.num_qs,
        horizon_length=config.chunk_size,
        action_chunking=config.action_chunking,
    )

    if config.agent == "acfql":
        return dict(
            base,
            agent_name="acfql",
            alpha=config.bc_flow_alpha,
            flow_steps=config.flow_steps,
            normalize_q_loss=False,
            encoder=None,
            actor_type=config.actor_type,
            actor_num_samples=config.actor_num_samples,
            use_fourier_features=config.use_fourier_features,
            fourier_feature_dim=config.fourier_feature_dim,
            weight_decay=config.weight_decay,
        )

    if config.agent == "acrlpd":
        return dict(
            base,
            agent_name="acrlpd",
            target_entropy=config.target_entropy,
            target_entropy_multiplier=config.target_entropy_multiplier,
            init_temp=config.init_temp,
            bc_alpha=config.bc_alpha,
        )

    raise ValueError(f"Unsupported agent: {config.agent}")
