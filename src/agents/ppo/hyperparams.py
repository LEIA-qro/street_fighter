# hyperparams.py
#
# Pure PPO configuration. Kept free of SB3 model construction so it can be
# unit tested and driven by Optuna without spawning an environment.
#
# Reference for the choices below:
#   Huang, Dossa, Raffin, Kanervisto & Wang, "The 37 Implementation Details of
#   Proximal Policy Optimization", ICLR Blog Track, 2022.
#   https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

from typing import Callable, Optional

from agents.ppo.config import N_STEPS, BATCH_SIZE, NET_ARCH


def linear_schedule(initial: float, final: float = 0.0) -> Callable[[float], float]:
    """SB3 schedule: called with progress_remaining, 1.0 -> 0.0 over training.

    Detail #4 of the 37 implementation details: annealing the learning rate
    linearly to zero is part of the canonical PPO setup, not an optional extra.
    """
    def schedule(progress_remaining: float) -> float:
        return final + progress_remaining * (initial - final)
    return schedule


def resolve_override(cli_value: Optional[float], phase_value: float) -> float:
    """CLI overrides beat phase values -- and 0.0 is a real value.

    The previous `if x > 0.0` sentinel made `--ent_coef 0` a silent no-op.
    """
    return phase_value if cli_value is None else float(cli_value)


def build_ppo_kwargs(lr: float, ent_coef: float, clip_range: float,
                     device: str = "cpu", anneal_lr: bool = True,
                     target_kl: Optional[float] = None,
                     gamma: float = 0.995, gae_lambda: float = 0.95,
                     n_epochs: int = 4) -> dict:
    """Complete PPO kwargs minus policy / env / tensorboard_log.

    Defaults that differ from the previous inline configuration, and why:

    * target_kl None (was 0.03) -- SB3 aborts the epoch loop once
      approx_kl > 1.5 * target_kl. With clip_range ~0.26 that fired within
      one or two of the ten configured epochs, so most of every 32,768-sample
      rollout was thrown away. Lower n_epochs instead of truncating them.
    * n_epochs 4 (was 10) -- the standard PPO value for on-policy control.
    * gamma 0.995 (was 0.99) -- 0.99 gives a ~100 agent-step horizon; at
      FRAME_SKIP=4 that is ~6.7 seconds, far short of a full SF2 round. 0.995
      roughly doubles it so the value head can see the KO from mid-round.
    * learning_rate annealed by default.
    * vf_coef / max_grad_norm / normalize_advantage made explicit so Optuna
      can reach them.
    """
    learning_rate = linear_schedule(lr) if anneal_lr else lr
    return {
        "learning_rate": learning_rate,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "target_kl": target_kl,
        "policy_kwargs": dict(net_arch=NET_ARCH),
        "device": device,
    }
