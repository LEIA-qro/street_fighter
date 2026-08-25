# test_dqn_qrdqn_warmup_parity.py
#
# Guards against the QRDQN(...) warmup/update schedule silently drifting
# between the production trainer (agents/dqn/agent.py) and the Optuna
# tuner (agents/dqn/optuna_study.py). Before Task 8's DQN->QR-DQN swap,
# both files omitted learning_starts/train_freq/gradient_steps/
# target_update_interval and so implicitly inherited the same
# stable-baselines3 defaults. agent.py now sets them explicitly per the
# brief; if optuna_study.py is ever edited without updating agent.py (or
# vice versa), the tuner would score hyperparameters under a warmup
# regime production training never uses -- exactly the bug this test
# would have caught.
#
# This is a static-source check (parses the QRDQN(...) constructor call
# via ast and compares literal keyword values) rather than one that
# constructs an SB3 model or a training environment, so it stays cheap
# and does not require an emulator.

import ast
import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

WARMUP_KEYS = ("learning_starts", "train_freq", "gradient_steps", "target_update_interval")


def _qrdqn_constructor_kwargs(module_path):
    """Return the keyword args of the first QRDQN(...) *constructor* call in
    a file (as opposed to QRDQN.load(...), which is an attribute call and
    thus has a different ast.Call.func shape).
    """
    source = Path(module_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=module_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "QRDQN":
            kwargs = {}
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                try:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
                except ValueError:
                    # Non-literal keyword (e.g. policy_kwargs=dict(net_arch=net_arch),
                    # env=env) -- irrelevant to the warmup-schedule comparison.
                    continue
            return kwargs
    raise AssertionError(f"No QRDQN(...) constructor call found in {module_path}")


def test_dqn_agent_and_optuna_study_share_the_same_qrdqn_warmup_schedule():
    agent_path = os.path.join(SRC_PATH, "agents", "dqn", "agent.py")
    optuna_path = os.path.join(SRC_PATH, "agents", "dqn", "optuna_study.py")

    agent_kwargs = _qrdqn_constructor_kwargs(agent_path)
    optuna_kwargs = _qrdqn_constructor_kwargs(optuna_path)

    for key in WARMUP_KEYS:
        assert key in agent_kwargs, f"{key} missing from the QRDQN(...) call in dqn/agent.py"
        assert key in optuna_kwargs, f"{key} missing from the QRDQN(...) call in dqn/optuna_study.py"
        assert agent_kwargs[key] == optuna_kwargs[key], (
            f"{key} diverges: dqn/agent.py sets {agent_kwargs[key]!r}, "
            f"dqn/optuna_study.py sets {optuna_kwargs[key]!r}"
        )
