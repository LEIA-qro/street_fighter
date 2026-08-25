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
# buffer_size joined WARMUP_KEYS for the same reason: both call sites now
# pass the imported BUFFER_SIZE constant from agents/dqn/config.py rather
# than a literal, so a plain ast.literal_eval can't see it. _resolve_kwarg
# below additionally follows a bare Name back through the file's
# `from module import name` statements and imports the real module to read
# the live value, so the test compares what each file ACTUALLY passes at
# call time, not just its literal source text -- a hardcoded literal on
# one side and the imported constant on the other will correctly compare
# unequal (or fail to resolve) rather than silently passing.
#
# This is a static-source check (parses the QRDQN(...) constructor call
# via ast) rather than one that constructs an SB3 model or a training
# environment, so it stays cheap and does not require an emulator.

import ast
import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

WARMUP_KEYS = ("learning_starts", "train_freq", "gradient_steps", "target_update_interval", "buffer_size")


def _imported_names(tree):
    """Map local names bound by top-level `from module import name [as alias]`
    statements to (module, original_name), so a bare Name reference to an
    imported module-level constant can be resolved back to its source.
    """
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports[alias.asname or alias.name] = (node.module, alias.name)
    return imports


def _resolve_kwarg(value_node, import_map, module_path):
    """Resolve a keyword's AST value to a concrete, comparable value.

    Literals resolve directly. A bare Name that was imported at module level
    (e.g. `buffer_size=BUFFER_SIZE` after `from agents.dqn.config import
    BUFFER_SIZE`) is resolved by actually importing that module and reading
    the live attribute -- this is what makes the comparison catch a
    hardcoded-literal-vs-imported-constant divergence rather than just
    comparing two unresolved Name nodes as trivially "equal names".

    Anything else (a local variable, a function call, an attribute chain)
    cannot be soundly resolved without executing the file, so it raises
    instead of being silently skipped -- an un-resolvable value for a key
    under test should fail loudly, not vanish from the comparison.
    """
    try:
        return ast.literal_eval(value_node)
    except ValueError:
        pass
    if isinstance(value_node, ast.Name) and value_node.id in import_map:
        module_name, attr_name = import_map[value_node.id]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise ValueError(
        f"Cannot statically resolve keyword value {ast.dump(value_node)!r} in {module_path}"
    )


def _qrdqn_constructor_kwargs(module_path, keys):
    """Return the resolved values (see _resolve_kwarg) of `keys` among the
    keyword args of the first QRDQN(...) *constructor* call in a file (as
    opposed to QRDQN.load(...), which is an attribute call and thus has a
    different ast.Call.func shape). Keywords not in `keys` are skipped
    entirely -- e.g. policy_kwargs=dict(net_arch=net_arch), env=env -- since
    they're irrelevant to the comparison this test makes.
    """
    source = Path(module_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=module_path)
    import_map = _imported_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "QRDQN":
            kwargs = {}
            for kw in node.keywords:
                if kw.arg not in keys:
                    continue
                kwargs[kw.arg] = _resolve_kwarg(kw.value, import_map, module_path)
            return kwargs
    raise AssertionError(f"No QRDQN(...) constructor call found in {module_path}")


def test_dqn_agent_and_optuna_study_share_the_same_qrdqn_warmup_schedule():
    agent_path = os.path.join(SRC_PATH, "agents", "dqn", "agent.py")
    optuna_path = os.path.join(SRC_PATH, "agents", "dqn", "optuna_study.py")

    agent_kwargs = _qrdqn_constructor_kwargs(agent_path, WARMUP_KEYS)
    optuna_kwargs = _qrdqn_constructor_kwargs(optuna_path, WARMUP_KEYS)

    for key in WARMUP_KEYS:
        assert key in agent_kwargs, f"{key} missing from the QRDQN(...) call in dqn/agent.py"
        assert key in optuna_kwargs, f"{key} missing from the QRDQN(...) call in dqn/optuna_study.py"
        assert agent_kwargs[key] == optuna_kwargs[key], (
            f"{key} diverges: dqn/agent.py sets {agent_kwargs[key]!r}, "
            f"dqn/optuna_study.py sets {optuna_kwargs[key]!r}"
        )
