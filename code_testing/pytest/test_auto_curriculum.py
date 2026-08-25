# test_auto_curriculum.py
#
# Robust offline unit test suite for AutoCurriculumCallback.
# Verifies:
#   1. Lottery pool weighting mathematical correctness
#   2. Telemetry outcome logging into decoupled per-state win buffers
#   3. Active-only evaluation stability gating triggers
#   4. Dynamic state introduction (2 at a time micro-steps)
#   5. Complete state JSON serialization & restoration (for resume safety)
#
# Runs strictly offline (no emulator subprocesses or socket connections needed).
#

import os
import sys
import shutil
import tempfile
import unittest
from collections import deque

from pathlib import Path

# Inject project root path dynamically
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import core.config as config
from agents.auto_curriculum_callback import AutoCurriculumCallback


class TestAutoCurriculum(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for scratch files
        self.test_dir = tempfile.mkdtemp()
        
        # Ensure residual stop signals do not disrupt tests
        stop_file = os.path.join(config.PROJECT_ROOT, ".stop_training")
        if os.path.exists(stop_file):
            try:
                os.remove(stop_file)
            except OSError:
                pass
        
        # Mappings of phases / levels
        self.phase_hyperparams = {
            0: {"lr": 1e-4, "ent_coef": 0.01, "clip": 0.2},
            1: {"lr": 8e-5, "ent_coef": 0.008, "clip": 0.2},
            2: {"lr": 5e-5, "ent_coef": 0.005, "clip": 0.15},
            3: {"lr": 2e-5, "ent_coef": 0.002, "clip": 0.15}
        }
        
        # Instantiate Callback offline (no model or env bound yet)
        self.callback = AutoCurriculumCallback(
            save_path=self.test_dir,
            phase_hyperparams=self.phase_hyperparams,
            start_level=1,
            eval_interval=500,
            save_interval=1000,
            algo="ppo",
            env_version="v2",
            model_name="test_agent",
            win_rate_threshold=0.75,
            stability_threshold=3,
            min_episodes_for_eval=10
        )

    def tearDown(self):
        # Remove temporary directory resources
        shutil.rmtree(self.test_dir)
        stop_file = os.path.join(config.PROJECT_ROOT, ".stop_training")
        if os.path.exists(stop_file):
            try:
                os.remove(stop_file)
            except OSError:
                pass

    def test_lottery_pool_weights(self):
        """Verify that state multiplicity replicates strictly match the weight ratios."""
        self.callback.current_level = 2
        
        # Introduce 2 states from Level 3
        introduced = ["RYU_BLANKA_R1_lvl3.State", "RYU_DHALSIM_R1_lvl3.State"]
        self.callback.introduced_states = introduced
        
        pool = self.callback.generate_weighted_lottery_pool()
        
        # 1. Mastered Level 1 states (Weight 1)
        for s in config.DIFFICULTY_LEVELS[1]:
            count = pool.count(s)
            self.assertEqual(count, 1, f"Expected rehearsal state {s} to have multiplicity 1, found {count}")
            
        # 2. Active Level 2 states (Weight 3)
        for s in config.DIFFICULTY_LEVELS[2]:
            count = pool.count(s)
            self.assertEqual(count, 3, f"Expected active state {s} to have multiplicity 3, found {count}")
            
        # 3. Introduced Level 3 states (Category weight 48 / 2 states = multiplicity 24)
        for s in introduced:
            count = pool.count(s)
            self.assertEqual(count, 24, f"Expected new state {s} to have multiplicity 24, found {count}")
            
        # 4. Unintroduced Level 3 states (Weight 0 - not in pool)
        unintroduced = [s for s in config.DIFFICULTY_LEVELS[3] if s not in introduced]
        for s in unintroduced:
            count = pool.count(s)
            self.assertEqual(count, 0, f"Expected unintroduced state {s} to be absent from pool, found {count}")

    def test_telemetry_capturing(self):
        """Verify that step outcomes feed correctly into decoupled per-state win buffers."""
        mock_infos = [
            {"win": 1, "state_file": "RYU_CHUNLI_R1_lvl1.State"},
            {"win": 0, "state_file": "RYU_CHUNLI_R1_lvl1.State"},
            {"win": 1, "state_file": "RYU_DHALSIM_R1_lvl1.State"}
        ]
        
        # Mock SB3 locals dict
        self.callback.locals = {"infos": mock_infos}
        
        # Trigger step callback logic
        self.callback._on_step()
        
        # Assertions
        chunli_buf = self.callback.state_win_buffers["RYU_CHUNLI_R1_lvl1.State"]
        dhalsim_buf = self.callback.state_win_buffers["RYU_DHALSIM_R1_lvl1.State"]
        
        self.assertEqual(len(chunli_buf), 2)
        self.assertEqual(list(chunli_buf), [1, 0])
        
        self.assertEqual(len(dhalsim_buf), 1)
        self.assertEqual(list(dhalsim_buf), [1])

    def test_stability_gating_and_promotion(self):
        """Verify that stability counters trigger 2-state micro-steps on mastery."""
        self.callback.current_level = 1
        self.callback.introduced_states = []
        self.callback.stability_counter = 0
        
        # Populate buffers of Level 1 states with high win rates (100% winrate)
        # Minimum episodes per state needed: 15 (min_samples_per_state)
        for i, state in enumerate(config.DIFFICULTY_LEVELS[1]):
            # Append 15 wins per state to clear confidence gating
            for _ in range(15):
                self.callback.state_win_buffers[state].append(1)

        # Mock locals to avoid crash on reward/status check
        self.callback.locals = {"infos": []}
        self.callback.num_timesteps = 500
        self.callback.last_eval_step = 0
        
        # Mock training_env and model methods
        class MockEnv:
            def env_method(self, name, arg): pass
            def save(self, path): pass
        class MockModel:
            def save(self, path): pass
            def get_env(self):
                return MockEnv()
            
        self.callback.model = MockModel()

        # Run step evaluations
        # First Eval: Win rate >= 75% -> stability counter = 1
        self.callback._on_step()
        self.assertEqual(self.callback.stability_counter, 1)
        self.assertEqual(len(self.callback.introduced_states), 0)

        # Second Eval: Win rate >= 75% -> stability counter = 2
        self.callback.num_timesteps = 1000
        self.callback._on_step()
        self.assertEqual(self.callback.stability_counter, 2)
        
        # Third Eval: Win rate >= 75% -> stability counter = 3 -> Triggers promotion!
        # Stability counter reset to 0, introduced_states size becomes 2 (first 2 of Level 2)
        self.callback.num_timesteps = 1500
        self.callback._on_step()
        
        self.assertEqual(self.callback.stability_counter, 0)
        self.assertEqual(len(self.callback.introduced_states), 2)
        
        # Target states introduced in alphabetical/sorted order
        expected_added = sorted(config.DIFFICULTY_LEVELS[2])[:2]
        self.assertEqual(self.callback.introduced_states, expected_added)

    def test_json_state_serialization_safety(self):
        """Verify that serialization accurately dumps and fully restores deques and trackers."""
        self.callback.current_level = 3
        self.callback.stability_counter = 2
        self.callback.introduced_states = ["RYU_BLANKA_R1_lvl4.State", "RYU_DHALSIM_R1_lvl4.State"]
        self.callback.num_timesteps = 45000
        self.callback.last_save_step = 40000
        self.callback.last_eval_step = 44500
        
        # Put some test values inside the win buffers
        self.callback.state_win_buffers["RYU_CHUNLI_R1_lvl1.State"].append(1)
        self.callback.state_win_buffers["RYU_CHUNLI_R1_lvl1.State"].append(0)
        self.callback.state_win_buffers["RYU_DHALSIM_R1_lvl1.State"].append(1)
        
        # Save state
        self.callback._save_curriculum_state()
        
        # Load state into a fresh, separate dictionary
        restored = AutoCurriculumCallback.load_state(self.test_dir)
        
        self.assertEqual(restored["current_level"], 3)
        self.assertEqual(restored["stability_counter"], 2)
        self.assertEqual(restored["introduced_states"], ["RYU_BLANKA_R1_lvl4.State", "RYU_DHALSIM_R1_lvl4.State"])
        self.assertEqual(restored["num_timesteps"], 45000)
        self.assertEqual(restored["last_save_step"], 40000)
        self.assertEqual(restored["last_eval_step"], 44500)
        
        # Check buffers restoration structure
        buffers_raw = restored["state_win_buffers"]
        self.assertEqual(buffers_raw["RYU_CHUNLI_R1_lvl1.State"], [1, 0])
        self.assertEqual(buffers_raw["RYU_DHALSIM_R1_lvl1.State"], [1])

    def test_apply_level_hyperparams_updates_model(self):
        """Verify that _apply_level_hyperparams correctly updates model learning_rate and clip_range."""
        import torch.nn as nn
        import torch.optim as optim

        class DummyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(1, 1)
                self.optimizer = optim.Adam(self.layer.parameters(), lr=1e-3)

        class DummyModel:
            def __init__(self):
                self.policy = DummyPolicy()
                self.learning_rate = 1e-3
                self.clip_range = 0.2
                self.ent_coef = 0.01

        self.callback.model = DummyModel()
        self.callback._apply_level_hyperparams(level=1)

        # Check that learning rate callable was assigned and returns expected value
        self.assertTrue(callable(self.callback.model.learning_rate))
        self.assertAlmostEqual(self.callback.model.learning_rate(1.0), 1e-4)
        self.assertAlmostEqual(self.callback.model.policy.optimizer.param_groups[0]["lr"], 1e-4)

        # Check that clip range callable was assigned and returns expected value
        self.assertTrue(callable(self.callback.model.clip_range))
        self.assertAlmostEqual(self.callback.model.clip_range(1.0), 0.2)

    def test_compatibility_methods(self):
        """Verify AutoCurriculumCallback satisfies the legacy callback interface."""
        # Populate dummy win buffers
        states = list(self.callback.state_win_buffers.keys())[:4]
        for s in states:
            self.callback.state_win_buffers[s].extend([1, 1, 0, 1])  # 3 wins / 4 = 75%
        
        # Test _win_rate()
        wr = self.callback._win_rate()
        self.assertIsInstance(wr, float)
        self.assertAlmostEqual(wr, 0.75)

        # Test current_phase property
        self.assertEqual(self.callback.current_phase, self.callback.current_level)

        # Test _save_phase_state() alias
        self.callback._save_phase_state()
        expected_json = os.path.join(self.test_dir, "auto_curriculum_state_test_agent.json")
        self.assertTrue(os.path.exists(expected_json))


if __name__ == "__main__":
    unittest.main()

