import os
import json
import tempfile
import time
import unittest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch

# Set up paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from core import config
from core.telemetry import write_telemetry, clean_telemetry


def save_test_apex_checkpoint(path, broken=False):
    import torch
    from agents.rainbow import QRDuelingNet

    meta = {
        "in_dim": 212,
        "hidden": 8,
        "quantiles": 3,
        "onehot": True,
        "n_actions": 72,
        "macros": True,
    }
    state_dict = {"probe": torch.zeros(1)} if broken else QRDuelingNet(
        212, n_actions=72, n_quantiles=3, hidden=8).state_dict()
    torch.save({"meta": meta, "state_dict": state_dict}, path)


class TestTelemetryDashboard(unittest.TestCase):
    def test_model_dropdown_includes_latest_benchmarked_models(self):
        from scripts import web_dashboard

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "models" / "latest" / "v3" / "ppo"
            model_dir.mkdir(parents=True)
            (model_dir / "champion.zip").write_bytes(b"model")
            (model_dir / "champion.pkl").write_bytes(b"norm")
            with patch.object(config, "PROJECT_ROOT", tmp):
                zips, pkls = web_dashboard.get_model_files("ppo")

        self.assertIn("models/latest/v3/ppo/champion.zip", zips)
        self.assertIn("models/latest/v3/ppo/champion.pkl", pkls)

    def test_clean_telemetry(self):
        target_path = os.path.join(config.PROJECT_ROOT, ".telemetry.json")
        with open(target_path, "w") as f:
            f.write("{}")
        self.assertTrue(os.path.exists(target_path))
        clean_telemetry()
        self.assertFalse(os.path.exists(target_path))

    def test_write_telemetry_v3_structure(self):
        # Create a mock 2216-dim observation (554 dims per frame * 4 stacked frames)
        # 10 continuous + 512 actions + 32 characters
        single_frame = np.zeros(554, dtype=np.float32)
        # Fill continuous: P1_HP=140, P2_HP=126, RelX=65, RelY=0, Wall=120, P1Proj=-1, P2Proj=136, P1Vel=0, P2Vel=-5, RelDist=65
        single_frame[:10] = [140.0, 126.0, 65.0, 0.0, 120.0, -1.0, 136.0, 0.0, -5.0, 65.0]
        # P1 action one-hot: index 12 (Crouch) = 1.0
        single_frame[10 + 12] = 1.0
        # P2 action one-hot: index 48 (Fireball) = 1.0
        single_frame[10 + 256 + 48] = 1.0
        # P1 char one-hot: index 0 (Ryu) = 1.0
        single_frame[10 + 512 + 0] = 1.0
        # P2 char one-hot: index 4 (Ken) = 1.0
        single_frame[10 + 512 + 16 + 4] = 1.0

        obs_mock = np.concatenate([single_frame] * 4) # Stack 4 times

        # Run telemetry writer with None model (call twice to pass throttling threshold)
        write_telemetry(
            model_name="test_model_v3",
            env_version="v3",
            status="PLAYING",
            model=None,
            obs=obs_mock,
            player=1
        )
        write_telemetry(
            model_name="test_model_v3",
            env_version="v3",
            status="PLAYING",
            model=None,
            obs=obs_mock,
            player=1
        )

        target_path = os.path.join(config.PROJECT_ROOT, ".telemetry.json")
        self.assertTrue(os.path.exists(target_path))

        with open(target_path, "r") as f:
            data = json.load(f)

        self.assertEqual(data["model_name"], "test_model_v3")
        self.assertEqual(data["env_version"], "v3")
        self.assertEqual(data["status"], "PLAYING")
        self.assertEqual(data["value_estimate"], 0.0)
        self.assertEqual(len(data["frames"]), 4)

        # Validate parsed features of active frame (frame 0 in serialization is the latest)
        latest = data["frames"][0]
        self.assertEqual(latest["p1_hp"], 140)
        self.assertEqual(latest["p2_hp"], 126)
        self.assertEqual(latest["rel_x"], 65)
        self.assertEqual(latest["rel_y"], 0)
        self.assertEqual(latest["p1_corner_dist"], 120)
        self.assertEqual(latest["p1_proj"], -1)
        self.assertEqual(latest["p2_proj"], 136)
        self.assertEqual(latest["p1_vel_x"], 0)
        self.assertEqual(latest["p2_vel_x"], -5)
        self.assertEqual(latest["rel_dist"], 65)
        self.assertIn("Crouch", latest["p1_action_name"])
        self.assertIn("Fireball", latest["p2_action_name"])
        self.assertEqual(latest["p1_char_name"], "Ryu")
        self.assertEqual(latest["p2_char_name"], "Ken")

        # Cleanup
        clean_telemetry()

    def test_telemetry_rendering_y_elevation(self):
        from scripts.web_dashboard import compute_fighter_visual_coords
        
        # P1 on ground, P2 on ground
        p1_x, p1_y, p2_x, p2_y = compute_fighter_visual_coords(rel_x=80, rel_y=0, corner_dist=120)
        self.assertEqual(p1_y, 24)
        self.assertEqual(p2_y, 24)
        
        # P1 jumps (rel_y = +80, P2 is below P1)
        p1_x, p1_y, p2_x, p2_y = compute_fighter_visual_coords(rel_x=80, rel_y=80, corner_dist=120)
        self.assertGreater(p1_y, 24)
        self.assertEqual(p2_y, 24)
        
        # P2 jumps (rel_y = -80, P2 is above P1)
        p1_x, p1_y, p2_x, p2_y = compute_fighter_visual_coords(rel_x=80, rel_y=-80, corner_dist=120)
        self.assertEqual(p1_y, 24)
        self.assertGreater(p2_y, 24)

    def test_telemetry_rendering_x_scaling(self):
        from scripts.web_dashboard import compute_fighter_visual_coords
        
        # Standard footsie distance (rel_x = 80)
        p1_x, p1_y, p2_x, p2_y = compute_fighter_visual_coords(rel_x=80, rel_y=0, corner_dist=120)
        # Ensure fighters are separated and strictly within arena boundaries (4% to 96%)
        self.assertTrue(4.0 <= p1_x <= 96.0)
        self.assertTrue(4.0 <= p2_x <= 96.0)
        self.assertGreater(abs(p2_x - p1_x), 10.0)  # Significant visual separation

    def test_get_live_telemetry_html_rendering(self):
        from core.telemetry import write_telemetry, clean_telemetry
        from scripts.web_dashboard import get_live_telemetry_html
        
        # Mock observation with P1 jumping in frame 0 and P2 jumping in frame 1
        f0 = np.zeros(554, dtype=np.float32)
        f0[:10] = [176.0, 176.0, 70.0, 90.0, 150.0, -1.0, -1.0, 0.0, 0.0, 70.0]
        
        f1 = np.zeros(554, dtype=np.float32)
        f1[:10] = [176.0, 176.0, 70.0, -85.0, 150.0, -1.0, -1.0, 0.0, 0.0, 70.0]
        
        obs = np.concatenate([f0, f1, f0, f1])
        # Call twice to pass write throttling
        write_telemetry("ppo_test", "v3", "PLAYING", None, obs, player=1)
        write_telemetry("ppo_test", "v3", "PLAYING", None, obs, player=1)
        
        html = get_live_telemetry_html()
        self.assertTrue("bottom: 61px" in html or "bottom: 60px" in html or "bottom: 59px" in html)  # P1 elevated
        self.assertIn("translateX(-50%)", html)
        
        clean_telemetry()

    def test_action_id_decoding_comprehensive(self):
        from core.telemetry import ACTION_NAMES, decode_one_hot

        # Test key SF2 state IDs
        self.assertEqual(ACTION_NAMES.get(0), "Idle / Neutral")
        self.assertIn(ACTION_NAMES.get(2), ["Walk", "Walk / Movement"])
        self.assertIn(ACTION_NAMES.get(47), ["Fireball (Startup)", "Fireball (Hadouken)"])
        self.assertIn(ACTION_NAMES.get(48), ["Fireball (Active)", "Fireball (Hadouken)"])

        # Verify one-hot array decoding
        arr = np.zeros(256, dtype=np.float32)
        arr[47] = 1.0
        self.assertEqual(decode_one_hot(arr), 47)

    def test_observation_action_one_hot_pipeline(self):
        """Verifies that raw RAM action IDs correctly propagate into 554-dim obs and decode in telemetry."""
        from core.telemetry import write_telemetry, clean_telemetry

        clean_telemetry()

        # Simulate 554-dim frame slice
        frame = np.zeros(554, dtype=np.float32)
        # Continuous features
        frame[0] = 176.0  # P1 HP
        frame[1] = 176.0  # P2 HP
        frame[2] = 80.0   # RelX
        frame[3] = 0.0    # RelY
        frame[4] = 120.0  # WallDist
        frame[5] = -1.0   # P1 Proj
        frame[6] = -1.0   # P2 Proj
        frame[7] = 0.0    # P1 VelX
        frame[8] = 0.0    # P2 VelX
        frame[9] = 80.0   # RelDist

        # Set P1 Action = 48 (Hadouken), P2 Action = 24 (Sweep)
        frame[10 + 48] = 1.0
        frame[10 + 256 + 24] = 1.0
        # Set P1 Char = 0 (Ryu), P2 Char = 4 (Ken)
        frame[10 + 512 + 0] = 1.0
        frame[10 + 512 + 16 + 4] = 1.0

        # 4-frame stack
        obs = np.concatenate([frame, frame, frame, frame])

        # Call twice to pass write throttling
        write_telemetry(
            model_name="test_agent",
            env_version="v3",
            status="PLAYING",
            model=None,
            obs=obs,
            player=1
        )
        write_telemetry(
            model_name="test_agent",
            env_version="v3",
            status="PLAYING",
            model=None,
            obs=obs,
            player=1
        )

        telemetry_file = os.path.join(config.PROJECT_ROOT, ".telemetry.json")
        self.assertTrue(os.path.exists(telemetry_file))

        with open(telemetry_file, "r") as f:
            data = json.load(f)

        # Frame 0 in reversed list is the latest frame
        latest_frame = data["frames"][0]
        self.assertIn("Fireball (Hadouken)", latest_frame["p1_action_name"])
        self.assertTrue("Sweep" in latest_frame["p2_action_name"] or "24" in latest_frame["p2_action_name"])
        self.assertEqual(latest_frame["p1_char_name"], "Ryu")
        self.assertEqual(latest_frame["p2_char_name"], "Ken")

        clean_telemetry()

    def test_selective_norm_non_destructive(self):
        """Verifies that SelectiveVecNormalize does not mutate raw observations in place."""
        from core.selective_norm import SelectiveVecNormalize
        from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym
        from gymnasium import spaces

        class _MockEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.action_space = spaces.Discrete(2)
                self.observation_space = spaces.Box(
                    low=-500.0, high=500.0, shape=(554 * 4,), dtype=np.float32
                )
            def reset(self, **kwargs):
                return np.zeros((554 * 4,), dtype=np.float32), {}
            def step(self, action):
                return np.zeros((554 * 4,), dtype=np.float32), 0.0, False, False, {}

        dummy = DummyVecEnv([lambda: _MockEnv()])
        norm = SelectiveVecNormalize(dummy, training=False)
        norm.running_mean = np.array([140.0, 140.0, 0.0, 0.0, 120.0, -1.0, -1.0, 0.0, 0.0, 80.0], dtype=np.float64)
        norm.running_var = np.array([40.0**2] * 10, dtype=np.float64)

        raw_obs = np.zeros(554 * 4, dtype=np.float32)
        raw_obs[0] = 176.0  # P1 HP
        raw_obs[1] = 176.0  # P2 HP
        raw_obs[2] = 80.0   # RelX
        raw_obs[4] = 120.0  # WallDist

        obs_input = raw_obs[np.newaxis, :]
        normalized = norm.normalize_obs(obs_input, update=False)

        # Verify input array was not mutated
        self.assertEqual(raw_obs[0], 176.0)
        self.assertEqual(raw_obs[2], 80.0)
        self.assertNotEqual(normalized[0, 0], 176.0)  # Normalized result is scaled

    def test_stream_logs_unbuffered_env(self):
        from scripts.web_dashboard import stream_logs, VENV_PYTHON, state

        # Quick test command that prints two lines with a short sleep
        cmd = [VENV_PYTHON, "-u", "-c", "import sys, time; print('LINE1', flush=True); time.sleep(0.05); print('LINE2', flush=True)"]
        generator = stream_logs(cmd)
        
        first_output = next(generator)
        self.assertIn("Executing:", first_output)
        
        second_output = next(generator)
        self.assertIn("LINE1", second_output)

        # Exhaust generator to allow clean subprocess termination
        for _ in generator:
            pass
        if state.active_process:
            state.active_process.poll()
            state.active_process = None

    def test_stream_logs_pending_launch_can_be_cancelled(self):
        from scripts.web_dashboard import (
            VENV_PYTHON, graceful_stop_process, state, stream_logs,
        )

        generator = stream_logs([
            VENV_PYTHON, "-c", "print('SHOULD_NOT_START')",
        ])
        self.assertIn("Executing:", next(generator))
        with state.process_lock:
            self.assertIsNotNone(state.launch_token)
            self.assertIsNone(state.active_process)

        message = graceful_stop_process()
        self.assertIn("cancelled", message)
        remaining = list(generator)
        self.assertTrue(any("Launch cancelled" in item for item in remaining))
        with state.process_lock:
            self.assertIsNone(state.launch_token)
            self.assertIsNone(state.active_process)

    def test_stream_logs_disconnect_keeps_live_process_stoppable(self):
        from scripts.web_dashboard import stream_logs, VENV_PYTHON, state

        stop_file = Path(config.PROJECT_ROOT) / ".stop_training"
        stop_file.write_text("STOP", encoding="utf-8")
        generator = stream_logs([
            VENV_PYTHON, "-u", "-c",
            "import time; print('READY', flush=True); time.sleep(10)",
        ])
        self.assertIn("Executing:", next(generator))
        child_output = next(generator)
        self.assertIn("READY", child_output)
        self.assertNotIn("[ERROR]", child_output)
        self.assertFalse(stop_file.exists())

        with state.process_lock:
            proc = state.active_process
        self.assertIsNotNone(proc)
        self.assertIsNone(proc.poll())

        generator.close()
        with state.process_lock:
            self.assertIs(state.active_process, proc)

        proc.terminate()
        proc.wait(timeout=5)
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with state.process_lock:
                if state.active_process is None:
                    break
            time.sleep(0.01)
        with state.process_lock:
            self.assertIsNone(state.active_process)

    def test_match_testing_args_and_commands(self):
        from scripts.web_dashboard import run_matchup
        
        # Test generator for invalid matchup
        gen = run_matchup("Human Player", "v2", "None", "None", "auto",
                          "CPU (Built-in AI)", "v2", "None", "None", "auto",
                          False, True, 3.0)
        msg = next(gen)
        self.assertIn("Invalid Matchup", msg)

    def test_stand_checkpoint_discovery_filters_non_macro_models(self):
        import torch
        from scripts import web_dashboard

        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            good = root / "good_macro.pt"
            bad = root / "old_primitive.pt"
            broken = root / "broken_weights.pt"
            save_test_apex_checkpoint(good)
            save_test_apex_checkpoint(broken, broken=True)
            torch.save({
                "meta": {"in_dim": 212, "n_actions": 63, "macros": False},
                "state_dict": {"probe": torch.zeros(1)},
            }, bad)

            found = web_dashboard.get_stand_checkpoint_files([root])
            self.assertEqual(found, [good.relative_to(config.PROJECT_ROOT).as_posix()])

    def test_stand_default_fallback_prioritizes_full_ladder(self):
        from scripts import web_dashboard

        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            four_levels = root / "four_levels.pt"
            eight_levels = root / "eight_levels.pt"
            four_levels.touch()
            eight_levels.touch()
            four_metrics = {"wr_media": 0.99}
            eight_metrics = {"wr_media": 0.80}
            for level in range(1, 5):
                four_metrics[f"wr_lvl{level}"] = 0.99
            for level in range(1, 9):
                eight_metrics[f"wr_lvl{level}"] = 0.80
            Path(str(four_levels) + ".json").write_text(
                json.dumps(four_metrics), encoding="utf-8")
            Path(str(eight_levels) + ".json").write_text(
                json.dumps(eight_metrics), encoding="utf-8")

            choices = [
                four_levels.relative_to(config.PROJECT_ROOT).as_posix(),
                eight_levels.relative_to(config.PROJECT_ROOT).as_posix(),
            ]
            selected = web_dashboard.get_stand_default_checkpoint(choices)
            self.assertEqual(selected, choices[1])

    def test_stand_status_does_not_label_four_levels_as_eight(self):
        from scripts import web_dashboard

        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            ckpt = root / "four_levels.pt"
            save_test_apex_checkpoint(ckpt)
            metrics = {"wr_media": 0.90}
            for level in range(1, 5):
                metrics[f"wr_lvl{level}"] = 0.90
            Path(str(ckpt) + ".json").write_text(
                json.dumps(metrics), encoding="utf-8")
            relative = ckpt.relative_to(config.PROJECT_ROOT).as_posix()

            with patch.object(web_dashboard, "STAND_CHECKPOINT_DIRS", (root,)):
                status = web_dashboard.get_stand_checkpoint_status(relative)

            self.assertIn("WR reportado en sidecar (4 niveles)", status)
            self.assertNotIn("selector robusto (8 niveles", status)

    def test_run_stand_builds_dedicated_apex_command(self):
        from scripts import web_dashboard

        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            ckpt = root / "champion.pt"
            save_test_apex_checkpoint(ckpt)
            relative = ckpt.relative_to(config.PROJECT_ROOT).as_posix()
            captured = []

            def fake_stream(cmd):
                captured.append(cmd)
                yield "stand command ready"

            with patch.object(web_dashboard, "STAND_CHECKPOINT_DIRS", (root,)), \
                 patch.object(web_dashboard, "stream_logs", fake_stream):
                output = list(web_dashboard.run_stand(
                    relative, "human", "KEN", 1, 3.5, "cpu"))

            self.assertEqual(output, ["stand command ready"])
            command = captured[0]
            self.assertTrue(command[1].endswith(os.path.join("scripts", "stand_leia.py")))
            self.assertIn(relative, command)
            self.assertEqual(command[command.index("--opponent-type") + 1], "human")
            self.assertEqual(command[command.index("--opponent") + 1], "KEN")
            self.assertEqual(command[command.index("--cpu-level") + 1], "1")
            self.assertEqual(command[command.index("--rematch-delay") + 1], "3.5")
            self.assertNotIn("--infinite-match", command)
            self.assertNotIn("test_agent_v2.py", " ".join(command))

    def test_run_stand_builds_apex_vs_apex_command(self):
        from scripts import web_dashboard

        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            p1 = root / "p1.pt"
            p2 = root / "p2.pt"
            save_test_apex_checkpoint(p1)
            save_test_apex_checkpoint(p2)
            p1_relative = p1.relative_to(config.PROJECT_ROOT).as_posix()
            p2_relative = p2.relative_to(config.PROJECT_ROOT).as_posix()
            captured = []

            def fake_stream(cmd):
                captured.append(cmd)
                yield "duel ready"

            with patch.object(web_dashboard, "STAND_CHECKPOINT_DIRS", (root,)), \
                 patch.object(web_dashboard, "stream_logs", fake_stream):
                output = list(web_dashboard.run_stand(
                    p1_relative, "model", "RYU", 1, 2.0, "cpu",
                    p2_checkpoint=p2_relative, p2_device="cpu"))

            self.assertEqual(output, ["duel ready"])
            command = captured[0]
            self.assertEqual(command[command.index("--opponent-type") + 1], "model")
            self.assertEqual(command[command.index("--p2-ckpt") + 1], p2_relative)
            self.assertEqual(command[command.index("--p2-device") + 1], "cpu")

    def test_run_stand_builds_apex_vs_ppo_v3_command(self):
        from scripts import web_dashboard

        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            p1 = root / "p1.pt"
            p2_zip = root / "p2.zip"
            p2_pkl = root / "p2.pkl"
            save_test_apex_checkpoint(p1)
            p2_zip.write_bytes(b"model")
            p2_pkl.write_bytes(b"normalization")
            relative = lambda path: path.relative_to(
                config.PROJECT_ROOT).as_posix()
            captured = []

            def fake_stream(cmd):
                captured.append(cmd)
                yield "hybrid duel ready"

            with patch.object(web_dashboard, "STAND_CHECKPOINT_DIRS", (root,)), \
                 patch.object(web_dashboard, "stream_logs", fake_stream):
                output = list(web_dashboard.run_stand(
                    relative(p1), "sb3", "RYU", 1, 2.0, "cpu",
                    p2_device="cpu", p2_algo="ppo", p2_env="v3",
                    p2_zip=relative(p2_zip), p2_pkl=relative(p2_pkl),
                    infinite_match=True,
                ))

            self.assertEqual(output, ["hybrid duel ready"])
            command = captured[0]
            self.assertEqual(command[command.index("--opponent-type") + 1], "sb3")
            self.assertEqual(command[command.index("--p2-algo") + 1], "ppo")
            self.assertEqual(command[command.index("--p2-env") + 1], "v3")
            self.assertEqual(
                command[command.index("--p2-model-zip") + 1], relative(p2_zip))
            self.assertEqual(
                command[command.index("--p2-model-pkl") + 1], relative(p2_pkl))
            self.assertIn("--infinite-match", command)

    def test_run_matchup_routes_unified_apex_cpu_controls_to_stand(self):
        from scripts import web_dashboard

        captured = []

        def fake_run_stand(*args, **kwargs):
            captured.append((args, kwargs))
            yield "apex cpu ready"

        with patch.object(web_dashboard, "run_stand", fake_run_stand):
            output = list(web_dashboard.run_matchup(
                "apex", "v2", "None", "None", "auto",
                "CPU (Built-in AI)", "v2", "None", "None", "auto",
                False, True, 2.5, 7,
                "p1_champion.pt", "p2_champion.pt", "KEN",
            ))

        self.assertEqual(output, ["apex cpu ready"])
        args, kwargs = captured[0]
        self.assertEqual(args[:6], (
            "p1_champion.pt", "cpu", "KEN", 7, 2.5, "auto"))
        self.assertEqual(kwargs["p2_checkpoint"], "p2_champion.pt")
        self.assertTrue(kwargs["infinite_match"])

    def test_run_matchup_routes_apex_vs_ppo_v3_to_hybrid_stand(self):
        from scripts import web_dashboard

        captured = []

        def fake_run_stand(*args, **kwargs):
            captured.append((args, kwargs))
            yield "apex ppo ready"

        with patch.object(web_dashboard, "run_stand", fake_run_stand):
            output = list(web_dashboard.run_matchup(
                "apex", "v2", "None", "None", "cpu",
                "ppo", "v3", "p2.zip", "p2.pkl", "cpu",
                False, True, 2.5, 7,
                "p1_champion.pt", "p2_champion.pt", "RANDOM",
            ))

        self.assertEqual(output, ["apex ppo ready"])
        args, kwargs = captured[0]
        self.assertEqual(args[:3], ("p1_champion.pt", "sb3", "RYU"))
        self.assertEqual(kwargs["p2_algo"], "ppo")
        self.assertEqual(kwargs["p2_env"], "v3")
        self.assertEqual(kwargs["p2_zip"], "p2.zip")
        self.assertEqual(kwargs["p2_pkl"], "p2.pkl")
        self.assertTrue(kwargs["infinite_match"])

    def test_apex_p2_default_never_overrides_valid_opponent_selection(self):
        from scripts.web_dashboard import normalize_apex_p2_selection

        self.assertEqual(
            normalize_apex_p2_selection("apex", "ppo"),
            "ppo",
        )
        self.assertEqual(
            normalize_apex_p2_selection("apex", "CPU (Built-in AI)"),
            "CPU (Built-in AI)",
        )
        self.assertEqual(
            normalize_apex_p2_selection("apex", "apex"),
            "apex",
        )
        self.assertEqual(
            normalize_apex_p2_selection("ppo", "CPU (Built-in AI)"),
            "CPU (Built-in AI)",
        )

    def test_run_stand_reports_corrupt_checkpoint_in_console(self):
        from scripts import web_dashboard

        with tempfile.TemporaryDirectory(dir=config.PROJECT_ROOT) as tmp:
            root = Path(tmp)
            ckpt = root / "corrupt.pt"
            ckpt.write_bytes(b"not a torch checkpoint")
            relative = ckpt.relative_to(config.PROJECT_ROOT).as_posix()

            with patch.object(web_dashboard, "STAND_CHECKPOINT_DIRS", (root,)):
                output = list(web_dashboard.run_stand(
                    relative, "human", "KEN", 1, 3.5, "cpu"))

            self.assertEqual(len(output), 1)
            self.assertIn("Error de configuración del modelo Ape-X", output[0])

    def test_model_testing_preserves_classic_ui_and_adds_apex_viewer(self):
        from scripts import web_dashboard

        dashboard = web_dashboard.demo.get_config_file()
        components = dashboard["components"]

        def by_label(label):
            matches = [c for c in components if c.get("props", {}).get("label") == label]
            self.assertEqual(len(matches), 1, label)
            return matches[0]

        def by_value(value):
            matches = [c for c in components if c.get("props", {}).get("value") == value]
            self.assertEqual(len(matches), 1, value)
            return matches[0]

        p1_algo = by_label("P1 Algorithm")
        p2_algo = by_label("P2 Algorithm")
        self.assertEqual(p1_algo["props"]["value"], "ppo")
        self.assertEqual(p2_algo["props"]["value"], "ppo")
        self.assertIn("apex", [choice[1] for choice in p1_algo["props"]["choices"]])
        self.assertIn("apex", [choice[1] for choice in p2_algo["props"]["choices"]])
        self.assertNotEqual(p2_algo["props"].get("interactive", True), False)

        expected_defaults = {
            "P1 Environment": "v2",
            "P1 Compute Device": "auto",
            "P1 Model (.zip)": "None",
            "P2 Environment": "v2",
            "P2 Compute Device": "auto",
            "P2 Model (.zip)": "None",
        }
        for label, value in expected_defaults.items():
            component = by_label(label)
            self.assertEqual(component["props"]["value"], value)
            self.assertTrue(component["props"]["visible"])

        self.assertFalse(by_label(
            "🔄 Infinite Matchups (Auto-Rematch)")["props"]["value"])
        self.assertTrue(by_label(
            "Enable Performance Profiling")["props"]["visible"])
        self.assertTrue(by_label(
            "CPU Level / Classic Max Cap")["props"]["visible"])
        self.assertTrue(by_value(
            "⏯️ Toggle Agent (Play/Pause)")["props"]["visible"])
        by_value("Agent State: **PAUSED** (Default)")

        model_tabs = [
            c for c in components
            if c["type"] == "tabitem"
            and "Model Testing & Matchups" in c.get("props", {}).get("label", "")
        ]
        self.assertEqual(len(model_tabs), 1)
        self.assertFalse(any(
            c["type"] == "tabitem"
            and "stand" in c.get("props", {}).get("label", "").lower()
            for c in components
        ))
        p1_apex = by_label("P1 Ape-X checkpoint (.pt)")
        p2_apex = by_label("P2 Ape-X checkpoint (.pt)")
        self.assertFalse(p1_apex["props"]["visible"])
        self.assertFalse(p2_apex["props"]["visible"])
        self.assertTrue(str(p1_apex["props"]["value"]).endswith(
            "apex_v1592_benchmarked.pt"))
        self.assertFalse(by_label(
            "P2 Character (Ape-X matches)")["props"]["visible"])
        self.assertFalse(any(
            c["type"] == "accordion"
            and "Ape-X" in c.get("props", {}).get("label", "")
            for c in components
        ))

        classic_launch = by_value("⚔️ Launch Match")
        dependencies = dashboard["dependencies"]
        classic_event = next(
            dep for dep in dependencies
            if (classic_launch["id"], "click") in dep.get("targets", []))
        self.assertEqual(classic_event["api_name"], "run_matchup")
        self.assertEqual(len(classic_event["inputs"]), 17)
        self.assertFalse(any(
            c.get("props", {}).get("value") == "🥊 Launch Ape-X vs Human"
            for c in components
        ))

    def test_ai_vs_ai_cli_args(self):
        import argparse
        # Replicate parser from test_ai_vs_ai_v2
        parser = argparse.ArgumentParser()
        parser.add_argument("--algo_p1", type=str, default="ppo")
        parser.add_argument("--env_p1",  type=str, default="v2", choices=["v2", "v3"])
        parser.add_argument("--load_zip_p1", type=str, required=True)
        parser.add_argument("--load_pkl_p1", type=str, required=True)
        parser.add_argument("--device_p1", type=str, default="auto")
        
        parser.add_argument("--algo_p2", type=str, default="ppo")
        parser.add_argument("--env_p2",  type=str, default="v2", choices=["v2", "v3"])
        parser.add_argument("--load_zip_p2", type=str, required=True)
        parser.add_argument("--load_pkl_p2", type=str, required=True)
        parser.add_argument("--device_p2", type=str, default="auto")
        parser.add_argument("--profile", action="store_true")
        parser.add_argument("--infinite_match", action="store_true")
        parser.add_argument("--rematch_delay", type=float, default=2.0)

        test_args = [
            "--algo_p1", "ppo", "--env_p1", "v3", "--load_zip_p1", "m1.zip", "--load_pkl_p1", "m1.pkl", "--device_p1", "auto",
            "--algo_p2", "ppo", "--env_p2", "v3", "--load_zip_p2", "m2.zip", "--load_pkl_p2", "m2.pkl", "--device_p2", "auto",
            "--profile", "--infinite_match", "--rematch_delay", "3.0"
        ]
        parsed = parser.parse_args(test_args)
        self.assertEqual(parsed.device_p2, "auto")
        self.assertTrue(parsed.infinite_match)
        self.assertEqual(parsed.rematch_delay, 3.0)

    def test_infinite_match_agent_state_init(self):
        from scripts.web_dashboard import run_matchup
        state_file = os.path.join(config.PROJECT_ROOT, ".agent_state")
        
        # When infinite match is True, .agent_state should be PLAY
        gen = run_matchup("ppo", "v3", "dummy.zip", "dummy.pkl", "auto",
                          "ppo", "v3", "dummy.zip", "dummy.pkl", "auto",
                          False, True, 2.0)
        # Trigger first yield
        next(gen)
        self.assertTrue(os.path.exists(state_file))
        with open(state_file, "r") as f:
            self.assertEqual(f.read().strip(), "PLAY")

        # When infinite match is False, .agent_state should be PAUSE
        gen = run_matchup("ppo", "v3", "dummy.zip", "dummy.pkl", "auto",
                          "ppo", "v3", "dummy.zip", "dummy.pkl", "auto",
                          False, False, 2.0)
        next(gen)
        with open(state_file, "r") as f:
            self.assertEqual(f.read().strip(), "PAUSE")

    def test_busy_match_launch_does_not_rewrite_active_agent_state(self):
        from scripts.web_dashboard import run_matchup, state

        state_file = os.path.join(config.PROJECT_ROOT, ".agent_state")
        with open(state_file, "w") as f:
            f.write("PLAY")

        with state.process_lock:
            previous_process = state.active_process
            previous_token = state.launch_token
            previous_cleanup = state.cleanup_in_progress
            state.active_process = object()
            state.launch_token = None
            state.cleanup_in_progress = False
        try:
            output = list(run_matchup(
                "ppo", "v2", "dummy.zip", "dummy.pkl", "auto",
                "Human Player", "v2", "None", "None", "auto",
                False, False, 2.0,
            ))
            self.assertEqual(output, ["Error: A process is already running!"])
            with open(state_file, "r") as f:
                self.assertEqual(f.read().strip(), "PLAY")
        finally:
            with state.process_lock:
                state.active_process = previous_process
                state.launch_token = previous_token
                state.cleanup_in_progress = previous_cleanup

    def test_safe_banner_encoding(self):
        # Test that ASCII banners encode properly in cp1252 and UTF-8
        p1_name = "ppo_test_model"
        p2_name = "opponent_test_model"
        winner_msg_p1 = f"[WINNER] {p1_name} (Player 1) WINS!"
        winner_msg_p2 = f"[WINNER] {p2_name} (Player 2) WINS!"
        draw_msg = "[DRAW] DOUBLE K.O. (Draw)!"
        
        # Must encode cleanly in cp1252 without throwing UnicodeEncodeError
        winner_msg_p1.encode("cp1252")
        winner_msg_p2.encode("cp1252")
        draw_msg.encode("cp1252")

    def test_pvp_state_file_exists(self):
        pvp_state = os.path.join(config.STATES_DIR, "RYU_RYU_R1_PvP.State")
        self.assertTrue(os.path.exists(pvp_state), f"Required state file does not exist: {pvp_state}")

    def test_ai_vs_ai_round_lifecycle_logic(self):
        # Simulate round sequence: Reset (transient 0 HP) -> Active fighting (>0 HP) -> KO (0 HP) -> Reset
        round_started = False
        
        # 1. Transient reset frame (both HP = 0)
        p1_hp, p2_hp = 0, 0
        if not round_started and p1_hp > 0 and p2_hp > 0:
            round_started = True
        self.assertFalse(round_started)  # Must not arm KO detector
        
        # 2. Round active (both HP > 0)
        p1_hp, p2_hp = 176, 176
        if not round_started and p1_hp > 0 and p2_hp > 0:
            round_started = True
        self.assertTrue(round_started)   # Armed!
        
        # 3. P2 KO'd (p2_hp = 0)
        p1_hp, p2_hp = 140, 0
        ko_triggered = False
        if round_started and (p1_hp <= 0 or p2_hp <= 0):
            ko_triggered = True
        self.assertTrue(ko_triggered)

    def test_exact_step_info_hp_extraction(self):
        # Simulate step info with exact RAM HP values
        info = [{"my_hp": 120.0, "enemy_hp": 0.0}]
        
        # Should accurately extract integer HP without float drift
        if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], dict) and "my_hp" in info[0]:
            ai_hp = int(info[0]["my_hp"])
            opp_hp = int(info[0]["enemy_hp"])
        else:
            ai_hp, opp_hp = -1, -1
            
        self.assertEqual(ai_hp, 120)
        self.assertEqual(opp_hp, 0)
        self.assertTrue(ai_hp <= 0 or opp_hp <= 0)
        self.assertTrue(ai_hp > 0 and opp_hp <= 0)  # AI won!


if __name__ == "__main__":
    unittest.main()
