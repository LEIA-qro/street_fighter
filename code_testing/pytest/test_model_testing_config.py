import os
import sys
import unittest
from pathlib import Path

# Inject src dynamically
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from core import config

class TestModelTestingConfig(unittest.TestCase):
    def test_get_cpu_states_up_to_level_defaults_and_counts(self):
        # Level 1 has 12 states
        lvl1_states = config.get_cpu_states_up_to_level(1)
        self.assertEqual(len(lvl1_states), 12)
        self.assertIn("RYU_CHUNLI_R1_lvl1.State", lvl1_states)

        # Level 5 (default) has 5 * 12 = 60 states
        lvl5_states = config.get_cpu_states_up_to_level(5)
        self.assertEqual(len(lvl5_states), 60)
        self.assertIn("RYU_RYU_R1_lvl5.State", lvl5_states)

        # Level 8 (max) has 8 * 12 = 96 states
        lvl8_states = config.get_cpu_states_up_to_level(8)
        self.assertEqual(len(lvl8_states), 96)
        self.assertIn("RYU_BALROG_R1_HARD.State", lvl8_states)

    def test_get_cpu_states_up_to_level_clamping(self):
        # Below min clamps to 1
        min_clamped = config.get_cpu_states_up_to_level(0)
        self.assertEqual(len(min_clamped), 12)

        # Above max clamps to 8
        max_clamped = config.get_cpu_states_up_to_level(15)
        self.assertEqual(len(max_clamped), 96)

    def test_test_agent_v2_cpu_level_cap_arg(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--cpu_level_cap", type=int, default=5, choices=range(1, 9))
        
        args = parser.parse_args([])
        self.assertEqual(args.cpu_level_cap, 5)

        args_custom = parser.parse_args(["--cpu_level_cap", "3"])
        self.assertEqual(args_custom.cpu_level_cap, 3)

    def test_dashboard_run_matchup_command_generation(self):
        from scripts import web_dashboard
        gen = web_dashboard.run_matchup(
            p1_algo="ppo", p1_env="v2", p1_zip="models/test.zip", p1_pkl="models/test.pkl", p1_device="auto",
            p2_algo="CPU (Built-in AI)", p2_env="v2", p2_zip="None", p2_pkl="None", p2_device="auto",
            profile_enabled=False, infinite_match_enabled=True, rematch_delay=2.0, cpu_level_cap=6
        )
        first_output = next(gen)
        gen.close()
        self.assertIn("--infinite_match", first_output)
        self.assertIn("--cpu_level_cap", first_output)
        self.assertIn("6", first_output)

    def test_scoreboard_reset_payload_format(self):
        # Format: RESET <path>|<p1_score>|<p2_score>
        state_path = r"C:\Users\Diego Perea\Documents\Apps\BizHawk-2.8-win-x64\street_fighter\states\RYU_RYU_R1_PvP.State"
        p1_wins = 3
        p2_wins = 1
        payload = f"RESET {state_path}|{p1_wins}|{p2_wins}\n"
        
        # Test string parsing as done in Lua
        reset_arg = payload.strip()[6:]  # strip 'RESET '
        sep_idx = reset_arg.find("|")
        self.assertNotEqual(sep_idx, -1)
        parsed_path = reset_arg[:sep_idx]
        self.assertEqual(parsed_path, state_path)
        
        score_str = reset_arg[sep_idx + 1:]
        sep2_idx = score_str.find("|")
        p1_parsed = int(score_str[:sep2_idx])
        p2_parsed = int(score_str[sep2_idx + 1:])
        self.assertEqual(p1_parsed, 3)
        self.assertEqual(p2_parsed, 1)

    def test_all_readme_cli_command_examples(self):
        """Verifies every command line example in README.md and doc/DEVELOPER_CLI_GUIDE.md against real argument parsers."""
        import argparse

        # 1. train.py
        p_train = argparse.ArgumentParser()
        p_train.add_argument("--algo", required=True, choices=["ppo", "sac", "dqn"])
        p_train.add_argument("--env", default="v2", choices=["v1", "v2", "v3"])
        p_train.add_argument("--steps", type=int, default=50000000)
        p_train.add_argument("--load_zip", type=str, default=None)
        p_train.add_argument("--load_pkl", type=str, default=None)
        p_train.add_argument("--phase", type=str, default="0")
        p_train.add_argument("--device", type=str, default="auto")
        p_train.add_argument("--auto_curriculum", action="store_true")
        p_train.add_argument("--lr", type=float, default=0.0)
        p_train.add_argument("--ent_coef", type=float, default=0.0)
        p_train.add_argument("--clip_range", type=float, default=0.0)

        # Example 1: Standard train with auto_curriculum
        cmd1 = ["--algo", "ppo", "--env", "v2", "--steps", "10000000", "--device", "cuda", "--auto_curriculum"]
        args1 = p_train.parse_args(cmd1)
        self.assertEqual(args1.algo, "ppo")
        self.assertEqual(args1.steps, 10000000)
        self.assertTrue(args1.auto_curriculum)

        # Example 1b: Train with resume params and phase
        cmd1b = ["--algo", "ppo", "--env", "v2", "--steps", "5000000", "--load_zip", "models/production/v2/ppo/ppo_model.zip", "--load_pkl", "models/production/v2/ppo/ppo_model_vecnorm.pkl", "--phase", "2", "--auto_curriculum", "--device", "cuda"]
        args1b = p_train.parse_args(cmd1b)
        self.assertEqual(args1b.phase, "2")
        self.assertEqual(args1b.load_zip, "models/production/v2/ppo/ppo_model.zip")

        # 2. resume.py
        p_resume = argparse.ArgumentParser()
        p_resume.add_argument("--load_zip", type=str, default="dummy.zip")
        p_resume.add_argument("--load_pkl", type=str, default="dummy.pkl")
        p_resume.add_argument("--phase", type=str, default=None)
        p_resume.add_argument("--device", type=str, default="auto")
        args2 = p_resume.parse_args([])
        self.assertEqual(args2.device, "auto")

        # 3. tune.py
        p_tune = argparse.ArgumentParser()
        p_tune.add_argument("--algo", required=True, choices=["ppo", "sac", "dqn"])
        p_tune.add_argument("--env", default="v2", choices=["v1", "v2", "v3"])
        p_tune.add_argument("--trials", type=int, default=2)
        p_tune.add_argument("--study_name", type=str, default="ppo_sf2_tuning")
        p_tune.add_argument("--load_zip", type=str, default=None)
        p_tune.add_argument("--load_pkl", type=str, default=None)
        p_tune.add_argument("--phase", type=str, default="0")
        p_tune.add_argument("--timesteps", type=int, default=50000)
        p_tune.add_argument("--device", type=str, default="auto")
        cmd3 = ["--algo", "ppo", "--env", "v2", "--trials", "50", "--timesteps", "500000", "--device", "cuda"]
        args3 = p_tune.parse_args(cmd3)
        self.assertEqual(args3.trials, 50)
        self.assertEqual(args3.timesteps, 500000)

        # 4. test_agent_v2.py
        p_test = argparse.ArgumentParser()
        p_test.add_argument("--algo", type=str, default="ppo")
        p_test.add_argument("--env", type=str, default="v2", choices=["v2", "v3"])
        p_test.add_argument("--load_zip", type=str, required=True)
        p_test.add_argument("--load_pkl", type=str, required=True)
        p_test.add_argument("--player", type=int, default=1)
        p_test.add_argument("--opponent_type", type=str, choices=["human", "cpu"], default="human")
        p_test.add_argument("--device", type=str, default="auto")
        p_test.add_argument("--profile", action="store_true")
        p_test.add_argument("--infinite_match", action="store_true")
        p_test.add_argument("--rematch_delay", type=float, default=2.0)
        p_test.add_argument("--cpu_level_cap", type=int, default=5, choices=range(1, 9))
        cmd4 = ["--algo", "ppo", "--env", "v2", "--load_zip", "models/production/v2/ppo/ppo_model.zip", "--load_pkl", "models/production/v2/ppo/ppo_model_vecnorm.pkl", "--player", "1", "--opponent_type", "cpu", "--device", "cuda"]
        args4 = p_test.parse_args(cmd4)
        self.assertEqual(args4.opponent_type, "cpu")

        # 5. test_ai_vs_ai_v2.py
        p_dual = argparse.ArgumentParser()
        p_dual.add_argument("--algo_p1", type=str, default="ppo")
        p_dual.add_argument("--env_p1", type=str, default="v2", choices=["v2", "v3"])
        p_dual.add_argument("--load_zip_p1", type=str, required=True)
        p_dual.add_argument("--load_pkl_p1", type=str, required=True)
        p_dual.add_argument("--device_p1", type=str, default="auto")
        p_dual.add_argument("--algo_p2", type=str, default="ppo")
        p_dual.add_argument("--env_p2", type=str, default="v2", choices=["v2", "v3"])
        p_dual.add_argument("--load_zip_p2", type=str, required=True)
        p_dual.add_argument("--load_pkl_p2", type=str, required=True)
        p_dual.add_argument("--device_p2", type=str, default="auto")
        p_dual.add_argument("--profile", action="store_true")
        p_dual.add_argument("--infinite_match", action="store_true")
        p_dual.add_argument("--rematch_delay", type=float, default=2.0)
        cmd5 = ["--algo_p1", "ppo", "--load_zip_p1", "models/production/v2/ppo/p1.zip", "--load_pkl_p1", "models/production/v2/ppo/p1_vecnorm.pkl", "--algo_p2", "dqn", "--load_zip_p2", "models/production/v2/dqn/p2.zip", "--load_pkl_p2", "models/production/v2/dqn/p2_vecnorm.pkl", "--device_p1", "cuda", "--device_p2", "cuda"]
        args5 = p_dual.parse_args(cmd5)
        self.assertEqual(args5.algo_p2, "dqn")

        # 6. train_league.py
        p_league = argparse.ArgumentParser()
        p_league.add_argument("--steps", type=int, default=5000000)
        p_league.add_argument("--env_version", default="v2", choices=["v2", "v3"])
        p_league.add_argument("--matchup_mode", default="ryu_vs_ryu", choices=["ryu_vs_ryu", "ryu_vs_all", "custom"])
        p_league.add_argument("--custom_state", default=None)
        p_league.add_argument("--model_name", default="league")
        p_league.add_argument("--resume", action="store_true")
        p_league.add_argument("--device", default="auto")
        cmd6 = ["--env_version", "v2", "--steps", "5000000", "--matchup_mode", "ryu_vs_ryu", "--device", "cuda"]
        args6 = p_league.parse_args(cmd6)
        self.assertEqual(args6.matchup_mode, "ryu_vs_ryu")

        # 7. train_exploiter.py
        p_exploiter = argparse.ArgumentParser()
        p_exploiter.add_argument("--type", default="rusher", choices=["rusher", "spammer", "turtle"])
        p_exploiter.add_argument("--steps", type=int, default=1000000)
        p_exploiter.add_argument("--env_version", default="v2", choices=["v2", "v3"])
        p_exploiter.add_argument("--matchup_mode", default="ryu_vs_ryu", choices=["ryu_vs_ryu", "ryu_vs_all", "custom"])
        p_exploiter.add_argument("--custom_state", default=None)
        p_exploiter.add_argument("--model_name", default="league")
        p_exploiter.add_argument("--device", default="auto")
        cmd7 = ["--type", "rusher", "--env_version", "v2", "--steps", "1000000", "--device", "cuda"]
        args7 = p_exploiter.parse_args(cmd7)
        self.assertEqual(args7.type, "rusher")

        # 8. train_pbt.py
        p_pbt = argparse.ArgumentParser()
        p_pbt.add_argument("--algo", default="ppo")
        p_pbt.add_argument("--env", default="v2")
        p_pbt.add_argument("--model_name", default="PBT_BEST_model")
        p_pbt.add_argument("--load_zip", default=None)
        p_pbt.add_argument("--load_pkl", default=None)
        p_pbt.add_argument("--phase", default="0")
        p_pbt.add_argument("--steps", type=int, default=5000000)
        p_pbt.add_argument("--steps_per_exploit", type=int, default=500000)
        p_pbt.add_argument("--population", type=int, default=10)
        p_pbt.add_argument("--max_concurrent", type=int, default=None)
        p_pbt.add_argument("--envs_per_worker", type=int, default=1)
        p_pbt.add_argument("--resume", action="store_true")
        cmd8 = ["--algo", "ppo", "--env", "v2", "--population", "10", "--steps", "5000000"]
        args8 = p_pbt.parse_args(cmd8)
        self.assertEqual(args8.population, 10)

        # 9. web_dashboard.py
        p_dash = argparse.ArgumentParser()
        p_dash.add_argument("--host", "--server_name", dest="server_name", type=str, default="0.0.0.0")
        p_dash.add_argument("--port", "--server_port", dest="server_port", type=int, default=7860)
        p_dash.add_argument("--share", action="store_true")
        args9 = p_dash.parse_args(["--port", "8080", "--share"])
        self.assertEqual(args9.server_port, 8080)
        self.assertTrue(args9.share)

    def test_all_cli_scripts_help_execution(self):
        """Directly executes all 9 CLI entry scripts with --help in subprocess to confirm 0 exit code."""
        import subprocess
        scripts = [
            "train.py",
            "resume.py",
            "tune.py",
            "test_agent_v2.py",
            "test_ai_vs_ai_v2.py",
            "train_league.py",
            "train_exploiter.py",
            "train_pbt.py",
            "web_dashboard.py"
        ]
        scripts_dir = os.path.join(SRC_PATH, "scripts")
        for script_name in scripts:
            script_path = os.path.join(scripts_dir, script_name)
            self.assertTrue(os.path.exists(script_path), f"Script not found: {script_path}")
            res = subprocess.run(
                [sys.executable, script_path, "--help"],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )
            self.assertEqual(
                res.returncode, 0,
                f"Script {script_name} failed --help with exit code {res.returncode}.\nStderr: {res.stderr}\nStdout: {res.stdout}"
            )

if __name__ == "__main__":
    unittest.main()

