import os, argparse, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from core import config
from agents.pbt import build_orchestrator

def update_ppo_config_var(key, value):
    ppo_config_path = os.path.join(config.SRC_DIR, "agents", "ppo", "config.py")
    if not os.path.exists(ppo_config_path): return False
    with open(ppo_config_path, "r") as f: content = f.read()
    formatted_value = f'"{value}"' if isinstance(value, str) else str(value)
    pattern = rf"^({key}\s*=\s*)(.*?)(\s*(?:#.*)?)$"
    if re.search(pattern, content, flags=re.MULTILINE):
        content = re.sub(pattern, rf"\g<1>{formatted_value}\g<3>", content, flags=re.MULTILINE)
        with open(ppo_config_path, "w") as f: f.write(content)
        return True
    return False

def main():
    config.generate_lua_config()

    import torch
    # Performance Optimization: Restrict PyTorch to 2 CPU threads during PBT run
    # Prevents logical core thrashing alongside active EmuHawk instances.
    torch.set_num_threads(2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="ppo")
    parser.add_argument("--env", default="v2")
    parser.add_argument("--model_name", default="PBT_BEST_model")
    parser.add_argument("--load_zip", default=None)
    parser.add_argument("--load_pkl", default=None)
    parser.add_argument("--phase", default="0")
    parser.add_argument("--steps", type=int, default=5000000)
    parser.add_argument("--steps_per_exploit", type=int, default=500000)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--max_concurrent", type=int, default=None, help="Max concurrent trials (default: population size)")
    parser.add_argument("--envs_per_worker", type=int, default=1, help="Number of environments per PBT agent")
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()

    # Set max_concurrent to population if not provided
    if args.max_concurrent is None:
        args.max_concurrent = args.population

    # FIX 3: Correct PB2 Population Guards
    # Max safe ports = 10015 - 9999 = 16. Total envs = max_concurrent * envs_per_worker.
    if args.population < 4:
        sys.exit(
            "Error: Population < 4 — PB2's Gaussian Process requires a minimum viable "
            "sample size. Use --population 4 or higher."
        )

    print(f"Starting PBT (Pop: {args.population}, Concurrent: {args.max_concurrent}, Envs/Worker: {args.envs_per_worker}, Steps: {args.steps})...")
    orchestrator = build_orchestrator()
    best_config = orchestrator.run(
        algo=args.algo, env_version=args.env, total_steps=args.steps, population_size=args.population,
        max_concurrent=args.max_concurrent, envs_per_worker=args.envs_per_worker,
        steps_per_exploit=args.steps_per_exploit, start_phase=args.phase, 
        base_zip=args.load_zip, base_pkl=args.load_pkl, model_name=args.model_name, resume=args.resume
    )

    if args.algo == "ppo":
        update_ppo_config_var("LR", best_config.get("lr"))
        update_ppo_config_var("ENT_COEF", best_config.get("ent_coef"))
        update_ppo_config_var("CLIP_RANGE", best_config.get("clip_range"))

if __name__ == "__main__":
    main()
