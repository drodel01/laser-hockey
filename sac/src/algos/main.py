from pathlib import Path
from sys import path

import wandb

path.insert(0, str(Path(__file__).resolve().parents[2].absolute()))

from src.environment.make_hockey_env import make_hockey_env_gym  # noqa: E402
from src.algos.cross_q.cross_q import CrossQ  # noqa: E402
from src.algos.sac.sac import SAC  # noqa: E402
from src.environment.hockey_env import BasicOpponent  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    env = make_hockey_env_gym(opponent=BasicOpponent(weak=False))
    test_env = make_hockey_env_gym(opponent=BasicOpponent(weak=False))

    match args.model.lower():
        case "sac":
            model = SAC(env)
        case "cross_q":
            model = CrossQ(env)
        case _:
            raise ValueError(f"Unknown model: {args.model}")

    run = wandb.init(project="hockey")
    model.learn(
        total_steps=500_000,
        test_env=test_env,
        wandb_run=run,
    )
    model_save_dir = Path(f"models/{run.id}")
    model_path = model_save_dir / "model"
    model.save_model(model_path=model_path)
    wandb.save(model_path, base_path=model_save_dir)
    run.finish()
