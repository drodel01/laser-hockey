from pathlib import Path
from sys import path

import wandb

path.insert(0, str(Path(__file__).resolve().parents[2].absolute()))

from src.environment.make_hockey_env import make_hockey_env_gym  # noqa: E402
from src.algos.cross_q.cross_q_self_play import CrossQForSelfPlay  # noqa: E402
from src.environment.hockey_env import BasicOpponent  # noqa: E402

if __name__ == "__main__":
    env = make_hockey_env_gym(opponent=BasicOpponent(weak=False))
    test_env = make_hockey_env_gym(opponent=BasicOpponent(weak=False))

    model = CrossQForSelfPlay(env=env)

    run = wandb.init(project="hockey")
    model.learn_via_self_play(
        total_steps=int(10e6),
        test_env=test_env,
        wandb_run=run,
    )
    model_save_dir = Path(f"models/{run.id}")
    model_path = model_save_dir / "model"
    model.save_model(model_path=model_path)
    wandb.save(model_path, base_path=model_save_dir)
    run.finish()
