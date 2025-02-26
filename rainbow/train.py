import argparse
import json
import os
import sys
import torch
import numpy as np
import random

sys.path.append("..")
from agent import DQNAgent
from hockey.hockey_env import HockeyEnv_BasicOpponent

ACTION_FOLDER = "action_configs"
ALGO_CONFIG_FOLDER = "algorithm_configs"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm_config", default="default_config.json", type=str)
    parser.add_argument("--actions_config", default="actions_full.json", type=str)
    args = parser.parse_args()

    with open(os.path.join(ALGO_CONFIG_FOLDER, args.algorithm_config), "r") as f:
        algo_config = json.load(f)

    with open(os.path.join(ACTION_FOLDER, args.actions_config), "r") as f:
        actions = json.load(f)

    # COPY CONFIGS TO BE SURE LATER ON, IN CASE OF CHANGES
    experiment_path = os.path.join(algo_config["checkpoint_base_path"], algo_config["experiment_name"])
    os.makedirs(experiment_path, exist_ok=True)
    with open(os.path.join(algo_config["checkpoint_base_path"], algo_config["experiment_name"], "algorithm_config.json"), "w") as f:
        json.dump(algo_config, f)
    with open(os.path.join(algo_config["checkpoint_base_path"], algo_config["experiment_name"], "action_config.json"), "w") as f:
        json.dump(actions, f)

    env = HockeyEnv_BasicOpponent(weak_opponent=algo_config['weak_opponent'])
    algo_config.pop("weak_opponent")

    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    seed = algo_config["seed"]
    if seed != "None":
        np.random.seed(seed)
        random.seed(seed)
        seed_torch(seed)

    # SELF PLAY:
    orig_dqn_weights_path = algo_config.get("dqn_weights_path")
    if "opponent_weights_list" in algo_config.keys():
        # only take as many weights as fit in the opponent_list_len, including 2 basic opponents        
        weights_list = algo_config["opponent_weights_list"][:(algo_config["opponent_list_len"]-2)]
        algo_config.pop("opponent_weights_list")
        opponent_list = list()
        for weights_path in weights_list:
            algo_config["dqn_weights_path"] = weights_path
            opponent = DQNAgent(env, actions, **algo_config)
            opponent.is_test = True
            opponent_list.append(opponent)
        algo_config["opponent_list"] = opponent_list

    algo_config["dqn_weights_path"] = orig_dqn_weights_path
    agent = DQNAgent(env, actions, **algo_config)

    agent.train()


if __name__ == "__main__":
    main()