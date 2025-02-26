import os
import sys
sys.path.append("..")
from hockey.hockey_env import BasicOpponent, HockeyEnv_BasicOpponent
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer, PRB
from network import LinearNetwork, DuelingNetwork
from random import choice as random_choice_from_list
import json
from tqdm import tqdm


class DQNAgent:
    def __init__(self, env, actions, memory_size, batch_size, target_update_freq, epsilon_decay, lr, 
                 seed, max_epsilon=1.0, min_epsilon=0.1, discount=0.95, device=torch.device('cpu'),
                 hidden_sizes=[128,128], start_training_at=500, checkpoint_base_path=r"checkpoints", 
                 experiment_name="test_experiment", max_timesteps_per_episode=500, 
                 num_timesteps=1e7, test_iterations=100, checkpoint_interval=50000, 
                 use_double_dqn=False, use_per=False, use_dueling_dqn=False, alpha=0.6, beta=0.4, 
                 train_freq=1, clip_norm_max=10.0, linear_beta_schedule=False, do_self_play=False,
                 opponent_choice_freq=2, opponent_list_len=10, self_play_min_basic_winrate=0.9,
                 keep_basic_op_performance=False, dqn_weights_path=None, opponent_list=None
    ):
        """
        Init
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update_freq (int): update frequency for target model
            epsilon_decay (float): step size to decrease epsilon
            lr_actor (float): learning rate for actor network
            lr_critic (float): learning rate for critic network
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            clip_norm_max (float): max value for gradient clipping
            discount (float): discount factor
            start_training_at (int): initial timesteps without network training (just experience gathering)
            train_freq (int): train agent with 1 batch every train_freq environment steps
            hidden_sizes (List[int]): hidden layer sizes for Q network
            max_timesteps_per_episode (int): after these timesteps the episode is aborted and reset
            num_timesteps (int): total number of timesteps for training
            test_iterations (int): number of episodes during model evaluation
            use_double_dqn (bool): whether to use double dqn
            use_dueling_dqn (bool): whether to use dueling dqn
            use_per (bool): whether to use prioritized experience replay
            linear_beta_schedule (bool): whether to use linear beta schedule instead of exponential
            alpha (float), beta(float): parameters for PER
            do_self_play (bool): whether to use self-play
            keep_basic_op_performance (bool): if true, only add the current agent to opponent_list, if it beats the previous best winrate AND wins more than 80% of the time against basic_strong
            dqn_weights_path (str, optional): if provided, will initialize dqn and dqn_target with these weights
            opponent_list (List[DQNAgent], optional): if provided, will start self play with these opponents
        """
        # General params
        observation_dim = env.observation_space.shape[0]        
        self.actions = actions
        self.env = env
        self.batch_size = int(batch_size)
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        if seed == "None":
            self.seed = None
        else:
            self.seed = seed
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.clip_norm_max = clip_norm_max
        self.discount = discount
        self.device = device
        print(self.device)

        # RAINBOW MODS
        self.use_double_dqn = use_double_dqn
        self.use_dueling_dqn = use_dueling_dqn
        self.beta = beta
        self.init_beta = beta
        self.linear_beta_schedule = linear_beta_schedule
        self.use_per = use_per
        if self.use_per:
            self.memory = PRB(observation_dim, int(memory_size), batch_size, alpha)
        else:
            self.memory = ReplayBuffer(observation_dim, int(memory_size), batch_size)

        # Train loop settings
        self.start_training_at = int(start_training_at)
        self.train_freq = int(train_freq)
        self.max_timesteps_per_episode = int(max_timesteps_per_episode)
        self.num_timesteps = int(num_timesteps)
        self.test_iterations = test_iterations
        self.checkpoint_interval = checkpoint_interval
        self.target_update_freq = target_update_freq

        # Self play settings
        self.do_self_play = do_self_play
        self.opponent_choice_freq = opponent_choice_freq
        self.opponent_list_len = opponent_list_len
        # self.min_basic_winrate = self_play_min_basic_winrate
        self.current_best_winrate = self_play_min_basic_winrate
        self.win_rates = list()
        self.win_rates_per_opponent = list()
        self.scores_per_opponent = list()
        self.keep_basic_op_performance = keep_basic_op_performance
        self.opponent_list = opponent_list

        # Checkpoint settings
        self.checkpoint_base_path = os.path.join(checkpoint_base_path, experiment_name)
        os.makedirs(self.checkpoint_base_path, exist_ok=True)
        self.stats_path = os.path.join(self.checkpoint_base_path, f"stats.json")
        self.mean_evals = list()
        self.win_percentages = list()

        # Network initialization
        if self.use_dueling_dqn:
            Network = DuelingNetwork
        else:
            Network = LinearNetwork
        self.dqn = Network(observation_dim, len(self.actions), hidden_sizes=hidden_sizes).to(self.device)
        if dqn_weights_path:
            self.dqn.load_state_dict(torch.load(dqn_weights_path, map_location=self.device))
        self.dqn_target = Network(observation_dim, len(self.actions), hidden_sizes=hidden_sizes).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)

        self.transition = list()
        # Network mode (determines if transitions are stored)
        self.is_test = False

    def act(self, state) -> np.ndarray:
        """
        Select an action from the input state
        
        Args:
            state (torch.tensor): current observed environment state

        Returns:
            selected_action (np.array): 
        """
        # Act epsilon-greedily
        if (self.epsilon > np.random.random()) and (not self.is_test):
            action = torch.randint(
                            high=len(self.actions), size=()
                        ).item()
        else:
            with torch.no_grad():
                action = self.dqn(
                    torch.tensor(state).to(torch.float32).to(self.device)
                ).argmax()
            action = action.detach().cpu().item()
        
        selected_action = np.array(self.actions[action])

        if not self.is_test:
            self.transition = [state, action]
        
        return selected_action

    def step(self, action):
        """
        Take an action and return the response of the env
        
        Args:
            action (np.array): action of an agent

        Returns:
            next_state (np.array): next state returned by env
            reward (float): reward returned by env
            done (bool): flag indicating whether episode is done (e.g. someone won)
        """
        next_state, reward, done, _, info = self.env.step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done, _, info

    def update_model(self):
        """
        Update the model by gradient descent
        
        Args:
            None
        
        Returns:
            loss (float): loss for sampled batch
        """

        samples = self.memory.sample_batch(beta=self.beta)
        # If no PER, weights will be torch.ones
        weights = (samples["weights"]).to(self.device)

        losses = self.compute_loss(samples)
        loss = torch.mean(losses*weights)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), self.clip_norm_max)
        self.optimizer.step()

        if self.use_per:
            # Add stabilization constant, ensures everything is sampled
            self.memory.update_priorities(samples["indices"], losses.detach().cpu().squeeze()+1e-6)

        return loss.item()
        
    def train(self, num_timesteps=None, checkpoint_interval=None):
        """
        Train the agent

        Args:
            num_timesteps (int): total number of timesteps taken in the environment
            checkpoint_interval (int): frequency for saving checkpoints and evaluating model
        """
        self.is_test = False

        print("#"*50)
        print("training model...")
        print("#"*50)

        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        if checkpoint_interval is None:
            checkpoint_interval = self.checkpoint_interval

        state, _ = self.env.reset(seed=self.seed)
        update_count = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        timestep_in_episode = 0
        # create opponent list (if no self play, the list only has one element)
        opponent_list = [BasicOpponent(weak=True)]
        if self.do_self_play:
            # start with weak and strong
            opponent_list.append(BasicOpponent(weak=False))  
            if self.opponent_list:
                opponent_list += self.opponent_list

        self.env.opponent = random_choice_from_list(opponent_list)
        episodes_since_opponent_switch = 0

        for timestep in tqdm(range(1, num_timesteps + 1), ascii=True):
            action = self.act(state)
            next_state, reward, done, _, info = self.step(action)

            state = next_state
            score += reward

            # Beta schedule
            if self.linear_beta_schedule:
                self.beta = self.init_beta + (timestep/num_timesteps) * (1-self.init_beta)
            else:
                self.beta = self.beta + (timestep/num_timesteps)* (1-self.beta)

            # Save and reset score and env, when episode ends
            if done or (timestep_in_episode > self.max_timesteps_per_episode):
                state, _ = self.env.reset(seed=self.seed)
                scores.append(score)
                score = 0
                timestep_in_episode = 0
                # update self play:
                episodes_since_opponent_switch += 1
                if episodes_since_opponent_switch % self.opponent_choice_freq == 0:  # TODO: choose opponent choice freq lower than normal, e.g. 50k
                    self.env.opponent = random_choice_from_list(opponent_list)
                    episodes_since_opponent_switch = 0

            # TRAIN
            if (timestep % self.train_freq == 0) and (timestep > self.start_training_at):
                loss = self.update_model()
                losses.append(loss)
                update_count += 1

                # Linear epsilon schedule
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                        self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)

                if update_count % self.target_update_freq == 0:
                    self.hard_update_target()

            # EVAL/SAVE
            if timestep % checkpoint_interval == 0:

                if self.do_self_play:
                    win_rate, did_improve = self.evaluate_self_play(timestep, opponent_list)
                    if did_improve and (
                        (self.keep_basic_op_performance==False) 
                        or (self.win_rates_per_opponent[-1][1]>0.8)  # don't unlearn basic opponent
                    ): 
                        agent_copy = deepcopy(self)
                        agent_copy.is_test = True
                        agent_copy.memory = None  # free space
                        opponent_list.append(agent_copy)
                        self.current_best_winrate = win_rate

                        if len(opponent_list) > self.opponent_list_len:
                            opponent_list.pop(2)  # always keep first two BasicOpponent's in the list

                        # set new opponent before returning to train loop
                        # (doesn't matter that this might be in the middle of an episode, because checkpoint_interval >> len(episode))
                        self.env.opponent = random_choice_from_list(opponent_list)  
                                         
                else:
                    self.save_checkpoint(timestep, scores, losses, epsilons)
                

        self.env.close()
                
    def test(self, seed=None):

        self.is_test = True
        if seed is None:
            seed = self.seed
                
        state, _ = self.env.reset(seed=seed)
        done = False
        score = 0
        
        for timestep in range(self.max_timesteps_per_episode):
            action = self.act(state)
            next_state, reward, done, _, info = self.step(action)

            state = next_state
            score += reward
            if done: break
        
        self.is_test = False
        
        return score, info["winner"]

    def compute_loss(self, samples):
        """Return dqn loss
        
        Args:
            samples (dict[str, tensor]): dictionary returned by sample_batch from the replay buffer

        Returns:
            loss (torch.tensor[float]): loss between predicted q-value and target
        """
        state = samples["observations"].to(self.device)
        next_state = samples["next_observations"].to(self.device)
        action = samples["actions"].view(-1, 1).to(torch.long).to(self.device)
        reward = samples["rewards"].view(-1, 1).to(self.device)
        done = samples["done"].view(-1, 1).to(self.device)

        curr_q_value = self.dqn(state).gather(1, action)
        if self.use_double_dqn:
            next_q_value = self.dqn_target(next_state).gather(
                1, self.dqn(next_state).argmax(dim=1, keepdim=True)
            ).detach()
        else:
            next_q_value = self.dqn_target(
                next_state
            ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.discount * next_q_value * mask).to(self.device)

        losses = F.smooth_l1_loss(curr_q_value, target, reduction="none")

        return losses

    def hard_update_target(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def save_checkpoint(self, timestep, scores, losses, epsilons):
        """
        Save checkpoint of the training progress

        Args:
            timestep (int): current timestep in training
            scores (List[float]): scores per game
            losses (List[float]): losses for every time step in training
            epsilons (List[float]): epsilons for every time step in training
        """
        # SAVE MODEL
        torch.save(self.dqn.state_dict(), os.path.join(self.checkpoint_base_path, f"dqn_{timestep}"))

        # EVAL
        print("#"*50)
        print("evaluating model...")
        print("#"*50)
        eval_scores = []
        num_wins = 0
        for _ in range(self.test_iterations):
            score, winner = self.test()
            eval_scores.append(score)
            if winner==1:
                num_wins += 1

        self.mean_evals.append(np.mean(eval_scores))     
        self.win_percentages.append(num_wins/self.test_iterations)   

        stats_dict = {
            "scores": scores,
            "losses": losses,
            "epsilons": epsilons,
            "eval_scores_this_episode": eval_scores,
            "win_percentages": self.win_percentages,
            "mean_eval_scores": self.mean_evals
        }
        # SAVE STATS
        with open(self.stats_path, "w") as f:
            json.dump(stats_dict, f)

    def evaluate_self_play(self, timestep, opponent_list):
        print("#"*50)
        print("evaluating model for self play...")
        print("#"*50)

        # SAVE MODEL
        torch.save(self.dqn.state_dict(), os.path.join(self.checkpoint_base_path, f"dqn_{timestep}"))

        win_rate_list = list()
        avg_score_per_opponent = list()
        for opponent in opponent_list:
            self.env.opponent = opponent
            num_wins_this_op = 0
            scores_this_op = list()
            for test_iter in range(self.test_iterations):  # TODO: remember to increase this for stability, e.g. to 250
                score, winner = self.test(seed=test_iter)
                scores_this_op.append(score)
                if winner==1:
                    num_wins_this_op += 1
            win_rate_list.append(num_wins_this_op / self.test_iterations)
            avg_score_per_opponent.append(float(np.mean(scores_this_op)))

        win_rate = float(np.mean(win_rate_list))
        self.win_rates.append(win_rate)
        self.win_rates_per_opponent.append(win_rate_list)
        self.scores_per_opponent.append(avg_score_per_opponent)
        did_improve = bool(win_rate > self.current_best_winrate)
        stats_dict = {
            "win_rates": self.win_rates,
            "win_rate_per_opponent": self.win_rates_per_opponent,
            "avg_score_per_opponent": self.scores_per_opponent,
            "did_improve": did_improve
        }

        # SAVE STATS
        with open(self.stats_path, "w") as f:
            json.dump(stats_dict, f)

        return win_rate, did_improve