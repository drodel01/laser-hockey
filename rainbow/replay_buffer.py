import numpy as np
import torch

from segment_tree import MinSegmentTree, SumSegmentTree


class ReplayBuffer:
    """Torch replay buffer"""

    def __init__(self, observation_dim, max_size, batch_size=32):
        """
        observation_dim (int): dimension of observation space
        max_size (int): length of replay buffer
        batch_size (int): batch size for sampling

        NOTE: code only works for DQN, dtypes should be adjusted for other methods
        """
        self.observation_buffer = torch.zeros((max_size, observation_dim), dtype=torch.float32)
        self.next_observation_buffer = torch.zeros((max_size, observation_dim), dtype=torch.float32)
        self.action_buffer = torch.zeros((max_size), dtype=torch.int16)
        self.reward_buffer = torch.zeros((max_size), dtype=torch.float32)
        self.done_buffer = torch.zeros((max_size), dtype=torch.float32)
        self.max_size = max_size
        self.batch_size = batch_size
        self.current_position = 0
        self.current_size = 0

    def store(self, observation, action, reward, next_observation, done):
        """
        Stores given transition in replay buffer.

        Args:
            observation (torch.tensor): current observation
            action (torch.tensor): current action
            reward (float): reward after taking action
            next_observation (torch.tensor): next received observation after taking action
            done (bool): flag that indicates whether episode is done (e.g. winning or losing)

        Returns:
            None
        """
        self.observation_buffer[self.current_position] = torch.from_numpy(observation)
        self.action_buffer[self.current_position] = action
        self.reward_buffer[self.current_position] = reward
        self.next_observation_buffer[self.current_position] = torch.from_numpy(next_observation)
        self.done_buffer[self.current_position] = done
        self.current_position = (self.current_position + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample_batch(self, **kwargs):
        """
        Sample batch of size self.batch_size uniformly from replay buffer

        Args:
            None
        
        Returns:
            dict: containing uniformly sampled experience from replay buffer
        """
        indices = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        # make compatible with PER
        weights = torch.ones(len(indices))
        return dict(observations=self.observation_buffer[indices],
                    next_observations=self.next_observation_buffer[indices],
                    actions=self.action_buffer[indices],
                    rewards=self.reward_buffer[indices],
                    done=self.done_buffer[indices],
                    weights=weights)

    def __len__(self):
        return self.current_size
    


class PRB(ReplayBuffer):
    def __init__(self, observation_dim, max_size, batch_size, alpha):
        """
        Args:
            observation_dim (int): dimension of observation space
            max_size (int): length of replay buffer
            batch_size (int): batch size for sampling
            alpha (float): prioritization coefficient. alpha=0 corresponds to uniform sampling. Larger alpha skew more towards high TD error samples.

        NOTE: code only works for DQN, dtypes should be adjusted for other methods
        """
        if alpha < 0:
            raise ValueError(f"In PER, alpha must be >=0, but received alpha={alpha}")
        
        super(PRB, self).__init__(observation_dim, max_size, batch_size)
        self.max_priority, self.position_in_tree = 1.0, 0
        self.alpha = alpha
        
        # capacity of tree is smallest 2^n that can hold max_size
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(self, observation, action, reward, next_observation, done):
        """
        Stores given transition in replay buffer.

        Args:
            observation (torch.tensor): current observation
            action (torch.tensor): current action
            reward (float): reward after taking action
            next_observation (torch.tensor): next received observation after taking action
            done (bool): flag that indicates whether episode is done (e.g. winning or losing)

        Returns:
            None
        """
        super().store(observation, action, reward, next_observation, done)

        self.sum_tree[self.position_in_tree] = self.max_priority ** self.alpha
        self.min_tree[self.position_in_tree] = self.max_priority ** self.alpha
        self.position_in_tree = (self.position_in_tree + 1) % self.max_size

    def sample_batch(self, beta=0.4):
        """
        Sample batch of size self.batch_size uniformly from replay buffer

        Args:
            beta (float): PER weight exponent. beta=1 corresponds to uniform sampling. Needs to be decayed during training.
        
        Returns:
            dict: containing uniformly sampled experience from replay buffer
        """
        if beta <= 0:
            raise ValueError(f"In PER, beta must be >0, but received beta={beta}")
        
        indices, weights = self.sample(beta)
            
        return dict(observations=self.observation_buffer[indices],
                    next_observations=self.next_observation_buffer[indices],
                    actions=self.action_buffer[indices],
                    rewards=self.reward_buffer[indices],
                    done=self.done_buffer[indices],
                    weights=weights,
                    indices=indices)
        
    def update_priorities(self, indices, priorities):
        """
        Update priorities of sampled transitions
        
        Args:
            indices (List[int]): indices for which we wish to update priorities
            priorities (torch.tensor): new priorities of these indices.
        
        Returns:
            None
        """
        if len(indices) != len(priorities):
            raise ValueError(f"In update_priorities, length of indices and priorities must match, but received len(indices)={len(indices)} and len(priorities)={len(priorities)}")

        for index, priority in zip(indices, priorities):
            if priority <= 0:
                raise ValueError(f"Priorities must be non-negative, but received priority {priority} for index {index}")
            if (0 > index) or (index > len(self)):
                raise ValueError(f"Index must be between 0 and len(self)={len(self)}, but received index={index}")

            self.sum_tree[index] = priority ** self.alpha
            self.min_tree[index] = priority ** self.alpha

            # update max
            self.max_priority = max(self.max_priority, priority)
            
    def sample(self, beta):
        """
        Sample indices based on proportions
        
        Args:
            beta (float): PER weight exponent, to be used for loss weight. beta=1 corresponds to uniform sampling. Needs to be decayed during training.
        
        Returns:
            List of int
        """
        indices = []
        weights = []
        total_priority_sum = self.sum_tree.sum(0, len(self)-1)
        length_of_probability_segment = total_priority_sum / self.batch_size

        min_probability = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (min_probability * len(self))**(-beta)
        
        for i in range(self.batch_size):
            # sample index
            left_interval_bound = length_of_probability_segment * i
            right_interval_bound = length_of_probability_segment * (i + 1)
            ####### DIRTY BUGFIX #############
            index = self.max_size
            tries = 0
            while index >= self.max_size and tries < 3:
                prefixsum = np.random.uniform(left_interval_bound, right_interval_bound)
                index = self.sum_tree.find_prefixsum_idx(prefixsum)
                tries += 1
            index = min(self.max_size-1, index)
            indices.append(index)
            ##################################
            # calculate weight
            this_index_priority = self.sum_tree[index] / total_priority_sum
            this_index_weight = (this_index_priority * len(self))**(-beta)
            weights.append(this_index_weight / max_weight)
            
        return indices, torch.tensor(weights)