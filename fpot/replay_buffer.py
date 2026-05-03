"""Circular replay buffer backed by numpy arrays."""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.masks    = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.idx  = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.masks[self.idx] = 1.0 - float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs, actions, rewards, next_obs, dones):
        batch_size = obs.shape[0]
        rewards = np.asarray(rewards).reshape(-1)
        dones = np.asarray(dones).reshape(-1)
        masks = 1.0 - dones.astype(np.float32)
        end_idx = self.idx + batch_size
        if end_idx <= self.capacity:
            sl = slice(self.idx, end_idx)
            self.obs[sl] = obs
            self.actions[sl] = actions
            self.rewards[sl] = rewards
            self.next_obs[sl] = next_obs
            self.masks[sl] = masks
        else:
            first = self.capacity - self.idx
            self.obs[self.idx:]     = obs[:first]
            self.actions[self.idx:] = actions[:first]
            self.rewards[self.idx:] = rewards[:first]
            self.next_obs[self.idx:] = next_obs[:first]
            self.masks[self.idx:]   = masks[:first]
            rest = batch_size - first
            self.obs[:rest]     = obs[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.next_obs[:rest] = next_obs[first:]
            self.masks[:rest]   = masks[first:]
        self.idx = end_idx % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> dict:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations":      self.obs[indices],
            "actions":           self.actions[indices],
            "rewards":           self.rewards[indices],
            "next_observations": self.next_obs[indices],
            "masks":             self.masks[indices],
        }

    def __len__(self):
        return self.size
