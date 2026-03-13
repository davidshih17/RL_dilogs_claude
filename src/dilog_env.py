"""
Gym environment for dilogarithm simplification.
Based on Section 4.1 of arXiv:2206.04115
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Optional
import pickle

from dilog_utils import (
    DilogExpression,
    apply_reflection,
    apply_inversion,
    apply_duplication,
    apply_cyclic_permutation,
)


class DilogEnv(gym.Env):
    """
    RL Environment for simplifying dilogarithm expressions.

    Actions:
        0: Reflection on first term
        1: Inversion on first term
        2: Duplication on first term
        3: Cyclic permutation (rotate first term to end)

    The agent can apply any identity to any term by first using cyclic
    permutation to rotate the desired term to the front.

    Observation: One-hot encoded prefix notation
    """

    def __init__(
        self,
        dataset_path: str,
        max_steps: int = 50,
        max_length: int = 512,
        use_cyclic_penalty: bool = True,
        penalty_lambda: float = 0.25
    ):
        super(DilogEnv, self).__init__()

        self.max_steps = max_steps
        self.max_length = max_length
        self.use_cyclic_penalty = use_cyclic_penalty
        self.penalty_lambda = penalty_lambda

        # Load dataset
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)

        # Define vocabulary for one-hot encoding
        self.vocab = [
            'add', 'sub', 'mul', 'div', 'pow',
            'polylog', 'x', '-',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '<PAD>', '<UNK>'
        ]
        self.vocab_size = len(self.vocab)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.pad_idx = self.token_to_idx['<PAD>']

        # Action space: 4 actions (3 identities on first term + cyclic permutation)
        self.action_space = spaces.Discrete(4)

        # Observation space: flattened one-hot encoded sequence + 3 additional features
        # (prev_action, prev_num_terms, min_num_terms)
        obs_size = self.max_length * self.vocab_size + 3
        self.observation_space = spaces.Box(
            low=-1, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        # Current state
        self.current_expr = None
        self.current_step = 0
        self.prev_action = None
        self.prev_num_terms = None
        self.min_num_terms = None

    def reset(self, seed=None, options=None):
        """Reset environment and sample a new expression"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Sample a random expression from dataset
        sample = np.random.choice(self.dataset)
        self.current_expr = sample['expression'].copy()

        self.current_step = 0
        self.prev_action = None
        self.prev_num_terms = self.current_expr.num_terms()
        self.min_num_terms = self.prev_num_terms

        return self._get_observation(), {}

    def set_sample(self, sample):
        """Set a specific sample for evaluation (used by parallel evaluation)."""
        self.current_expr = sample['expression'].copy()
        self.current_step = 0
        self.prev_action = None
        self.prev_num_terms = self.current_expr.num_terms()
        self.min_num_terms = self.prev_num_terms

    def step(self, action: int):
        """Execute one step in the environment"""
        self.current_step += 1

        # Save previous expression in case action fails
        prev_expr = self.current_expr.copy()

        # Apply action with error handling for complex numbers
        try:
            if action == 0:  # Reflection
                self.current_expr = apply_reflection(self.current_expr, 0)
            elif action == 1:  # Inversion
                self.current_expr = apply_inversion(self.current_expr, 0)
            elif action == 2:  # Duplication
                self.current_expr = apply_duplication(self.current_expr, 0)
            elif action == 3:  # Cyclic permutation
                self.current_expr = apply_cyclic_permutation(self.current_expr)

            # Try to get observation to check if expression is valid
            current_num_terms = self.current_expr.num_terms()
            obs = self._get_observation()  # This will raise TypeError if complex

        except (TypeError, ValueError, AttributeError) as e:
            # Action produced invalid expression (e.g., complex numbers)
            # Revert to previous expression and give negative reward
            self.current_expr = prev_expr
            current_num_terms = self.prev_num_terms
            reward = -0.5
            terminated = False
            truncated = False

            # Update state
            self.prev_action = action
            self.prev_num_terms = current_num_terms

            # Get observation from reverted expression
            obs = self._get_observation()

            info = {
                'num_terms': current_num_terms,
                'min_num_terms': self.min_num_terms,
                'simplified': False,
                'error': str(e)
            }

            return obs, reward, terminated, truncated, info

        # Calculate reward (Eq. 66 in paper)
        # Sparse reward: +1 ONLY when reaching a NEW global minimum
        # r_t = 1 if N^dilogs_t < N^dilogs_t' for ALL t' < t, else 0
        if current_num_terms < self.min_num_terms:
            reward = 1.0
            self.min_num_terms = current_num_terms
        else:
            reward = 0.0

        # Cyclic penalty (Eq. 68 in paper)
        if self.use_cyclic_penalty:
            if (self.prev_action is not None and
                action == self.prev_action and
                current_num_terms >= self.prev_num_terms):
                reward -= self.penalty_lambda

        # Check termination conditions
        terminated = False
        truncated = False
        if current_num_terms == 0:  # Simplified to 0
            terminated = True
        elif self.current_step >= self.max_steps:
            truncated = True

        # Update state
        self.prev_action = action
        self.prev_num_terms = current_num_terms

        # Observation
        obs = self._get_observation()

        # Info
        info = {
            'num_terms': current_num_terms,
            'min_num_terms': self.min_num_terms,
            'simplified': current_num_terms == 0
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Convert current expression to one-hot encoded observation.
        Also includes previous action, previous num_terms, and min_num_terms.
        """
        # Get prefix notation
        tokens = self.current_expr.to_prefix_notation()

        # Convert to indices
        indices = []
        for token in tokens[:self.max_length]:
            if token in self.token_to_idx:
                indices.append(self.token_to_idx[token])
            else:
                indices.append(self.token_to_idx['<UNK>'])

        # Pad to max_length
        while len(indices) < self.max_length:
            indices.append(self.pad_idx)

        # Create one-hot encoding
        obs = np.zeros((self.max_length, self.vocab_size), dtype=np.float32)
        for i, idx in enumerate(indices):
            obs[i, idx] = 1.0

        # Flatten for compatibility
        obs = obs.reshape(-1)

        # Add additional state information
        # (prev_action, prev_num_terms, min_num_terms)
        additional_info = np.array([
            self.prev_action if self.prev_action is not None else -1,
            self.prev_num_terms if self.prev_num_terms is not None else 0,
            self.min_num_terms if self.min_num_terms is not None else 0
        ], dtype=np.float32)

        obs = np.concatenate([obs, additional_info])

        return obs

    def render(self, mode='human'):
        """Render the current state"""
        print(f"Step {self.current_step}: {self.current_expr}")
        print(f"  Num terms: {self.current_expr.num_terms()}")


if __name__ == "__main__":
    # Test the environment
    print("Testing DilogEnv...")

    env = DilogEnv(dataset_path="../data/train_set.pkl")

    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    env.render()

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nAction: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        env.render()
        print(f"Info: {info}")

        if terminated or truncated:
            break
