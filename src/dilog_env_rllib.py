"""
Dilogarithm simplification environment wrapped for Ray RLlib.
Uses Dict observation space with action_mask for RLlib's action masking.

This wraps the base DilogEnvEquivariant environment to be compatible
with RLlib's action masking RLModule.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
from typing import Optional, Tuple, Dict, Any

from dilog_env_equivariant import DilogEnvEquivariant, extract_term_features


class DilogEnvRLlib(gym.Env):
    """
    RLlib-compatible Dilogarithm simplification environment.

    Key difference from DilogEnvEquivariant:
    - Observation space is a Dict with 'observations' and 'action_mask' keys
    - This is required for RLlib's action masking RLModule

    Observation space:
        Dict({
            'observations': Box(shape=(obs_dim,)),  # The actual state features
            'action_mask': Box(shape=(n_actions,))  # Binary mask for valid actions
        })

    Action space: Discrete(3 * max_terms)
        - Actions 0 to max_terms-1: Apply reflection to term i
        - Actions max_terms to 2*max_terms-1: Apply inversion to term i
        - Actions 2*max_terms to 3*max_terms-1: Apply duplication to term i
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the RLlib-compatible environment.

        Args:
            config: RLlib environment config dict with keys:
                - dataset_path: Path to pickle file with training data
                - max_steps: Maximum steps per episode (default: 50)
                - max_terms: Maximum number of terms to handle (default: 15)
                - max_poly_degree: Max polynomial degree for features (default: 6)
                - use_cyclic_penalty: Whether to penalize repeated actions (default: True)
                - penalty_lambda: Penalty coefficient (default: 0.25)
        """
        super().__init__()

        # Handle RLlib config
        if config is None:
            config = {}

        self.dataset_path = config.get('dataset_path', '../data/paper_train_set.pkl')
        self.max_steps = config.get('max_steps', 50)
        self.max_terms = config.get('max_terms', 15)
        self.max_poly_degree = config.get('max_poly_degree', 6)
        self.use_cyclic_penalty = config.get('use_cyclic_penalty', True)
        self.penalty_lambda = config.get('penalty_lambda', 0.25)

        # Create the base environment
        self._base_env = DilogEnvEquivariant(
            dataset_path=self.dataset_path,
            max_steps=self.max_steps,
            max_terms=self.max_terms,
            max_poly_degree=self.max_poly_degree,
            use_cyclic_penalty=self.use_cyclic_penalty,
            penalty_lambda=self.penalty_lambda
        )

        # Copy dimensions from base env
        self.n_actions = self._base_env.n_actions
        self.obs_dim = self._base_env.obs_dim
        self.term_feature_dim = self._base_env.term_feature_dim

        # Define action space (same as base)
        self.action_space = spaces.Discrete(self.n_actions)

        # Define observation space as Dict for RLlib action masking
        self.observation_space = spaces.Dict({
            'observations': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.obs_dim,),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0.0, high=1.0,
                shape=(self.n_actions,),
                dtype=np.float32
            )
        })

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        obs, info = self._base_env.reset(seed=seed, options=options)
        action_mask = self._base_env.action_masks()

        # Return observation as dict for RLlib
        rllib_obs = {
            'observations': obs,
            'action_mask': action_mask
        }

        return rllib_obs, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self._base_env.step(action)
        action_mask = self._base_env.action_masks()

        # Return observation as dict for RLlib
        rllib_obs = {
            'observations': obs,
            'action_mask': action_mask
        }

        return rllib_obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """Render the environment."""
        return self._base_env.render(mode)

    @property
    def current_expr(self):
        """Access current expression from base env."""
        return self._base_env.current_expr

    def set_sample(self, sample):
        """Set a specific sample for evaluation."""
        self._base_env.set_sample(sample)


# Register the environment with gymnasium
def register_dilog_env():
    """Register the DilogEnvRLlib environment with gymnasium."""
    try:
        gym.envs.registration.register(
            id='DilogSimplify-v0',
            entry_point='dilog_env_rllib:DilogEnvRLlib',
        )
    except gym.error.Error:
        # Already registered
        pass


if __name__ == "__main__":
    print("Testing DilogEnvRLlib...")

    # Test with config dict (as RLlib would pass it)
    config = {
        'dataset_path': '../data/paper_train_set.pkl',
        'max_steps': 50,
        'max_terms': 15,
    }

    env = DilogEnvRLlib(config)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"\nReset observation keys: {obs.keys()}")
    print(f"Observations shape: {obs['observations'].shape}")
    print(f"Action mask shape: {obs['action_mask'].shape}")
    print(f"Valid actions: {obs['action_mask'].sum():.0f}")

    # Test a few steps
    for step in range(5):
        # Sample from valid actions only
        valid_actions = np.where(obs['action_mask'] > 0)[0]
        action = np.random.choice(valid_actions)

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step + 1}: action={action}, reward={reward:.2f}, terms={info['num_terms']}")

        if terminated or truncated:
            break

    print("\nTest passed!")
