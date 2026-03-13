"""
Dilogarithm simplification environment with:
- Graph-based term features
- 3N action space (3 identities × N terms) with masking
- Permutation equivariant representation

Action space: 3 * max_terms actions
- Actions 0 to max_terms-1: Apply reflection to term i
- Actions max_terms to 2*max_terms-1: Apply inversion to term i
- Actions 2*max_terms to 3*max_terms-1: Apply duplication to term i

Invalid actions (terms that don't exist) are masked.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import sympy as sp
from sympy import symbols, Poly, fraction
from typing import Optional

from dilog_utils import (
    DilogExpression, apply_reflection, apply_inversion,
    apply_duplication, x
)


def extract_term_features(coeff: float, arg: sp.Expr, max_poly_degree: int = 6) -> np.ndarray:
    """
    Extract a fixed-size feature vector from a single Li_2 term.
    Features are normalized/clipped to prevent overflow.
    """
    features = []

    # Coefficient features - use log scale for magnitude, sign separate
    abs_coeff = abs(float(coeff))
    # Log scale with small epsilon to handle coeff=0
    log_coeff = np.log1p(abs_coeff)  # log(1 + |coeff|), bounded and stable
    features.append(np.clip(log_coeff, -10, 10))
    features.append(1.0 if coeff >= 0 else -1.0)

    try:
        num, den = fraction(arg)
        num_poly = Poly(num, x) if num.has(x) else Poly(num, x, domain='ZZ')
        den_poly = Poly(den, x) if den.has(x) else Poly(den, x, domain='ZZ')

        num_deg = num_poly.degree() if num.has(x) else 0
        den_deg = den_poly.degree() if den.has(x) else 0
        features.append(float(num_deg))
        features.append(float(den_deg))

        num_coeffs = [float(c) for c in num_poly.all_coeffs()] if num.has(x) else [float(num)]
        den_coeffs = [float(c) for c in den_poly.all_coeffs()] if den.has(x) else [float(den)]

        num_coeffs = num_coeffs[:max_poly_degree] + [0.0] * (max_poly_degree - len(num_coeffs))
        den_coeffs = den_coeffs[:max_poly_degree] + [0.0] * (max_poly_degree - len(den_coeffs))

        # Apply log-scale transformation to polynomial coefficients too
        # sign(x) * log(1 + |x|) preserves sign and prevents overflow
        def safe_log_transform(val):
            sign = 1.0 if val >= 0 else -1.0
            return sign * np.log1p(abs(val))

        num_coeffs = [np.clip(safe_log_transform(c), -20, 20) for c in num_coeffs]
        den_coeffs = [np.clip(safe_log_transform(c), -20, 20) for c in den_coeffs]

        features.extend(num_coeffs)
        features.extend(den_coeffs)

    except Exception:
        features.extend([0.0] * (2 + 2 * max_poly_degree))

    return np.array(features, dtype=np.float32)


class DilogEnvEquivariant(gym.Env):
    """
    Dilogarithm simplification with 3N action space and equivariant observations.

    Observation: (max_terms, term_feature_dim + 1) where +1 is validity mask
    Action: 3 * max_terms discrete actions (masked for invalid terms)
    """

    def __init__(
        self,
        dataset_path: str,
        max_steps: int = 50,
        max_terms: int = 15,
        max_poly_degree: int = 6,
        use_cyclic_penalty: bool = True,
        penalty_lambda: float = 0.25
    ):
        super().__init__()

        self.max_steps = max_steps
        self.max_terms = max_terms
        self.max_poly_degree = max_poly_degree
        self.use_cyclic_penalty = use_cyclic_penalty
        self.penalty_lambda = penalty_lambda

        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)

        # Term features: 2 (coeff) + 2 (degrees) + 2*max_poly_degree (poly coeffs)
        self.term_feature_dim = 2 + 2 + 2 * max_poly_degree

        # Action space: 3 identities × max_terms
        self.n_actions = 3 * max_terms
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation: per-term features + validity mask + global info
        # Shape: (max_terms * (term_feature_dim + 1)) + 3 global features
        self.obs_dim = max_terms * (self.term_feature_dim + 1) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # State
        self.current_expr: Optional[DilogExpression] = None
        self.current_step = 0
        self.prev_action = None
        self.prev_num_terms = None
        self.min_num_terms = None
        self._action_mask = np.zeros(self.n_actions, dtype=np.float32)

    def action_masks(self) -> np.ndarray:
        """Return action mask for MaskablePPO (required method name)"""
        return self._get_action_mask()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Use expression from options if provided, otherwise pick random sample
        if options is not None and 'expression' in options:
            self.current_expr = options['expression'].copy()
        else:
            idx = np.random.randint(len(self.dataset))
            sample = self.dataset[idx]
            self.current_expr = sample['expression'].copy()

        self.current_step = 0
        self.prev_action = None
        self.prev_num_terms = self.current_expr.num_terms()
        self.min_num_terms = self.prev_num_terms

        obs = self._get_observation()
        info = {
            'num_terms': self.prev_num_terms,
            'action_mask': self._get_action_mask()
        }

        return obs, info

    def set_sample(self, sample):
        """Set a specific sample for evaluation"""
        self.current_expr = sample['expression'].copy()
        self.current_step = 0
        self.prev_action = None
        self.prev_num_terms = self.current_expr.num_terms()
        self.min_num_terms = self.prev_num_terms

    def _get_action_mask(self) -> np.ndarray:
        """Get mask for valid actions (1 = valid, 0 = invalid)"""
        mask = np.zeros(self.n_actions, dtype=np.float32)
        n_terms = min(len(self.current_expr.terms), self.max_terms)

        # For each identity, only terms 0 to n_terms-1 are valid
        for identity_idx in range(3):
            for term_idx in range(n_terms):
                action = identity_idx * self.max_terms + term_idx
                if action < self.n_actions:  # Safety check
                    mask[action] = 1.0

        return mask

    def _decode_action(self, action: int):
        """Decode action into (identity_idx, term_idx)"""
        identity_idx = action // self.max_terms
        term_idx = action % self.max_terms
        return identity_idx, term_idx

    def step(self, action: int):
        self.current_step += 1

        identity_idx, term_idx = self._decode_action(action)
        n_terms = len(self.current_expr.terms)

        # Check if action is valid
        if term_idx >= n_terms:
            # Invalid action - penalize and skip
            reward = -0.5
            obs = self._get_observation()
            info = {
                'num_terms': n_terms,
                'min_num_terms': self.min_num_terms,
                'simplified': False,
                'action_mask': self._get_action_mask(),
                'invalid_action': True
            }
            terminated = False
            truncated = self.current_step >= self.max_steps
            return obs, reward, terminated, truncated, info

        # Save previous expression
        prev_expr = self.current_expr.copy()

        # Apply identity directly to the specified term (no rotation needed)
        try:
            if identity_idx == 0:  # Reflection
                expr = apply_reflection(self.current_expr, term_idx)
            elif identity_idx == 1:  # Inversion
                expr = apply_inversion(self.current_expr, term_idx)
            elif identity_idx == 2:  # Duplication
                expr = apply_duplication(self.current_expr, term_idx)

            self.current_expr = expr
            current_num_terms = self.current_expr.num_terms()
            obs = self._get_observation()

        except (TypeError, ValueError, AttributeError) as e:
            self.current_expr = prev_expr
            current_num_terms = self.prev_num_terms
            reward = -0.5
            terminated = False
            truncated = self.current_step >= self.max_steps

            obs = self._get_observation()
            info = {
                'num_terms': current_num_terms,
                'min_num_terms': self.min_num_terms,
                'simplified': False,
                'action_mask': self._get_action_mask(),
                'error': str(e)
            }
            return obs, reward, terminated, truncated, info

        # Reward: +1 for new global minimum
        if current_num_terms < self.min_num_terms:
            reward = 1.0
            self.min_num_terms = current_num_terms
        else:
            reward = 0.0

        # Cyclic penalty
        if self.use_cyclic_penalty:
            if (self.prev_action is not None and
                action == self.prev_action and
                current_num_terms >= self.prev_num_terms):
                reward -= self.penalty_lambda

        terminated = current_num_terms == 0
        truncated = self.current_step >= self.max_steps

        self.prev_action = action
        self.prev_num_terms = current_num_terms

        info = {
            'num_terms': current_num_terms,
            'min_num_terms': self.min_num_terms,
            'simplified': current_num_terms == 0,
            'action_mask': self._get_action_mask()
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get observation with per-term features and validity mask"""
        # Per-term features + validity flag
        term_obs = np.zeros((self.max_terms, self.term_feature_dim + 1), dtype=np.float32)

        n_terms = len(self.current_expr.terms)
        for i, (coeff, arg) in enumerate(self.current_expr.terms[:self.max_terms]):
            try:
                features = extract_term_features(coeff, arg, self.max_poly_degree)
                # Safety: replace any NaN/Inf with 0
                features = np.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0)
                term_obs[i, :-1] = features
                term_obs[i, -1] = 1.0  # Valid flag
            except Exception:
                term_obs[i, -1] = 0.0  # Invalid

        # Flatten term observations
        flat_terms = term_obs.reshape(-1)

        # Global features
        global_features = np.array([
            float(n_terms),
            float(self.prev_action) if self.prev_action is not None else -1.0,
            float(self.min_num_terms) if self.min_num_terms is not None else 0.0
        ], dtype=np.float32)

        obs = np.concatenate([flat_terms, global_features])
        # Final safety check
        obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
        return obs

    def render(self, mode='human'):
        print(f"Step {self.current_step}: {self.current_expr}")
        print(f"  Num terms: {self.current_expr.num_terms()}")


if __name__ == "__main__":
    print("Testing DilogEnvEquivariant...")

    env = DilogEnvEquivariant(dataset_path="../data/train_set.pkl")

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space size: {env.n_actions}")
    print(f"Valid actions: {info['action_mask'].sum():.0f}")
    env.render()

    # Test a few steps with valid actions
    for step in range(5):
        # Sample from valid actions only
        mask = env._get_action_mask()
        valid_actions = np.where(mask > 0)[0]
        action = np.random.choice(valid_actions)

        identity_idx, term_idx = env._decode_action(action)
        identity_names = ['Reflection', 'Inversion', 'Duplication']

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nAction: {identity_names[identity_idx]} on term {term_idx}")
        print(f"Reward: {reward}, Terms: {info['num_terms']}, Valid actions: {info['action_mask'].sum():.0f}")

        if terminated or truncated:
            break
