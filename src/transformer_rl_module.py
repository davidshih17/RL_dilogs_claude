"""
Transformer-based model for dilogarithm simplification.

Uses self-attention to allow each term to "see" all other terms,
enabling the model to recognize pairwise relationships like potential
cancellations between terms.

Architecture:
1. Per-term embedding (linear projection)
2. Transformer encoder (self-attention layers)
3. Per-term contextualized representations
4. Policy head (per-term action logits)
"""

import numpy as np
import torch
import torch.nn as nn
import math
from typing import Any, Dict, Optional, List, Union

# Constant for masking invalid actions
FLOAT_MIN = -3.4e38


def ortho_init(module: nn.Module, gain: float = 1.0):
    """Orthogonal initialization matching SB3's default."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for dilogarithm terms.

    Each term attends to all other terms, allowing the model to learn
    pairwise relationships (e.g., which terms might cancel).

    Architecture:
    - Linear embedding of term features
    - Stack of transformer encoder layers
    - Output: per-term contextualized representations + global representation
    """

    def __init__(
        self,
        max_terms: int = 15,
        term_feature_dim: int = 17,  # 16 features + 1 validity flag
        embed_dim: int = 64,         # Transformer hidden dimension
        num_heads: int = 4,          # Number of attention heads
        num_layers: int = 3,         # Number of transformer layers
        ff_dim: int = 128,           # Feedforward dimension
        dropout: float = 0.1,
        features_dim: int = 128,     # Output dimension for compatibility
    ):
        super().__init__()

        self.max_terms = max_terms
        self.term_feature_dim = term_feature_dim
        self.embed_dim = embed_dim

        # Term embedding: project raw features to embed_dim
        self.term_embedding = nn.Linear(term_feature_dim, embed_dim)

        # Learnable [CLS] token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection (for compatibility with existing code)
        # Takes CLS token + global features -> features_dim
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim + 3, features_dim),  # +3 for metadata
            nn.ReLU(),
        )

        self.output_dim = features_dim
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        sqrt2 = np.sqrt(2.0)
        ortho_init(self.term_embedding, gain=sqrt2)
        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                ortho_init(layer, gain=sqrt2)

    def forward(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            observations: (batch, obs_dim) where obs_dim = max_terms * term_feature_dim + 3

        Returns:
            Dict with:
                'features': (batch, features_dim) global features for value function
                'term_features': (batch, max_terms, embed_dim) per-term features for policy
        """
        batch_size = observations.shape[0]

        # Split observation: term features + metadata
        term_obs_flat = observations[:, :-3]
        metadata = observations[:, -3:]  # (batch, 3)

        # Reshape to (batch, max_terms, term_feature_dim)
        term_obs = term_obs_flat.view(batch_size, self.max_terms, self.term_feature_dim)

        # Validity mask is last feature of each term
        term_mask = term_obs[:, :, -1]  # (batch, max_terms)

        # Embed terms
        term_embeddings = self.term_embedding(term_obs)  # (batch, max_terms, embed_dim)

        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        sequence = torch.cat([cls_tokens, term_embeddings], dim=1)  # (batch, max_terms+1, embed_dim)

        # Create attention mask: CLS can attend to everything, terms attend based on validity
        # Shape: (batch, max_terms+1) - True means IGNORE this position
        cls_mask = torch.zeros(batch_size, 1, device=observations.device, dtype=torch.bool)
        term_attn_mask = (term_mask == 0)  # True where invalid (should be masked)
        attn_mask = torch.cat([cls_mask, term_attn_mask], dim=1)  # (batch, max_terms+1)

        # Apply transformer
        # Note: src_key_padding_mask expects True for positions to IGNORE
        transformed = self.transformer(sequence, src_key_padding_mask=attn_mask)

        # Extract outputs
        cls_output = transformed[:, 0, :]  # (batch, embed_dim) - global representation
        term_outputs = transformed[:, 1:, :]  # (batch, max_terms, embed_dim) - per-term

        # Mask invalid terms in output
        term_outputs = term_outputs * term_mask.unsqueeze(-1)

        # Global features for value function
        global_input = torch.cat([cls_output, metadata], dim=-1)  # (batch, embed_dim + 3)
        global_features = self.output_projection(global_input)  # (batch, features_dim)

        return {
            'features': global_features,
            'term_features': term_outputs,
            'term_mask': term_mask,
        }


# ============================================================================
# SFT version (for supervised training on oracle trajectories)
# ============================================================================

class TransformerPolicySFT(nn.Module):
    """
    Transformer policy for SFT training.

    Same architecture as TransformerActionMaskingRLModule but without
    RLlib dependencies, for supervised learning on oracle trajectories.
    """

    def __init__(
        self,
        max_terms: int = 15,
        term_feature_dim: int = 17,
        n_actions: int = 45,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 128,
        dropout: float = 0.1,
        features_dim: int = 128,
        pi_hidden: List[int] = None,
    ):
        super().__init__()

        if pi_hidden is None:
            pi_hidden = [64, 64]

        self.max_terms = max_terms
        self.n_identities = 3

        # Transformer encoder
        self.encoder = TransformerEncoder(
            max_terms=max_terms,
            term_feature_dim=term_feature_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            features_dim=features_dim,
        )

        # Policy head: per-term to per-identity logits
        self.pi = nn.Sequential(
            nn.Linear(embed_dim, pi_hidden[0]),
            nn.ReLU(),
            nn.Linear(pi_hidden[0], pi_hidden[1]),
            nn.ReLU(),
            nn.Linear(pi_hidden[1], self.n_identities),
        )

        self._init_policy_head()

    def _init_policy_head(self):
        sqrt2 = np.sqrt(2.0)
        pi_layers = list(self.pi)
        for i, layer in enumerate(pi_layers):
            if isinstance(layer, nn.Linear):
                if i == len(pi_layers) - 1:
                    ortho_init(layer, gain=0.01)
                else:
                    ortho_init(layer, gain=sqrt2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim)

        Returns:
            logits: (batch, n_actions) action logits
        """
        batch_size = obs.shape[0]

        encoder_out = self.encoder(obs)
        term_features = encoder_out['term_features']  # (batch, max_terms, embed_dim)

        # Per-term, per-identity logits
        term_logits = self.pi(term_features)  # (batch, max_terms, 3)

        # Reshape to (batch, 3 * max_terms)
        logits = term_logits.permute(0, 2, 1).reshape(batch_size, -1)

        return logits
