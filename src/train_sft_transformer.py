"""
SFT training using oracle trajectories with TRANSFORMER architecture.

Uses self-attention to allow each term to see all other terms,
enabling the model to learn pairwise relationships.

Usage:
    python train_sft_transformer.py --output_dir ../models/sft_transformer_v1
"""

import argparse
import os
import sys
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dilog_env_equivariant import extract_term_features
from dilog_utils import DilogExpression
from transformer_rl_module import TransformerPolicySFT


@dataclass
class OracleTransition:
    obs: np.ndarray
    action_mask: np.ndarray
    expert_action: int


class OracleDataset(torch.utils.data.Dataset):
    def __init__(self, transitions: List[OracleTransition]):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        return {
            'obs': torch.tensor(t.obs, dtype=torch.float32),
            'action_mask': torch.tensor(t.action_mask, dtype=torch.float32),
            'expert_action': torch.tensor(t.expert_action, dtype=torch.long),
        }


def load_oracle_data(oracle_dir: str) -> List[Dict]:
    """Load oracle trajectory data from pickle files."""
    all_samples = []
    oracle_path = Path(oracle_dir)
    pkl_files = sorted(oracle_path.glob('*.pkl'))

    print(f"Found {len(pkl_files)} oracle data files")
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            samples = pickle.load(f)
            print(f"  Loaded {len(samples)} samples from {pkl_file.name}")
            all_samples.extend(samples)

    print(f"Total: {len(all_samples)} samples with oracle trajectories")
    return all_samples


def oracle_to_transitions(
    samples: List[Dict],
    max_terms: int = 15,
    max_poly_degree: int = 6,
) -> List[OracleTransition]:
    """Convert oracle samples to training transitions."""
    transitions = []
    term_feature_dim = 2 + 2 + 2 * max_poly_degree  # 16 features

    for sample in samples:
        for step in sample['trajectory']:
            state = step['state']
            action_idx = step['action_idx']
            term_idx = step['term_idx']

            n_terms = state.num_terms()

            # Term observations (16 features + 1 validity flag = 17)
            term_obs = np.zeros((max_terms, term_feature_dim + 1), dtype=np.float32)
            for i, (coeff, arg) in enumerate(state.terms[:max_terms]):
                try:
                    features = extract_term_features(coeff, arg, max_poly_degree)
                    features = np.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0)
                    term_obs[i, :-1] = features
                    term_obs[i, -1] = 1.0  # Valid flag
                except Exception:
                    term_obs[i, -1] = 0.0

            flat_terms = term_obs.reshape(-1)

            # Global features (3 values)
            global_features = np.array([
                float(n_terms),
                -1.0,  # prev_action
                float(n_terms),  # min_num_terms
            ], dtype=np.float32)

            obs = np.concatenate([flat_terms, global_features])
            obs = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)

            # Action mask
            n_actions = 3 * max_terms
            action_mask = np.zeros(n_actions, dtype=np.float32)
            for identity_idx in range(3):
                for ti in range(min(n_terms, max_terms)):
                    action_mask[identity_idx * max_terms + ti] = 1.0

            # Convert (action_idx, term_idx) to single action
            action = action_idx * max_terms + term_idx

            transitions.append(OracleTransition(
                obs=obs,
                action_mask=action_mask,
                expert_action=action,
            ))

    return transitions


def train_sft(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: str,
    output_dir: str,
    warmup_epochs: int = 5,
    start_epoch: int = 0,
    best_val_acc: float = 0.0,
):
    """Train the model using supervised learning on oracle data."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler with warmup (adjusted for start_epoch)
    def lr_lambda(epoch):
        actual_epoch = epoch + start_epoch
        if actual_epoch < warmup_epochs:
            return (actual_epoch + 1) / warmup_epochs
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    FLOAT_MIN = -3.4e38

    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            obs = batch['obs'].to(device)
            action_mask = batch['action_mask'].to(device)
            expert_actions = batch['expert_action'].to(device)

            optimizer.zero_grad()
            logits = model(obs)

            # Apply action mask
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
            masked_logits = logits + inf_mask

            loss = F.cross_entropy(masked_logits, expert_actions)
            loss.backward()

            # Gradient clipping for transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * obs.size(0)
            preds = masked_logits.argmax(dim=-1)
            train_correct += (preds == expert_actions).sum().item()
            train_total += expert_actions.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                action_mask = batch['action_mask'].to(device)
                expert_actions = batch['expert_action'].to(device)

                logits = model(obs)
                inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
                masked_logits = logits + inf_mask

                loss = F.cross_entropy(masked_logits, expert_actions)
                val_loss += loss.item() * obs.size(0)
                preds = masked_logits.argmax(dim=-1)
                val_correct += (preds == expert_actions).sum().item()
                val_total += expert_actions.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
              f"lr={current_lr:.6f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'pi_state_dict': model.pi.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'pi_state_dict': model.pi.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pt'))

    return best_val_loss, best_val_acc


def train(args):
    """Main training function."""
    print("=" * 70)
    print("SFT Training with Transformer Architecture")
    print(f"Started: {datetime.now()}")
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Transformer: embed_dim={args.embed_dim}, num_heads={args.num_heads}, "
          f"num_layers={args.num_layers}, ff_dim={args.ff_dim}")
    print("=" * 70)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load oracle data
    print("\nLoading oracle data...")
    samples = load_oracle_data(args.oracle_dir)

    print("\nConverting to transitions...")
    transitions = oracle_to_transitions(
        samples,
        max_terms=args.max_terms,
        max_poly_degree=6,
    )
    print(f"Created {len(transitions)} training transitions")

    # Verify observation dimensions
    obs_dim = transitions[0].obs.shape[0]
    term_feature_dim = (obs_dim - 3) // args.max_terms
    print(f"Observation dimension: {obs_dim}")
    print(f"Inferred term_feature_dim: {term_feature_dim}")

    # Train/val split
    random.shuffle(transitions)
    val_size = int(len(transitions) * args.val_split)
    val_transitions = transitions[:val_size]
    train_transitions = transitions[val_size:]

    print(f"Train: {len(train_transitions)}, Val: {len(val_transitions)}")

    train_dataset = OracleDataset(train_transitions)
    val_dataset = OracleDataset(val_transitions)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create transformer model
    n_actions = 3 * args.max_terms
    model = TransformerPolicySFT(
        max_terms=args.max_terms,
        term_feature_dim=term_feature_dim,
        n_actions=n_actions,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        features_dim=128,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"Policy head parameters: {sum(p.numel() for p in model.pi.parameters()):,}")

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        print(f"  Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    print("\n" + "=" * 70)
    print("Starting SFT training...")
    print("=" * 70)
    start_time = time.time()

    best_loss, best_acc = train_sft(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
        warmup_epochs=args.warmup_epochs,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best validation accuracy: {best_acc:.4f}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'pi_state_dict': model.pi.state_dict(),
        'args': vars(args),
        'best_val_loss': best_loss,
        'best_val_acc': best_acc,
    }, os.path.join(args.output_dir, 'final_model.pt'))

    print(f"\nFinal model saved to {args.output_dir}")
    print(f"Training completed: {datetime.now()}")


def main():
    parser = argparse.ArgumentParser(description='SFT training with transformer architecture')

    # Data paths
    parser.add_argument('--oracle_dir', type=str,
                        default='data/harder_train_100k_oracle')
    parser.add_argument('--output_dir', type=str,
                        default='models/sft_transformer_v1')

    # Training config
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)  # Smaller batch for transformer
    parser.add_argument('--lr', type=float, default=1e-4)       # Lower lr for transformer
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--max_terms', type=int, default=15)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # Transformer architecture
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Transformer hidden dimension')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--ff_dim', type=int, default=128,
                        help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1)

    # Resume from checkpoint
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Misc
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    print(f"Using random seed: {args.seed}")
    train(args)


if __name__ == '__main__':
    main()
