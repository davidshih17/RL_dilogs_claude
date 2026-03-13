"""
Evaluate SFT Transformer with anti-cycle inference on HARDER test set.

Combines:
- Anti-cycle inference (mask out visited state-action pairs)
- Harder test data support (non-zero target ns)

Success = reducing expression to target ns terms or fewer.
"""

import argparse
import pickle
import numpy as np
import torch
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from dilog_env_rllib import DilogEnvRLlib
from transformer_rl_module import TransformerEncoder


def obs_to_hash(obs):
    """Hash observation for cycle detection."""
    return hash(obs['observations'].tobytes())


ACTION_NAMES = ['reflection', 'inversion', 'duplication']


def evaluate_with_anticycle(
    checkpoint_path: str,
    test_data_path: str,
    max_steps: int = 50,
    verbose: bool = False,
    verbose_all: bool = False
):
    """
    Evaluate the SFT model with anti-cycle inference on harder data.

    Args:
        checkpoint_path: Path to model checkpoint
        test_data_path: Path to test data pickle
        max_steps: Maximum steps per episode
        verbose: Print per-sample failure lines
        verbose_all: Print full per-step details for every sample

    Returns:
        results dict with solve rates by ns
    """
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    encoder = TransformerEncoder(
        max_terms=15,
        term_feature_dim=17,
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        ff_dim=128,
        dropout=0.1,
        features_dim=128,
    )

    pi = torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 3),
    )

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    pi.load_state_dict(checkpoint['pi_state_dict'])
    encoder.eval()
    pi.eval()

    FLOAT_MIN = -3.4e38

    # Load test data
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    # Set up env
    env_config = {
        'dataset_path': test_data_path,
        'max_steps': max_steps,
        'max_terms': 15,
        'max_poly_degree': 6,
        'use_cyclic_penalty': False,
    }
    env = DilogEnvRLlib(env_config)

    # Track results by ns and by scramble depth
    solved_by_ns = defaultdict(int)
    total_by_ns = defaultdict(int)
    steps_by_ns = defaultdict(list)
    solved_by_scr = defaultdict(int)
    total_by_scr = defaultdict(int)
    steps_by_scr = defaultdict(list)
    failures = []

    num_samples = len(test_data)
    print(f"\nEvaluating {num_samples} samples...")

    running_solved = 0

    for sample_idx in range(num_samples):
        sample = test_data[sample_idx]

        # Get target expression and compute actual ns from it
        target_expr = sample.get('target_expression', None)
        if target_expr is not None and hasattr(target_expr, 'num_terms'):
            ns = target_expr.num_terms()
        elif target_expr is not None and hasattr(target_expr, 'terms'):
            ns = len(target_expr.terms)
        else:
            ns = 0
        ns_metadata = sample.get('num_terms_simple', 0)
        n_scr = sample.get('num_scrambles', -1)
        total_by_ns[ns] += 1
        if n_scr >= 0:
            total_by_scr[n_scr] += 1

        # Reset with expression and target
        obs, info = env.reset(options={
            'expression': sample['expression'],
            'target_expression': target_expr,
        })
        done = False

        # Track visited (state_hash, action) pairs for anti-cycle
        visited_state_actions = set()
        state_hash = obs_to_hash(obs)
        episode_steps = 0
        min_terms_reached = info.get('num_terms', 15)
        first_solved_step = None  # Track when target was first reached

        if verbose_all:
            print(f"\n{'='*80}")
            ns_note = f" (metadata={ns_metadata})" if ns != ns_metadata else ""
            print(f"Sample {sample_idx}: ns={ns}{ns_note}, scramble_depth={n_scr}, "
                  f"starting_terms={info.get('num_terms', '?')}")
            print(f"  Target: {target_expr}")
            print(f"  Start:  {env.current_expr}")
            for ti, (c, a) in enumerate(env.current_expr.terms):
                print(f"    Term {ti}: coeff={c}, arg={a}")

        while not done:
            observations = torch.tensor(obs['observations'], dtype=torch.float32).unsqueeze(0)
            action_mask = torch.tensor(obs['action_mask'], dtype=torch.float32).unsqueeze(0).clone()

            # Anti-cycle: mask out actions that would revisit a state
            n_masked = 0
            for action_idx in range(action_mask.shape[1]):
                if action_mask[0, action_idx] > 0 and (state_hash, action_idx) in visited_state_actions:
                    action_mask[0, action_idx] = 0.0
                    n_masked += 1

            # Fallback if all actions masked
            anticycle_fallback = False
            if action_mask.sum() == 0:
                action_mask = torch.tensor(obs['action_mask'], dtype=torch.float32).unsqueeze(0)
                anticycle_fallback = True

            with torch.no_grad():
                encoder_out = encoder(observations)
                term_features = encoder_out['term_features']
                term_logits = pi(term_features)
                logits = term_logits.permute(0, 2, 1).reshape(1, -1)
                inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
                logits = logits + inf_mask
                action = torch.argmax(logits, dim=-1).item()

            # Decode action
            action_type_idx = action // env.max_terms
            term_idx = action % env.max_terms
            action_type_name = ACTION_NAMES[action_type_idx] if action_type_idx < 3 else f"unknown({action_type_idx})"

            # Record this (state, action) pair
            visited_state_actions.add((state_hash, action))

            obs, reward, terminated, truncated, info = env.step(action)
            state_hash = obs_to_hash(obs)
            episode_steps += 1
            current_terms = info.get('num_terms', 15)
            min_terms_reached = min(min_terms_reached, current_terms)

            if verbose_all:
                fallback_str = " [ANTICYCLE FALLBACK]" if anticycle_fallback else ""
                print(f"  Step {episode_steps}: {action_type_name}(term {term_idx}) -> "
                      f"{current_terms} terms, reward={reward:.3f}{fallback_str}")
                print(f"    Expr: {env.current_expr}")

            # Record the first step at which target was reached
            if first_solved_step is None and current_terms <= ns:
                first_solved_step = episode_steps

            done = terminated or truncated

        # Check if solved (reached target ns or fewer terms at any point)
        # Use min_terms_reached since model may overshoot past the target
        solved = (min_terms_reached <= ns) or info.get('simplified', False)

        solve_steps = first_solved_step if first_solved_step is not None else episode_steps
        if solved:
            solved_by_ns[ns] += 1
            steps_by_ns[ns].append(solve_steps)
            if n_scr >= 0:
                solved_by_scr[n_scr] += 1
                steps_by_scr[n_scr].append(solve_steps)
            running_solved += 1
            if verbose_all:
                print(f"  >> SOLVED in {solve_steps} steps (min_terms={min_terms_reached}, target_ns={ns})")
        else:
            failures.append((sample_idx, ns, n_scr, min_terms_reached))
            if verbose or verbose_all:
                print(f"  >> FAILED: Sample {sample_idx}, ns={ns}, min_terms={min_terms_reached}, steps={episode_steps}")


        # Print progress every 10 samples
        if (sample_idx + 1) % 10 == 0:
            running_rate = 100 * running_solved / (sample_idx + 1)
            print(f"[{sample_idx + 1}/{num_samples}] Solved: {running_solved}/{sample_idx + 1} ({running_rate:.1f}%)", flush=True)

    env.close()

    # Print results
    total_solved = sum(solved_by_ns.values())
    total = len(test_data)

    print(f"\nOverall: {total_solved}/{total} solved ({100*total_solved/total:.1f}%)")
    print(f"\nBy ns (actual target terms):")
    for ns in sorted(total_by_ns.keys()):
        solved = solved_by_ns[ns]
        total_ns = total_by_ns[ns]
        avg_steps = np.mean(steps_by_ns[ns]) if steps_by_ns[ns] else 0
        print(f"  ns={ns}: {solved}/{total_ns} ({100*solved/total_ns:.1f}%), avg_steps={avg_steps:.1f}")

    if total_by_scr:
        print(f"\nBy scramble depth:")
        for scr in sorted(total_by_scr.keys()):
            solved = solved_by_scr[scr]
            total_scr = total_by_scr[scr]
            avg_steps = np.mean(steps_by_scr[scr]) if steps_by_scr[scr] else 0
            print(f"  {scr} scrambles: {solved}/{total_scr} ({100*solved/total_scr:.1f}%), avg_steps={avg_steps:.1f}")

    return {
        'total_solved': total_solved,
        'total': total,
        'solve_rate': total_solved / total,
        'solved_by_ns': dict(solved_by_ns),
        'total_by_ns': dict(total_by_ns),
        'failures': failures,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate SFT with anti-cycle on harder data')
    parser.add_argument('--checkpoint', type=str,
                        default='models/sft_harder_100k_v1/best_model.pt')
    parser.add_argument('--test_data', type=str,
                        default='data/harder_test_5k_oracle.pkl')
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--verbose_all', action='store_true',
                        help='Print full per-step details for every sample')

    args = parser.parse_args()

    print("=" * 70)
    print("SFT Transformer Evaluation with Anti-Cycle Inference (Harder Data)")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")

    results = evaluate_with_anticycle(
        args.checkpoint,
        args.test_data,
        args.max_steps,
        args.verbose,
        args.verbose_all
    )

    print(f"\nFirst 10 failures: {results['failures'][:10]}")


if __name__ == '__main__':
    main()
