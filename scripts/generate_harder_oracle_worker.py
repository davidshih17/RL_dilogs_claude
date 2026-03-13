#!/usr/bin/env python
"""
Worker script for parallel generation of harder test data with oracle trajectories.
Each worker generates a portion of the dataset with a unique random seed.

Usage:
    python generate_harder_oracle_worker.py --worker_id 0 --num_workers 100 \
        --total_samples 5000 --output_dir data/harder_oracle_chunks
"""

import argparse
import os
import sys
import pickle
import random

import numpy as np
import sympy as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), 'src'))
sys.path.insert(0, SCRIPT_DIR)

from generate_harder_with_oracle import generate_harder_sample_with_oracle, generate_zero_target_with_oracle


def main():
    parser = argparse.ArgumentParser(description='Worker for parallel harder data generation')
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0 to num_workers-1)')
    parser.add_argument('--num_workers', type=int, default=100, help='Total number of workers')
    parser.add_argument('--total_samples', type=int, required=True, help='Total samples to generate across all workers')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for chunks')
    parser.add_argument('--max_degree', type=int, default=2)
    parser.add_argument('--max_coeff', type=int, default=2)
    parser.add_argument('--seed_offset', type=int, default=0, help='Offset for random seed (use different for train/test)')
    args = parser.parse_args()

    # Set random seed based on worker_id
    seed = args.worker_id + args.seed_offset
    random.seed(seed)
    np.random.seed(seed)

    # Calculate samples for this worker
    samples_per_worker = args.total_samples // args.num_workers
    extra = args.total_samples % args.num_workers
    # First 'extra' workers get one more sample
    if args.worker_id < extra:
        my_samples = samples_per_worker + 1
    else:
        my_samples = samples_per_worker

    # Distribute samples across ns values (0,1,2,3)
    samples_per_ns = my_samples // 4
    extra_ns = my_samples % 4

    print(f"Worker {args.worker_id}/{args.num_workers}: generating {my_samples} samples (seed={seed})")

    x = sp.Symbol('x')
    action_list = ['inversion', 'reflection', 'duplication']

    # Configuration for each ns value
    configs = [
        {'ns': 0, 'num_zeros': 2, 'max_scr': 5},
        {'ns': 1, 'num_zeros': 1, 'max_scr': 5},
        {'ns': 2, 'num_zeros': 1, 'max_scr': 6},
        {'ns': 3, 'num_zeros': 1, 'max_scr': 7},
    ]

    all_samples = []

    for i, config in enumerate(configs):
        ns = config['ns']
        num_zeros = config['num_zeros']
        max_scr = config['max_scr']

        # Distribute extra samples to first few ns values
        target_samples = samples_per_ns + (1 if i < extra_ns else 0)

        if target_samples == 0:
            continue

        print(f"  ns={ns}: generating {target_samples} samples")

        samples = []
        attempts = 0
        max_attempts = target_samples * 50

        while len(samples) < target_samples and attempts < max_attempts:
            attempts += 1

            result = generate_harder_sample_with_oracle(
                ns, num_zeros, max_scr,
                args.max_degree, args.max_coeff, x, action_list
            )

            if result is None:
                continue

            if len(result['trajectory']) < 1:
                continue

            samples.append(result)

        print(f"    Generated {len(samples)} samples for ns={ns}")
        all_samples.extend(samples)

    # Save this worker's chunk
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'chunk_{args.worker_id:04d}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(all_samples, f)

    print(f"Worker {args.worker_id}: saved {len(all_samples)} samples to {output_path}")


if __name__ == '__main__':
    main()
