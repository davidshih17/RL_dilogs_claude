#!/usr/bin/env python
"""
Merge chunks from parallel generation into a single dataset.

Usage:
    python merge_harder_oracle_chunks.py --input_dir data/harder_oracle_chunks \
        --output_path data/harder_test_oracle.pkl
"""

import argparse
import os
import sys
import pickle
import glob
import random
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), 'src'))


def main():
    parser = argparse.ArgumentParser(description='Merge chunks from parallel generation')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing chunk files')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for merged dataset')
    parser.add_argument('--shuffle', action='store_true', default=True, help='Shuffle the merged dataset')
    args = parser.parse_args()

    # Find all chunk files
    chunk_files = sorted(glob.glob(os.path.join(args.input_dir, 'chunk_*.pkl')))
    print(f"Found {len(chunk_files)} chunk files in {args.input_dir}")

    if len(chunk_files) == 0:
        print("No chunk files found!")
        return

    # Load and merge all chunks
    all_samples = []
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            samples = pickle.load(f)
        print(f"  {os.path.basename(chunk_file)}: {len(samples)} samples")
        all_samples.extend(samples)

    print(f"\nTotal samples: {len(all_samples)}")

    # Distribution by ns
    ns_counts = defaultdict(int)
    for s in all_samples:
        ns_counts[s['num_terms_simple']] += 1
    print("\nDistribution by ns:")
    for ns in sorted(ns_counts.keys()):
        print(f"  ns={ns}: {ns_counts[ns]} ({100*ns_counts[ns]/len(all_samples):.1f}%)")

    # Shuffle if requested
    if args.shuffle:
        random.shuffle(all_samples)
        print("\nShuffled dataset")

    # Save merged dataset
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(all_samples, f)
    print(f"\nSaved to {args.output_path}")

    # Print some statistics
    if all_samples:
        avg_traj = sum(len(s['trajectory']) for s in all_samples) / len(all_samples)
        avg_terms = sum(s['expression'].num_terms() for s in all_samples) / len(all_samples)
        print(f"\nStatistics:")
        print(f"  Avg trajectory length: {avg_traj:.1f}")
        print(f"  Avg num terms (scrambled): {avg_terms:.1f}")


if __name__ == '__main__':
    main()
