"""
Extract the 164 overlap indices: test samples whose expression matches
any training state (starting expression or trajectory intermediate),
using sorted-term symbolic hashing (permutation-invariant).

Outputs a pickle file with the set of overlap indices.
"""
import pickle
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(BASE, 'src'))
from dilog_utils import DilogExpression
TRAIN_PATH = os.path.join(BASE, 'data/harder_train_100k_oracle.pkl')
TEST_PATH = os.path.join(BASE, 'data/paper_transformer_test_full.pkl')
OUT_PATH = os.path.join(BASE, 'data/overlap_indices_164.pkl')


def expr_to_hashable(expr):
    """Hash a DilogExpression by sorted (str(coeff), str(arg)) tuples."""
    terms = []
    for coeff, arg in expr.terms:
        terms.append((str(coeff), str(arg)))
    return tuple(sorted(terms))


t0 = time.time()

# Phase 1: Build hash set from all training states
print("Loading training data...", flush=True)
with open(TRAIN_PATH, 'rb') as f:
    train_data = pickle.load(f)
print(f"  {len(train_data)} samples")

print("Hashing all training states...", flush=True)
train_hashes = set()
n_total = 0

for i, s in enumerate(train_data):
    h = expr_to_hashable(s['expression'])
    train_hashes.add(h)
    n_total += 1

    for step in s['trajectory']:
        state = step['state']
        if isinstance(state, DilogExpression):
            h = expr_to_hashable(state)
            train_hashes.add(h)
            n_total += 1

    if (i + 1) % 20000 == 0:
        print(f"  {i+1}/{len(train_data)}, {n_total} states, {len(train_hashes)} unique", flush=True)

print(f"  Total: {n_total} states, {len(train_hashes)} unique hashes")
del train_data

# Phase 2: Check test set
print("\nLoading test data...", flush=True)
with open(TEST_PATH, 'rb') as f:
    test_data = pickle.load(f)
print(f"  {len(test_data)} test samples")

overlap_indices = set()
for i, s in enumerate(test_data):
    h = expr_to_hashable(s['expression'])
    if h in train_hashes:
        overlap_indices.add(i)

print(f"\nOverlap indices: {len(overlap_indices)}")
print(f"Indices: {sorted(overlap_indices)}")

# Save
with open(OUT_PATH, 'wb') as f:
    pickle.dump(sorted(overlap_indices), f)
print(f"\nSaved to {OUT_PATH}")
print(f"Total time: {time.time()-t0:.1f}s")
