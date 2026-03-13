#!/usr/bin/env python
"""
Generate paper tables and figures for dilogarithm simplification results.
v3: Fixed ns to use actual target expression term count (not metadata).
    71 samples had metadata ns > actual ns due to sympy auto-simplification
    during data generation.

Compares our model vs Dersy et al. under both source-relative and target-relative criteria.
Excludes:
  - 164 overlap samples (test expressions appearing in training trajectories)
  - 128 pathological samples (source terms <= target terms)

Outputs:
  - Per-n_s and per-scramble-depth failure count breakdowns for both models
  - Figure: solve rate vs scramble depth
  - Figure: solve rate by n_s
"""
import re
import sys
import os
import pickle
import numpy as np
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(BASE, 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
TEST_PATH = os.path.join(BASE, 'data/paper_transformer_test_full.pkl')
OVERLAP_INDICES_PATH = os.path.join(BASE, 'data/overlap_indices_164.pkl')
DERSY_EVAL_PATH = os.path.join(BASE, 'data/dsz_baseline/eval.func_simple.test.0')
FIGURE_DIR = os.path.join(BASE, 'figures')

# Our model's known failures on the FULL 5000 test set
# Target-relative failures (min_terms > n_s):
OUR_TARGET_REL_FAILURES = {251, 2258, 3153, 3560, 4100, 4809}
# Source-relative failures (no reduction at all, min_terms == start_terms):
# Subset of target-rel failures: 3153 (4->4) and 4100 (3->3)
OUR_SOURCE_REL_FAILURES = {3153, 4100}

# ============================================================================
# Load test set metadata
# ============================================================================
print("Loading test set...", flush=True)
with open(TEST_PATH, 'rb') as f:
    test_data = pickle.load(f)
print(f"  Full test set: {len(test_data)} samples")

# Parse metadata for each sample, computing ns from ACTUAL target expression
sample_meta = []
ns_mismatch_count = 0
for idx, sample in enumerate(test_data):
    # Get actual ns from target expression
    target_expr = sample.get('target_expression', None)
    if target_expr is not None and hasattr(target_expr, 'num_terms'):
        actual_ns = target_expr.num_terms()
    elif target_expr is not None and hasattr(target_expr, 'terms'):
        actual_ns = len(target_expr.terms)
    else:
        actual_ns = 0

    # Get metadata ns for comparison
    info = sample.get('info', '')
    m = re.match(r'(\d+),simple(\d+)zeros(\d+)scrambles(\d+)', info)
    if m:
        metadata_ns = int(m.group(2))
        scramble_depth = int(m.group(4))
    else:
        metadata_ns = sample.get('num_terms_simple', -1)
        scramble_depth = sample.get('num_scrambles', -1)

    if actual_ns != metadata_ns:
        ns_mismatch_count += 1

    # Use actual ns, not metadata
    sample_meta.append({
        'idx': idx, 'ns': actual_ns, 'ns_metadata': metadata_ns,
        'scramble_depth': scramble_depth, 'info': info
    })

print(f"  ns metadata mismatches: {ns_mismatch_count} (using actual target term count)")

# ============================================================================
# Identify overlap and pathological indices
# ============================================================================
# Overlap: test expressions that appear (up to permutation) in any training state
# (starting expressions + all trajectory intermediates = 578k states, 418k unique)
print("Loading overlap indices...", flush=True)
with open(OVERLAP_INDICES_PATH, 'rb') as f:
    overlap_indices = set(pickle.load(f))
print(f"  Overlap indices: {len(overlap_indices)}")

# Pathological: source terms <= target terms (simplification is undefined)
# Identified by counting 'polylog' occurrences in the Dersy eval file strings
print("Parsing Dersy eval file...", flush=True)
dersy_results = {}
with open(DERSY_EVAL_PATH, 'r') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith('Equation '):
        m = re.match(r'Equation (\d+) \((\d+)/(\d+)\)', line)
        if m:
            eq_idx = int(m.group(1))
            n_valid = int(m.group(2))
            # Parse src
            i += 1
            src_line = lines[i].strip()
            src = src_line[4:] if src_line.startswith('src=') else src_line
            # Parse tgt
            i += 1
            tgt_line = lines[i].strip()
            tgt = tgt_line[4:] if tgt_line.startswith('tgt=') else tgt_line
            # Parse hypothesis
            i += 1
            hyp_line = lines[i].strip()
            parts = hyp_line.split(' ', 2)
            valid = int(parts[0])
            dersy_results[eq_idx] = {
                'src': src, 'tgt': tgt, 'valid': valid,
                'src_terms': src.count('polylog') if src != '0' else 0,
                'tgt_terms': tgt.count('polylog') if tgt != '0' else 0,
            }
    i += 1
print(f"  Parsed {len(dersy_results)} equations")

pathological_indices = set()
for idx, r in dersy_results.items():
    if r['src_terms'] <= r['tgt_terms']:
        pathological_indices.add(idx)
print(f"  Pathological indices: {len(pathological_indices)}")

# Report overlap between the two exclusion sets
both = overlap_indices & pathological_indices
excluded = overlap_indices | pathological_indices
print(f"\n  Overlap AND pathological: {len(both)}")
print(f"  Total excluded (union): {len(excluded)}")
if both:
    print(f"  Indices in both: {sorted(both)}")

# Clean set
clean_indices = set(range(5000)) - excluded
print(f"  Clean set: {len(clean_indices)} samples")

# ============================================================================
# Compute per-sample results for both models on the clean set
# ============================================================================
print("\nComputing results on clean set...", flush=True)

# Our model
our_target_rel_failures_clean = OUR_TARGET_REL_FAILURES & clean_indices
our_source_rel_failures_clean = OUR_SOURCE_REL_FAILURES & clean_indices
print(f"  Our target-rel failures (clean): {len(our_target_rel_failures_clean)} -> {sorted(our_target_rel_failures_clean)}")
print(f"  Our source-rel failures (clean): {len(our_source_rel_failures_clean)} -> {sorted(our_source_rel_failures_clean)}")

# Check which of our failures were excluded
our_tgt_excluded = OUR_TARGET_REL_FAILURES & excluded
our_src_excluded = OUR_SOURCE_REL_FAILURES & excluded
if our_tgt_excluded:
    print(f"  NOTE: {len(our_tgt_excluded)} of our target-rel failures were excluded: {sorted(our_tgt_excluded)}")
if our_src_excluded:
    print(f"  NOTE: {len(our_src_excluded)} of our source-rel failures were excluded: {sorted(our_src_excluded)}")

# Dersy model
dersy_success_clean = set()
dersy_failure_clean = set()
for idx in clean_indices:
    if idx in dersy_results and dersy_results[idx]['valid']:
        dersy_success_clean.add(idx)
    else:
        dersy_failure_clean.add(idx)
print(f"  Dersy successes (clean): {len(dersy_success_clean)}")
print(f"  Dersy failures (clean): {len(dersy_failure_clean)}")

# ============================================================================
# Breakdowns by n_s and scramble_depth (FAILURE COUNTS)
# ============================================================================
ns_values = sorted(set(sample_meta[idx]['ns'] for idx in clean_indices))
scr_values = sorted(set(sample_meta[idx]['scramble_depth'] for idx in clean_indices))
n_total = len(clean_indices)

# By n_s
table_by_ns = {}
for ns in ns_values:
    idx_set = set(idx for idx in clean_indices if sample_meta[idx]['ns'] == ns)
    n = len(idx_set)
    our_src_fail = len(our_source_rel_failures_clean & idx_set)
    our_tgt_fail = len(our_target_rel_failures_clean & idx_set)
    dersy_fail = len(dersy_failure_clean & idx_set)
    table_by_ns[ns] = {'n': n, 'our_src_fail': our_src_fail, 'our_tgt_fail': our_tgt_fail, 'dersy_fail': dersy_fail}

# By scramble depth
table_by_scr = {}
for scr in scr_values:
    idx_set = set(idx for idx in clean_indices if sample_meta[idx]['scramble_depth'] == scr)
    n = len(idx_set)
    our_src_fail = len(our_source_rel_failures_clean & idx_set)
    our_tgt_fail = len(our_target_rel_failures_clean & idx_set)
    dersy_fail = len(dersy_failure_clean & idx_set)
    table_by_scr[scr] = {'n': n, 'our_src_fail': our_src_fail, 'our_tgt_fail': our_tgt_fail, 'dersy_fail': dersy_fail}

# Grouped 1-5
idx_1_5 = set(idx for idx in clean_indices if sample_meta[idx]['scramble_depth'] <= 5)
n_1_5 = len(idx_1_5)
grp_1_5 = {
    'n': n_1_5,
    'our_src_fail': len(our_source_rel_failures_clean & idx_1_5),
    'our_tgt_fail': len(our_target_rel_failures_clean & idx_1_5),
    'dersy_fail': len(dersy_failure_clean & idx_1_5),
}

# Overall
overall = {
    'n': n_total,
    'our_src_fail': len(our_source_rel_failures_clean),
    'our_tgt_fail': len(our_target_rel_failures_clean),
    'dersy_fail': len(dersy_failure_clean),
}

# Print failure count table
print("\n" + "=" * 80)
print("FAILURE COUNTS BY n_s (using actual target term count)")
print("=" * 80)
print(f"\n{'ns':>4} {'N':>6} {'Us-Src':>7} {'Us-Tgt':>7} {'Them-Src':>9} {'Them-Tgt':>9}")
print("-" * 50)
for ns in ns_values:
    d = table_by_ns[ns]
    print(f"{ns:>4} {d['n']:>6} {d['our_src_fail']:>7} {d['our_tgt_fail']:>7} {d['dersy_fail']:>9} {d['dersy_fail']:>9}")
d = overall
print(f"{'All':>4} {d['n']:>6} {d['our_src_fail']:>7} {d['our_tgt_fail']:>7} {d['dersy_fail']:>9} {d['dersy_fail']:>9}")

print("\n" + "=" * 80)
print("FAILURE COUNTS BY SCRAMBLE DEPTH")
print("=" * 80)
print(f"\n{'Depth':>6} {'N':>6} {'Us-Src':>7} {'Us-Tgt':>7} {'Them-Src':>9} {'Them-Tgt':>9}")
print("-" * 52)
for scr in scr_values:
    d = table_by_scr[scr]
    print(f"{scr:>6} {d['n']:>6} {d['our_src_fail']:>7} {d['our_tgt_fail']:>7} {d['dersy_fail']:>9} {d['dersy_fail']:>9}")
d = grp_1_5
print(f"{'1-5':>6} {d['n']:>6} {d['our_src_fail']:>7} {d['our_tgt_fail']:>7} {d['dersy_fail']:>9} {d['dersy_fail']:>9}")
d = overall
print(f"{'All':>6} {d['n']:>6} {d['our_src_fail']:>7} {d['our_tgt_fail']:>7} {d['dersy_fail']:>9} {d['dersy_fail']:>9}")

# ============================================================================
# LaTeX table output (failure counts)
# ============================================================================
print("\n" + "=" * 80)
print("LATEX TABLE (failure counts, for paper)")
print("=" * 80)
print()

def fmt(n, nfail):
    pct = 100 * (n - nfail) / n
    if pct > 99.95:
        return f"{nfail} ($>$99.9\\%)"
    else:
        return f"{nfail} ({pct:.1f}\\%)"

print("% --- Table with failure counts ---")
for ns in ns_values:
    d = table_by_ns[ns]
    n = d['n']
    print(f"% n_s={ns}: N={n}, us_src_fail={d['our_src_fail']}, us_tgt_fail={d['our_tgt_fail']}, dersy_fail={d['dersy_fail']}")
for scr in scr_values:
    d = table_by_scr[scr]
    n = d['n']
    print(f"% depth={scr}: N={n}, us_src_fail={d['our_src_fail']}, us_tgt_fail={d['our_tgt_fail']}, dersy_fail={d['dersy_fail']}")
d = grp_1_5
print(f"% depth=1-5: N={d['n']}, us_src_fail={d['our_src_fail']}, us_tgt_fail={d['our_tgt_fail']}, dersy_fail={d['dersy_fail']}")
d = overall
print(f"% Overall: N={d['n']}, us_src_fail={d['our_src_fail']}, us_tgt_fail={d['our_tgt_fail']}, dersy_fail={d['dersy_fail']}")

# ============================================================================
# Figure: Solve rate vs scramble depth
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING FIGURES")
print("=" * 80)

fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

depths = sorted(table_by_scr.keys())
our_src_rates = []
our_tgt_rates = []
dersy_rates = []
for scr in depths:
    d = table_by_scr[scr]
    n = d['n']
    our_src_rates.append(100 * (n - d['our_src_fail']) / n)
    our_tgt_rates.append(100 * (n - d['our_tgt_fail']) / n)
    dersy_rates.append(100 * (n - d['dersy_fail']) / n)

ax.plot(depths, our_tgt_rates, 'o-', color='#1f77b4', label='Ours (target-rel)', markersize=5, linewidth=1.5)
ax.plot(depths, our_src_rates, 's--', color='#1f77b4', label='Ours (source-rel)', markersize=4, linewidth=1.0, alpha=0.7)
ax.plot(depths, dersy_rates, '^-', color='#ff7f0e', label='Ref. [DSZ]', markersize=5, linewidth=1.5)

ax.set_xlabel('Scramble depth')
ax.set_ylabel('Solve rate (%)')
ax.set_xticks(depths)
ax.set_ylim(78, 101)
ax.legend(loc='lower left', fontsize=8)
ax.axvspan(0.5, 7.5, alpha=0.08, color='gray', label='_nolegend_')
ax.text(4, 79.5, 'training range', ha='center', fontsize=7, color='gray')
ax.grid(True, alpha=0.3)

plt.tight_layout()
figpath = os.path.join(FIGURE_DIR, 'dilog_solve_rate_vs_depth.pdf')
plt.savefig(figpath, dpi=150, bbox_inches='tight')
print(f"  Saved: {figpath}")

# Figure 2: Solve rate by n_s
fig2, ax2 = plt.subplots(1, 1, figsize=(4.5, 3.5))

x_pos = np.arange(len(ns_values))
width = 0.25

our_tgt_ns_rates = [100 * (table_by_ns[ns]['n'] - table_by_ns[ns]['our_tgt_fail']) / table_by_ns[ns]['n'] for ns in ns_values]
our_src_ns_rates = [100 * (table_by_ns[ns]['n'] - table_by_ns[ns]['our_src_fail']) / table_by_ns[ns]['n'] for ns in ns_values]
dersy_ns_rates = [100 * (table_by_ns[ns]['n'] - table_by_ns[ns]['dersy_fail']) / table_by_ns[ns]['n'] for ns in ns_values]

bars1 = ax2.bar(x_pos - width, our_tgt_ns_rates, width, label='Ours (target-rel)', color='#1f77b4')
bars2 = ax2.bar(x_pos, our_src_ns_rates, width, label='Ours (source-rel)', color='#1f77b4', alpha=0.5)
bars3 = ax2.bar(x_pos + width, dersy_ns_rates, width, label='Ref. [DSZ]', color='#ff7f0e')

ax2.set_xlabel('Target terms ($n_s$)')
ax2.set_ylabel('Solve rate (%)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([str(ns) for ns in ns_values])
ax2.set_ylim(80, 101)
ax2.legend(fontsize=8)
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
figpath2 = os.path.join(FIGURE_DIR, 'dilog_solve_rate_by_ns.pdf')
plt.savefig(figpath2, dpi=150, bbox_inches='tight')
print(f"  Saved: {figpath2}")

# ============================================================================
# Detailed summary for paper
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED DATA FOR PAPER")
print("=" * 80)

o = overall
print(f"\nFull test set: 5000 samples")
print(f"Excluded: {len(overlap_indices)} overlap + {len(pathological_indices)} pathological "
      f"- {len(both)} in both = {len(excluded)} total excluded")
print(f"Clean test set: {n_total} samples")

print(f"\nOur model:")
print(f"  Source-rel failures: {o['our_src_fail']}/{n_total} ({100*o['our_src_fail']/n_total:.2f}%)")
print(f"  Source-rel solve rate: {100*(n_total - o['our_src_fail'])/n_total:.2f}%")
print(f"  Target-rel failures: {o['our_tgt_fail']}/{n_total} ({100*o['our_tgt_fail']/n_total:.2f}%)")
print(f"  Target-rel solve rate: {100*(n_total - o['our_tgt_fail'])/n_total:.2f}%")

print(f"\nDersy et al. (seq2seq, greedy):")
print(f"  Failures: {o['dersy_fail']}/{n_total} ({100*o['dersy_fail']/n_total:.1f}%)")
print(f"  Solve rate: {100*(n_total - o['dersy_fail'])/n_total:.1f}%")

print(f"\nPer-n_s breakdown (actual target terms):")
print(f"{'ns':>4} {'N':>6} {'Us-Src%':>8} {'Us-Tgt%':>8} {'DSZ%':>8}")
for ns in ns_values:
    d = table_by_ns[ns]
    n = d['n']
    us_src = 100 * (n - d['our_src_fail']) / n
    us_tgt = 100 * (n - d['our_tgt_fail']) / n
    dsz = 100 * (n - d['dersy_fail']) / n
    print(f"{ns:>4} {n:>6} {us_src:>7.1f}% {us_tgt:>7.1f}% {dsz:>7.1f}%")

print(f"\nPer-scramble-depth breakdown:")
print(f"{'Depth':>6} {'N':>6} {'Us-Src%':>8} {'Us-Tgt%':>8} {'DSZ%':>8}")
for scr in depths:
    d = table_by_scr[scr]
    n = d['n']
    us_src = 100 * (n - d['our_src_fail']) / n
    us_tgt = 100 * (n - d['our_tgt_fail']) / n
    dsz = 100 * (n - d['dersy_fail']) / n
    print(f"{scr:>6} {n:>6} {us_src:>7.1f}% {us_tgt:>7.1f}% {dsz:>7.1f}%")

print("\nDONE")
