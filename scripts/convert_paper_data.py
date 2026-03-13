"""
Convert paper's train_data.txt and test_data.txt to our pickle format.

Command being run:
cd /home/shih/work/RL_dilogs_claude && PYTHONUNBUFFERED=1 python -u scripts/convert_paper_data.py
"""

import sys
sys.path.insert(0, '/home/shih/work/RL_dilogs_claude/src')

import pickle
import sympy as sp
from sympy import polylog
from dilog_utils import DilogExpression


def parse_paper_data_file(filepath):
    """Parse paper's data file format.

    Format:
    Example N: X scrambles on Y different terms
    Simple expression : 0
    Scrambled expression : <sympy expr>
    <blank line>
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    expressions = []
    i = 0
    while i < len(lines):
        if lines[i].startswith('Example'):
            # Parse info line
            info_line = lines[i].strip()
            parts = info_line.split()
            # "Example 0: 1 scrambles on 2 different terms"
            num_scrambles = int(parts[2])
            num_terms = int(parts[5])

            # Skip "Simple expression : 0"
            i += 1

            # Get scrambled expression
            i += 1
            if i < len(lines) and 'Scrambled expression :' in lines[i]:
                expr_str = lines[i].split(': ', 1)[1].strip()
                expressions.append({
                    'expr_str': expr_str,
                    'num_scrambles': num_scrambles,
                    'num_terms': num_terms
                })
            i += 1
            # Skip blank line
            if i < len(lines) and lines[i].strip() == '':
                i += 1
        else:
            i += 1

    return expressions


def parse_term(term):
    """Parse a single term to extract coefficient and polylog argument."""
    if term.func == sp.Mul:
        coeff = 1
        polylog_arg = None
        for factor in term.args:
            if factor.func == polylog:
                polylog_arg = factor.args[1]
            elif factor.is_number:
                try:
                    val = complex(factor)
                    if val.imag != 0:
                        return None, None  # Skip complex coefficients
                    coeff = coeff * val.real
                except:
                    coeff = coeff * factor
            else:
                coeff = coeff * factor

        if polylog_arg is None:
            return None, None

        try:
            coeff_val = float(coeff)
            return coeff_val, polylog_arg
        except (TypeError, ValueError):
            return None, None

    elif term.func == polylog:
        return 1.0, term.args[1]

    return None, None


def sympy_to_dilog_expr(sympy_expr):
    """Convert sympy polylog expression to DilogExpression."""
    terms = []

    if sympy_expr.func == polylog:
        terms.append((1.0, sympy_expr.args[1]))
        return DilogExpression(terms)

    if sympy_expr.func == sp.Add:
        for arg in sympy_expr.args:
            coeff, polylog_arg = parse_term(arg)
            if coeff is not None and polylog_arg is not None:
                terms.append((coeff, polylog_arg))
    else:
        coeff, polylog_arg = parse_term(sympy_expr)
        if coeff is not None and polylog_arg is not None:
            terms.append((coeff, polylog_arg))

    return DilogExpression(terms)


def convert_paper_data_to_pickle(input_path, output_path):
    """Convert paper's data file to our pickle format."""
    print(f"Reading {input_path}...")
    expressions = parse_paper_data_file(input_path)
    print(f"  Parsed {len(expressions)} expressions")

    dataset = []
    skipped = 0
    skipped_reasons = {'empty': 0, 'constant_polylogs': 0, 'error': 0}

    for i, expr_data in enumerate(expressions):
        if (i + 1) % 1000 == 0:
            print(f"  Processing {i + 1}/{len(expressions)}...")

        try:
            # Count polylogs in original string (before sympy evaluation)
            original_polylog_count = expr_data['expr_str'].count('polylog(')

            # Parse sympy expression
            sympy_expr = sp.sympify(expr_data['expr_str'])

            # Convert to DilogExpression
            dilog_expr = sympy_to_dilog_expr(sympy_expr)
            converted_count = dilog_expr.num_terms()

            if converted_count == 0:
                skipped += 1
                skipped_reasons['empty'] += 1
            elif converted_count < original_polylog_count:
                # Some polylogs were lost to auto-evaluation (constant polylogs)
                skipped += 1
                skipped_reasons['constant_polylogs'] += 1
            else:
                dataset.append({
                    'expression': dilog_expr,
                    'num_scrambles': expr_data['num_scrambles'],
                    'num_terms': converted_count
                })
        except Exception as e:
            skipped += 1
            skipped_reasons['error'] += 1
            if skipped_reasons['error'] <= 5:
                print(f"  Error on expr {i}: {e}")

    print(f"\nConversion summary:")
    print(f"  Converted: {len(dataset)}")
    print(f"  Skipped: {skipped}")
    print(f"    - Empty: {skipped_reasons['empty']}")
    print(f"    - Constant polylogs: {skipped_reasons['constant_polylogs']}")
    print(f"    - Errors: {skipped_reasons['error']}")

    # Save to pickle
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    print("Done!")

    return dataset


if __name__ == '__main__':
    print("=" * 60)
    print("Converting paper's data to our pickle format")
    print("=" * 60)

    # Convert training data
    print("\n--- Converting training data ---")
    train_dataset = convert_paper_data_to_pickle(
        '/home/shih/work/ML_Polylogarithms_original/data/train_data.txt',
        '/home/shih/work/RL_dilogs_claude/data/paper_train_set.pkl'
    )

    # Convert test data
    print("\n--- Converting test data ---")
    test_dataset = convert_paper_data_to_pickle(
        '/home/shih/work/ML_Polylogarithms_original/data/test_data.txt',
        '/home/shih/work/RL_dilogs_claude/data/paper_test_set.pkl'
    )

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Training set: {len(train_dataset)} samples -> data/paper_train_set.pkl")
    print(f"Test set: {len(test_dataset)} samples -> data/paper_test_set.pkl")
    print("=" * 60)
