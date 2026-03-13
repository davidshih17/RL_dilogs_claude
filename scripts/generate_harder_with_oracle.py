#!/usr/bin/env python
"""
Generate HARDER test data (non-zero targets) WITH oracle trajectories.

This combines the functionality of:
1. generate_transformer_starts from utils_env.py - generates expressions with ns>0 targets
2. generate_with_oracle.py - records states during scrambling and finds reverse actions

Strategy:
1. Generate num_terms_simple base polylog terms (the simple target)
2. Scramble each base term while recording intermediate states
3. Add zero-target scrambled terms (num_zero additions)
4. Track all states so we can recover oracle trajectories

Each sample will contain:
- expression: the final scrambled DilogExpression
- target_expression: the simplified DilogExpression (num_terms_simple terms)
- num_scrambles: total number of scrambles applied
- num_terms_simple: number of terms in simplified form (ns)
- trajectory: list of (state, action_idx, term_idx) for solving
"""

import argparse
import os
import sys
import pickle
import random
from fractions import Fraction

import sympy as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(SCRIPT_DIR), 'src'))
from dilog_utils import DilogExpression, apply_reflection, apply_inversion, apply_duplication


ACTION_NAMES = ['reflection', 'inversion', 'duplication']
ACTION_COEFF_MAP = {'inversion': -1, 'reflection': -1, 'duplication': 1}


def get_inv_arg_poly(polylog_expr):
    polylog_arg = polylog_expr.args[-1]
    return polylog_expr.replace(polylog_arg, sp.cancel(1 / polylog_arg))


def get_refl_arg_poly(polylog_expr):
    polylog_arg = polylog_expr.args[-1]
    return polylog_expr.replace(polylog_arg, sp.cancel(1 - polylog_arg))


def get_dupli_arg_poly(polylog_expr):
    polylog_arg = polylog_expr.args[-1]
    new_expr = sp.Rational(1, 2) * polylog_expr.replace(polylog_arg, sp.cancel(polylog_arg * polylog_arg))
    new_expr = new_expr - polylog_expr.replace(polylog_arg, sp.cancel(-polylog_arg))
    return new_expr


def act_arg_poly(polylog_expr, action_name):
    if action_name == 'inversion':
        return get_inv_arg_poly(polylog_expr)
    elif action_name == 'reflection':
        return get_refl_arg_poly(polylog_expr)
    elif action_name == 'duplication':
        return get_dupli_arg_poly(polylog_expr)


def act_arg(arg, action_name):
    if action_name == 'inversion':
        return sp.cancel(1 / arg)
    elif action_name == 'reflection':
        return sp.cancel(1 - arg)
    elif action_name == 'duplication':
        return sp.cancel(arg * arg)


def get_polylog_terms_with_coeffs(expr):
    """Extract all (coeff, polylog_term) pairs from expression."""
    terms = []
    if expr == 0:
        return terms
    if isinstance(expr, sp.Add):
        for arg in expr.args:
            if isinstance(arg, sp.polylog):
                terms.append((sp.Integer(1), arg))
            elif isinstance(arg, sp.Mul):
                coeff = sp.Integer(1)
                polylog = None
                for subarg in arg.args:
                    if isinstance(subarg, sp.polylog):
                        polylog = subarg
                    else:
                        coeff = coeff * subarg
                if polylog:
                    terms.append((coeff, polylog))
    elif isinstance(expr, sp.Mul):
        coeff = sp.Integer(1)
        polylog = None
        for arg in expr.args:
            if isinstance(arg, sp.polylog):
                polylog = arg
            else:
                coeff = coeff * arg
        if polylog:
            terms.append((coeff, polylog))
    elif isinstance(expr, sp.polylog):
        terms.append((sp.Integer(1), expr))
    return terms


def generate_random_argument(max_degree, max_coeff, variable):
    deg_num = random.randint(0, max_degree)
    deg_denom = random.randint(0, max_degree)
    coeffs_num = [random.randint(-max_coeff, max_coeff) for _ in range(deg_num + 1)]
    coeffs_denom = [random.randint(-max_coeff, max_coeff) for _ in range(deg_denom + 1)]
    if all(c == 0 for c in coeffs_num):
        coeffs_num[0] = random.choice([-1, 1])
    if all(c == 0 for c in coeffs_denom):
        coeffs_denom[0] = random.choice([-1, 1])
    num = sum(c * variable**i for i, c in enumerate(coeffs_num))
    denom = sum(c * variable**i for i, c in enumerate(coeffs_denom))
    return sp.cancel(num / denom)


def partition(n, k):
    """Partition n into k parts, each >= 1."""
    if k == 1:
        return [[n]]
    if n < k:
        return []
    result = []
    for i in range(1, n - k + 2):
        for p in partition(n - i, k - 1):
            result.append([i] + p)
    return result


def partition_with_zeros(n, k):
    """Partition n into k parts, allowing zeros."""
    if k == 1:
        return [[n]]
    if n == 0:
        return [[0] * k]
    result = []
    for i in range(0, n + 1):
        for p in partition_with_zeros(n - i, k - 1):
            result.append([i] + p)
    return result


def sympy_to_dilog_expression(sympy_expr):
    """Convert sympy expression to DilogExpression."""
    terms = []
    if sympy_expr == 0:
        return DilogExpression([])

    sympy_expr = sp.expand(sympy_expr)

    if isinstance(sympy_expr, sp.polylog):
        terms.append((Fraction(1), sympy_expr.args[1]))
        return DilogExpression(terms)

    if isinstance(sympy_expr, sp.Mul):
        coeff = 1
        polylog_term = None
        for arg in sympy_expr.args:
            if isinstance(arg, sp.polylog):
                polylog_term = arg
            else:
                coeff *= arg
        if polylog_term is not None:
            try:
                coeff_rat = sp.Rational(coeff)
                coeff_frac = Fraction(int(coeff_rat.p), int(coeff_rat.q))
            except:
                coeff_frac = Fraction(float(coeff))
            terms.append((coeff_frac, polylog_term.args[1]))
        return DilogExpression(terms)

    if isinstance(sympy_expr, sp.Add):
        for term in sympy_expr.args:
            if isinstance(term, sp.polylog):
                terms.append((Fraction(1), term.args[1]))
            elif isinstance(term, sp.Mul):
                coeff = 1
                polylog_term = None
                for arg in term.args:
                    if isinstance(arg, sp.polylog):
                        polylog_term = arg
                    else:
                        coeff *= arg
                if polylog_term is not None:
                    try:
                        coeff_rat = sp.Rational(coeff)
                        coeff_frac = Fraction(int(coeff_rat.p), int(coeff_rat.q))
                    except:
                        coeff_frac = Fraction(float(coeff))
                    terms.append((coeff_frac, polylog_term.args[1]))

    return DilogExpression(terms)


def expressions_equal(expr1: DilogExpression, expr2: DilogExpression) -> bool:
    """Check if two DilogExpressions are equivalent."""
    if expr1.num_terms() != expr2.num_terms():
        return False
    if expr1.num_terms() == 0 and expr2.num_terms() == 0:
        return True

    # Sort terms by string representation for comparison
    terms1 = sorted([(float(c), str(sp.cancel(a))) for c, a in expr1.terms])
    terms2 = sorted([(float(c), str(sp.cancel(a))) for c, a in expr2.terms])

    for (c1, a1), (c2, a2) in zip(terms1, terms2):
        if abs(c1 - c2) > 1e-9:
            return False
        if a1 != a2:
            return False
    return True


def find_action_between_states(state_from: DilogExpression, state_to: DilogExpression) -> tuple:
    """
    Brute force search to find which action transforms state_from -> state_to.

    Returns (action_idx, term_idx) or None if not found.
    action_idx: 0=reflection, 1=inversion, 2=duplication
    """
    n_terms = state_from.num_terms()

    for term_idx in range(n_terms):
        # Try reflection
        try:
            result = apply_reflection(state_from, term_idx)
            if expressions_equal(result, state_to):
                return (0, term_idx)  # reflection
        except:
            pass

        # Try inversion
        try:
            result = apply_inversion(state_from, term_idx)
            if expressions_equal(result, state_to):
                return (1, term_idx)  # inversion
        except:
            pass

        # Try duplication
        try:
            result = apply_duplication(state_from, term_idx)
            if expressions_equal(result, state_to):
                return (2, term_idx)  # duplication
        except:
            pass

    return None


def generate_harder_sample_with_oracle(num_terms_simple, num_zero, max_scr, max_degree, max_coeff, variable, action_list):
    """
    Generate a sample with non-zero target (ns = num_terms_simple) with oracle trajectory.

    For ns=0 (num_terms_simple=0), this is equivalent to the zero-target case.
    For ns>0, we generate base terms and scramble them.

    Strategy:
    1. Generate num_terms_simple base polylog terms (these form the simple target)
    2. Scramble these terms while recording states
    3. Add num_zero zero-target scrambled pairs
    4. Track all states to recover oracle trajectory
    """
    if num_terms_simple == 0:
        # Fall back to zero-target generation (similar to generate_with_oracle.py)
        return generate_zero_target_with_oracle(num_zero, max_scr, max_degree, max_coeff, variable, action_list)

    # Generate the simple (target) arguments
    simple_args = [generate_random_argument(max_degree, max_coeff, variable) for _ in range(num_terms_simple)]
    simple_coeffs = [random.randint(1, max_coeff * 4) * random.choice([-1, 1]) for _ in range(num_terms_simple)]

    # Distribute scrambles between non-zero terms and zero additions
    if num_zero > 0:
        zero_scr = random.randint(num_zero, max_scr) if max_scr >= num_zero else num_zero
    else:
        zero_scr = 0
    non_zero_scr = max(1, max_scr - zero_scr)  # At least 1 scramble for non-zero terms

    # Partition non-zero scrambles among the simple terms
    if num_terms_simple > 1:
        parts_list = partition_with_zeros(non_zero_scr, num_terms_simple)
        parts_list = [p for p in parts_list if sum(p) == non_zero_scr]
        if not parts_list:
            scr_per_term = [non_zero_scr // num_terms_simple] * num_terms_simple
        else:
            scr_per_term = random.choice(parts_list)
    else:
        scr_per_term = [non_zero_scr]

    # Build the simple target expression
    simple_expr = sp.Integer(0)
    for arg, coeff in zip(simple_args, simple_coeffs):
        simple_expr += coeff * sp.polylog(2, arg)

    # Now scramble each simple term while recording states
    # States will be recorded as we build up the expression
    states = []
    current_expr = sp.Integer(0)

    # Track scrambled versions of each simple term
    scrambled_terms = []

    for i, (arg, coeff, num_scr_i) in enumerate(zip(simple_args, simple_coeffs, scr_per_term)):
        # Start with the original polylog
        term = sp.polylog(2, arg)
        coeff_mul = 1

        past_action = None
        past_arg = None

        for scr_step in range(num_scr_i):
            # Get polylog terms in current term
            term_polylogs = get_polylog_terms_with_coeffs(term)
            if not term_polylogs:
                break

            # Pick a random polylog to act on
            local_idx = random.randint(0, len(term_polylogs) - 1)
            _, polylog_term = term_polylogs[local_idx]
            term_arg = polylog_term.args[-1]

            # Pick action (avoiding immediate reversal)
            valid = False
            for _ in range(100):
                action = random.choice(action_list)
                if past_action is None or action != past_action:
                    valid = True
                    break
                try:
                    if past_arg is not None and sp.simplify(term_arg - past_arg) != 0:
                        valid = True
                        break
                except:
                    valid = True
                    break
            if not valid:
                continue

            # Apply scramble
            new_polylog = act_arg_poly(polylog_term, action)
            term = term.replace(polylog_term, new_polylog * ACTION_COEFF_MAP[action])
            term = sp.expand(term)

            # Update full expression and record state
            test_expr = current_expr + coeff * coeff_mul * term
            for j, (a, c, s_t) in enumerate(scrambled_terms):
                test_expr += c * s_t
            test_expr = sp.expand(test_expr)

            if test_expr != 0:
                dilog_state = sympy_to_dilog_expression(test_expr)
                if dilog_state.num_terms() > 0:
                    states.append(dilog_state)

            past_action = action
            past_arg = act_arg(term_arg, action)

        scrambled_terms.append((arg, coeff, term))

    # Build the scrambled expression from non-zero terms
    scrambled_expr = sp.Integer(0)
    for arg, coeff, term in scrambled_terms:
        scrambled_expr += coeff * term
    scrambled_expr = sp.expand(scrambled_expr)

    # Now add zero-target terms if needed
    if num_zero > 0 and zero_scr >= num_zero:
        zero_expr, zero_states = generate_zero_terms_with_states(
            num_zero, zero_scr, max_degree, max_coeff, variable, action_list, scrambled_expr
        )
        scrambled_expr += zero_expr
        scrambled_expr = sp.expand(scrambled_expr)
        states.extend(zero_states)

    if scrambled_expr == 0:
        return None

    if len(states) < 1:
        return None

    # Final expressions
    final_expr = sympy_to_dilog_expression(scrambled_expr)
    target_expr = sympy_to_dilog_expression(simple_expr)

    if final_expr.num_terms() == 0:
        return None

    # Build oracle trajectory by finding actions between consecutive states
    # States go from initial -> ... -> final_expr
    # For solving, we go backwards: final_expr -> ... -> target_expr

    trajectory = []
    reversed_states = list(reversed(states))

    for i in range(len(reversed_states)):
        current_state = reversed_states[i]

        if i < len(reversed_states) - 1:
            next_state = reversed_states[i + 1]
        else:
            next_state = target_expr

        # Find action that transforms current_state -> next_state
        action_info = find_action_between_states(current_state, next_state)

        if action_info is None:
            # Could not find action - skip this sample
            return None

        action_idx, term_idx = action_info
        trajectory.append({
            'state': current_state,
            'action_idx': action_idx,
            'term_idx': term_idx,
        })

    return {
        'expression': final_expr,
        'target_expression': target_expr,
        'num_scrambles': max_scr,
        'num_terms_simple': num_terms_simple,
        'num_zeros': num_zero,
        'trajectory': trajectory,
    }


def generate_zero_terms_with_states(num_zero, zero_scr, max_degree, max_coeff, variable, action_list, base_expr):
    """Generate zero-target terms and return states during scrambling."""
    if zero_scr < num_zero:
        num_zero = zero_scr

    parts = partition(zero_scr, num_zero)
    if not parts:
        return sp.Integer(0), []
    part = random.choice(parts)

    states = []
    zero_expr = sp.Integer(0)

    for i in range(num_zero):
        random_arg = generate_random_argument(max_degree, max_coeff, variable)
        random_coeff = random.randint(1, max_coeff * 4)

        # term1 gets scrambled, term2 stays as is (so term1 - term2 = 0 when unscrambled)
        term1 = sp.polylog(2, random_arg)
        term2 = sp.polylog(2, random_arg)

        past_action = None
        past_arg = None

        for scr_step in range(part[i]):
            term_polylogs = get_polylog_terms_with_coeffs(term1)
            if not term_polylogs:
                break

            local_idx = random.randint(0, len(term_polylogs) - 1)
            _, polylog_term = term_polylogs[local_idx]
            term_arg = polylog_term.args[-1]

            valid = False
            for _ in range(100):
                action = random.choice(action_list)
                if past_action is None or action != past_action:
                    valid = True
                    break
                try:
                    if past_arg is not None and sp.simplify(term_arg - past_arg) != 0:
                        valid = True
                        break
                except:
                    valid = True
                    break
            if not valid:
                continue

            new_polylog = act_arg_poly(polylog_term, action)
            term1 = term1.replace(polylog_term, new_polylog * ACTION_COEFF_MAP[action])
            term1 = sp.expand(term1)

            # Record state
            test_expr = base_expr + zero_expr + random_coeff * term1 - random_coeff * term2
            test_expr = sp.expand(test_expr)

            if test_expr != 0:
                dilog_state = sympy_to_dilog_expression(test_expr)
                if dilog_state.num_terms() > 0:
                    states.append(dilog_state)

            past_action = action
            past_arg = act_arg(term_arg, action)

        zero_expr += random_coeff * term1 - random_coeff * term2

    return sp.expand(zero_expr), states


def generate_zero_target_with_oracle(num_add_zero, max_scr, max_degree, max_coeff, variable, action_list):
    """Generate zero-target sample with oracle (similar to generate_with_oracle.py)."""
    if max_scr < num_add_zero:
        num_add_zero = max_scr

    parts = partition(max_scr, num_add_zero)
    if not parts:
        return None
    part = random.choice(parts)

    pairs = []
    for i in range(num_add_zero):
        random_arg = generate_random_argument(max_degree, max_coeff, variable)
        random_coeff = random.randint(1, max_coeff * 4)
        pairs.append({
            'arg': random_arg,
            'coeff': random_coeff,
            'num_scr': part[i],
            'scrambled_term': sp.polylog(2, random_arg),
        })

    states = []
    scramble_order = []
    for i, p in enumerate(pairs):
        scramble_order.extend([i] * p['num_scr'])
    random.shuffle(scramble_order)

    past_action = None
    past_arg = None

    for pair_idx in scramble_order:
        pair = pairs[pair_idx]
        pair_terms = get_polylog_terms_with_coeffs(pair['scrambled_term'])
        if not pair_terms:
            continue

        local_idx = random.randint(0, len(pair_terms) - 1)
        _, polylog_term = pair_terms[local_idx]
        term_arg = polylog_term.args[-1]

        valid = False
        for _ in range(100):
            action = random.choice(action_list)
            if past_action is None or action != past_action:
                valid = True
                break
            try:
                if past_arg is not None and sp.simplify(term_arg - past_arg) != 0:
                    valid = True
                    break
            except:
                valid = True
                break
        if not valid:
            continue

        new_polylog = act_arg_poly(polylog_term, action)
        pair['scrambled_term'] = pair['scrambled_term'].replace(
            polylog_term, new_polylog * ACTION_COEFF_MAP[action]
        )
        pair['scrambled_term'] = sp.expand(pair['scrambled_term'])

        after_expr = sp.Integer(0)
        for p in pairs:
            after_expr += p['scrambled_term'] * p['coeff'] - sp.polylog(2, p['arg']) * p['coeff']
        after_expr = sp.expand(after_expr)

        if after_expr == 0:
            continue

        dilog_expr = sympy_to_dilog_expression(after_expr)
        if dilog_expr.num_terms() > 0:
            states.append(dilog_expr)

        past_action = action
        past_arg = act_arg(term_arg, action)

    if len(states) < 1:
        return None

    final_expr = states[-1]
    zero_state = DilogExpression([])

    trajectory = []
    reversed_states = list(reversed(states))

    for i in range(len(reversed_states)):
        current_state = reversed_states[i]

        if i < len(reversed_states) - 1:
            next_state = reversed_states[i + 1]
        else:
            next_state = zero_state

        action_info = find_action_between_states(current_state, next_state)

        if action_info is None:
            return None

        action_idx, term_idx = action_info
        trajectory.append({
            'state': current_state,
            'action_idx': action_idx,
            'term_idx': term_idx,
        })

    return {
        'expression': final_expr,
        'target_expression': zero_state,
        'num_scrambles': max_scr,
        'num_terms_simple': 0,
        'num_zeros': num_add_zero,
        'trajectory': trajectory,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate harder test data with oracle trajectories')
    parser.add_argument('--num_terms_simple', type=int, required=True,
                        help='Number of terms in simplified form (ns). 0 = simplifies to zero.')
    parser.add_argument('--num_zeros', type=int, required=True,
                        help='Number of zero additions')
    parser.add_argument('--max_scr', type=int, required=True,
                        help='Maximum total scrambles')
    parser.add_argument('--num_samples', type=int, default=250,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str,
                        default='data/harder_oracle/')
    parser.add_argument('--max_degree', type=int, default=2)
    parser.add_argument('--max_coeff', type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    x = sp.Symbol('x')
    action_list = ['inversion', 'reflection', 'duplication']

    print(f'Generating {args.num_samples} samples:')
    print(f'  num_terms_simple (ns): {args.num_terms_simple}')
    print(f'  num_zeros: {args.num_zeros}')
    print(f'  max_scr: {args.max_scr}')
    print()

    samples = []
    attempts = 0
    max_attempts = args.num_samples * 50

    while len(samples) < args.num_samples and attempts < max_attempts:
        attempts += 1

        result = generate_harder_sample_with_oracle(
            args.num_terms_simple, args.num_zeros, args.max_scr,
            args.max_degree, args.max_coeff, x, action_list
        )

        if result is None:
            continue

        if len(result['trajectory']) < 1:
            continue

        samples.append(result)

        if len(samples) % 50 == 0:
            print(f"  Generated {len(samples)}/{args.num_samples}", flush=True)

    output_path = os.path.join(
        args.output_dir,
        f'samples_ns{args.num_terms_simple}_z{args.num_zeros}_scr{args.max_scr}.pkl'
    )
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)

    print(f"\nSaved {len(samples)} samples to {output_path}")

    if samples:
        avg_traj = sum(len(s['trajectory']) for s in samples) / len(samples)
        avg_terms = sum(s['expression'].num_terms() for s in samples) / len(samples)
        avg_target_terms = sum(s['target_expression'].num_terms() for s in samples) / len(samples)
        print(f"Average trajectory length: {avg_traj:.1f}")
        print(f"Average num terms (scrambled): {avg_terms:.1f}")
        print(f"Average num terms (target): {avg_target_terms:.1f}")


if __name__ == '__main__':
    main()
