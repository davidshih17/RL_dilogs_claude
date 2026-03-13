"""
Utility functions for dilogarithm manipulation and identities.
Based on "Simplifying Polylogarithms with Machine Learning" (arXiv:2206.04115)
"""

import sympy as sp
from sympy import symbols, simplify, expand
from typing import List, Tuple
import numpy as np

x = symbols('x')

class DilogExpression:
    """Represents a linear combination of dilogarithms"""

    def __init__(self, terms: List[Tuple[float, sp.Expr]]):
        """
        Args:
            terms: List of (coefficient, argument) pairs for Li_2(argument)
        """
        self.terms = terms
        self._simplify_terms()

    def _simplify_terms(self):
        """Combine like terms and remove zero coefficients"""
        term_dict = {}
        for coeff, arg in self.terms:
            # Use cancel() instead of simplify() - much faster for rational functions
            # cancel() handles p/q forms efficiently without expensive simplification
            arg_simplified = sp.cancel(arg)
            if arg_simplified in term_dict:
                term_dict[arg_simplified] += coeff
            else:
                term_dict[arg_simplified] = coeff

        # Remove zero coefficients
        self.terms = [(c, a) for a, c in term_dict.items() if abs(float(c)) > 1e-10]

    def num_terms(self) -> int:
        """Return number of distinct dilogarithm terms"""
        return len(self.terms)

    def to_prefix_notation(self) -> List[str]:
        """Convert to prefix notation for neural network input"""
        tokens = []

        if len(self.terms) == 0:
            return ['0']

        # Build sum of terms
        for i, (coeff, arg) in enumerate(self.terms):
            if i > 0:
                tokens.append('add')

            # Add coefficient * polylog term
            if coeff != 1:
                tokens.append('mul')
                tokens.extend(self._number_to_tokens(coeff))

            tokens.append('polylog')
            tokens.append('2')  # Weight 2 (dilogarithm)
            tokens.extend(self._expr_to_tokens(arg))

        return tokens

    def _number_to_tokens(self, num: float) -> List[str]:
        """Convert number to tokens using numeral decomposition"""
        # Handle infinity and NaN
        if np.isinf(num) or np.isnan(num):
            raise ValueError(f"Invalid number in expression: {num}")

        if num < 0:
            return ['-'] + self._number_to_tokens(-num)

        # Integer representation
        num_int = int(num)
        if abs(num - num_int) < 1e-10:
            digits = str(abs(num_int))
            return list(digits)
        else:
            return [str(num)]

    def _expr_to_tokens(self, expr: sp.Expr) -> List[str]:
        """Convert sympy expression to prefix tokens"""
        if expr == x:
            return ['x']
        elif expr.is_number:
            # Handle rational numbers specially
            if isinstance(expr, sp.Rational):
                return self._number_to_tokens(float(expr))
            # Check if it's actually a real number (not complex)
            # Note: is_real can be None (unknown), True, or False
            if hasattr(expr, 'is_real') and expr.is_real is True:
                return self._number_to_tokens(float(expr))
            elif hasattr(expr, 'is_real') and expr.is_real is False:
                # Definitely complex - shouldn't happen with rational functions
                raise ValueError(f"Complex number in expression: {expr}")
            # If sympy can't determine (is_real is None), try to evaluate numerically
            try:
                val = complex(expr)
                if abs(val.imag) < 1e-10:  # Effectively real
                    return self._number_to_tokens(val.real)
                else:
                    # Complex number - this shouldn't happen with rational functions
                    raise ValueError(f"Complex number in expression: {expr}")
            except (TypeError, ValueError):
                # Fall through to structural parsing
                pass
        elif isinstance(expr, sp.Add):
            tokens = []
            args = expr.args
            for i, arg in enumerate(args):
                if i > 0:
                    tokens.append('add')
                tokens.extend(self._expr_to_tokens(arg))
            return tokens
        elif isinstance(expr, sp.Mul):
            tokens = []
            args = expr.args
            for i, arg in enumerate(args):
                if i > 0:
                    tokens.append('mul')
                tokens.extend(self._expr_to_tokens(arg))
            return tokens
        elif isinstance(expr, sp.Pow):
            base, exp = expr.args
            tokens = ['pow']
            tokens.extend(self._expr_to_tokens(base))
            tokens.extend(self._expr_to_tokens(exp))
            return tokens
        else:
            return [str(expr)]

    def copy(self):
        """Create a copy of this expression"""
        return DilogExpression([(c, a) for c, a in self.terms])

    def to_sympy(self) -> sp.Expr:
        """Convert to a sympy expression (sum of polylogs)"""
        if len(self.terms) == 0:
            return sp.Integer(0)

        result = sp.Integer(0)
        for coeff, arg in self.terms:
            # Convert coeff to sympy Rational if possible for cleaner output
            if isinstance(coeff, float) and coeff == int(coeff):
                coeff_sympy = sp.Integer(int(coeff))
            elif isinstance(coeff, float):
                # Try to convert to rational
                coeff_sympy = sp.Rational(coeff).limit_denominator(1000)
            else:
                coeff_sympy = sp.Rational(coeff)
            result = result + coeff_sympy * sp.polylog(2, arg)

        return result

    def __repr__(self):
        if len(self.terms) == 0:
            return "0"

        parts = []
        for coeff, arg in self.terms:
            if coeff == 1:
                parts.append(f"Li_2({arg})")
            elif coeff == -1:
                parts.append(f"-Li_2({arg})")
            else:
                parts.append(f"{coeff}*Li_2({arg})")
        return " + ".join(parts).replace("+ -", "- ")


def apply_reflection(expr: DilogExpression, term_idx: int) -> DilogExpression:
    """Apply reflection identity: Li_2(x) = -Li_2(1-x) + pi^2/6 - ln(x)*ln(1-x)"""
    if term_idx >= len(expr.terms):
        return expr.copy()

    new_terms = []
    for i, (coeff, arg) in enumerate(expr.terms):
        if i == term_idx:
            new_arg = 1 - arg
            new_terms.append((-coeff, sp.cancel(new_arg)))
        else:
            new_terms.append((coeff, arg))

    return DilogExpression(new_terms)


def apply_inversion(expr: DilogExpression, term_idx: int) -> DilogExpression:
    """Apply inversion identity: Li_2(x) = -Li_2(1/x) - pi^2/6 - ln^2(-x)/2"""
    if term_idx >= len(expr.terms):
        return expr.copy()

    new_terms = []
    for i, (coeff, arg) in enumerate(expr.terms):
        if i == term_idx:
            new_arg = 1 / arg
            new_terms.append((-coeff, sp.cancel(new_arg)))
        else:
            new_terms.append((coeff, arg))

    return DilogExpression(new_terms)


def apply_duplication(expr: DilogExpression, term_idx: int) -> DilogExpression:
    """Apply duplication identity: Li_2(x) = -Li_2(-x) + (1/2)*Li_2(x^2)"""
    if term_idx >= len(expr.terms):
        return expr.copy()

    new_terms = []
    for i, (coeff, arg) in enumerate(expr.terms):
        if i == term_idx:
            new_terms.append((-coeff, sp.cancel(-arg)))
            new_terms.append((coeff / 2, sp.cancel(arg**2)))
        else:
            new_terms.append((coeff, arg))

    return DilogExpression(new_terms)


def apply_cyclic_permutation(expr: DilogExpression) -> DilogExpression:
    """Cyclically permute the terms (move first term to end)"""
    if len(expr.terms) <= 1:
        return expr.copy()

    new_terms = expr.terms[1:] + [expr.terms[0]]
    return DilogExpression(new_terms)


def generate_random_rational(max_degree: int = 2, max_coeff: int = 2) -> sp.Expr:
    """Generate a random rational function of x with limited degree and coefficients"""
    num_degree = np.random.randint(0, max_degree + 1)
    num_coeffs = np.random.randint(-max_coeff, max_coeff + 1, num_degree + 1)
    numerator = sum(c * x**i for i, c in enumerate(num_coeffs) if c != 0)

    den_degree = np.random.randint(0, max_degree + 1)
    den_coeffs = np.random.randint(-max_coeff, max_coeff + 1, den_degree + 1)
    if all(c == 0 for c in den_coeffs):
        den_coeffs[0] = 1
    denominator = sum(c * x**i for i, c in enumerate(den_coeffs) if c != 0)

    if numerator == 0:
        numerator = x
    if denominator == 0:
        denominator = 1

    return simplify(numerator / denominator)
