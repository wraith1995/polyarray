"""Integer-valued IR atom.

:class:`IntAtom` is an opaque integer-valued symbolic input — distinct
from :class:`RationalFunction`, which carries real rational polynomials.
Its only operation is to be the scrutinee of a :func:`Interpreter.select_x`
typed switch.

Used by a selector-driven API so that a symbolic integer selector value
can branch the emitted IR over the canonical order of the selector's domain
at run time, without participating in any rational arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntAtom:
    """Integer-valued IR atom.

    ``name`` is the binding key the runtime uses to look up the atom's
    concrete integer value at evaluation time.  ``domain`` is the range
    of admissible values — typically ``range(n)`` for an integer selector
    over ``n`` cases.  The dataclass is frozen so it hashes by
    value, making it usable as a dictionary key alongside concrete ints.
    """

    name: str
    domain: range


__all__ = ["IntAtom"]
