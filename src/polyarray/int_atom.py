"""The integer-valued IR atom.

An :class:`IntAtom` is an opaque integer-valued symbolic input whose only operation is to
be the scrutinee of a :func:`Interpreter.select_x` typed switch. It stands alongside
:class:`RationalFunction`, the atom that carries real rational polynomials, and takes no
part in rational arithmetic. A selector-driven API uses it so that a symbolic integer
selector value branches the emitted IR over the canonical order of the selector's domain
at run time.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntAtom:
    """An integer-valued IR atom bound by name and confined to a small integer domain.

    The ``name`` is the key the runtime binds the atom's concrete integer value under at
    evaluation time, and the ``domain`` is the range of admissible values — typically
    ``range(n)`` for a selector over ``n`` cases. The dataclass is frozen, so it hashes by
    value and serves as a dictionary key alongside concrete ints.
    """

    name: str
    domain: range


__all__ = ["IntAtom"]
