"""polyarray — the standalone symbolic-numeric array IR.

Extracted faithfully from chartlib's ``_symbolic`` core (see
``VENDORED.md``).  This package is the single source of truth for the
``Program`` / ``Stmt`` / ``SymArray`` / ``RationalFunction`` IR plus all
polynomial backends (sympy / native_py / native_value / cython cpp) and
the analyze / partial_eval IR passes.

The IR is executable as-is: build a :class:`Program`, then call
``program.run({name: array})`` to get ``{name: ndarray}``.  See
``PUBLIC_API.md`` for the committed surface.
"""
from __future__ import annotations

# --- Construction / values --------------------------------------------------
from .ir import (
    Program,
    SymInput,
    SymbolEnv,
    Provenance,
    SymArray,
    OutSpec,
    DimAtom,
    is_dynamic,
    SymbolicBudget,
    budget_override,
    current_budget_override,
    allocate_input,
)

# --- Op vocabulary (the Stmt fns) -------------------------------------------
from .ir import (
    DetOp,
    InvOp,
    PinvOp,
    SolveOp,
    SvdOp,
    GSvdOp,
    QrOp,
    AssertOp,
    SqrtOp,
    AbsOp,
    SignOp,
    TensordotOp,
    MoveaxisOp,
    EinsumOp,
    EinsumStmtOp,
    SwitchOp,
    CallOp,
    WhileOp,
    IdentityOp,
)

# --- Rational lane ----------------------------------------------------------
from .rational import RationalFunction, eager_cancel

# --- Integer selectors ------------------------------------------------------
from .int_atom import IntAtom

# --- IR passes --------------------------------------------------------------
from .forward import analyze, partial_eval
from .simplify import specialize, fold_numeric, bind_inputs, substitute
from .budget import SimplifyBudget
from .sparsity import propagate_sparsity, block_zero_mask, SparsityReport

# --- Batched execution ------------------------------------------------------
from .ir import vmap

# --- Code generation --------------------------------------------------------
from .numpy_source import to_numpy_source

__all__ = [
    # construction / values
    "Program",
    "SymInput",
    "SymbolEnv",
    "Provenance",
    "SymArray",
    "OutSpec",
    "DimAtom",
    "is_dynamic",
    "SymbolicBudget",
    "budget_override",
    "current_budget_override",
    "allocate_input",
    "RationalFunction",
    "eager_cancel",
    "IntAtom",
    # op vocabulary
    "DetOp",
    "InvOp",
    "PinvOp",
    "SolveOp",
    "SvdOp",
    "GSvdOp",
    "QrOp",
    "AssertOp",
    "SqrtOp",
    "AbsOp",
    "SignOp",
    "TensordotOp",
    "MoveaxisOp",
    "EinsumOp",
    "EinsumStmtOp",
    "SwitchOp",
    "CallOp",
    "WhileOp",
    "IdentityOp",
    # IR passes
    "analyze",
    "partial_eval",
    # batched execution
    "vmap",
    # simplify workstream
    "specialize",
    "fold_numeric",
    "bind_inputs",
    "substitute",
    "SimplifyBudget",
    "propagate_sparsity",
    "block_zero_mask",
    "SparsityReport",
    # code generation
    "to_numpy_source",
]
