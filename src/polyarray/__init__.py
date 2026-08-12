"""polyarray — the standalone symbolic-numeric array IR.

polyarray is an imperative array IR in which numeric and symbolic values share one
representation. A :class:`Program` is an ordered list of :class:`Stmt` statements over
:class:`SymArray` values, and a ``SymArray`` is either an array of cells or a single bulk
tensor; a cell is an exact :class:`RationalFunction`, a plain float, or a reference to a
prior result::

    Program   ──▶  Stmt, Stmt, …  ──▶  named outputs
    SymArray  =  array of cells  |  bulk tensor
      cell    =  RationalFunction (exact)  |  float (numeric)  |  Ref (prior output)

The package is the single source of truth for this IR together with its polynomial
backends (sympy, native_py, native_value, cython cpp) and the ``analyze`` and
``partial_eval`` passes over programs.

The IR is executable as-is: build a :class:`Program` and call
``program.run({name: array})`` to get back ``{name: ndarray}``. See ``PUBLIC_API.md`` for
the committed surface.
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
    TransposeOp,
    SinvFullOp,
    GSvdFullOp,
    BlockDiagOp,
    BlockRepeatOp,
    DynBlockRepeatOp,
    DynEyeOp,
    DynZerosOp,
    DynEyeTensorOp,
    ProdShapeOp,
    SumShapeOp,
    SumDimOp,
    ProdDimOp,
    ScaleAxisDimOp,
    MulAxisDimOp,
    CompRankOp,
    HStackOp,
    ColStackOp,
    ScaleOp,
    ScaleByOp,
    AddOp,
    ConcatOp,
    AxisLenOp,
    ReshapeOp,
    ConstOp,
    EyeOp,
    FirstColsOp,
    LastColsOp,
    ProjectOp,
    EmbedOp,
    KronOp,
    KronFreeOp,
    InvTransposeOp,
    ComposeViaStdOp,
    SqrtSpdOp,
    RankOp,
    MetricOrthonormalOp,
)

# --- Rational lane ----------------------------------------------------------
from .rational import RationalFunction, eager_cancel

# --- Integer selectors ------------------------------------------------------
from .int_atom import IntAtom

# --- IR passes --------------------------------------------------------------
from .forward import analyze, partial_eval
from .simplify import specialize, fold_numeric, bind_inputs, partial_eval_numeric, partial_eval_numeric_symarray, substitute, evaluate_cone, dependency_cone, is_structurally_constant, NonExactFoldWarning, NonDeterministicFoldWarning, symarray_atoms
from .budget import SimplifyBudget
from .sparsity import propagate_sparsity, block_zero_mask, SparsityReport
from .degree import program_degree

# --- Linear algebra ---------------------------------------------------------
from .schur import mask_zeros, sound_sparsity_mask, symbolic_inverse

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
    "TransposeOp",
    "SinvFullOp",
    "GSvdFullOp",
    "BlockDiagOp",
    "BlockRepeatOp",
    "DynBlockRepeatOp",
    "DynEyeOp",
    "DynZerosOp",
    "DynEyeTensorOp",
    "ProdShapeOp",
    "SumShapeOp",
    "SumDimOp",
    "ProdDimOp",
    "ScaleAxisDimOp",
    "MulAxisDimOp",
    "CompRankOp",
    "HStackOp",
    "ColStackOp",
    "ScaleOp",
    "ScaleByOp",
    "AddOp",
    "ConcatOp",
    "AxisLenOp",
    "ReshapeOp",
    "ConstOp",
    "EyeOp",
    "FirstColsOp",
    "LastColsOp",
    "ProjectOp",
    "EmbedOp",
    "KronOp",
    "KronFreeOp",
    "InvTransposeOp",
    "ComposeViaStdOp",
    "SqrtSpdOp",
    "RankOp",
    "MetricOrthonormalOp",
    # IR passes
    "analyze",
    "partial_eval",
    # batched execution
    "vmap",
    # simplify workstream
    "specialize",
    "fold_numeric",
    "partial_eval_numeric",
    "partial_eval_numeric_symarray",
    "evaluate_cone",
    "dependency_cone",
    "is_structurally_constant",
    "symarray_atoms",
    "NonExactFoldWarning",
    "NonDeterministicFoldWarning",
    "bind_inputs",
    "substitute",
    "SimplifyBudget",
    "propagate_sparsity",
    "block_zero_mask",
    "SparsityReport",
    # linear algebra
    "mask_zeros",
    "sound_sparsity_mask",
    "symbolic_inverse",
    # code generation
    "to_numpy_source",
]
