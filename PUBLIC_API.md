# polyarray — public API

Committed surface, faithful to chartlib's `_symbolic` exports at the
extraction commit (see `VENDORED.md`). No names were added beyond what
chartlib already exposed; signatures are reproduced verbatim from the
extracted source.

Everything below is importable from the top-level package:

```python
from polyarray import Program, SymArray, RationalFunction, ...
```

## Construction

```python
class Program:
    def __init__(self, name: str = "anon",
                 inputs: Sequence[SymInput] = (),
                 env: SymbolEnv | None = None,
                 budget: SymbolicBudget | None = None) -> None
    def input(self, name: str) -> SymArray
    def add_output(self, name: str, cells: Any) -> SymArray
    def add_stmt(self, stmt: Stmt) -> int
    def emit_stmt(self, fn: Callable[..., Any] | Program,
                  in_refs: Sequence[Ref | SymArray],
                  out_specs: Sequence[OutSpec],
                  note: str = "", bulk: bool = True) -> tuple[SymArray, ...]
    def declare_int_atom(self, name: str, domain: range) -> IntAtom
    def run(self, values: Mapping[str, Any]) -> dict[str, np.ndarray]
    def build_runtime_bindings(self, values: Mapping[str, Any]) -> dict[str, float]
    def copy(self) -> Program

@dataclass(frozen=True)
class SymInput:
    name: str
    # A shape entry may be a runtime DimAtom (a dynamic input axis, e.g. an
    # FFS-typed Grassmann input a : Λᵏ). A dynamic SymInput is allocated as a
    # single bulk SymArray (no per-cell atoms); its DimAtom axes are bound from
    # the provided array's shape at Program.run time. Static inputs unchanged.
    shape: tuple[int | DimAtom, ...]
    provenance: Provenance | Callable[[tuple[int, ...]], Provenance]

@dataclass(frozen=True)
class Provenance:
    kind: ProvenanceKind        # "vertex" | "param_dof" | "stmt_out" | ...
    origin: Any
    index: tuple[int, ...]
    label: str

@dataclass(frozen=True)
class OutSpec:
    name: str
    # A shape entry may be a runtime DimAtom (dynamic shape); see below.
    shape: tuple[int | DimAtom, ...] = ()

@dataclass(frozen=True)
class DimAtom:                   # a runtime array dimension (e.g. SVD rank δ_f)
    name: str                    # provenance, e.g. "rank:Alt"
    # A tagged, hashable tuple identifying the run-time origin (the
    # dim_bindings key):
    #   ("stmt", stmt_idx, out_idx)  — a prior Stmt output (Stage B)
    #   ("in",   input_name, axis)   — a dynamic SymInput axis (Stage C)
    # Compat: a bare (stmt_idx, out_idx) 2-tuple is normalised to the
    # ("stmt", ...) form. Build input-axis atoms via DimAtom.from_input.
    source: tuple[Any, ...]
    @staticmethod
    def from_input(name: str, input_name: str, axis: int) -> DimAtom: ...

def is_dynamic(shape: tuple[int | DimAtom, ...]) -> bool
    # True iff any shape entry is a DimAtom (a runtime dimension).  An
    # OutSpec/output with a dynamic shape is always emitted bulk, its axis
    # sizes resolved at Program.run time.  Fully-concrete shapes are not
    # dynamic, so static-shape programs are byte-identical.

class SymbolEnv:                 # owned by Program; shared across cells
    ...

def allocate_input(env: SymbolEnv, spec: SymInput) -> np.ndarray
```

## Values

```python
class SymArray:                  # object/float ndarray of cells, bound to a Program
    def matmul(self, other: SymArray | np.ndarray) -> SymArray
    def matvec(self, v: SymArray | np.ndarray) -> SymArray
    def transpose(self) -> SymArray
    def det(self, budget: SymbolicBudget | None = None) -> SymArray
    def inverse(self, budget: SymbolicBudget | None = None) -> SymArray
    def pinv(self) -> SymArray
    def solve(self, b: SymArray | np.ndarray) -> SymArray
    def sqrt(self) -> SymArray
    def abs(self) -> SymArray
    def sign(self) -> SymArray
    def __getitem__(self, idx) -> Any            # slicing
    def evaluate(self, bindings: Mapping[str, np.ndarray | float]) -> np.ndarray

class RationalFunction:          # num / den over a poly ring (the rational lane)
    @classmethod
    def atom(cls, name: str) -> RationalFunction
    # + ring arithmetic (+, -, *, /), clean()/try_cancel, structural-zero preds
```

## Op vocabulary (the `Stmt.fn` types)

All frozen / hashable dataclasses.

```python
@dataclass(frozen=True) class DetOp                  # -> np.linalg.det
@dataclass(frozen=True) class InvOp                  # -> np.linalg.inv
@dataclass(frozen=True) class PinvOp                 # -> np.linalg.pinv
@dataclass(frozen=True) class SolveOp                # -> np.linalg.solve

@dataclass(frozen=True)
class SvdOp:                     # -> (U, S, Vh, rank); 4-output
    rcond: float | None = None   # rank threshold (np.linalg.matrix_rank rule)
    full_matrices: bool = False  # rank is the thresholded numerical rank (δ_f)

@dataclass(frozen=True)
class GSvdOp:                    # metric-aware GSVD of A:V->W; -> (U, UI, V, VI, S, rank); 6-output
    rcond: float | None = None   # rank threshold on the *whitened* singular values
    # __call__(A, M_V, M_W) with SPD domain/codomain metrics.
    #   U  = image basis in W  (Uᵀ M_W U = I);  UI = coker basis (M_W-⊥ complement)
    #   V  = coimg basis in V  (Vᵀ M_V V = I);  VI = ker  basis  (A·VI ≈ 0)
    #   full factors are [U|UI], [V|VI]; reconstruction:
    #       A = [U|UI] · Sfull · [V|VI]ᵀ · M_V    (trailing M_V; contravariant coords)
    #   M_V = M_W = I  ⇒  reduces to SvdOp (A = U diag(S) Vᵀ, same S/rank).

@dataclass(frozen=True)
class QrOp:                      # -> (Q, R); 2-output
    mode: str = "reduced"

@dataclass(frozen=True)
class AssertOp:                  # passthrough predicate check; returns first input
    kind: str                    # "shape_eq" | "rank_eq" | "spd" | "square_full_rank"
    msg: str = ""                # AssertionError(msg + detail) on failure
@dataclass(frozen=True) class SqrtOp                 # per-cell sqrt
@dataclass(frozen=True) class AbsOp                  # per-cell abs
@dataclass(frozen=True) class SignOp                 # per-cell sign

@dataclass(frozen=True)
class TensordotOp:
    # construct via TensordotOp.from_axes(axes) or TensordotOp((a, b))
    @classmethod def from_axes(cls, axes: Any) -> TensordotOp

@dataclass(frozen=True)
class MoveaxisOp:
    @classmethod def from_spec(cls, source: Any, destination: Any) -> MoveaxisOp

@dataclass(frozen=True)
class EinsumOp:                  # one symbolic + one captured-numeric operand
    spec: str; rhs_shape: tuple; rhs_dtype: str; rhs_bytes: bytes

@dataclass(frozen=True)
class EinsumStmtOp:              # >= 2 symbolic operands
    spec: str
    optimize: bool = True

@dataclass(frozen=True)
class SwitchOp:                  # pick branch by integer scrutinee (IntAtom)
    n_branches: int

@dataclass(frozen=True)
class CallOp:                    # opaque callable or sub-Program
    fn: Callable[..., Any] | Program

@dataclass(frozen=True)
class WhileOp:
    cond: Callable[..., Any] | Program
    body: Callable[..., Any] | Program
    max_iters: int = 100000

@dataclass(frozen=True) class IdentityOp             # capture heavy intermediate as atom

@dataclass(frozen=True)
class IntAtom:                   # integer-valued selector (SwitchOp scrutinee)
    name: str
    domain: range
```

## Budget / lanes

```python
@dataclass(frozen=True)
class SymbolicBudget:
    naive_inverse_max_size: int = 6
    inverse_max_degree: int | None = None
    cancel_num_dom: bool = False
    iszero_tol: float = 0.0
    surface_frame: bool = False
    einsum_bag_threshold: int | None = None
    defer_phi_jet: bool = True
    defer_covariant: bool = True
    freeze: bool = True
    defer_frame_contraction: bool = False

    @classmethod def legacy(cls) -> SymbolicBudget
    @classmethod def build_big_symbols(cls, *, retain_covariant: bool = False,
                                       surface_frame: bool | None = None,
                                       **overrides: Any) -> SymbolicBudget
    @classmethod def force_stmts(cls, **overrides: Any) -> SymbolicBudget
        # "no symbolic budget": drive every modeled op to a Stmt
        # (naive_inverse_max_size=0, inverse_max_degree=0,
        #  einsum_bag_threshold=1, freeze=True).  The build-then-simplify
        # entry point for Grassman lowering.
```

## IR passes (`forward.py`)

```python
def analyze(program: Program) -> IRReport
    # post-build structural cost report (stmt counts, per-cell monomial
    # mass, provenance histogram).

def partial_eval(program: Program, *, max_cell_size: int) -> Program
    # cost-driven, exactness-preserving collapse: output cells over the
    # monomial bound become IdentityOp atoms re-evaluated at run time.
    # max_cell_size=0 collapses all symbolic outputs to numeric.
    # partial_eval(p).run(x) == p.run(x).

```python
def partial_eval_numeric(program: Program, *, probes: int = 3, seed: int = 0,
                         rtol: float = 1e-9, atol: float = 1e-12) -> Program
def partial_eval_numeric_symarray(sa: SymArray, **kw) -> SymArray
    # Probe-and-freeze (simplify.py): fold every Stmt whose outputs are numerically
    # INVARIANT under the symbolic inputs — discovered by running the program at
    # `probes` random bindings (polynomial identity testing; probabilistic, NOT
    # exact-by-construction; measure-zero false freezes for rational cells).
    # Strictly stronger than the dataflow fold_numeric: collapses e.g. A·inv(A) ≡ I
    # and a metric-free grass_dof whose symbolic Jacobian input provably cancels.
    # The _symarray form also folds the cells (invariant atom -> numeric cell).

def dependency_cone(program: Program, target: SymArray) -> set[int]
    # The statement indices `target` transitively depends on (via input Refs +
    # DimAtom shape sources).
def evaluate_cone(program: Program, target: SymArray, values: Mapping) -> np.ndarray
    # Evaluate `target` by running ONLY its dependency cone at `values` — statements
    # outside the cone are NEVER executed, so a singular / failing op elsewhere in a
    # (possibly partially-built) shared program cannot crash or affect the probe.
    # Equals target.evaluate(values) whenever the full run would succeed.
    # (Program.build_runtime_bindings gained `only=<stmt indices>` for this lane.)
```
```

## Degree estimation (`degree.py`)

```python
def program_degree(program: Program, seed: Mapping[str, float], *,
                   gen_deg: Callable[[str], int] | None = None,
                   zero_ops: frozenset[str] = frozenset(),
                   passthrough_ops: frozenset[str] = frozenset(),
                   multilinear_ops: frozenset[str] = frozenset()) -> float
    # Whole-program POLYNOMIAL degree of the output in the seeded inputs
    # (fem task #9, lifted from pointwise). Sound over-estimation:
    # multilinear ops SUM, passthrough/additive MAX, all-constant
    # operands short-circuit to 0, det of an (n,n) degree-d operand is
    # n*d, CallOp (vmap) bodies are unwrapped and recursed; genuinely
    # rational/algebraic ops (inv/pinv/solve/sqrt/svd/...) on a
    # seed-dependent value give inf — the caller supplies its own order
    # there. The *_ops sets EXTEND the native categories with a front
    # end's op names (the `to_numpy_source(op_renderers=)` pattern);
    # domain seeding (FE degrees, affine geometry gates) stays with the
    # caller (pointwise `estimate_degree`).
```

## Backends

Selected at import time of `polyarray.poly_backend` from the
`CHARTLIB_POLY_BACKEND` (`sympy` | `flint` | `native_py` | `native_cpp`)
and `CHARTLIB_POLY_COEFF` (`double` | `mpf` | `quad`) environment
variables — **preserved verbatim from chartlib** (see VENDORED.md). The
sympy / native_py / native_value backends are pure Python; `native_cpp`
requires the built Cython `.so` (ship via wheels or `make cython`) and
falls back with a clear error if absent; `flint` requires the optional
`python-flint` and is auto-detected (default when present).

## Not included

`Program.fingerprint()` — referenced in chartlib comments but never
implemented; explicitly out of scope (plan §3b / §8). No caching layer
was added. Ops remain frozen / hashable as extracted.
```
