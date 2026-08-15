# polyarray — public API

The committed public surface. Everything an addition adds must be recorded
here; this file is the contract, and the generated
[API reference](api/index.rst) is its browsable form.

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
    # A shape entry may be a runtime DimAtom (a dynamic input axis whose size
    # is a runtime rank). A dynamic SymInput is allocated as a single bulk
    # SymArray (no per-cell atoms); its DimAtom axes are bound from the provided
    # array's shape at Program.run time.
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
class DimAtom:                   # a runtime array dimension (e.g. an SVD rank)
    name: str                    # provenance, e.g. "rank:svd"
    # A tagged, hashable tuple identifying the run-time origin (the
    # dim_bindings key):
    #   ("stmt", stmt_idx, out_idx)  — a prior Stmt output
    #   ("in",   input_name, axis)   — a dynamic SymInput axis
    # Compat: a bare (stmt_idx, out_idx) 2-tuple is normalised to the
    # ("stmt", ...) form. Build input-axis atoms via DimAtom.from_input.
    source: tuple[Any, ...]
    @staticmethod
    def from_input(name: str, input_name: str, axis: int) -> DimAtom: ...

def is_dynamic(shape: tuple[int | DimAtom, ...]) -> bool
    # True iff any shape entry is a DimAtom (a runtime dimension).  An
    # OutSpec/output with a dynamic shape is always emitted bulk, its axis
    # sizes resolved at Program.run time.  Fully-concrete shapes are not
    # dynamic, so static-shape programs are unaffected.

class SymbolEnv:                 # owned by Program; shared across cells
    ...

def allocate_input(env: SymbolEnv, spec: SymInput) -> np.ndarray
```

## Values

```python
class SymArray:                  # object/float ndarray of cells, bound to a Program
    def matmul(self, other: SymArray | np.ndarray) -> SymArray
    def matvec(self, v: SymArray | np.ndarray) -> SymArray
    def einsum(self, subscripts: str, *others: SymArray | np.ndarray) -> SymArray  # np.einsum on cells, program threaded;
                                                                                  # BULK-PRESERVING: a bulk operand routes to
                                                                                  # runtime_einsum_multi (one EinsumStmtOp), never unpacks
    def transpose(self) -> SymArray
    def reshape(self, shape: tuple[int, ...] | list[int]) -> SymArray   # BULK-PRESERVING: emits ReshapeOp, never unpacks
    def expand_dims(self, axis: int) -> SymArray                        # the deferral-safe `arr[:, np.newaxis]`
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

## Linear algebra

```python
def symbolic_inverse(matrix: SymArray | np.ndarray, *, mask: np.ndarray | None = None) -> SymArray
    # Block-triangular Schur inverse — the SPARSITY-AWARE sibling of SymArray.inverse. Exploits a
    # (row-reordered) block-lower-triangular / block-diagonal structure so a structurally sparse symbolic
    # matrix inverts to small RationalFunctions instead of a dense cofactor blow-up (or a numeric InvOp).
    # `mask` steers the split (a caller's cheap/exact sparsity); omitted → resolved by deterministic probing
    # of a program-carrying SymArray, else syntactic simple_zero. A conservative (denser) mask is never
    # wrong, only less aggressive. Leaf inverses / Schur-combine products above
    # SymbolicBudget.schur_{inverse,matmul}_stmt_size defer to numeric Stmts.

def sound_sparsity_mask(matrix: SymArray) -> np.ndarray
    # The mask `symbolic_inverse` resolves for itself, as a value a caller can hold: a False entry is a
    # cell PROVED zero (syntactic, or a constant within roundoff after the exact fold), a True entry
    # claims nothing. For taking the sparsity where it is still visible — e.g. before a Program.graft,
    # which re-homes cells as fresh atoms and so hides every provable zero.

def mask_zeros(arr: SymArray, mask: np.ndarray) -> SymArray
    # `arr` with every cell `mask` proves zero replaced by an EXACT zero of the matching lane, riding the
    # same program. The companion of sound_sparsity_mask: writes a mask BACK INTO the matrix, so every
    # reader (exact fold, degree walk, codegen, the consumer's own arithmetic) sees the sparsity — rather
    # than only the one reader a `mask=` argument reaches.
```

## Op vocabulary (the `Stmt.fn` types)

All frozen / hashable dataclasses.

### The union — `StmtFn` / `STMT_FN_OPS` / `is_builtin_op`

```python
StmtFn: TypeAlias = DetOp | InvOp | ... | WhileOp   # every op class listed below
STMT_FN_OPS: tuple[type, ...]                       # = get_args(StmtFn); the isinstance tuple
def is_builtin_op(fn: object) -> TypeGuard[StmtFn]
```

`Stmt.fn` is `Callable | Program | None` — deliberately **open**, so a front end above
polyarray can put its own op class, a `vmap` closure or a plain callable there. `StmtFn`
names the **closed** part: the ops polyarray itself owns. A pass over the vocabulary is
written as `if is_builtin_op(fn): <exhaustive match closed by typing.assert_never>`, with
the open cases handled *before* the match. Adding an op to `StmtFn` then makes every pass
that has not decided about it a **mypy error** — the guarantee an isinstance ladder cannot
give (an op missing from a ladder is silently opaque; `KronOp`/`KronFreeOp` and `SwitchOp`
each cost a day that way).

Adding an op means, together: this list, `StmtFn`, `polyarray/__init__`'s exports,
`degree.DEFAULT_DEGREE_KINDS`, the `pyab` + `to_numpy_source` render lanes, and whatever
mypy names. `tests/test_op_union.py` checks every one of those mirrors against the union.

```python
@dataclass(frozen=True) class DetOp                  # -> np.linalg.det
@dataclass(frozen=True) class InvOp                  # -> np.linalg.inv
@dataclass(frozen=True) class PinvOp                 # -> np.linalg.pinv
@dataclass(frozen=True) class SolveOp                # -> np.linalg.solve

@dataclass(frozen=True)
class SvdOp:                     # -> (U, S, Vh, rank); 4-output
    rcond: float | None = None   # rank threshold (np.linalg.matrix_rank rule)
    full_matrices: bool = False  # rank is the thresholded numerical rank

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

# Generic array builtins (relocated front-end lowering ops; both lanes render).
@dataclass(frozen=True) class TransposeOp            # A.T (full reverse; value-preserving)

@dataclass(frozen=True)
class SinvFullOp:                # rectangular-diagonal S⁻¹: 1/Sᵢ (i<rank) else 0
    nrows: int; ncols: int       # __call__(S, rank) -> (nrows×ncols)

@dataclass(frozen=True)
class GSvdFullOp:                # GSvdOp then [U|UI],[V|VI]; -> (Ufull, Vfull, S, rank); 4-output
    rcond: float | None = None   # __call__(A, M_V, M_W); full-width de-whitened factors

@dataclass(frozen=True) class BlockDiagOp            # diag(A, B, …) of the operands
@dataclass(frozen=True)
class BlockRepeatOp:             # n block-diagonal copies of A = kron(eye(n), A)
    n: int
@dataclass(frozen=True) class DynBlockRepeatOp       # __call__(A, n): runtime-n block-repeat

# Batch-2 relocated generic array builtins (both lanes render; several 0-d int → DimAtom source).
@dataclass(frozen=True)
class DynEyeOp:                  # eye(ref.shape[axis]) — runtime identity sized by a ref axis
    axis: int = 1
@dataclass(frozen=True)
class DynZerosOp:                # zeros((refs[i].shape[axes[i]], …)) — a symbolic ℝⁿ zero
class DynReshapeOp:              # refs[0].reshape(refs[1+i].shape[axes[i]], …) — dynamic un-fuse
    axes: tuple[int, ...]
@dataclass(frozen=True)
class DynEyeTensorOp:            # eye(∏dᵢ).reshape(∏dᵢ, d₀, …) — multi-axis DimVar seed identity
    axes: tuple[int, ...]
@dataclass(frozen=True)
class ProdShapeOp:               # static · ∏ refs[i].shape[axes[i]] as a 0-d int
    axes: tuple[int, ...]; static: int = 1
@dataclass(frozen=True)
class SumShapeOp:                # static + Σ refs[i].shape[axes[i]] as a 0-d int
    axes: tuple[int, ...]; static: int = 0
@dataclass(frozen=True)
class SumDimOp:                  # Σ operands' axis lengths as a 0-d int
    axis: int = 0
@dataclass(frozen=True)
class ProdDimOp:                 # ∏ operands' axis lengths as a 0-d int
    axis: int = 0
@dataclass(frozen=True)
class ScaleAxisDimOp:            # n · mat.shape[axis] as a 0-d int (static n)
    n: int; axis: int = 0
@dataclass(frozen=True)
class MulAxisDimOp:              # __call__(n, mat): int(n) · mat.shape[axis] as a 0-d int
    axis: int = 0
@dataclass(frozen=True)
class CompRankOp:                # __call__(rank): ambient − int(rank) as a 0-d int
    ambient: int
@dataclass(frozen=True) class HStackOp               # [A | B | …] concat matrices on axis 1
@dataclass(frozen=True) class ColStackOp             # stack flattened vectors as columns (axis 1)
@dataclass(frozen=True)
class ScaleOp:                   # factor · x (static scalar)
    factor: float
@dataclass(frozen=True) class ScaleByOp              # __call__(x, s): s · x (runtime scalar)
@dataclass(frozen=True)
class AddOp:                     # left-fold x0 + x1 + … over n operands
    n: int
@dataclass(frozen=True) class ConcatOp               # flatten each operand, concatenate on axis 0
@dataclass(frozen=True)
class AxisLenOp:                 # x.shape[axis] as a 0-d int
    axis: int = 0
@dataclass(frozen=True)
class ReshapeOp:                 # A.reshape(shape) (static shape)
    shape: tuple[int, ...]
@dataclass(frozen=True)
class ConstOp:                   # frozen numeric constant from raw bytes (no args)
    key: str; data_bytes: bytes; shape: tuple[int, ...]; dtype: str
@dataclass(frozen=True)
class EyeOp:                     # static n×n identity np.eye(n) (no args)
    n: int
@dataclass(frozen=True) class FirstColsOp            # __call__(A, rank): A[:, :int(rank)]
@dataclass(frozen=True) class LastColsOp             # __call__(A, rank): A[:, int(rank):]

# Batch-3 relocated generic array / linalg builtins (both lanes render; linalg via _ns_call).
@dataclass(frozen=True) class ProjectOp              # __call__(P, v): Pᵀ @ v.reshape(-1) (ambient→sub)
@dataclass(frozen=True)
class EmbedOp:                   # __call__(P, vsub): (P @ vsub).reshape(shape) (sub→ambient)
    shape: tuple[int, ...] = ()
@dataclass(frozen=True)
class KronOp:                    # chained Kronecker product kron(mats[0], mats[1], …)
    n: int
@dataclass(frozen=True)
class KronFreeOp:                # __call__(F, G): block-Kron on space axes, outer on trailing free axes
    nf_free: int; ng_free: int
@dataclass(frozen=True) class InvTransposeOp         # __call__(A): inv(A).T (dual-basis cob)
@dataclass(frozen=True) class ComposeViaStdOp        # __call__(R_to, R_from): solve(R_to, R_from)
@dataclass(frozen=True) class SqrtSpdOp              # __call__(G): SPD operator sqrt via eigh (SPD-guarded)
@dataclass(frozen=True)
class RankOp:                    # __call__(A): numeric column rank as a 0-d int
    tol: float = 1e-9
@dataclass(frozen=True) class MetricOrthonormalOp    # __call__(A, G): A·L⁻ᵀ, LLᵀ=AᵀGA (Cholesky)

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
```

```python
def specialize(program, *, bind=None, subs=None, budget=None) -> Program
def fold_numeric(program) -> Program        # = specialize (empty bind)
def bind_inputs(program, bind) -> Program   # = specialize(bind=...)
    # Exactness-preserving partial evaluation (simplify.py): folds every Stmt whose
    # inputs all resolve numeric, dropping it; folds `known` into surviving refs /
    # outputs; descends partially-numeric sub-Program / CallOp bodies.
    # Dynamic dims: a Stmt that CREATES a runtime δ (DimAtom) — SvdOp/GSvdOp/QrOp/
    # pinv/… — is folded UNIFORMLY when its inputs are all numeric (a value-
    # invariant map with a statically-knowable rank). The δ it creates is resolved
    # from the concrete folded-output shape and SUBSTITUTED (→ concrete int) across
    # every remaining shape (later Stmt outputs, bulk handles, input refs), so no
    # dynamic δ lingers downstream. A δ from a Stmt with a symbolic input is NOT
    # folded — it survives unchanged (conservative).

def partial_eval_numeric(program: Program, *, probes: int = 3, seed: int = 0,
                         rtol: float = 1e-9, atol: float = 1e-12,
                         mode: str | None = None, work_budget: int | None = None,
                         max_sym_mass: int | None = None,
                         time_budget: float | None = None) -> Program
def partial_eval_numeric_symarray(sa: SymArray, **kw) -> SymArray
class NonExactFoldWarning(UserWarning)
class NonDeterministicFoldWarning(NonExactFoldWarning)
    # Fold every Stmt whose outputs are INVARIANT under the symbolic inputs —
    # strictly stronger than the dataflow fold_numeric: collapses e.g. A·inv(A) ≡ I
    # and one whose symbolic input provably cancels.
    # The _symarray form also folds the cells (invariant atom -> numeric cell).
    # `mode` selects HOW invariance is certified (default: env
    # POLYARRAY_PARTIAL_EVAL_MODE, else "hybrid"; the parameter always wins):
    #   "exact"  — exact_fold.py: each output entry brought to rational normal form
    #              over the feed atoms (flint fmpq — exact-by-construction; the
    #              rational op set + exact Gauss inv/det/solve + sub-Program descent;
    #              numeric-closed subgraphs run their real ops deterministically).
    #              Non-normalizable (opaque-op) statements are left symbolic; provably
    #              non-constant ones are REFUTED. Cost is bounded by BOTH `work_budget`
    #              (DETERMINISTIC work units, charged between ops) and an operand-size cap
    #              checked BEFORE each symbolic op (`exact_fold._MAX_SYM_MASS` monomials —
    #              one einsum / Gauss pass over RF cells cannot be interrupted mid-flight).
    #              Oversized or out-of-budget statements degrade to unresolved ⇒ probe
    #              fallback. `work_budget=None` reads POLYARRAY_EXACT_WORK_BUDGET, else
    #              `exact_fold._DEFAULT_WORK_BUDGET`; `work_budget=0` means unbounded.
    #
    #              The budget is WORK, NOT SECONDS — that is the contract. What certifies
    #              is a function of the program alone, so the same input yields the same
    #              certificate on any machine under any load. It used to be wall-clock
    #              seconds, which made a certificate a property of the box: one leg gave
    #              6874 / 6895 / 7726 frozen statements on three runs. A generous
    #              wall-clock BACKSTOP survives only to catch a mis-calibrated cost model,
    #              and raises NonDeterministicFoldWarning when it fires — never silent,
    #              because it reintroduces exactly that machine dependence.
    #
    #              `time_budget=` (seconds) is ACCEPTED for backward compatibility but no
    #              longer selects certificates: it sizes the backstop, i.e. it still bounds
    #              how long the call may run. Passing it emits a DeprecationWarning.
    #   "hybrid" — exact first; ONLY unresolved statements fall back to the probe
    #              pass, and every probe freeze raises ONE aggregated
    #              NonExactFoldWarning naming the sites. Exactly-refuted statements
    #              are never probed (closes the colluding-probe false freeze).
    #   "probe"  — the legacy probe-and-freeze, unchanged and silent: `probes`
    #              random bindings (polynomial identity testing; probabilistic, NOT
    #              exact-by-construction; measure-zero false freezes) — for
    #              diagnostic/performance sites that don't need exactness.
    # The _symarray form additionally applies the ENTRY-LEVEL exact fold in
    # exact/hybrid modes: a cell whose rational normal form is degree-0 folds to its
    # exact constant even when no single statement is invariant.

def dependency_cone(program: Program, target: SymArray) -> set[int]
    # The statement indices `target` transitively depends on (via input Refs +
    # DimAtom shape sources).
def evaluate_cone(program: Program, target: SymArray, values: Mapping) -> np.ndarray
    # Evaluate `target` by running ONLY its dependency cone at `values` — statements
    # outside the cone are NEVER executed, so a singular / failing op elsewhere in a
    # (possibly partially-built) shared program cannot crash or affect the probe.
    # Equals target.evaluate(values) whenever the full run would succeed.
    # (Program.build_runtime_bindings gained `only=<stmt indices>` for this lane.)

def is_structurally_constant(program: Program, target: SymArray) -> bool
    # SOUND, sampling-free test that `target`'s value is a build-time constant:
    # True iff its dependency_cone is a CLOSED CONSTANT subprogram — no Stmt input is
    # an InputRef / IntAtomRef (a runtime feed) and no cell generator anywhere in the
    # cone (nor in `target` itself) has provenance kind other than "stmt_out"
    # (i.e. no vertex/point/coeff/… feed atom is read). EXACT, not heuristic; never
    # calls a feed-varying value constant. CONSERVATIVE: any ref/generator that cannot
    # be positively classified constant-safe — or a cone that fails to enumerate —
    # forces False. Folds without any evaluation/codegen.
```

## Degree estimation (`degree.py`)

```python
def program_degree(program: Program, seed: Mapping[str, float], *,
                   gen_deg: Callable[[str], int] | None = None,
                   zero_ops: frozenset[str] = frozenset(),
                   passthrough_ops: frozenset[str] = frozenset(),
                   multilinear_ops: frozenset[str] = frozenset()) -> float
    # Whole-program POLYNOMIAL degree of the output in the seeded inputs.
    # Sound over-estimation:
    # multilinear ops SUM, passthrough/additive MAX, all-constant
    # operands short-circuit to 0, det of an (n,n) degree-d operand is
    # n*d, CallOp (vmap) bodies are unwrapped and recursed; genuinely
    # rational/algebraic ops (inv/pinv/solve/sqrt/svd/...) on a
    # seed-dependent value give inf — the caller supplies its own order
    # there. The *_ops sets EXTEND the native categories with a front
    # end's op names (the `to_numpy_source(op_renderers=)` pattern);
    # domain seeding stays with the caller.
```

## Compile observability (`observe.py`)

The staged compile trace that a front end instruments against. It lives here
because this package owns the measurement primitives (`forward.analyze`,
`degree.program_degree`, `ir._cell_size`) and the pyab→torch dump hand-off.

```python
class Level(IntEnum): OFF, WARN, INFO, DEBUG, DUMP     # from FEM_OBSERVE, default WARN

def measure(obj, *, degree_seed=None) -> Measurement
    # Size / degree / provenance of a Program, SymArray, ndarray, sequence, or wrapper
    # exposing `.program`/`.prog`. FAILURE-TOLERANT: a probe that raises yields a
    # Measurement carrying `.error`, never propagates. Never forces a bulk node.

@contextmanager
def observe_compile(name, *, level=None, dump_root=None, report=True) -> CompileTrace
def scope(name, **kw) -> CompileTrace    # reuse an enclosing trace, else open one
def trace() -> CompileTrace              # the ambient trace; NEVER None (null when off)
def active() -> bool                     # guard EXPENSIVE ctx gathering only
def stage(name, obj=None, *, detail=None, **ctx)   # record on the ambient trace
def phase(name, *, detail=None, **ctx)             # time a block; result -> box[0]
    # `detail` is a zero-arg callable returning extra text for the stage's DUMP —
    # what the stage produced in the caller's own vocabulary (which basis, which
    # bindings). Called only at `dump`, only once per stage; being a thunk it may
    # close over locals bound inside the phase.
def dump_dir_for(name) -> Path | None    # pass straight to pyab.compile_torch(dump_dir=)

class CompileTrace:
    .stage / .phase / .dump_dir_for / .report() / .write_report()
    .peak_stage()      # largest mass — where the IR is biggest
    .jump_stage()      # largest mass multiplier over its parent — where it GOT big
    .rational_stage()  # where degree first went inf (stopped being polynomial)
```

Stages repeated in a loop are **rolled up** by `(depth, name)`: `Snapshot.m` is the
largest MEASURED occurrence, `.count` the number of runs, `.n_measured` how many were
measured — one row and one dump directory per stage however many times it ran. Below
`debug` the occurrences are sampled geometrically (1, 2, 4, 8, …); `debug`/`dump`
measure all of them.

At `dump` each stage directory gets `stage.txt` (numbers + `IRReport`), `program.py`
(the stage's polyarray IR rendered via `numpy_source.to_numpy_source`), and `detail.txt`
if the call site supplied one.

Env: `FEM_OBSERVE` (level), `FEM_OBSERVE_DIR` (dump root), `FEM_OBSERVE_MASS_CEILING` /
`_OPERAND_CEILING` / `_CELLS_CEILING` / `_DEGREE_CEILING` (warning thresholds).

## Backends

Selected at import time of `polyarray.poly_backend` from the
`CHARTLIB_POLY_BACKEND` (`sympy` | `flint` | `native_py` | `native_cpp`)
and `CHARTLIB_POLY_COEFF` (`double` | `mpf` | `quad`) environment
variables. The
sympy / native_py / native_value backends are pure Python; `native_cpp`
requires the built Cython `.so` (ship via wheels or `make cython`) and
falls back with a clear error if absent; `flint` requires the optional
`python-flint` and is auto-detected (default when present).

## Code emission

Two optional, additive lowerings turn a `Program` into runnable code. Both
keep their target dependency **out of `import polyarray`** (front-end ops are
supplied by the caller, never imported by polyarray):

- `numpy_source.to_numpy_source(program, func_name="f", op_renderers=None)` —
  a standalone numpy `.py` source string. The emitted function takes one
  parameter per program input, one per `IntAtom`, and one per **free feed atom**
  (a cell generator bound by neither an input nor a statement output — for
  example per-cell vertex atoms in a mid-pipeline program; appended last in
  sorted generator-name order, and threaded into sub-`Program`/`vmap` bodies that
  are open over them). Front-end Stmt ops emit via a `__numpy_source__(self,
  args) -> str` hook on the op class — the same shape as pyab's
  `__pyab_lower__`, discovered without any threading — or via `op_renderers`
  keyed by op class name, which takes precedence over the hook.
- `pyab` — lower a `Program` to **PyArrayBackend IR** (torch-compilable).
  Requires `pyarraybackend` (imported lazily); `torch` only when compiling
  through the torch backend. Public surface:
  `pyab.lower_program_into(program, builder, arg_exprs, *, opts)` (inline a
  program into a PyAB `StmtBuilder`), `pyab.as_function_def(program, name, opts)`
  (emit a callable `FunctionDefStmt` + helper defs),
  `pyab.call_lowered(program, builder, arg_exprs, *, place, in_dims, out_dim, ...)`
  (emit a callable + a *placed* call — `place="plain"|"vmap"|"fuse"` chosen at
  the call site), and `pyab.compile_torch` / `pyab.compile_numpy` conveniences.
  `pyab.LowerOpts(target=, small_qr=, op_lowerings=)` configures it;
  `SmallQrOpts(max_dim=4)` intercepts small fixed-size `QrOp` as unrolled scalar
  Householder QR (LAPACK sign convention) instead of a `linalg.qr` call. The
  full op vocabulary lowers, including `SvdOp`/`GSvdOp` (composite cholesky/svd/
  solve + rank split — the data-dependent rank runs eager, i.e. a `@torch.compile`
  fusion boundary, not a hard failure), `WhileOp` (→ `torch.while_loop`, cond/body
  must be sub-Programs) and nested (multi-var) `vmap` (→ nested `torch.vmap`). A
  `Program` op with no lowering raises `NotImplementedError` (extend via
  `op_lowerings`). **Backend prep** (`pyab.prepare(program)`, run automatically
  before lowering; opt out via `LowerOpts(collapse_vmap=/simplify_gsvd=False)`)
  eliminates avoidable work: `pyab.collapse_vmap` replaces a `vmap` over a single
  leading-batch op (det/inv/pinv/solve/sqrt/abs/sign/einsum) with the batched op
  applied directly — **numerically verified equivalent on a probe before
  rewriting** (so e.g. numpy-2.0 vector-`solve`, which no longer batches, is left
  alone); and identity metrics fold out of a `GSvdOp` (collapsing it toward a
  plain SVD). Both are semantics-preserving and also speed up `Program.run` /
  `to_numpy_source`.

## Not included

`Program.fingerprint()` — explicitly out of scope; there is no caching
layer. Ops remain frozen and hashable.
