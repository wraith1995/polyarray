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
    shape: tuple[int, ...]
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
    shape: tuple[int, ...] = ()

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
