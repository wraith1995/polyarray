"""Import isolation: polyarray must be self-contained.

Importing the package must pull in only its declared runtime dependencies and
the standard library, nothing from another project. The probe runs in a fresh
interpreter and measures what ``import polyarray`` adds, so a module another
test already imported cannot mask a leak.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def test_polyarray_imports_only_declared_dependencies() -> None:
    probe = textwrap.dedent(
        """
        import sys
        before = {m.split(".")[0] for m in sys.modules}
        import polyarray  # noqa: F401
        added = {m.split(".")[0] for m in sys.modules} - before
        allowed = {"polyarray", "numpy", "sympy", "mpmath", "gmpy2",
                   "flint", "cython_runtime"}
        foreign = sorted(
            t for t in added
            if t not in allowed
            and t not in sys.stdlib_module_names
            and not t.startswith("_")
        )
        print(" ".join(foreign))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True,
    )
    foreign = out.stdout.split()
    assert not foreign, f"polyarray pulled in undeclared modules: {foreign}"


def test_public_surface_is_importable() -> None:
    import polyarray as pa

    expected = {
        "Program", "SymInput", "SymbolEnv", "Provenance", "SymArray",
        "OutSpec", "SymbolicBudget", "allocate_input", "RationalFunction",
        "IntAtom", "DetOp", "InvOp", "PinvOp", "SolveOp", "SqrtOp", "AbsOp",
        "SignOp", "TensordotOp", "MoveaxisOp", "EinsumOp", "EinsumStmtOp",
        "SwitchOp", "CallOp", "WhileOp", "IdentityOp", "analyze", "partial_eval",
    }
    assert expected <= set(pa.__all__)
    for name in expected:
        assert hasattr(pa, name), f"polyarray.{name} missing"
