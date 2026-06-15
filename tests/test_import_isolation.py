"""Import isolation: polyarray must not pull in chartlib (plan §7)."""
from __future__ import annotations

import sys


def test_import_polyarray_does_not_load_chartlib() -> None:
    import polyarray  # noqa: F401

    leaked = [
        m for m in sys.modules
        if m == "chartlib" or m.startswith("chartlib.")
    ]
    assert leaked == [], f"chartlib leaked into sys.modules: {leaked}"


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
