"""Backend matrix: run a representative golden program under each
available poly backend and assert matching results (plan §6/§7).

The backend is fixed at *import time* of ``polyarray.poly_backend`` from
the ``CHARTLIB_POLY_BACKEND`` / ``CHARTLIB_POLY_COEFF`` environment
variables (preserved verbatim from chartlib). To exercise multiple
backends in one test run we launch a fresh subprocess per backend, each
running the same rational-lane program (symbolic 2x2 inverse, which goes
through the poly ring) and printing the result. The parent compares.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

# Program that exercises the polynomial ring (rational lane): a symbolic
# 2x2 inverse evaluated at a concrete matrix. Each backend must produce
# the same numbers.
_CHILD = r"""
import numpy as np
from polyarray import Program, SymInput, Provenance

prog = Program("rat", inputs=[SymInput("M", (2, 2),
    Provenance("vertex", "M", (), "M"))])
inv = prog.input("M").inverse()
prog.add_output("inv", inv.cells)
res = prog.run({"M": np.array([[2.0, 1.0], [1.0, 3.0]])})
flat = np.asarray(res["inv"], dtype=float).ravel()
print(",".join(repr(float(x)) for x in flat))
import polyarray.poly_backend as pb
import sys
print("BACKEND=" + pb.BACKEND_NAME, file=sys.stderr)
"""

# (env overrides, human label). native_value == native_py over the mpf
# value handle (the value-handle layer); native_py default uses the
# double handle.
_BACKENDS = [
    ({"CHARTLIB_POLY_BACKEND": "sympy"}, "sympy"),
    ({"CHARTLIB_POLY_BACKEND": "native_py", "CHARTLIB_POLY_COEFF": "double"}, "native_py"),
    ({"CHARTLIB_POLY_BACKEND": "native_py", "CHARTLIB_POLY_COEFF": "mpf"}, "native_value"),
    ({"CHARTLIB_POLY_BACKEND": "native_cpp", "CHARTLIB_POLY_COEFF": "double"}, "native_cpp_double"),
    ({"CHARTLIB_POLY_BACKEND": "native_cpp", "CHARTLIB_POLY_COEFF": "mpf"}, "native_cpp_mpf"),
]

# Hand-computed reference: inv([[2,1],[1,3]]) = 1/5 [[3,-1],[-1,2]].
_EXPECTED = np.array([3.0, -1.0, -1.0, 2.0]) / 5.0


def _run_backend(env_overrides):
    env = dict(os.environ)
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True, text=True, env=env,
    )
    return proc


@pytest.mark.parametrize("env_overrides,label", _BACKENDS, ids=[b[1] for b in _BACKENDS])
def test_backend_matches_reference(env_overrides, label):
    proc = _run_backend(env_overrides)
    if proc.returncode != 0:
        # native_cpp requires the built .so; if absent, skip rather than fail.
        if "native_cpp" in label and "Cython extension" in proc.stderr:
            pytest.skip(f"{label}: native_cpp extension not built")
        pytest.fail(
            f"backend {label} failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    line = proc.stdout.strip().splitlines()[-1]
    got = np.array([float(x) for x in line.split(",")])
    np.testing.assert_allclose(got, _EXPECTED, atol=1e-9,
        err_msg=f"backend {label} mismatch (stderr: {proc.stderr})")


def test_at_least_sympy_and_native_py_run():
    # Sanity: the always-available pure-Python backends must work.
    for env_overrides, label in _BACKENDS[:3]:
        proc = _run_backend(env_overrides)
        assert proc.returncode == 0, f"{label} failed: {proc.stderr}"
