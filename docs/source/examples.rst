Examples
========

The following are short, runnable illustrations of the ideas in
:doc:`philosophy`. Every block is a doctest: the code and the output shown below
it are checked by ``make -C docs doctest``, so this page fails the build if the
API drifts from what it claims.

Building and running a program
------------------------------

One builds a :class:`~polyarray.ir.Program` over symbolic inputs and then runs it
under NumPy. The array operations are the ones one expects; here the determinant
of a symbolic ``2 × 2`` matrix is computed in closed form and then evaluated on a
concrete array.

.. doctest::

   >>> import numpy as np
   >>> from polyarray import Program, SymInput, Provenance
   >>> prog = Program("m", inputs=[SymInput("A", (2, 2), Provenance("vertex", "A", (), "A"))])
   >>> _ = prog.add_output("det", prog.input("A").det().cells)
   >>> prog.run({"A": np.array([[1., 2.], [3., 4.]])})["det"].tolist()
   -2.0

Exact symbolic cells
--------------------

A cell of an array need not be a number. It may be a
:class:`~polyarray.RationalFunction`, a quotient of polynomials over a ring, and
its arithmetic is exact: a common factor cancels rather than accruing a rounding
error.

.. doctest::

   >>> from polyarray import RationalFunction
   >>> x = RationalFunction.atom("x")
   >>> print((x * x - 1) / (x - 1))
   (x**2 - 1)/(x - 1)
   >>> print(((x * x - 1) / (x - 1)).clean())
   x + 1

Control flow
------------

A program may branch on an integer scrutinee. ``declare_int_atom`` names the
selector, :class:`~polyarray.ir.SwitchOp` picks a branch by its value, and the
value is supplied at run time alongside the arrays.

.. doctest::

   >>> import numpy as np
   >>> from polyarray import Program, SymArray, SwitchOp, OutSpec
   >>> from polyarray.ir import IntAtomRef
   >>> prog = Program("switch", inputs=[])
   >>> _ = prog.declare_int_atom("o", range(3))
   >>> branches = [SymArray(np.array([1., 2.]), program=prog),
   ...             SymArray(np.array([3., 4.]), program=prog),
   ...             SymArray(np.array([5., 6.]), program=prog)]
   >>> [out] = prog.emit_stmt(SwitchOp(n_branches=3), [IntAtomRef("o"), *branches],
   ...                        [OutSpec("chosen", (2,))], note="select(o)")
   >>> _ = prog.add_output("chosen", out.cells)
   >>> prog.run({"o": 0})["chosen"].tolist()
   [1.0, 2.0]
   >>> prog.run({"o": 2})["chosen"].tolist()
   [5.0, 6.0]

Analyzing a program
-------------------

Because a program is data, one can measure it before running it.
``program_degree`` reports the polynomial degree of an output in its inputs (with
every atom weighted as degree one, the determinant above is degree two), and
``analyze`` returns a report of the program's shape -- here ``n_defer`` is the
number of statements that fell back to a numeric op, which is zero because the
determinant stayed symbolic. The same measurement primitives back the
``FEM_OBSERVE`` tracing surface.

.. doctest::

   >>> import numpy as np
   >>> from polyarray import Program, SymInput, Provenance, program_degree, analyze
   >>> prog = Program("m", inputs=[SymInput("A", (2, 2), Provenance("vertex", "A", (), "A"))])
   >>> _ = prog.add_output("det", prog.input("A").det().cells)
   >>> program_degree(prog, seed={}, gen_deg=lambda name: 1)
   2.0
   >>> analyze(prog).n_defer
   0

Discovering sparsity
--------------------

polyarray reads the structural-zero pattern of a matrix -- the cells it can
prove are zero -- rather than being told where they are. Here a symbolic input
is restricted to its lower triangle (a Hadamard product with a lower-triangular
pattern of ones and zeros), so the entries above the diagonal become exact
zeros. ``sound_sparsity_mask`` recovers that pattern: a ``0`` entry is a cell
proved zero.

.. doctest::

   >>> import numpy as np
   >>> from polyarray import Program, SymInput, Provenance, sound_sparsity_mask
   >>> prog = Program("m", inputs=[SymInput("A", (2, 2), Provenance("vertex", "A", (), "A"))])
   >>> lower = np.tril(np.ones((2, 2)))                 # keep the lower triangle
   >>> tri = prog.input("A").einsum("ij,ij->ij", lower)
   >>> tri.cells[0, 1]                                  # the dropped entry is an exact zero
   RationalFunction(0)
   >>> sound_sparsity_mask(tri).astype(int).tolist()
   [[1, 0], [1, 1]]

Exploiting structure: a symbolic inverse
----------------------------------------

``symbolic_inverse`` is the sparsity-aware inverse. It resolves that same
structural-zero pattern itself -- no mask is supplied -- and inverts the matrix
block by block with a Schur procedure, so each cell is a small rational function
and the zero above the diagonal is preserved rather than filled in.

.. doctest::

   >>> from polyarray import symbolic_inverse
   >>> inv = symbolic_inverse(tri)          # reuses ``tri`` from above; no mask
   >>> print(inv.cells[0, 0])
   1/A_0_0
   >>> print(inv.cells[0, 1])
   0
   >>> print(inv.cells[1, 0])
   -A_1_0/(A_0_0*A_1_1)

Partial evaluation
------------------

Once a program's structural inputs are fixed, ``partial_eval`` folds away what it
can, returning a program that computes the same thing.

.. doctest::

   >>> import numpy as np
   >>> from polyarray import Program, SymInput, Provenance, partial_eval
   >>> prog = Program("m", inputs=[SymInput("A", (2, 2), Provenance("vertex", "A", (), "A"))])
   >>> _ = prog.add_output("det", prog.input("A").det().cells)
   >>> folded = partial_eval(prog, max_cell_size=1000)
   >>> folded.run({"A": np.array([[1., 2.], [3., 4.]])})["det"].tolist()
   -2.0

Emitting the program elsewhere
------------------------------

The same IR can be rendered back to readable NumPy source rather than executed
in place.

.. doctest::

   >>> import numpy as np
   >>> from polyarray import Program, SymInput, Provenance, to_numpy_source
   >>> prog = Program("m", inputs=[SymInput("A", (2, 2), Provenance("vertex", "A", (), "A"))])
   >>> _ = prog.add_output("det", prog.input("A").det().cells)
   >>> src = to_numpy_source(prog, func_name="det2x2")
   >>> "def det2x2(A):" in src
   True
   >>> "(A[0, 0])*(A[1, 1])" in src
   True
