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

Exploiting structure: a symbolic inverse
----------------------------------------

``symbolic_inverse`` is the sparsity-aware inverse. Given a matrix that is
structurally block-triangular -- here the top-right entry is a structural zero --
it inverts block by block with a Schur procedure, so the result is a small
rational function in each cell and the zero block is preserved rather than filled
in. (In a real pipeline these matrices come from the algebra layers above
polyarray; the array is built by hand here only to keep the example short.)

.. doctest::

   >>> import numpy as np
   >>> from polyarray import RationalFunction, SymArray, symbolic_inverse
   >>> x = RationalFunction.atom("x")
   >>> one, zero = RationalFunction.constant(1.0), RationalFunction.constant(0.0)
   >>> M = SymArray(np.array([[x + one, zero], [x, one + one]], dtype=object))
   >>> inv = symbolic_inverse(M)
   >>> print(inv.cells[0, 0])
   1/(x + 1)
   >>> print(inv.cells[0, 1])
   0

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
