Use Case
========

Polyarray is a compiler IR that represents Numpy/JAX/PyTorch style array programs with 
both a traditional array language IR and an expression language of rational functions.
By representing parts of programs with rational functions, we can symbolically simplify
programs using computer algebra techniques. By maintaining a traditional array language IR,
we can represent complex computations elegantly and utilize high-performance array language targets
such as Jax or PyTorch. If you want to symbolically simplify part of a program using a computer algebra
system and then lower the whole thing to an array language, this is your tool.

For instance, build a small matrix of symbolic parameters, invert it
symbolically, and lower the whole program to a NumPy function:

.. doctest::

   >>> import numpy as np
   >>> from polyarray import Program, SymInput, Provenance, symbolic_inverse, to_numpy_source
   >>> prog = Program("inv", inputs=[SymInput("A", (2, 2), Provenance("vertex", "A", (), "A"))])
   >>> lower = np.tril(np.ones((2, 2)))                 # keep the lower triangle
   >>> A = prog.input("A").einsum("ij,ij->ij", lower)   # zero the entries above the diagonal
   >>> inv = symbolic_inverse(A)                         # invert it symbolically
   >>> print(inv.cells[0, 0])                            # each cell is a rational function
   1/A_0_0
   >>> print(inv.cells[0, 1])                            # the structural zero is preserved
   0
   >>> print(inv.cells[1, 0])
   -A_1_0/(A_0_0*A_1_1)
   >>> _ = prog.add_output("inv", inv.cells)
   >>> print(to_numpy_source(prog, func_name="inv2x2"))  # lower the whole program to NumPy
   import numpy as np
   <BLANKLINE>
   def inv2x2(A):
       """Generated from polyarray Program 'inv' by to_numpy_source."""
       inv = np.array([[((1.0) / ((A[0, 0]))), 0.0], [((-(A[1, 0])) / ((A[0, 0])*(A[1, 1]))), ((1.0) / ((A[1, 1])))]], dtype=float)
       return inv
   <BLANKLINE>

Each cell of the inverse is an exact rational function, and the zero above the
diagonal survives rather than being filled in, so the structure is now explicit
in the program. That same symbolic form is what can grow: a rational inverse of
a larger matrix quickly dwarfs the program that produced it.

Since CAS computations and representations can be quite expensive, Polyarray provides tools to balance
the utility against the expense. In particular, Polyarray parametrizes symbolic computation with a budget
that limits how many array operations are converted to rational functions. To symbolically simplify a program,
especially one with complex meaningful numerical constants, we might partially evaluate part of the program to produce
a rational function form. For small programs, we can use this to detect structural sparsity or reduce computation,
but we risk exploding the size of the program.

How big is that in the dumb case? Invert a dense ``n x n`` matrix of distinct
symbols by Cramer's rule. The determinant is a sum over all ``n!`` permutations
-- ``n!`` monomials, each a product of ``n`` symbols -- and every one of the
``n^2`` inverse entries is an ``(n-1) x (n-1)`` cofactor of ``(n-1)!`` terms over
that shared determinant, so the inverse runs to on the order of ``n * n!``
monomials in all. The determinant alone is ``2`` terms at ``n = 2``, ``120`` at
``n = 5``, and about ``3.6`` million at ``n = 10`` -- already far past the size
of the ``10 x 10`` array it inverts. This factorial growth is what the budget is
there to avoid.

We offer the following features:

* An array IR that follows NumPy. It supplies the usual elementwise and
  linear-algebra operations (``matmul``, ``matvec``, ``einsum``, ``transpose``,
  ``reshape``, ``det``, ``inverse``, ``solve``, ``pinv``, ``sqrt``, and
  slicing), together with the control flow one expects from PyTorch or JAX:
  branching on an integer scrutinee (``declare_int_atom`` with
  :class:`~polyarray.ir.SwitchOp`), bounded loops (``WhileOp``), and calls into
  opaque functions or nested programs (``CallOp``). Unlike an ordinary array
  library, a cell need not be a number: it may be a
  :class:`~polyarray.RationalFunction`, a quotient of polynomials over a ring,
  with exact arithmetic and structural tests for cancellation and for a zero. A
  single program therefore carries floating-point tensors and exact symbolic
  entries at once.

* A builder in the style of the tracing APIs of PyTorch and JAX: one constructs
  a :class:`~polyarray.Program` by emitting statements over
  :class:`~polyarray.SymArray` operands. In doing so the builder must decide how
  much of a computation to keep as rational functions. Keeping more is better
  for later simplification, but a fully expanded rational program is frequently
  enormous, so this is a genuine trade-off rather than a fixed choice. We expose
  it as a ``SymbolicBudget`` (with controls such as ``naive_inverse_max_size``,
  ``inverse_max_degree``, ``einsum_bag_threshold``, and ``freeze``): an
  operation is carried symbolically while it remains small and is deferred to a
  numeric statement once expanding it would not be.

* Passes that analyze and rewrite a program. We estimate the polynomial degree
  of an output in its inputs (``program_degree``), discover and record sparsity
  (``sound_sparsity_mask`` and the routines that write a mask back into a
  matrix), and partially evaluate a program once its structural inputs are
  fixed, either exactly (``exact_fold``, ``partial_eval``) or by numeric probing
  (``partial_eval_numeric``). These passes recover the structure of a
  matrix-construction program, and that structure can be exploited: a
  structurally sparse symbolic matrix is inverted block by block by a Schur
  procedure (``symbolic_inverse``) into small rational functions, rather than
  through a dense cofactor expansion or an opaque numeric inverse.

* Several ways to execute or emit the same IR. A program runs directly under
  NumPy (``Program.run``), renders back to readable NumPy source
  (``numpy_source.to_numpy_source``), or compiles to a batched PyTorch kernel
  (``pyab.compile_torch``). The symbolic lane itself runs over whichever
  polynomial backend is present -- the pure-Python ``native_py`` and
  ``native_value``, a Cython/C++ ``native_cpp``, SymPy, or the exact-rational
  ``flint`` -- selected at import and requiring no change to the program.

* Instrumentation for the way these programs fail. A symbolic IR tends to fail
  by growing: it stops being polynomial, or its mass concentrates in one place
  and expands. We provide a tracing interface (``FEM_OBSERVE``) and measurement
  primitives that report where a program grows, where it ceases to be
  polynomial, and where its mass lies, so that such a blow-up can be located
  rather than merely encountered.
