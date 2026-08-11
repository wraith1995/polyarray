Philosophy
==========

Polyarray is an IR toolkit for programs that sit between tensor programming, in
the machine-learning sense, and tensor programming, in the computer algebra sense. We want a representation that
is expressive, that executes efficiently, that reuses existing tensor-programming
software such as JAX or PyTorch, and that can nonetheless be analyzed and
simplified symbolically in the way one would in a system like SymPy. No single
existing tool occupies this middle ground, so polyarray provides the following.

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
