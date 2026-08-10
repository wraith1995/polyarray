polyarray
=========

The symbolic-numeric array IR: a :class:`~polyarray.ir.Program` of statements
over :class:`~polyarray.ir.SymArray`\ s whose cells are rational functions,
floats, or references to opaque ops. It is the lowering target for the algebra
layers above it, and it executes as-is — build a program, then call
``program.run({name: array})``.

.. toctree::
   :maxdepth: 2
   :caption: Narrative

   philosophy
   lanes
   ops

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
