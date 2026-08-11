polyarray
=========

Polyarray is a symbolic-numeric array IR. A :class:`~polyarray.ir.Program` is a
sequence of statements over :class:`~polyarray.ir.SymArray`\ s, and a cell of an
array is a rational function, a float, or a reference to an opaque op, so one
program holds both symbolic and numeric work. It is the lowering target for the
algebra layers above it, and it executes as it stands: one builds a program and
then calls ``program.run({name: array})``. The :doc:`philosophy` explains what
we are trying to balance and why; the :doc:`API reference <api/index>` documents
the surface.

.. toctree::
   :maxdepth: 2
   :caption: Narrative

   philosophy
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
