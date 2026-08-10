Philosophy
==========

.. note::

   This page is for the maintainer to write. The headings below mark the
   questions the rest of the documentation assumes an answer to; nothing here
   is generated from the source, and nothing should be.

Why an IR at all
----------------

.. todo:: What building a program buys over evaluating eagerly.

One program, two lanes
----------------------

.. todo:: Why numeric and symbolic are a property of a value rather than two
   code paths, and what goes wrong when they fork.

Exactness
---------

.. todo:: What "exact" means here, which passes preserve it, and where the
   boundary with floating point sits.

Budgets
-------

.. todo:: Why size is a policy rather than a heuristic, and how the build-time
   and post-build budgets differ.

Closed vocabularies
-------------------

.. todo:: Why the op union and the ref union are closed, and what an omission
   costs.
