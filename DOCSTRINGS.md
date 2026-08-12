# Docstring style

Docstrings describe what the code is and does, in prose, for a reader who does
not already know the codebase. The API reference is generated from them, so they
are the reference. These principles govern them; the two worked examples below
are the template.

## Principles

1. **Complete sentences.** Every statement is a full sentence with a subject and
   a verb. No fragments, no telegraphic phrases, no trailing appositive dumps.

2. **Prose, not an index.** Describe the thing and how it relates to the rest;
   do not enumerate a roster of members or restate the signature. The reference
   already lists the names — the docstring explains how they fit.

3. **State what it is, not what it isn't.** Open with the positive meaning.
   Introduce a contrast only after the meaning stands, and only when the contrast
   itself teaches something.

4. **Open with one self-contained summary sentence.** The first line is what the
   index shows, so it must read on its own. Expand underneath.

5. **Explain the mechanism when that is the useful part.** Say how it works, not
   just what category it belongs to — "extends that program, rewriting cells
   where the result is closed-form and appending a `Stmt` where it is not,"
   not "IR structure: `Stmt`, `Program`."

6. **A small diagram where it helps.** When a structure is easier shown than
   told — a module overview, a containment, a pipeline — a short ASCII diagram in
   a `::` literal block earns its space.

7. **Present tense, current behavior only.** No history, migration, roadmap, or
   provenance: no "used to", "replaces the former", "byte-identical", no plan /
   stage / slice / step, no measured anecdotes.

8. **polyarray's own vocabulary only.** Name the real types and mechanisms —
   program, statement, cell, ring, atom, rational function, bulk tensor. Do not
   borrow another layer's domain terms (geometry, DoF, φ-jet, finite element) or
   name other packages; reduce a motivating concept to the array/IR property it
   actually is. `pyab` is the one exception, since it is part of this repo.

9. **Keep the real contracts, and the terse register.** Exact facts stay — what
   `run` returns, what a `False` mask entry means, that dynamic shapes resolve at
   run time — as does genuine mathematics (metric, Σ, rational function). Dense
   and precise, never padded or breezy.

10. **Emphasis is earned.** Mark a genuine term of art on first use. Drop
    decorative bold fragments, shouting, and em-dash pile-ups used for tone.

## Template

**A module docstring — prose and a diagram, not three lists of class names.**

```
"""The imperative IR: programs of statements over symbolic-numeric arrays.

A :class:`Program` is an ordered list of imperative :class:`Stmt` statements over
:class:`SymArray` values, with named :class:`SymInput` inputs and named outputs::

    Program
      ├─ inputs   named SymInputs
      ├─ body     an ordered list of Stmts,  each  fn(Ref, …) → SymArray, …
      └─ outputs  named SymArrays

    SymArray  =  a numpy array of cells  |  a single bulk tensor
      cell    =  RationalFunction (exact)  |  float (numeric)  |  Ref (a prior output)

A :class:`SymArray` is the one value type, and it takes one of two forms. …
"""
```

**A class docstring — say what it is, not what it is not.**

Instead of *"A bulk output is NOT scattered into per-cell atom RationalFunctions:
…"*, state the meaning directly:

```
"""Handle for a :class:`Stmt` output carried as one whole tensor.

A *bulk* output stays a single numeric tensor: at :meth:`Program.run` time the
producing statement's result is bound under one name, and the owning
:class:`SymArray` evaluates directly to that tensor. …
"""
```

**A short attribute docstring — plain, no insider shorthand.**

Instead of *"The same vocabulary as a runtime `isinstance` tuple — derived from
`StmtFn`, so the two can never drift."*, name what it holds:

```
#: The op classes named by :data:`StmtFn`, collected into a tuple for
#: :func:`isinstance` checks. …
```
