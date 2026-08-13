# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

Initial public release.

### Added
- A symbolic-numeric array IR: `SymArray` values whose cells are rational
  functions or floats, threaded through a `Program`.
- Interchangeable polynomial backends (sympy and native), selected by
  environment at import time.
- A simplification budget that bounds symbolic work.
- Source emission: `to_numpy_source` / `compile` render a `Program` to
  standalone numpy source, with an `op_renderers` extension point for a
  consumer's own ops.

[Unreleased]: https://github.com/wraith1995/polyarray/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wraith1995/polyarray/releases/tag/v0.1.0
