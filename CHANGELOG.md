# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.2.0] - 2025-08-08

### Added
- Evaluator support for Dot (`.`) and Character Classes (`[...]`) with ranges and escapes (e.g., `\d`, `\w`, `\s`).
- New unit tests for regex-like atoms and DSL-level tests for dot/charclass behavior.
- CI coverage artifact generation and Codecov upload; README badges for CI and coverage.

### Changed
- Consolidated demo CLI workflow into main CI matrix (Windows, macOS, Linux).
- Parser: normalized resolver args/flags/optional handling; improved typing.

### Fixed
- Lint (ruff) warnings and mypy typing issues across parser and tests.
- Resolver tests minor issues (exception handling, indentation).

## [0.2.1] - 2025-08-08

### Fixed
- PyPI README logo: use absolute PNG URL so the image renders on PyPI.

## [0.1.0] - 2025-08-01

### Added
- Initial release of OmegaOMG DSL v1.0 with AST evaluator, resolver pipeline, CLI, and demo.
