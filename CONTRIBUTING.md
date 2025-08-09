# Contributing to OmegaOMG

Thanks for your interest in contributing!

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
```

## Common tasks

- Run tests with coverage:
  ```powershell
  pytest --cov
  ```
- Lint:
  ```powershell
  ruff check .
  pylint dsl omg.py highlighter.py
  ```
- Type check:
  ```powershell
  mypy dsl
  ```

## Development notes

- Python 3.9+ is supported; CI runs a version matrix.
- Keep public behavior backwards compatible; add tests for new features.
- Rules must include at least one ListMatch; unbounded quantifiers (*, +) are disallowed.
- Entity resolution has deterministic tie‑breaking; see RESOLUTION.md before changing resolver logic.

## Pull requests

1. Create a feature branch.
2. Include focused tests for the change (happy‑path + 1 edge case).
3. Ensure `pytest`, `ruff`, `pylint`, and `mypy` are clean.
4. Open a PR; the CI workflow will run automatically.

## Reporting issues

Please include:
- Repro steps
- Expected vs actual behavior
- Relevant DSL snippets and environment details

Thanks for helping improve OmegaOMG!
