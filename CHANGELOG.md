# Change Log

## [0.1.0] - 2026-09-14

- Changed
  - Renamed the package from `bootstrapped` to `bootstrap2`
  - Migrated packaging from `setup.py` / `setup.cfg` to `pyproject.toml`
  - Limited supported Python versions to currently supported releases
    (3.10–3.13)
  - Raised the minimum versions of `numpy`, `scipy`, `pandas`, and `matplotlib`
    to the oldest releases that ship Python 3.10 wheels
  - Kept the original BSD-3-Clause license, adding a copyright line for the
    bootstrap2 maintainer, and declared it as an SPDX license expression in
    `pyproject.toml`
  - Added modern project scaffolding: `tox.ini`, `.pre-commit-config.yaml`,
    GitHub Actions CI (Linux / macOS / Windows), and this changelog
  - Rewrote all docstrings in NumPy style, formatted by `format-docstring` and
    checked by `pydoclint` (new `pydoclint` tox env, run in CI)
- Full diff
  - https://github.com/jsh9/bootstrap2/compare/bd19cae...0.1.0
