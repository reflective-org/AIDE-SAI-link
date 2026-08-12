# Contributing to aide_sai_core

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.
All contributors are expected to follow our [Code of Conduct](./CODE_OF_CONDUCT.md).

## Getting Started

1. Fork and clone the repository:

```bash
git clone https://github.com/reflective-org/aide_sai_core.git
cd aide_sai_core
```

2. Install
Follow the installation instructions in [the README](./README.md#installation).

## Development Workflow

1. Create a branch for your work using one of these prefixes:

```bash
git checkout -b feat/your-feature-name       # new feature
git checkout -b fix/your-bug-fix             # bug fix
git checkout -b docs/what-you-documented     # documentation only
git checkout -b test/what-you-tested         # adding/improving tests
git checkout -b refactor/what-you-refactored # restructure without behavior change
git checkout -b perf/what-you-optimized      # performance improvement
```

2. Make your changes and ensure they pass all tests.

3. Commit your changes with a clear message:

```bash
git commit -m "Add brief description of change"
```

4. Push and open a pull request against the `dev` branch.

## Code Style
- PEP 8 with 4-space indentation
- Type hints encouraged for public APIs
- No enforced formatter yet; please match the surrounding style

## Running Tests

**This model does not fail loudly.** A broken run still completes, stays finite,
and draws a plausible-looking figure — four transport bugs once cost a mass leak
that went unnoticed for months, with no crash and no warning. Run these before
you push:

```bash
python3 validation/test_conservation.py     # transport does not leak mass
python3 validation/test_physics_math.py     # settling math unchanged
```

Seconds each, no GPU and no CESM data needed, and non-zero exit on failure. CI
runs both on every push, so this is for getting the answer first. Run them
whenever you touch transport (`fast_advection/`) or `settling.py`.

The rest of `validation/` are investigations rather than pass/fail tests: they
load a real run and print diagnostics for a human to read, and need the CESM
archive and usually a checkpoint. See
[docs/VALIDATION.md](./docs/VALIDATION.md) for what each one asks, how to run
it, and the precision rule to follow if you change advection.

### Writing Tests

There is no pytest in this repo, and no `tests/` directory. Automated tests live
in `validation/` as standalone scripts following the pattern in
`test_conservation.py`:

- a module-level `check(name, ok, detail)` that prints `PASS`/`FAIL` and appends
  failures to a `FAILED` list;
- one function per group of related assertions;
- a `__main__` block that runs them and ends with `sys.exit(1 if FAILED else 0)`.

The exit code is what makes a script usable in CI, so it is not optional.

To be runnable in CI a test must be **self-contained**: inputs built from a
fixed seed (`np.random.default_rng(0)`), no GPU, no CESM archive, and no
`tomas-jax` / `jax-rrtmgp` import. Anything needing those is an investigation
rather than a test — write it as one, and say so in its docstring. If you add a
self-contained harness, add a step for it to `.github/workflows/ci.yml`.

## Reporting Issues

When reporting a bug, please include:

- Python version
- Package version
- Steps to reproduce the issue
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 license.
