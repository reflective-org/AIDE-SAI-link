# Validation

## Why bother

This model does not fail loudly. Its characteristic failure is a run that
completes, stays finite, and produces fields that look entirely reasonable while
being wrong.

That is not hypothetical. Two examples from this tree's own history:

- Four separate transport bugs added up to a **mass leak** that went unnoticed
  for months (fixed 2026-07-25): the aerosol burden fell faster than the run's
  own budgeted sinks could account for. Every one was silent — no crash, no NaN,
  no warning. The model just quietly lost aerosol.
- Two positivity-limiter bugs were **invisible in float64 and fatal in float32**.
  Testing at the module defaults would have found nothing; production runs in
  f32.

A figure cannot tell you about either. Both sides of any comparison you draw come
out of the same code, so a plot of a broken model looks like a plot of a working
one. That is what the checks here are for: they assert invariants the physics
must satisfy regardless of what the answer turns out to be — mass in equals mass
out, a droplet is bigger than its dry core, fall speed rises with size.

## Two kinds of thing in `validation/`

They share a folder but answer different questions.

### Automated tests — "did I break something?"

Self-contained: they build their inputs from a fixed random seed, need no GPU, no
CESM archive and no submodules, run in seconds, and **exit non-zero on
failure**. Safe to run anytime, and CI runs them on every push.

| file | what it asserts |
|---|---|
| `test_conservation.py` | Lin-Rood transport conserves tracer mass to roundoff for any wind field; the cos(φ) area metric, grid spacing, continuity-derived omega and polar caps all behave. Guards the four bugs behind a leak |
| `test_physics_math.py` | closed-form settling physics: Tang solution density, wet growth factor (1.087–2.566, cross-checked against the run header), slip-corrected fall speed, and that the implicit sweep changes the burden *only* by its bottom outflow |

```bash
python3 $REPO/validation/test_conservation.py     # ~12 s
python3 $REPO/validation/test_physics_math.py     # ~2 s
```

What they do **not** cover: microphysics, radiation, the coupled loop, and
anything that needs real meteorology. They are regression guards, not proof the
physics is right — several reference values come from the same papers the code
implements, so agreement means "still computes what it did," not "is correct."

### Investigations — "what is actually going on here?"

These load a real run and print diagnostics for a human to read. There is no
pass/fail because there is no fixed right answer. Run them when chasing a
specific question, not routinely.

| file | the question | needs |
|---|---|---|
| `validate_vpos_f32.py` | is the `ADV_VPOS` positivity limiter doing its job at production precision? | CESM archive + checkpoint |
| `floor_anatomy.py` | where does the number floor come from, and is it fatal for multi-year runs? | CESM archive + checkpoint |
| `validate_radiation.py` | how does our radiation compare against CESM's own output? | CESM archive + `jax-rrtmgp` + GPU. Writes a `.png` |
| `test_radiation.py` | does the radiation driver run on one real CESM hour without NaNs? | CESM archive + `jax-rrtmgp` + GPU |

`test_radiation.py` is named like an automated test but behaves like one of
these — it prints, it does not gate.

## Rules that matter

### Production runs in single precision — validate there

> [!IMPORTANT]
> **A change to advection confirmed only in float64 is not confirmed.** Two bugs
> in the vertical positivity limiter were invisible in f64 and fatal in f32.

Why single precision changes the answer, not just the speed:

The PPM remap in `fct_lr.py` is exactly *conservative* but not *positive* — it
can put −δ in one cell and +δ in its neighbour with the sum untouched.
`coupling.py` then clips those negatives to zero, which silently **creates**
particles (mass has a budgeted `floor` term; number's never had one). `ADV_VPOS`
is the limiter that stops it.

The negatives are rounding-scale, and that scale is the whole problem. In f64
they are ~1e-16 — indistinguishable from zero, so a broken limiter looks
perfectly healthy. In f32 they are ~1e-7, nine orders of magnitude larger, and
the same bug is fatal. A higher CFL compounds it: material moves further per
substep, gradients are steeper, and the limiter is under more stress.

**What to actually set.** In practice, nothing:

- `validate_vpos_f32.py` already defaults to `ADV_CFL=0.5` and `ADV_F32=1`, so
  prefixing those variables is a no-op. It is written for this test.
- Both production entry points pin the same values explicitly — `coupling.py`
  and `driver_fast.py` each wrap `advect_hour_batch` in a `functools.partial`
  with `cfl=0.5, dtype=float32`.

The one place the distinction bites is **calling `fct_lr.advect_hour_batch`
directly**, e.g. in a new test. Its signature defaults are `cfl=0.2,
dtype=float64`, which no run uses — so a bare call exercises a configuration the
model is never in, and passes for reasons that do not transfer.

Those signature defaults are deliberately left alone. Changing them would not
touch production (both callers override), and it would break
`test_conservation.py`: that harness asserts Lin-Rood conservation as a
*mathematical* property, which is only visible at f64. Measured 2026-08-12,
flipping the default to f32 moves its drop-in check from `+4.9e-16` to
`-9.9e-08` and the test fails — correctly, since that residual is f32 roundoff
rather than a scheme defect.

**Run from your runs directory, not the repo.** The harnesses that read a
checkpoint (`validate_vpos_f32.py`, `floor_anatomy.py`) take it from the working
directory by default, and `validate_radiation.py` writes its figure there. Point
them elsewhere with `STATE=/path/to/coupled_state_<TAG>_ckpt.npz`, or pass a path
as the first argument to `floor_anatomy.py`. They locate the repo from their own
file location, so invoking them by absolute path from anywhere works.

## What CI covers

CI runs the two automated tests plus a lint on every push. GitHub's runners have
no GPU and not the ~23 TB CESM archive, so **no investigation runs there** — the
badge means transport and settling math are intact, and nothing more. Everything
in the second table stays a manual, on-the-box activity.
