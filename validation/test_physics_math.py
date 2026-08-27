"""Regression tests for the closed-form physics in settling.py (CPU, seconds).

WHAT THESE ARE. Every check here is an invariant or a pinned reference value for
a PURE function -- no CESM archive, no checkpoint, no GPU, no tomas-jax. That is
the whole point: it makes them the only physics in this tree that CI can run.
They guard against a coefficient being retyped, a unit conversion inverting, or
the implicit sweep silently ceasing to conserve mass.

WHAT THESE ARE NOT. They do not validate the physics. The reference values come
from the same papers settling.py implements, so agreement means "the code still
computes what it did", not "the parameterization is right". Real validation
needs the GPU harnesses and a run: validate_vpos_f32.py, test_radiation.py.

equilibrium_wt_field() is deliberately NOT covered -- it imports
tomas_jax.fast.water, so it cannot run without the sibling repo, which defeats
the purpose of this file.

Run:  python3 validation/test_physics_math.py   (no GPU, no CESM files needed)
"""
import sys
import os
import numpy as np

# settling.py lives in the REPO ROOT, one level up from this file. Python puts
# the SCRIPT's directory on sys.path, never the cwd, so this insert is what makes
# `import settling` work regardless of where the harness is launched from.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402 -- puts src/ subdirs on sys.path
import jax.numpy as jnp
import settling as S

RHO_DRY = 1770.0                 # kg/m3, coupling.py's RHO_AER (pure sulfate)
FAILED = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def dry_diameter(m_kg, rho=RHO_DRY):
    """The dry-sphere diameter the wet growth factor is measured against."""
    return (m_kg / rho * (6.0 / np.pi)) ** (1.0 / 3.0)


# ------------------------------------------------------- solution density
def test_tang_density():
    """Tang (1997) polynomial: endpoints, monotonicity, physical range."""
    # x=0 is pure water and the polynomial's constant term is 0.9971 g/cm3, so
    # this pins the units (g/cm3 -> kg/m3) as much as the value. A factor-of-1000
    # slip here would otherwise sail through every ratio downstream.
    rho_w = float(S.tang_density(0.0))
    check('pure water endpoint is 997.1 kg/m3',
          abs(rho_w - 997.1) < 1e-6, f'got {rho_w:.4f}')

    x = np.linspace(0.0, 80.0, 81)
    rho = np.asarray(S.tang_density(x))
    # denser with more acid, everywhere -- the polynomial is a fit and a sign
    # error in any coefficient shows up as a turning point inside the range
    check('density increases monotonically over 0-80 wt%',
          bool(np.all(np.diff(rho) > 0)),
          f'min step {np.diff(rho).min():.3e} kg/m3 per wt%')

    # 80 wt% H2SO4 is ~1.73 g/cm3 in the literature; the fit gives 1716.5. A
    # wide band, because this is a units/typo guard, not a fit-quality check.
    rho80 = float(S.tang_density(80.0))
    check('80 wt% density is physical (1650-1800 kg/m3)',
          1650.0 < rho80 < 1800.0, f'got {rho80:.1f}')


# ------------------------------------------------------------- wet sizing
def test_wet_size():
    """Growth factor D_wet/D_dry against the range the model reports itself."""
    m = 1.0e-18                              # kg dry SO4/particle, mid-range bin

    # These two numbers are NOT invented for this test: run_prod.sh's own header
    # prints "15 wt% nodes 10-80%, D_wet/D_dry 1.087-2.566" at startup, so the
    # endpoints below are cross-checked against an independent artifact of the
    # production run. If a coefficient in wet_size or tang_density moves, these
    # are what catch it.
    for wt, want in ((10.0, 2.566), (80.0, 1.087)):
        dpw, rho = S.wet_size(m, wt, RHO_DRY)
        gf = float(dpw) / dry_diameter(m)
        check(f'growth factor at {wt:.0f} wt% is {want}',
              abs(gf - want) < 1e-3, f'got {gf:.4f}')

    # dilute solution -> bigger, lighter droplet; concentrated -> smaller, denser.
    # Size and density MUST move oppositely: settling_velocity multiplies them, so
    # if they ever moved together the fall speed would be doubly wrong.
    wts = np.linspace(10.0, 80.0, 15)
    dpw, rho = S.wet_size(m, wts, RHO_DRY)
    dpw = np.asarray(dpw); rho = np.asarray(rho)
    check('droplet shrinks as it concentrates',
          bool(np.all(np.diff(dpw) < 0)), f'D {dpw[0]*1e9:.1f} -> {dpw[-1]*1e9:.1f} nm')
    check('solution densifies as it concentrates',
          bool(np.all(np.diff(rho) > 0)), f'rho {rho[0]:.0f} -> {rho[-1]:.0f} kg/m3')

    # the wet droplet can never be smaller than its own dry core
    check('wet diameter always exceeds the dry core',
          bool(np.all(dpw > dry_diameter(m))),
          f'min GF {dpw.min()/dry_diameter(m):.3f}')


# ------------------------------------------------------ settling velocity
def test_settling_velocity():
    """Slip-corrected Stokes: sign, monotonicity, and the slip regime itself."""
    T = jnp.full((1, 1, 1), 210.0)           # K, mid-stratosphere
    P = jnp.full((1, 1, 1), 5000.0)          # Pa = 50 hPa, near the injection level
    dp = jnp.asarray(np.array([0.05, 0.1, 0.5, 1.0, 2.0]) * 1e-6)
    v = np.asarray(S.settling_velocity(dp, T, P, RHO_DRY))[:, 0, 0, 0]

    check('fall speed is positive (downward) in every bin',
          bool(np.all(v > 0)), f'min {v.min():.3e} m/s')
    check('fall speed increases with particle size',
          bool(np.all(np.diff(v) > 0)),
          f'{v[0]*1e3:.4f} -> {v[-1]*1e3:.4f} mm/s over 0.05-2 um')

    # A 1 um particle at 50 hPa falls ~21 m/day. Pinned as a regression value
    # with a generous band: it is the one place a broken viscosity fit, a wrong
    # mean-free-path, or a lost factor of g would show up as a NUMBER rather
    # than as a monotonicity break.
    v1 = float(np.asarray(S.settling_velocity(
        jnp.asarray([1.0e-6]), T, P, RHO_DRY))[0, 0, 0, 0])
    m_per_day = v1 * 86400.0
    check('1 um at 50 hPa falls 15-30 m/day',
          15.0 < m_per_day < 30.0, f'got {m_per_day:.1f} m/day ({v1*1e3:.4f} mm/s)')

    # THE STRATOSPHERIC POINT (module docstring): mean free path grows as pressure
    # falls, so Kn >> 1 and the slip correction dominates. The same particle must
    # fall FASTER higher up. Without Cc this ratio would be exactly 1.
    P_hi = jnp.full((1, 1, 1), 50000.0)      # 500 hPa
    v_lo = float(np.asarray(S.settling_velocity(
        jnp.asarray([1.0e-6]), T, P, RHO_DRY))[0, 0, 0, 0])
    v_hi = float(np.asarray(S.settling_velocity(
        jnp.asarray([1.0e-6]), T, P_hi, RHO_DRY))[0, 0, 0, 0])
    check('slip correction makes the same particle fall faster at lower pressure',
          v_lo > 2.0 * v_hi, f'{v_lo/v_hi:.2f}x faster at 50 hPa than 500 hPa')


# --------------------------------------------------------- the sweep itself
def _column(nbins=3, nlev=8, nlat=2, nlon=2, seed=0):
    rng = np.random.default_rng(seed)
    num = jnp.asarray(rng.random((nbins, nlev, nlat, nlon)) + 0.5)
    mas = jnp.asarray(rng.random((nbins, nlev, nlat, nlon)) * 1e-12 + 1e-13)
    # level 0 = band top; pressure and thickness both increase downward
    pres = jnp.asarray(np.broadcast_to(
        np.linspace(2000.0, 14000.0, nlev)[:, None, None], (nlev, nlat, nlon)))
    temp = jnp.full((nlev, nlat, nlon), 215.0)
    dp = jnp.asarray(np.full(nlev, 1500.0))
    return num, mas, temp, pres, dp


def test_settle_step_conserves():
    """The module's own contract: burden changes ONLY by the bottom outflow.

    settling.py's docstring states interior transfers move q*dp between adjacent
    levels exactly, so sum(q*dp) must change by the returned bottom-face outflow
    and nothing else, to float64 roundoff. That single identity is what lets
    coupling.py's budget line close -- the 'settle' stage is taken straight from
    this return value, so a leak here would be invisible in the budget and show
    up only as the model quietly losing mass.
    """
    num, mas, temp, pres, dp = _column()
    dpb = jnp.asarray(np.array([0.1, 0.5, 2.0]) * 1e-6)
    dt = 6 * 3600.0

    n2, m2, on, om = S.settle_step(num, mas, temp, pres, dp, dt, dpb, RHO_DRY)

    for label, q0, q1, out in (('number', num, n2, on), ('mass', mas, m2, om)):
        b0 = float(jnp.sum(q0 * dp[None, :, None, None]))
        b1 = float(jnp.sum(q1 * dp[None, :, None, None]))
        lost = float(jnp.sum(out))
        rel = abs((b0 - b1) - lost) / max(abs(b0), 1e-300)
        check(f'{label} burden changes only by the bottom outflow',
              rel < 1e-13, f'residual/burden {rel:.2e}')

    check('settling removes material rather than creating it',
          float(jnp.sum(on)) > 0 and float(jnp.sum(n2)) < float(jnp.sum(num)),
          f'outflow {float(jnp.sum(on)):.3e}')
    check('all outputs stay finite and non-negative',
          bool(jnp.all(jnp.isfinite(n2)) and jnp.all(jnp.isfinite(m2))
               and jnp.all(n2 >= 0) and jnp.all(m2 >= 0)))


def test_settle_step_top_is_sealed():
    """Nothing settles IN from above the band (zero-flux top face).

    With material only in the top level, after one step the column total must be
    unchanged except for what left the bottom -- and the top level must only
    ever lose. A sign error in the scan's flux direction would show up here as
    the top level gaining.
    """
    num, mas, temp, pres, dp = _column()
    num = num.at[:, 1:].set(0.0)             # everything in the topmost level
    mas = mas.at[:, 1:].set(0.0)
    dpb = jnp.asarray(np.array([0.1, 0.5, 2.0]) * 1e-6)

    n2, _, on, _ = S.settle_step(num, mas, temp, pres, dp, 6 * 3600.0, dpb, RHO_DRY)
    check('top level only loses material, never gains',
          bool(jnp.all(n2[:, 0] <= num[:, 0] + 1e-15)),
          f'max gain {float(jnp.max(n2[:, 0] - num[:, 0])):.3e}')
    check('material appears below, so the sweep moves it downward',
          float(jnp.sum(n2[:, 1:])) > 0)

    # dt=0 -> a no-op. Guards the degenerate branch a "settling off"
    # configuration relies on. NOT bit-exact, and should not be asserted as
    # such: the update still evaluates (q*dp)/dp, which is q only to within a
    # rounding of the product and another of the quotient (measured: 1 ulp,
    # rel 1.6e-16). The OUTFLOW is exactly zero, though -- it is dt*F, so the
    # multiply by zero is exact -- and that one is worth pinning hard.
    n3, m3, on3, om3 = S.settle_step(num, mas, temp, pres, dp, 0.0, dpb, RHO_DRY)
    rel = float(jnp.max(jnp.abs(n3 - num) / jnp.maximum(num, 1e-300)))
    check('dt=0 leaves the field unchanged to roundoff',
          rel < 1e-14, f'max rel change {rel:.2e}')
    check('dt=0 removes exactly nothing',
          float(jnp.sum(on3)) == 0.0 and float(jnp.sum(om3)) == 0.0)


if __name__ == '__main__':
    for fn in (test_tang_density, test_wet_size, test_settling_velocity,
               test_settle_step_conserves, test_settle_step_top_is_sealed):
        print(f"{fn.__name__}:")
        fn()
    print(f"\n{'ALL PASS' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)
