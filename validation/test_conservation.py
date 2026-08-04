"""Mass-conservation regression tests for the transport core (CPU, seconds).

These guard the four bugs that produced the coupling's long-standing ~-0.3%/day
mass leak (fixed 2026-07-25). All four were SILENT -- the model ran, stayed
finite, and produced plausible fields -- so they need explicit asserts:

  1. grid spacing: cx/cy must use the real cell width, not 1 degree
  2. y-sweep must carry the cos(phi) area metric
  3. omega must satisfy discrete continuity (else the advective-form update
     converts the residual straight into a mass leak)
  4. the polar caps must be stirred BEFORE the x-sweep (cx ~ 1/cos(lat)
     overflows int32 at the +-90 rows and destroys a non-uniform cap)

plus the Lin-Rood flux-form module (fct_lr.py), which must conserve tracer mass
to roundoff for ANY wind field and be a drop-in for fct_fast.

Run:  python3 validation/test_conservation.py   (no GPU, no CESM files needed)
"""
import sys
import os
import numpy as np

# The modules under test live in the REPO ROOT and in fast_advection/, one level
# up from this file -- inserting this file's own directory (what the line here
# did until 2026-07-30) put only validation/ on the path, so `import fct_fast`
# died before a single test ran. Python puts the SCRIPT's directory on sys.path,
# never the cwd, so launching from the repo root does not save it either.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'fast_advection'))
import jax.numpy as jnp
import fct_fast as F

RAD, DEG = F.RAD, F.DEG
LAT = np.linspace(-90.0, 90.0, 192)
NLON = 288
FAILED = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------- metric
def test_metric_identity():
    """The metric y-sweep must change sum(ac*q) by EXACTLY the advective-form
    divergence term sum(q*(C-Cl)) and nothing else -- for an arbitrary wind."""
    rng = np.random.default_rng(0)
    mf, ac = F.grid_metric(LAT)
    q0 = rng.random(LAT.size) + 0.1
    cy = 0.4 * (2 * rng.random(LAT.size) - 1)
    cR = cy.copy(); cR[-1] = 0.0
    C = cR * mf
    expect = float((q0 * (C - np.concatenate([[0.0], C[:-1]]))).sum())
    q = F._ppm_frac_step_nonper(jnp.asarray(q0[None, :]), jnp.asarray(cy[None, :]),
                               mf=jnp.asarray(mf), ac=jnp.asarray(ac))
    got = float((np.asarray(q)[0] * ac).sum() - (q0 * ac).sum())
    rel = abs(got - expect) / max(abs(expect), 1e-300)
    check('y-sweep conserves sum(ac*q) up to the divergence term', rel < 1e-12,
          f'rel residual {rel:.1e}')


def test_area_weight_sums_to_sphere():
    """ac must be the true cell-mean cos(phi): sum(ac)*dphi == 2 (hemisphere
    integral of cos), and strictly positive everywhere INCLUDING the +-90 rows.
    coupling.py's burden weight is built from the same formula, so a mismatch
    here means the mass budget cannot close."""
    _, ac = F.grid_metric(LAT)
    dphi = (LAT[1] - LAT[0]) * DEG
    check('sum(ac)*dphi == 2 (full sphere)', abs(ac.sum() * dphi - 2.0) < 1e-12,
          f'{ac.sum()*dphi:.15f}')
    check('ac > 0 at every row (incl. +-90)', (ac > 0).all(),
          f'min ac {ac.min():.3e} vs cos(lat[0])={np.cos(LAT[0]*DEG):.1e}')


def test_cap_internal_faces_closed():
    mf_o, _ = F.grid_metric(LAT)
    mf_c, _ = F.grid_metric(LAT, 80.0)
    pol = np.abs(LAT) > 80.0
    internal = pol[:-1] & pol[1:]
    check('grid_metric(lat_freeze) closes the caps internal y-faces',
          (mf_c[:-1][internal] == 0).all() and (mf_o[:-1][internal] != 0).any(),
          f'{internal.sum()} faces closed')


# ---------------------------------------------------------------- spacing
def test_grid_spacing():
    """A pure zonal wind must advect a feature by u*t. With the legacy 1-degree
    assumption on this 1.25-degree grid the displacement is 25% too large."""
    nlev = 1
    lat = LAT
    j = int(np.argmin(np.abs(lat - 0.0)))          # equator row
    u = np.zeros((nlev, lat.size, NLON)); u[:] = 20.0
    v = np.zeros_like(u); w = np.zeros_like(u)
    q0 = np.zeros((1, nlev, lat.size, NLON)); q0[0, 0, j, NLON // 2] = 1.0
    dp = np.array([1000.0])
    T = 6 * 3600.0
    out = {}
    for tag, dxfix in (('fixed', True), ('legacy', False)):
        q, _ = F.advect_hour_batch(jnp.asarray(q0), u, v, w, u, v, w, lat=lat,
                                   dp=dp, qfrozb=jnp.zeros_like(jnp.asarray(q0)),
                                   lat_freeze=80.0, cfl=0.2, dt_total=T,
                                   metric=True, dxfix=dxfix, polar_mode='zonal',
                                   wcont=False)
        row = np.asarray(q)[0, 0, j]
        idx = np.arange(NLON)
        shift = ((idx - NLON // 2 + NLON // 2) % NLON - NLON // 2)
        out[tag] = float((row * shift).sum() / row.sum())      # cells moved
    dlon_m = RAD * np.cos(lat[j] * DEG) * (2 * np.pi / NLON)
    want = 20.0 * T / dlon_m
    check('true grid spacing gives the right zonal displacement',
          abs(out['fixed'] - want) / want < 0.02,
          f"moved {out['fixed']:.2f} cells, expected {want:.2f} "
          f"(legacy {out['legacy']:.2f} = {out['legacy']/want:.2f}x)")


# ---------------------------------------------------------------- continuity
def test_continuity_omega_kills_divergence():
    """With omega rederived from continuity the discrete 3-D divergence -- the
    exact thing the advective-form update turns into a mass leak -- must vanish
    to roundoff in the interior."""
    rng = np.random.default_rng(3)
    nlev = 11
    lat = LAT
    dp = np.full(nlev, 700.0)
    shape = (nlev, lat.size, NLON)
    # smooth, non-trivial winds
    lo = np.linspace(0, 2 * np.pi, NLON, endpoint=False)
    u = 30 * np.sin(lo)[None, None, :] * np.cos(lat * DEG)[None, :, None] * np.ones((nlev, 1, 1))
    v = 10 * np.cos(2 * lo)[None, None, :] * np.cos(lat * DEG)[None, :, None] * np.ones((nlev, 1, 1))
    w = 0.01 * rng.standard_normal(shape)
    mf, ac = F.grid_metric(lat, 80.0)
    pol = np.abs(lat) > 80.0
    dlam = 2 * np.pi / NLON
    dphi = (lat[1] - lat[0]) * DEG

    uf = 0.5 * (u + np.roll(u, -1, 2))
    Sx = (uf - np.roll(uf, 1, 2)) / (RAD * np.cos(lat * DEG) * dlam)[None, :, None]
    vf = 0.5 * (v + np.roll(v, -1, 1))
    cyr = vf / (RAD * dphi); cyr[:, -1, :] = 0.0
    C = cyr * mf[None, :, None]
    Sy = (C - np.concatenate([np.zeros_like(C[:, :1]), C[:, :-1]], 1)) / ac[None, :, None]

    def resid(wf):
        D = Sx + Sy + (wf[1:] - wf[:-1]) / dp[:, None, None]
        return float(np.sqrt((D[:, ~pol] ** 2).mean()))

    wf_leg = np.concatenate([np.zeros((1,) + shape[1:]), 0.5 * (w[:-1] + w[1:]),
                             np.zeros((1,) + shape[1:])], 0)
    wf_new = np.asarray(F._omega_continuity(
        jnp.asarray(u), jnp.asarray(v), jnp.asarray(w), jnp.asarray(lat),
        jnp.asarray(dp), jnp.asarray(pol), jnp.asarray(mf), jnp.asarray(ac),
        dlam, dphi))
    r_leg, r_new = resid(wf_leg), resid(wf_new)
    check('continuity omega zeroes the interior divergence residual',
          r_new < 1e-18 and r_new < r_leg * 1e-6,
          f'{r_leg:.2e} -> {r_new:.2e} 1/s')


# ---------------------------------------------------------------- polar caps
def test_polar_cap_survives_x_sweep():
    """A non-uniform polar cap must NOT be destroyed by the x-sweep. cx reaches
    ~1e18 at the +-90 rows, so K=round(cx).astype(int32) overflows; the cap is
    only safe because it is stirred to zonal uniformity BEFORE the sweep."""
    rng = np.random.default_rng(7)
    nlev = 3
    lat = LAT
    dp = np.full(nlev, 700.0)
    q0 = 1.0 + 0.5 * rng.random((1, nlev, lat.size, NLON))    # non-uniform caps
    u = 30 * rng.standard_normal((nlev, lat.size, NLON))
    v = 5 * rng.standard_normal((nlev, lat.size, NLON))
    w = np.zeros((nlev, lat.size, NLON))
    _, ac = F.grid_metric(lat, 80.0)
    wgt = ac[None, :, None] * dp[:, None, None]
    pol = np.abs(lat) > 80.0

    q, _ = F.advect_hour_batch(jnp.asarray(q0), u, v, w, u, v, w, lat=lat, dp=dp,
                               qfrozb=jnp.asarray(q0), lat_freeze=80.0, cfl=0.5,
                               dt_total=6 * 3600.0, metric=True, dxfix=True,
                               polar_mode='zonal', wcont=False)
    qn = np.asarray(q)[0]
    grew = qn.max() / q0.max()
    check('polar caps do not blow up in the x-sweep',
          np.isfinite(qn).all() and grew < 1.5 and qn.min() > 0,
          f'max grew {grew:.2f}x, min {qn.min():+.2e}')

    # the cap mixing itself must be exactly mass conserving, per cap
    qm = np.asarray(F._mix_caps(jnp.asarray(q0[0]), jnp.asarray(pol),
                                jnp.asarray(lat), jnp.asarray(ac)))
    for nm, m in (('south', pol & (lat < 0)), ('north', pol & (lat > 0))):
        a = float((q0[0] * wgt * m[None, :, None]).sum())
        b = float((qm * wgt * m[None, :, None]).sum())
        check(f'cap stirring conserves mass ({nm})', abs(b - a) / a < 1e-13,
              f'rel {abs(b-a)/a:.1e}')
    # ...and must not move mass between the two caps
    a = float((q0[0] * wgt * (pol & (lat < 0))[None, :, None]).sum())
    b = float((qm * wgt * (pol & (lat < 0))[None, :, None]).sum())
    check('no mass teleports between the poles', abs(b - a) / a < 1e-13)


# ---------------------------------------------------------------- legacy path
def test_legacy_bit_identical():
    """The legacy code path must be untouched, so old runs stay reproducible."""
    try:
        import fct_core as B                    # repo root: put on sys.path above
    except Exception as e:                      # pragma: no cover
        check('legacy sweeps bit-identical to fct_core', False, f'import failed: {e}')
        return
    rng = np.random.default_rng(11)
    q = jnp.asarray(rng.random((7, 40)) + 0.1)
    cf = jnp.asarray(0.4 * (2 * rng.random((7, 40)) - 1))
    same_np = np.array_equal(np.asarray(F._ppm_frac_step_nonper(q, cf)),
                             np.asarray(B._ppm_frac_step_nonper(q, cf)))
    same_p = np.array_equal(np.asarray(F._ppm_frac_step_per(q, cf)),
                            np.asarray(B._ppm_frac_step_per(q, cf)))
    check('legacy sweeps bit-identical to fct_core', same_np and same_p)


# ---------------------------------------------------------------- Lin-Rood
def test_linrood_exact():
    """Flux form must conserve sum(ac*rho*q) to ROUNDOFF for an arbitrary --
    including wildly divergent -- wind, with no help from continuity. This is the
    property the advective form cannot have, and the reason fct_lr exists."""
    import fct_lr as L
    rng = np.random.default_rng(5)
    mf, ac = F.grid_metric(LAT, 80.0)
    n = LAT.size
    q0 = 1.0 + rng.random(n)
    phi = LAT * DEG
    # Two Courant fields, both genuinely divergent (so the advective form would
    # leak) but SPATIALLY COHERENT, as any real wind is. An uncorrelated
    # cell-by-cell random cf is not a wind: iterated, it drives the air mass rho
    # through ZERO, and q = rho*q/rho then loses precision catastrophically
    # (measured: rho in [-107, 103], drift 2e-9). The flux telescoping stays
    # algebraically exact there, but the scheme is meaningless with rho <= 0, so
    # that is outside its validity domain -- see the rho>0 assert below.
    for tag, cf in (('smooth, |cf|<0.35', 0.35 * np.sin(3 * phi)),
                    ('sheared, |cf|<0.8', 0.8 * np.sin(7 * phi) * np.cos(phi))):
        r = jnp.asarray(np.ones((1, n))); t = jnp.asarray((q0)[None].copy())
        m0 = float((np.asarray(t)[0] * ac).sum())
        for _ in range(50):
            r, t = L._lr_sweep(r, t, jnp.asarray(cf[None]), periodic=False,
                               mf=jnp.asarray(mf), ac=jnp.asarray(ac))
        m1 = float((np.asarray(t)[0] * ac).sum())
        check(f'flux-form y-sweep conserves sum(ac*rho*q) exactly ({tag})',
              abs(m1 - m0) / m0 < 1e-13, f'rel drift {abs(m1-m0)/m0:.1e}')
        # NB: rho itself is NOT expected to stay near 1 here. A steady divergent
        # 1-D sweep with no vertical motion to compensate piles up (or drains)
        # air indefinitely -- that is the correct air-mass response, not an error.
        # rho health is only meaningful in 3-D, where the continuity omega
        # compensates the horizontal divergence; it is asserted there.

    # consistency: a uniform mixing ratio must stay uniform (to roundoff)
    cf = 0.5 * (2 * rng.random(n) - 1)
    r = jnp.asarray(np.ones((1, n))); t = jnp.asarray(np.full((1, n), 3.0))
    for _ in range(20):
        r, t = L._lr_sweep(r, t, jnp.asarray(cf[None]), periodic=False,
                           mf=jnp.asarray(mf), ac=jnp.asarray(ac))
    q = np.asarray(t)[0] / np.asarray(r)[0]
    err = np.abs(q / 3.0 - 1.0).max()
    check('flux form preserves a uniform mixing ratio', err < 1e-11,
          f'max rel err {err:.1e} over 20 sweeps')


def test_linrood_dropin():
    """With rho_reset (the default) the CALLER's ordinary sum(A*q) burden must be
    conserved exactly, so fct_lr is a drop-in for fct_fast with no extra state."""
    import fct_lr as L
    rng = np.random.default_rng(9)
    nlev = 6
    lat = LAT
    dp = np.full(nlev, 700.0)
    _, ac = F.grid_metric(lat, 80.0)
    wgt = ac[None, :, None] * dp[:, None, None]
    q0 = (1.0 + rng.random((1, nlev, lat.size, NLON)))
    lo = np.linspace(0, 2 * np.pi, NLON, endpoint=False)
    u = 25 * np.sin(lo)[None, None, :] * np.cos(lat * DEG)[None, :, None] * np.ones((nlev, 1, 1))
    v = 8 * np.cos(2 * lo)[None, None, :] * np.cos(lat * DEG)[None, :, None] * np.ones((nlev, 1, 1))
    w = 0.005 * rng.standard_normal((nlev, lat.size, NLON))
    M0 = (q0[0] * wgt).sum()
    q = jnp.asarray(q0); ft = fb = 0.0
    for _ in range(2):
        q, ns, vf = L.advect_hour_batch(q, u, v, w, u, v, w, lat=lat, dp=dp,
                                        qfrozb=jnp.asarray(q0), lat_freeze=80.0,
                                        cfl=0.5, dt_total=6 * 3600.0,
                                        polar_mode='zonal', return_vflux=True)
        vfn = np.asarray(vf)[0]
        ft += (vfn[0] * ac[:, None]).sum(); fb += (vfn[1] * ac[:, None]).sum()
    qn = np.asarray(q)[0]
    resid = ((qn * wgt).sum() - M0) - (ft - fb)
    check('drop-in: sum(A*q) change == face flux only, to roundoff',
          abs(resid) / M0 < 1e-11 and np.isfinite(qn).all(),
          f'residual/M0 {resid/M0:+.1e}, nsub={ns}, q min {qn.min():+.2e}')

    # rho health in 3-D: with the continuity omega compensating the horizontal
    # divergence, the air mass must stay near 1. This is the number that says
    # whether flux form is trustworthy for a given wind source -- if a future
    # emulator wind drives rho toward 0, q = rho*q/rho loses precision.
    _, _, _, rho = L.advect_hour_batch(jnp.asarray(q0), u, v, w, u, v, w,
                                       lat=lat, dp=dp, qfrozb=jnp.asarray(q0),
                                       lat_freeze=80.0, cfl=0.5,
                                       dt_total=6 * 3600.0, polar_mode='zonal',
                                       return_vflux=True, return_rho=True,
                                       rho_reset=False)
    rn = np.asarray(rho)
    check('3-D air mass stays near 1 under continuity omega',
          rn.min() > 0.5 and rn.max() < 2.0,
          f'rho in [{rn.min():.4f}, {rn.max():.4f}], rms(rho-1) '
          f'{np.sqrt(((rn-1)**2).mean()):.2e}')


if __name__ == '__main__':
    print(__doc__.splitlines()[0])
    print(f"\ngrid {LAT.size} x {NLON}  "
          f"(dlat {LAT[1]-LAT[0]:.4f} deg, dlon {360/NLON:.4f} deg)\n")
    for fn in (test_metric_identity, test_area_weight_sums_to_sphere,
               test_cap_internal_faces_closed, test_grid_spacing,
               test_continuity_omega_kills_divergence,
               test_polar_cap_survives_x_sweep, test_legacy_bit_identical,
               test_linrood_exact, test_linrood_dropin):
        print(f"{fn.__name__}:")
        fn()
    print(f"\n{'ALL PASS' if not FAILED else 'FAILURES: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)
