"""Unit tests for the semi-Lagrangian transport (sl_core.py + sl_lr.py). CPU, seconds.

Gate 7.1 of docs/TRANSPORT_SL_HANDOFF.md, on the model's own 192 x 288 grid
WITH nodes at the poles (the case dinosaur refuses and sl_core patches):

  1. a constant field is returned unchanged to roundoff, including across the
     poles (sanity check of the halo/pole patch);
  2. Williamson et al. (1992) case 1 -- cosine bell, solid-body rotation,
     alpha = 0 (along the equator) and alpha = 90 deg (over both poles), one
     full revolution in 48 x 6 h steps: L2 error a few %, max <= 1, min >= 0
     (the quasi-monotone limiter creates no new extrema);
  3. after the mass fixer, |M1 - M0|/M0 < 1e-12 for a random divergent 3-D
     wind on the closed extended column, global and local fixers, float32
     gathers; and the local fixer keeps positivity;
  4. bin consistency: (N, M) advected together with M/N inside the bin bounds;
     count the violations before the coupler's two-moment clip;
  5. the coupling-facing driver: signature, hourly winds, polar reset, K;
  6. the SPLIT operator (ADV_SL_VERT=flux, the default): constant preserved
     through open faces, column mass == M1 + face fluxes to roundoff (the
     flux-form target), positivity, bin consistency, and with u = v = 0 it
     reproduces fct_lr's vertical remap (same operator, same omega).

Run:  python3 scripts/validation/test_sl_transport.py   (no GPU, no CESM files)
test_conservation.py is untouched (it gates fct_lr; this file gates sl_lr).
"""
import os
import sys
import time

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402,F401 -- puts src/ subdirs on sys.path
import jax                         # noqa: E402
import jax.numpy as jnp            # noqa: E402
import sl_core as S                # noqa: E402
import sl_lr                       # noqa: E402

RAD = 6.371e6
LAT = np.linspace(-90.0, 90.0, 192)
NLON = 288
LON = np.arange(NLON) * (360.0 / NLON)
GRID = S.SLGrid.from_latitudes(LAT, NLON, RAD)
FAILED = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def mesh():
    lon = np.deg2rad(LON)[:, None]
    lat = np.deg2rad(LAT)[None, :]
    return np.broadcast_to(lon, (NLON, LAT.size)), np.broadcast_to(lat, (NLON, LAT.size))


# ---------------------------------------------------------------- 0. grid
def test_grid_and_halo():
    print("grid / pole halo")
    check('pole rows detected', GRID.south_pole_node and GRID.north_pole_node)
    ext = S._latitude_halo_axis(GRID)
    check('halo axis strictly increasing', bool(np.all(np.diff(ext) > 0)),
          f"{np.rad2deg(ext[:3]).round(2)} ... {np.rad2deg(ext[-3:]).round(2)}")
    # the halo rows mirror rows 1,2 / -2,-3, so the extended axis has the
    # same spacing as the grid everywhere
    dphi = np.deg2rad(LAT[1] - LAT[0])
    check('halo spacing == grid spacing', np.allclose(np.diff(ext), dphi))
    # a field with f(lon, lat) = antipodal symmetry across the pole must give a
    # halo that continues it smoothly: f = sin(lat)*cos(lon) -> beyond the
    # north pole at (lon, pi - phi) = f(lon+pi, phi) = -cos(lon) sin(phi)
    lon, lat = mesh()
    f = np.sin(lat) * np.cos(lon)
    fe = np.asarray(S._extend_with_pole_halo(jnp.asarray(f), GRID))
    expect = np.sin(ext[None, :]) * np.cos(lon[:, :1]) * np.where(ext > np.pi / 2, 1, 1)
    # beyond the pole sin(pi - phi) = sin(phi) and the roll gives cos(lon+pi)
    expect[:, -2:] = -np.sin(np.pi - ext[None, -2:]) * np.cos(lon[:, :1])
    expect[:, :2] = -np.sin(-np.pi - ext[None, :2]) * np.cos(lon[:, :1])
    check('halo values are the antipodal continuation',
          np.allclose(fe, expect, atol=1e-12), f"max|diff| {np.abs(fe-expect).max():.2e}")


# ---------------------------------------------------------------- 1. constant
def test_constant_field():
    print("constant field (2-D over the poles, 3-D with random omega)")
    rng = np.random.default_rng(1)
    lon, lat = mesh()
    # solid-body rotation over the poles, big step: every row crosses a pole
    u0 = 2 * np.pi * RAD / (12 * 86400.0)
    u = (u0 * np.sin(lat) * np.cos(lon)).astype(np.float32)
    v = (-u0 * np.sin(lon)).astype(np.float32)
    dep = S.horizontal_departure_points(u, v, GRID, 6 * 3600.0)
    f = jnp.full((NLON, LAT.size), 3.0, jnp.float32)
    g = np.asarray(S.transport_scalar_2d(f, dep, GRID, 'cubic', 'quasi_monotone'))
    check('2-D cubic+limiter: constant preserved', np.abs(g - 3.0).max() < 1e-5,
          f"max|err| {np.abs(g-3.0).max():.2e}")
    g = np.asarray(S.transport_scalar_2d(f, dep, GRID, 'cubic', None))
    check('2-D cubic unlimited: constant preserved', np.abs(g - 3.0).max() < 1e-5,
          f"max|err| {np.abs(g-3.0).max():.2e}")
    # 3-D through the driver (f32 gathers, f64 carry): constant to f64 roundoff
    nlev = 24
    plev = 120.0 * 1.25 ** np.arange(nlev)                         # Pa, 1.2 -> 200 hPa-ish
    dp = np.gradient(plev)
    u3 = np.broadcast_to(u.T[None], (nlev, LAT.size, NLON)).copy()
    v3 = np.broadcast_to(v.T[None], (nlev, LAT.size, NLON)).copy()
    w3 = (0.02 * rng.standard_normal((nlev, LAT.size, NLON)) * plev[:, None, None] / 1e4).astype(np.float32)
    q = np.full((3, nlev, LAT.size, NLON), 2.5)
    for vert in ('sl', 'flux'):
        for fixer in ('none', 'global', 'local'):
            out, K = sl_lr.advect_step_batch(q, u3, v3, w3, u3, v3, w3, LAT, dp, q, 21600.0,
                                             lat_freeze=90.0, nsub=2, fixer=fixer, vert=vert,
                                             plev=plev)
            err = float(np.abs(np.asarray(out) - 2.5).max())
            # flux: fct_lr's vertical sweep integrates cumulative column mass
            # in float32 (production precision), ~3e-5 relative at the thin
            # top levels (eps * p_total/dp_k); the 3-D gather is exact
            tol = 1e-10 if vert == 'sl' else 2.5 * 2e-4
            check(f'driver vert={vert} fixer={fixer}: constant preserved', err < tol,
                  f"max|err| {err:.2e}")


# ---------------------------------------------------------------- 2. Williamson 1
def williamson1(alpha_deg, steps=48, dt=6 * 3600.0, order='cubic', limiter='quasi_monotone'):
    lon, lat = mesh()
    a = np.deg2rad(alpha_deg)
    u0 = 2 * np.pi * RAD / (12 * 86400.0)                 # one revolution in 12 days
    u = u0 * (np.cos(lat) * np.cos(a) + np.sin(lat) * np.cos(lon) * np.sin(a))
    v = -u0 * np.sin(lon) * np.sin(a)
    lon_c, lat_c = 1.5 * np.pi, 0.0
    r = RAD * np.arccos(np.clip(np.sin(lat_c) * np.sin(lat)
                                + np.cos(lat_c) * np.cos(lat) * np.cos(lon - lon_c), -1, 1))
    R = RAD / 3
    h = np.where(r < R, 0.5 * (1 + np.cos(np.pi * r / R)), 0.0)
    dep = S.horizontal_departure_points(jnp.asarray(u, jnp.float32), jnp.asarray(v, jnp.float32),
                                        GRID, dt, iterations=2)
    step = jax.jit(lambda f: S.transport_scalar_2d(f, dep, GRID, order, limiter))
    f = jnp.asarray(h, jnp.float32)
    assert abs(steps * dt - 12 * 86400.0) < 1e-6
    for _ in range(steps):
        f = step(f)
    f = np.asarray(f, np.float64)
    w = np.cos(lat)
    l2 = np.sqrt((w * (f - h) ** 2).sum() / (w * h ** 2).sum())
    return l2, f.max(), f.min()


def test_williamson():
    print("Williamson case 1 (cosine bell, one revolution, 48 x 6 h)")
    for alpha in (0.0, 90.0):
        t0 = time.time()
        l2, mx, mn = williamson1(alpha)
        check(f'alpha={alpha:g}: L2 < 10%', l2 < 0.10, f"L2 {l2:.3f}  ({time.time()-t0:.1f}s)")
        check(f'alpha={alpha:g}: no new maximum', mx <= 1.0 + 1e-6, f"max {mx:.6f}")
        check(f'alpha={alpha:g}: no new minimum', mn >= -1e-7, f"min {mn:.2e}")
    # the unlimited cubic is more accurate but overshoots -- documents what the
    # limiter costs and buys
    l2, mx, mn = williamson1(90.0, limiter=None)
    print(f"        (unlimited cubic, alpha=90: L2 {l2:.3f}, max {mx:.4f}, min {mn:.2e})")


# ---------------------------------------------------------------- 3. mass fixer
def _random_3d(rng, nlev):
    """Smooth, divergent, level-dependent 3-D winds and random tracers. The
    winds are large-scale (divergence ~1e-5/s, like the stratosphere), not
    per-cell noise: the continuity-derived face omega integrates the
    divergence down the column, and grid-scale noise of 5 m/s is a
    divergence of 1e-4/s that flushes the whole band in an hour."""
    lon, lat = mesh()
    plev = 120.0 * 1.25 ** np.arange(nlev)
    dp = np.gradient(plev)
    ph = np.linspace(0, np.pi, nlev)[:, None, None]
    LO, LA = lon.T[None], lat.T[None]                      # (1, nlat, nlon)
    # the wave parts carry cos(lat) so the vector field is smooth at the poles
    u = 40 * np.cos(LA) + 15 * np.sin(3 * LO + ph) * np.cos(2 * LA) * np.cos(LA)
    v = 12 * np.cos(3 * LO - ph) * np.cos(LA) ** 2
    # CESM-style center omega: ~0.3 dp/h, smooth, random sign by level
    w = 0.3 * dp[:, None, None] / 3600.0 * np.sin(2 * LO + ph) * np.cos(LA)
    q = np.exp(rng.standard_normal((4, nlev, LAT.size, NLON)))
    q[1] = np.where(rng.random(q[1].shape) < 0.3, 0.0, q[1])          # a sparse tracer with zeros
    q[2] = (np.sin(2 * lon) * np.cos(lat) ** 3).T[None] + 1.5          # smooth, positive
    return (u.astype(np.float32), v.astype(np.float32), w.astype(np.float32), q, plev, dp)


def test_mass_fixer():
    print("mass fixer on the closed extended column (random divergent 3-D wind)")
    from fct_fast import grid_metric
    rng = np.random.default_rng(2)
    nlev = 24
    u, v, w, q, plev, dp = _random_3d(rng, nlev)
    _, ac = grid_metric(LAT, None)
    wgt = dp[:, None, None] * ac[None, :, None]
    M0 = (q * wgt).sum(axis=(1, 2, 3))
    for fixer in ('none', 'global', 'local'):
        out, K = sl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, q, 21600.0,
                                         lat_freeze=90.0, nsub=6, fixer=fixer, vert='sl',
                                         plev=plev)
        out = np.asarray(out)
        M1 = (out * wgt).sum(axis=(1, 2, 3))
        rel = np.abs(M1 / M0 - 1)
        if fixer == 'none':
            print(f"        raw operator drift over 6 h, per tracer: {rel}")
            check('raw operator is finite and non-negative', np.isfinite(out).all() and out.min() >= 0)
        else:
            check(f'fixer={fixer}: |M1-M0|/M0 < 1e-12', rel.max() < 1e-12, f"max {rel.max():.1e}")
            check(f'fixer={fixer}: positivity', out.min() >= 0, f"min {out.min():.2e}")
        if fixer == 'local':
            # the correction is local: the smooth tracer should be nearly untouched
            raw, _ = sl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, q, 21600.0,
                                             lat_freeze=90.0, nsub=6, fixer='none', vert='sl',
                                             plev=plev)
            d = np.abs(out[2] - np.asarray(raw)[2]) / 1.5
            frac = (d > 1e-6).mean()
            print(f"        local fixer touched {100*frac:.2f}% of the smooth tracer's cells "
                  f"(max rel change {d.max():.1e})")


# ---------------------------------------------------------------- 4. bin consistency
def test_bin_consistency():
    print("bin consistency: (N, M) with M/N inside [mlo, mhi]")
    rng = np.random.default_rng(3)
    nlev = 12
    u, v, w, q, plev, dp = _random_3d(rng, nlev)
    mlo, mhi = 1.0, 2.0
    N = q[0]
    mbar = mlo + (mhi - mlo) * (0.5 + 0.45 * np.sin(3 * np.deg2rad(LON))[None, None, :]
                                * np.cos(np.deg2rad(LAT))[None, :, None])
    M = N * mbar
    stack = np.stack([N, M])
    for vert in ('sl', 'flux'):
      for fixer in ('global', 'local'):
        out, _ = sl_lr.advect_step_batch(stack, u, v, w, u, v, w, LAT, dp, stack, 21600.0,
                                         lat_freeze=90.0, nsub=6, fixer=fixer, vert=vert,
                                         plev=plev)
        out = np.asarray(out)
        n, m = out[0], out[1]
        ok = n > 1e-30
        ratio = np.where(ok, m / np.where(ok, n, 1.0), mlo)
        viol = ((ratio < mlo * (1 - 1e-9)) | (ratio > mhi * (1 + 1e-9))) & ok
        worst = max(float((mlo - ratio[viol]).max() / mlo) if viol.any() else 0.0,
                    float((ratio[viol] - mhi).max() / mhi) if viol.any() else 0.0)
        frac = viol.mean()
        check(f'vert={vert} fixer={fixer}: < 1% of cells outside the bin bounds', frac < 0.01,
              f"{int(viol.sum())} cells ({100*frac:.4f}%) violate, worst excursion {100*worst:.2f}%")


# ---------------------------------------------------------------- 5. driver contract
def test_driver_contract():
    print("coupling-facing driver")
    import inspect
    sig = inspect.signature(sl_lr.advect_step_batch)
    check('has return_vflux (ADV_VFLUX probe answers True under ADV_WCONT)',
          'return_vflux' in sig.parameters)
    check('accepts cfl (ignored) and dtype like fct_lr',
          'cfl' in sig.parameters and 'dtype' in sig.parameters)
    rng = np.random.default_rng(4)
    nlev = 6
    u, v, w, q, plev, dp = _random_3d(rng, nlev)
    qfroz = np.full_like(q, 7.0)
    hourly = [(u * (1 + 0.1 * k), v, w) for k in range(7)]
    out, K = sl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, qfroz, 21600.0,
                                     lat_freeze=80.0, cfl=0.5, hourly_winds=hourly, plev=plev)
    out = np.asarray(out)
    check('returns (qb, K) with K = ADV_SL_K', K == sl_lr.ADV_SL_K and out.shape == q.shape)
    check('output is float64', out.dtype == np.float64)
    pol = np.abs(LAT) > 80.0
    check('ADV_POLAR default (zonal): caps well mixed, not reset',
          not np.all(out[:, :, pol, :] == 7.0)
          and np.allclose(out[:, :, pol, :], out[:, :, pol, :].mean(axis=3, keepdims=True), rtol=1e-12))
    outf, _ = sl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, qfroz, 21600.0,
                                      lat_freeze=80.0, hourly_winds=hourly, plev=plev,
                                      polar_mode='freeze')
    outf = np.asarray(outf)
    check('polar_mode=freeze: polar rows overwritten from qfrozb', np.all(outf[:, :, pol, :] == 7.0))
    check('non-polar rows advected (not qfrozb)', not np.all(out[:, :, ~pol, :] == 7.0))
    # hourly winds with n = K+1 are used exactly: K=6 with 7 hourly snapshots
    # must equal passing the bracketing pair when the hourly series IS linear
    lin = [(u * (1 - a) + u * 1.6 * a, v, w) for a in np.linspace(0, 1, 7)]
    o1, _ = sl_lr.advect_step_batch(q, u, v, w, u * 1.6, v, w, LAT, dp, qfroz, 21600.0,
                                    lat_freeze=80.0, nsub=6, plev=plev)
    o2, _ = sl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, qfroz, 21600.0,
                                    lat_freeze=80.0, nsub=6, hourly_winds=lin, plev=plev)
    d = float(np.abs(np.asarray(o1) - np.asarray(o2)).max())
    # float32 gathers on O(50) fields: the two wind paths round differently
    check('linear hourly series == bracketing interpolation (f32)', d < 1e-3, f"max|diff| {d:.1e}")
    # resampling: 7 hourly snapshots onto K=3 sub-steps still runs
    o3, K3 = sl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, qfroz, 21600.0,
                                     lat_freeze=80.0, nsub=3, hourly_winds=hourly, plev=plev)
    check('hourly winds resampled to K=3', K3 == 3 and np.isfinite(np.asarray(o3)).all())


# ---------------------------------------------------------------- 6. the split operator
def test_split_operator():
    print("split operator (SL horizontal + fct_lr flux-form vertical, open faces)")
    import fct_lr
    from fct_fast import grid_metric
    rng = np.random.default_rng(5)
    nlev = 24
    u, v, w, q, plev, dp = _random_3d(rng, nlev)
    # a reservoir that differs from the band so inflow is visible
    qfroz = q * (1.0 + 0.5 * rng.random(q.shape))
    _, ac = grid_metric(LAT, None)
    wgt = dp[:, None, None] * ac[None, :, None]
    M0 = (q * wgt).sum(axis=(1, 2, 3))
    for fixer in ('global', 'local'):
        out, K, vfl = sl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, qfroz, 21600.0,
                                              lat_freeze=90.0, nsub=6, fixer=fixer, vert='flux',
                                              plev=plev, return_vflux=True)
        out = np.asarray(out); vfl = np.asarray(vfl)
        check(f'fixer={fixer}: vflux shape (ntr, 2, nlat, nlon)', vfl.shape == (q.shape[0], 2, LAT.size, NLON))
        M1 = (out * wgt).sum(axis=(1, 2, 3))
        face = ((vfl[:, 0] - vfl[:, 1]) * ac[None, :, None]).sum(axis=(1, 2))
        rel = np.abs((M1 - M0 - face) / M0)
        check(f'fixer={fixer}: dM == f_top - f_bot to roundoff', rel.max() < 1e-12, f"max {rel.max():.1e}")
        # fct_lr's positivity limiter leaves float32-roundoff negatives (the
        # coupling floors and budgets them, as under lr); nothing larger
        check(f'fixer={fixer}: positivity to f32 roundoff', out.min() >= -1e-6 * out.max(),
              f"min {out.min():.2e} (max {out.max():.2e})")
        print(f"        face exchange over 6 h: {np.round(face / M0, 4)} of M0")
    # with the production caps (|lat| > 80) stirred, the whole step still
    # conserves to the face fluxes: nothing is discarded at the poles
    out, _, vfl = sl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, qfroz, 21600.0,
                                          lat_freeze=80.0, nsub=6, fixer='global', vert='flux',
                                          plev=plev, return_vflux=True, polar_mode='zonal')
    out = np.asarray(out); vfl = np.asarray(vfl)
    M1 = (out * wgt).sum(axis=(1, 2, 3))
    face = ((vfl[:, 0] - vfl[:, 1]) * ac[None, :, None]).sum(axis=(1, 2))
    rel = np.abs((M1 - M0 - face) / M0)
    check('lat_freeze=80, zonal caps: dM == f_top - f_bot to roundoff', rel.max() < 1e-12, f"max {rel.max():.1e}")
    # u = v = 0: the split reduces to fct_lr's vertical remap on the same
    # continuity omega (zero divergence -> the top-face anchor in every column).
    # lat_freeze=89.5, not 90: fct_lr sizes its CFL sub-step on every kept row
    # and the pole row has cos(90) ~ 6e-17, i.e. millions of sub-steps. The two
    # pole rows are then caps (stirred by fct_lr, reset here) and are excluded.
    # split='lie' with the air mass frozen: the identity being tested is "no
    # horizontal step => exactly fct_lr's single full-dt sweep". Strang (the
    # default) takes two half sweeps with the air-mass bookkeeping between them
    # and this w is not the continuity omega of u = v = 0, so it cannot match.
    z = np.zeros_like(u)
    out, _, vfl = sl_lr.advect_step_batch(q, z, z, w, z, z, w, LAT, dp, qfroz, 21600.0,
                                          lat_freeze=89.5, nsub=1, fixer='global', vert='flux',
                                          plev=plev, return_vflux=True, split='lie', rho=False)
    ref, n, vref = fct_lr.advect_step_batch(q, z, z, w, z, z, w, LAT, dp, qfroz, 21600.0,
                                            lat_freeze=89.5, cfl=0.5, dtype=jnp.float32,
                                            return_vflux=True)
    keep = np.abs(LAT) < 89.5
    out = np.asarray(out)[:, :, keep]; ref = np.asarray(ref)[:, :, keep]
    d = np.abs(out - ref).max() / np.abs(ref).max()
    vfl = np.asarray(vfl)[:, :, keep]; vref = np.asarray(vref)[:, :, keep]
    dv = np.abs(vfl - vref).max() / (np.abs(vref).max() + 1e-300)
    # both run the sweep in float32; they round differently (advective form
    # divides by rho') at the ~1e-4 level of the thin-top-level noise above
    check('u=v=0: split == fct_lr vertical (f32)', d < 5e-4 and n == 1,
          f"max rel diff {d:.1e}, vflux {dv:.1e}, fct_lr nsub={n}")
    # a uniform field through open faces with a uniform reservoir stays uniform
    # EXACTLY (f32) with the air mass frozen at 1 (lie, rho=0: the advective
    # form divides it out); with the air mass carried (lie rho=1, strang) the
    # remainder is O((dt div)^2) and test_rho_carry checks it with that tolerance
    c = np.full_like(q, 3.0)
    out, _ = sl_lr.advect_step_batch(c, u, v, w, u, v, w, LAT, dp, c, 21600.0,
                                     lat_freeze=90.0, nsub=6, fixer='global', vert='flux', plev=plev,
                                     split='lie', rho=False)
    err = float(np.abs(np.asarray(out) - 3.0).max())
    check('uniform field + uniform reservoir stays uniform (f32, lie rho=0)', err < 3 * 2e-4, f"max|err| {err:.1e}")


# ---------------------------------------------------------------- 8. air mass through the horizontal step
def test_rho_carry():
    print("air mass through the horizontal step (ADV_SL_RHO): per-level fixer load under divergent wind")
    from fct_fast import grid_metric
    rng = np.random.default_rng(5)
    nlev = 24
    u, v, w, q, plev, dp = _random_3d(rng, nlev)
    # a plume-like blob (a field orthogonal to the divergence pattern, like the
    # zonal wave in _random_3d, would show no load at all)
    lon, lat = mesh()
    blob = np.exp(-(((lat - np.deg2rad(30)) / np.deg2rad(10)) ** 2
                    + (np.cos(lat) * (lon - np.pi) / np.deg2rad(30)) ** 2)) + 0.02
    q = np.broadcast_to(blob.T[None, None], (1, nlev, LAT.size, NLON))   # (1, nlev, nlat, nlon)
    dt = 3600.0
    mf, ac = grid_metric(LAT, None)
    sw = lambda a: jnp.swapaxes(jnp.asarray(a), -1, -2)                # (lev, lat, lon) -> (lev, lon, lat)
    pol = jnp.asarray(np.abs(LAT) >= 89.99)
    wf = sl_lr._omega_continuity(jnp.asarray(u), jnp.asarray(v), jnp.asarray(w), jnp.asarray(LAT, jnp.float32),
                                 jnp.asarray(dp, jnp.float32), pol, jnp.asarray(mf, jnp.float32),
                                 jnp.asarray(ac, jnp.float32), 2 * np.pi / NLON,
                                 float(LAT[1] - LAT[0]) * sl_lr.DEG)
    rho_h = sl_lr._rho_h(dt * sw(wf), dp)                              # (nlev, nlon, nlat)
    ac2 = jnp.asarray(ac, jnp.float64)[None, :]
    qs = sw(q).astype(jnp.float64)
    M0 = np.asarray(jnp.sum(ac2[None, None] * qs, axis=(2, 3)))[0]
    qH, _ = sl_lr._horizontal_substep(qs, sw(u).astype(jnp.float32), sw(v).astype(jnp.float32), GRID, dt, ac2,
                                      'cubic', 'quasi_monotone', 2, 'none', jnp.dtype('float32'))
    e_old = np.abs(np.asarray(jnp.sum(ac2[None, None] * qH, axis=(2, 3)))[0] / M0 - 1)
    e_new = np.abs(np.asarray(jnp.sum(ac2[None, None] * rho_h[None] * qH, axis=(2, 3)))[0] / M0 - 1)
    div = np.abs(np.asarray(rho_h) - 1).max()
    print(f"        max |dt div_h v| {div:.1e}; per-level fixer load: rho=1 {e_old.max():.1e}, "
          f"rho_h {e_new.max():.1e} (ratio {e_new.max() / e_old.max():.2f})")
    check('fixer load with rho=1 is well above roundoff (O(dt div))', e_old.max() > 1e-4, f"{e_old.max():.1e}, dt div {div:.1e}")
    check('fixer load with rho_h is much smaller (O(dt^2))', e_new.max() < 0.3 * e_old.max(), f"ratio {e_new.max() / e_old.max():.2f}")
    # driver contract with the air mass carried: faces, uniform field, positivity
    u, v, w, q, plev, dp = _random_3d(rng, nlev)
    qfroz = np.full_like(q, 2.0); q[0] = 2.0
    _, ac = grid_metric(LAT, None)
    out, n, vfl = sl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, qfroz, 21600.0, lat_freeze=80.0,
                                          nsub=6, fixer='global', vert='flux', plev=plev, rho=True,
                                          return_vflux=True)
    out = np.asarray(out); vfl = np.asarray(vfl)
    wgt = dp[:, None, None] * ac[None, :, None]
    dM = (out * wgt).sum(axis=(1, 2, 3)) - (q * wgt).sum(axis=(1, 2, 3))
    fl = (ac[None, :, None] * (vfl[:, 0] - vfl[:, 1])).sum(axis=(1, 2))
    rel = np.abs(dM - fl) / np.abs(q * wgt).sum(axis=(1, 2, 3))
    check('rho=1 carry: dM == f_top - f_bot to roundoff', rel.max() < 1e-12, f"max {rel.max():.1e}")
    # the remainder is second order, (dt div_h v)^2 -- the PPM remap of rho_h
    # against the flux-form balance -- and these synthetic winds carry 10x the
    # stratospheric divergence (dt div 2.7e-2 vs ~4e-3), so the floor scales
    # with it: 7e-4 here, ~1e-5 in production, under the f32 noise
    tol = 6e-4 + 2.0 * div ** 2
    check('rho=1 carry: uniform field stays uniform (to (dt div)^2)', np.abs(out[0] - 2.0).max() < tol * 2.0,
          f"max dev {np.abs(out[0] - 2.0).max() / 2.0:.1e} (tol {tol:.1e})")
    check('rho=1 carry: positivity to f32 roundoff', out.min() >= -1e-6 * out.max(), f"min {out.min():.2e}")
    # Strang V/2-H-V/2: same contract
    out, n, vfl = sl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, qfroz, 21600.0, lat_freeze=80.0,
                                          nsub=6, fixer='global', vert='flux', plev=plev, split='strang',
                                          return_vflux=True)
    out = np.asarray(out); vfl = np.asarray(vfl)
    dM = (out * wgt).sum(axis=(1, 2, 3)) - (q * wgt).sum(axis=(1, 2, 3))
    fl = (ac[None, :, None] * (vfl[:, 0] - vfl[:, 1])).sum(axis=(1, 2))
    rel = np.abs(dM - fl) / np.abs(q * wgt).sum(axis=(1, 2, 3))
    check('strang: dM == f_top - f_bot to roundoff', rel.max() < 1e-12, f"max {rel.max():.1e}")
    check('strang: uniform field stays uniform (to (dt div)^2)', np.abs(out[0] - 2.0).max() < tol * 2.0,
          f"max dev {np.abs(out[0] - 2.0).max() / 2.0:.1e} (tol {tol:.1e})")
    check('strang: positivity to f32 roundoff', out.min() >= -1e-6 * out.max(), f"min {out.min():.2e}")
    # the default path (Jacobian air mass + remainder sweep): the uniform
    # field is now exact up to the f32 sweep noise, not just (dt div)^2
    check('strang + jac: uniform field stays uniform to f32 roundoff', np.abs(out[0] - 2.0).max() < 1e-4 * 2.0,
          f"max dev {np.abs(out[0] - 2.0).max() / 2.0:.1e}")
    out_e, _, vfl_e = sl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, qfroz, 21600.0, lat_freeze=80.0,
                                              nsub=6, fixer='global', vert='flux', plev=plev, split='strang',
                                              jac=False, return_vflux=True)
    out_e = np.asarray(out_e); vfl_e = np.asarray(vfl_e)
    dM = (out_e * wgt).sum(axis=(1, 2, 3)) - (q * wgt).sum(axis=(1, 2, 3))
    fl = (ac[None, :, None] * (vfl_e[:, 0] - vfl_e[:, 1])).sum(axis=(1, 2))
    rel = np.abs(dM - fl) / np.abs(q * wgt).sum(axis=(1, 2, 3))
    check('strang, jac=False (Eulerian air mass): dM == f_top - f_bot to roundoff', rel.max() < 1e-12,
          f"max {rel.max():.1e}")


if __name__ == '__main__':
    print(f"jax {jax.__version__} on {jax.default_backend()}; grid {LAT.size} x {NLON} with pole rows")
    t0 = time.time()
    test_grid_and_halo()
    test_constant_field()
    test_williamson()
    test_mass_fixer()
    test_bin_consistency()
    test_driver_contract()
    test_split_operator()
    test_rho_carry()
    print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}  ({time.time()-t0:.0f}s)")
    sys.exit(1 if FAILED else 0)
