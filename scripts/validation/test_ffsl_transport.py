#!/usr/bin/env python3
"""Gate for the large-Courant Lin-Rood transport (ADV_SCHEME=ffsl, advection/ffsl_lr.py).

CPU, ~1 min:  JAX_PLATFORMS=cpu python3 scripts/validation/test_ffsl_transport.py

  1. |c| < 1: the FFSL sweep IS fct_lr._lr_sweep (bit-identical in f64; on a
     bounded row with the 'face' sliver measure, O(dphi^2) with 'cell'/'exact')
  2. any c (|c| up to 8, divergent, both periodic and bounded rows with the
     spherical metric): sum(a rho q) telescopes to the boundary fluxes to
     roundoff; a uniform mixing ratio stays uniform; rho q stays non-negative
  3. a bell translated at c = 3.7 per step in a periodic row: lands where the
     exact shift puts it, with the PPM accuracy of a c = 0.7 step
  4. the full 3-D driver on the model grid with random divergent winds at ONE
     6-hour sub-step and at K = 6: dM == f_top - f_bot to roundoff, uniform
     field preserved, positivity; hourly_winds accepted; return_rho
  5. Williamson case 1 (solid-body rotation over the poles) at 6 h steps, K = 6
     (Courant up to ~6 at the cap edge): L2 against the analytic solution at
     least as good as fct_lr at CFL 0.5, and far fewer sub-steps
"""
import os
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'src'))
import _paths                      # noqa: E402,F401
import jax                         # noqa: E402
import jax.numpy as jnp            # noqa: E402
import fct_lr                      # noqa: E402
import ffsl_lr                     # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_sl_transport import LAT, NLON, mesh, _random_3d   # noqa: E402
from fct_fast import grid_metric   # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def test_sweep_small_courant():
    print("1. |c| < 1: FFSL sweep == fct_lr._lr_sweep")
    rng = np.random.default_rng(0)
    n = 64
    rho = jnp.asarray(1 + 0.1 * rng.standard_normal((5, n)))
    rhoq = jnp.asarray(np.exp(rng.standard_normal((5, n))))
    for periodic in (True, False):
        # smooth c with |c| < 1 and a 1-D divergence < 1 per step (a cell cannot
        # lose more than its content), as the real winds satisfy
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        c = jnp.asarray(0.9 * np.sin(x[None] + 6 * rng.random((5, 1))))
        if periodic:
            mf = ac = None
        else:
            lat = np.linspace(-80, 80, n)
            _mf, _ac = grid_metric(lat); mf = jnp.asarray(_mf); ac = jnp.asarray(_ac)
            c = c * jnp.asarray(np.cos(np.deg2rad(lat)))[None]
        a1 = fct_lr._lr_sweep(rho, rhoq, c, periodic, mf=mf, ac=ac)
        a2 = ffsl_lr._ffsl_sweep(rho, rhoq, c, periodic, mf=mf, ac=ac)
        d = max(float(jnp.abs(a1[0] - a2[0]).max()), float(jnp.abs(a1[1] - a2[1]).max()))
        if periodic:
            check("periodic: bit-identical to fct_lr._lr_sweep", d < 1e-12, f"max |diff| {d:.1e}")
        else:
            # the fraction is measured in the donor cell's area, fct_lr's in the
            # face length: the same divergence to O(dphi^2) on the smooth AIR
            # field (the random tracer also sees the limiter flip at a few cells)
            # (this test row ends at +-80 with half-cells, where fct_lr's m/a = 2
            # end treatment and the donor-area measure legitimately differ; the
            # model's rows end at the poles inside the stirred caps)
            dr = float(jnp.abs(a1[0] - a2[0])[..., 2:-2].max())
            check("bounded (cell measure): interior air field equals fct_lr._lr_sweep to O(dphi^2)", dr < 1e-2, f"max |drho| {dr:.1e} (tracer {d:.1e})")
            # with the cell edges supplied, the 'face' measure IS fct_lr's
            # c * m_face convention (bit-identical), and the 'exact' sliver
            # area sits between the two (2026-11-05)
            edge = jnp.asarray(ffsl_lr._edges(lat)); dphi = float(np.deg2rad(lat[1] - lat[0]))
            a3 = ffsl_lr._ffsl_sweep(rho, rhoq, c, periodic, mf=mf, ac=ac, edge=edge, dphi=dphi, frac='face')
            d3 = max(float(jnp.abs(a1[0] - a3[0]).max()), float(jnp.abs(a1[1] - a3[1]).max()))
            check("bounded (face measure): bit-identical to fct_lr._lr_sweep", d3 < 1e-12, f"max |diff| {d3:.1e}")
            a4 = ffsl_lr._ffsl_sweep(rho, rhoq, c, periodic, mf=mf, ac=ac, edge=edge, dphi=dphi, frac='exact')
            d4 = float(jnp.abs(a1[0] - a4[0])[..., 2:-2].max())
            M1 = jnp.sum(ac * a1[1], -1); M4 = jnp.sum(ac * a4[1], -1); M0 = jnp.sum(ac * rhoq, -1)
            rel = float(jnp.abs(M4 / M0 - 1).max())
            check("bounded (exact measure): air field within O(dphi^2) of fct_lr, mass conserved", d4 < 1e-2 and rel < 1e-12, f"max |drho| {d4:.1e}, rel dM {rel:.1e}")


def test_sweep_large_courant():
    print("2. any Courant number: conservation, uniform field, positivity")
    rng = np.random.default_rng(1)
    n = 96
    for periodic in (True, False):
        rho = jnp.asarray(1 + 0.2 * rng.random((6, n)))
        q = jnp.asarray(np.exp(2 * rng.standard_normal((6, n))))
        q = q.at[1].set(2.5)                                           # a uniform tracer
        q = q.at[2].set(jnp.where(rng.random(n) < 0.4, 0.0, q[2]))     # a sparse one
        # smooth divergent Courant field up to +-8
        x = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # divergence < 1 per cell per step (a cell cannot be emptied twice), as
        # the hourly CESM winds satisfy by a wide margin (dt div <= 0.3)
        c = jnp.asarray(6.0 * np.sin(x[None] + rng.random((6, 1)) * 6) + 2.0 * np.cos(2 * x[None] + rng.random((6, 1))))
        if not periodic:
            # v -> 0 at the poles: no cell at a row end is drained with nothing
            # behind it (that case is guarded in the sweep, but its mixing
            # ratio is then arbitrary, so it is excluded from the uniform check)
            c = c * jnp.asarray(np.sin(np.linspace(0, np.pi, n)))[None]
        if periodic:
            mf = ac = None; a = jnp.ones(n)
        else:
            lat = np.linspace(-80, 80, n) * np.pi / 180
            mf = jnp.asarray(np.cos(lat + 0.5 * (lat[1] - lat[0]))); ac = jnp.asarray(np.cos(lat)); a = ac
        rho_n, rhoq_n = ffsl_lr._ffsl_sweep(rho, rho * q, c, periodic, mf=mf, ac=ac)
        M0 = jnp.sum(a * rho * q, -1); M1 = jnp.sum(a * rhoq_n, -1)
        if periodic:
            rel = float(jnp.abs(M1 / M0 - 1).max())
            check(f"periodic: sum(a rho q) conserved to roundoff at |c| up to 8", rel < 1e-12, f"max rel {rel:.1e}")
        else:
            # bounded: the far face is zeroed, the near face too (shift-left), so closed
            rel = float(jnp.abs(M1 / M0 - 1).max())
            check(f"bounded: sum(a rho q) conserved to roundoff at |c| up to 8", rel < 1e-12, f"max rel {rel:.1e}")
        qn = rhoq_n / rho_n
        du = float(jnp.abs(qn[1] - 2.5).max())
        check(f"periodic={periodic}: uniform mixing ratio stays uniform", du < 1e-10, f"max dev {du:.1e}")
        check(f"periodic={periodic}: rho' > 0 and rho q' >= 0", float(rho_n.min()) > 0 and float(rhoq_n.min()) >= -1e-12 * float(rhoq_n.max()),
              f"rho min {float(rho_n.min()):.3f}, rhoq min {float(rhoq_n.min()):.1e}")


def test_translation():
    print("3. bell translated at c = 3.7 per step (periodic row)")
    n = 128
    x = np.arange(n)
    bell = np.exp(-((x - 30) / 6.0) ** 2)
    rho = jnp.ones((1, n)); rhoq = jnp.asarray(bell[None])
    c = jnp.full((1, n), 3.7)
    for _ in range(10):
        rho, rhoq = ffsl_lr._ffsl_sweep(rho, rhoq, c, True)
    exact = np.exp(-((((x - 30 - 37) + n / 2) % n - n / 2) / 6.0) ** 2)
    out = np.asarray(rhoq[0] / rho[0])
    l2 = np.sqrt(np.mean((out - exact) ** 2)) / np.sqrt(np.mean(exact ** 2))
    # the same ten steps at c = 0.7 (the fractional part alone) measure the PPM diffusion
    rho2 = jnp.ones((1, n)); rhoq2 = jnp.asarray(bell[None]); c2 = jnp.full((1, n), 0.7)
    for _ in range(10):
        rho2, rhoq2 = ffsl_lr._ffsl_sweep(rho2, rhoq2, c2, True)
    exact2 = np.exp(-((x - 37) / 6.0) ** 2)
    l2b = np.sqrt(np.mean((np.asarray(rhoq2[0]) - exact2) ** 2)) / np.sqrt(np.mean(exact2 ** 2))
    check("lands at the exact shift with PPM accuracy (L2 < 3 %)", l2 < 0.03, f"L2 {l2:.4f} (c=0.7 alone: {l2b:.4f})")
    check("integer part is exact: L2(c=3.7) == L2(c=0.7) to 1e-3", abs(l2 - l2b) < 1e-3, f"{abs(l2 - l2b):.1e}")
    check("peak kept >= 0.95", out.max() >= 0.95, f"peak {out.max():.4f}")


def test_driver_3d():
    print("4. 3-D driver on the model grid, random divergent winds")
    rng = np.random.default_rng(3)
    nlev = 24
    u, v, w, q, plev, dp = _random_3d(rng, nlev)
    u = 3 * u                                            # Courant up to ~6 at the cap edge in 1 h
    qfroz = q * (1.0 + 0.5 * rng.random(q.shape))
    q[2] = 2.0; qfroz[2] = 2.0                           # uniform tracer with uniform reservoir
    _, ac = grid_metric(LAT, 80.0)
    wgt = dp[:, None, None] * ac[None, :, None]
    for K, split in ((6, 'strang'), (6, 'lie'), (12, 'strang')):
        out, n, vfl, rho = ffsl_lr.advect_step_batch(q, u, v, w, u, v, w, LAT, dp, qfroz, 21600.0,
                                                     lat_freeze=80.0, nsub=K, split=split,
                                                     return_vflux=True, return_rho=True)
        out = np.asarray(out); vfl = np.asarray(vfl); rho = np.asarray(rho)
        check(f"K={K} {split}: returns K sub-steps", n == K)
        dM = (out * wgt).sum(axis=(1, 2, 3)) - (q * wgt).sum(axis=(1, 2, 3))
        fl = (ac[None, :, None] * (vfl[:, 0] - vfl[:, 1])).sum(axis=(1, 2))
        rel = np.abs(dM - fl) / np.abs(q * wgt).sum(axis=(1, 2, 3))
        check(f"K={K} {split}: dM == f_top - f_bot to roundoff (f32 sweeps)", rel.max() < 1e-5, f"max {rel.max():.1e}")
        # uniform field: with rho_reset the returned q' = rho q, so the uniform
        # tracer is uniform up to rho - 1 (the pressure-fixer rescale, as in fct_lr)
        du = np.abs(out[2] / rho[None] - 2.0).max()
        check(f"K={K} {split}: uniform q stays uniform in rho q / rho (f32)", du < 1e-4, f"max dev {du:.1e}, rho in [{rho.min():.3f}, {rho.max():.3f}]")
        check(f"K={K} {split}: positivity", out.min() >= -1e-6 * out.max(), f"min {out.min():.2e}")
    # hourly winds: 7 snapshots == bracketing interpolation when they are collinear
    hours = [(u * (1 - t) + 1.2 * u * t, v, w) for t in np.linspace(0, 1, 7)]
    a, _ = ffsl_lr.advect_step_batch(q, u, v, w, 1.2 * u, v, w, LAT, dp, qfroz, 21600.0, lat_freeze=80.0, nsub=6)
    b, _ = ffsl_lr.advect_step_batch(q, u, v, w, 1.2 * u, v, w, LAT, dp, qfroz, 21600.0, lat_freeze=80.0, nsub=6,
                                     hourly_winds=hours)
    d = float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(a)).max())
    check("hourly_winds (collinear) == bracketing interpolation", d < 1e-5, f"max rel diff {d:.1e}")


def test_williamson():
    print("5. Williamson case 1 at 6 h steps (12 days, one revolution)")
    from fct_fast import RAD
    lon, lat = mesh()                                     # (nlat? ) -> see test_sl_transport.mesh
    nlev = 2
    dp = np.array([1000.0, 1000.0]); plev = np.array([5000.0, 6000.0])
    # a 13.7-day revolution: 0.876 cells per hourly sub-step at every latitude
    # (12 days would be exactly one cell per hour: an exact integer shift, and
    # the test would prove nothing about the fractional PPM part)
    T = 13.7 * 86400.0; U0 = 2 * np.pi * RAD / T
    NSTEP = 48; theta = 2 * np.pi * NSTEP * 21600.0 / T       # angle travelled
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])   # (3, nlon, nlat)
    p0 = np.array([0.0, 1.0, 0.0])                             # bell centre (lon pi/2, lat 0)
    Rb = RAD / 3
    def bell_at(th, d):                                        # centre moved by th along direction d
        pc = np.cos(th) * p0 + np.sin(th) * d
        r = RAD * np.arccos(np.clip(np.einsum('i,ijk->jk', pc, xyz), -1, 1))
        return np.where(r < Rb, 0.5 * (1 + np.cos(np.pi * r / Rb)), 0.0)
    for alpha_deg in (0.0, 90.0):
        d = np.array([-1.0, 0.0, 0.0]) if alpha_deg == 0 else np.array([0.0, 0.0, -1.0])
        bell = bell_at(0.0, d); bell_T = bell_at(theta, d)
        al = np.deg2rad(alpha_deg)
        u = U0 * (np.cos(lat) * np.cos(al) + np.sin(lat) * np.cos(lon) * np.sin(al))
        v = -U0 * np.sin(lon) * np.sin(al)
        # layout (nlev, nlat, nlon)
        u3 = np.broadcast_to(u.T[None], (nlev, LAT.size, NLON)).astype(np.float32)
        v3 = np.broadcast_to(v.T[None], (nlev, LAT.size, NLON)).astype(np.float32)
        z = np.zeros_like(u3)
        q0 = np.broadcast_to(bell.T[None, None], (1, nlev, LAT.size, NLON)).astype(np.float64)
        qf = jnp.asarray(q0)
        qa = q0; qb = q0
        t0 = time.time(); nsub_a = 0
        for _ in range(NSTEP):
            qa, n = ffsl_lr.advect_step_batch(qa, u3, v3, z, u3, v3, z, LAT, dp, qf, 21600.0, lat_freeze=80.0, nsub=6)
            nsub_a += n
        ta = time.time() - t0
        t0 = time.time(); nsub_b = 0
        for _ in range(NSTEP):
            qb, n = fct_lr.advect_step_batch(qb, u3, v3, z, u3, v3, z, LAT, dp, qf, 21600.0, lat_freeze=80.0,
                                             cfl=0.5, dtype=jnp.float32)
            nsub_b += n
        tb = time.time() - t0
        keep = np.abs(LAT) < 80
        ex = bell_T.T[keep]
        l2a = np.sqrt(np.mean((np.asarray(qa)[0, 0][keep] - ex) ** 2) / np.mean(ex ** 2))
        l2b = np.sqrt(np.mean((np.asarray(qb)[0, 0][keep] - ex) ** 2) / np.mean(ex ** 2))
        print(f"        alpha={alpha_deg:.0f}: ffsl L2 {l2a:.4f} peak {np.asarray(qa)[0, 0].max():.4f} ({nsub_a} sub-steps, {ta:.0f} s)"
              f"  |  lr L2 {l2b:.4f} peak {np.asarray(qb)[0, 0].max():.4f} ({nsub_b} sub-steps, {tb:.0f} s)")
        check(f"alpha={alpha_deg:.0f}: ffsl L2 <= lr L2 * 1.05", l2a <= 1.05 * l2b, f"{l2a:.4f} vs {l2b:.4f}")
        check(f"alpha={alpha_deg:.0f}: ffsl uses < 1/5 of lr's sub-steps", nsub_a * 5 < nsub_b, f"{nsub_a} vs {nsub_b}")


if __name__ == '__main__':
    t0 = time.time()
    test_sweep_small_courant()
    test_sweep_large_courant()
    test_translation()
    test_driver_3d()
    test_williamson()
    dt = time.time() - t0
    if FAILS:
        print(f"FAILED: {', '.join(FAILS)}  ({dt:.0f}s)"); sys.exit(1)
    print(f"ALL PASS  ({dt:.0f}s)")
