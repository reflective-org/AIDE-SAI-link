"""Lin-Rood flux form at LARGE Courant number (ADV_SCHEME=ffsl, 2026-10-14).

WHY THIS EXISTS
---------------
fct_lr is the production transport and, on the budget quantities the model is
used for (burden, AOD, forcing), it is the converged reference: the split
semi-Lagrangian scheme (sl_lr) converges TO it at the open bottom face as its
sub-step count grows (docs/PROCESSES.md 1.8). Its only defect is cost: both
horizontal sweeps are limited to |Courant| < 1 and the 24 km longitude spacing
at the 80 deg cap edge forces 70-250 sub-steps per 6 h step, 6.6 s of a 21 s
emulator step.

This module is fct_lr with that limit removed -- Lin & Rood (1996) section 3,
the flux-form semi-Lagrangian (FFSL) extension that FV3 uses near its poles.
The flux through a face whose Courant number is c = N + f (integer N, fraction
f, both in the wind's direction) is the content of the N WHOLE upwind cells
plus the PPM sliver of fraction f of the next one:

    F_{i+1/2} = sum_{k=0}^{N-1} a_{i-k} X_{i-k}  +  f a_{i-N} <X>_{i-N}(f)        (c > 0)

for a cell content X (the air rho, or rho q) and a the cell area
(fct_fast.grid_metric; the fraction is measured in its own cell's area so a
whole-cell shift is exact on the sphere -- fct_lr's c m_face X convention is
the same divergence to O(dphi^2); the two sweeps are bit-identical on
periodic rows, where a = m = 1). Every flux still appears twice with opposite
signs, so sum(a rho q) telescopes to the boundary fluxes alone -- exactly, for
any Courant number -- and the tracer flux rides on the air flux, so a uniform
mixing ratio stays uniform. For |c| < 1 the whole-cell sum is empty and the
sweep IS fct_lr._lr_sweep. (The advective-form integer shift of fct_fast is a
permutation of cell contents and is not conservative; that is what the note in
fct_lr._lr_step_3d refers to. The flux form above is.) The vertical sweep,
fct_lr._lr_vert, is a conservative cell-integrated remap with no Courant
limit of its own, so the sub-step count is free: ADV_FFSL_K (default 24 since
2026-10-15, the production operating point; 6 is one per hourly CESM wind)
instead of 70-250. Fewer PPM remaps also means LESS
numerical diffusion than production fct_lr, not more; raising K is a plain
convergence knob here (each sub-step costs ~1/200 of a production lr step),
unlike the SL scheme where every extra gather smooths.

ORDER OF THE SWEEPS (ADV_FFSL_SPLIT)
------------------------------------
At 100 s sub-steps the x -> y -> z order of fct_lr is harmless. At one hour it
is not: the vertical sweep then sees the whole hour's horizontal convergence
(dt div ~ 0.3 at the 143 hPa face) on the air mass before it moves anything,
and the face flux is first order in the sub-step (the sl_lr Lie-rho A/B: +37 %
export at K = 6). `strang` (default) does z(dt/2) -> horizontal -> z(dt/2) and
alternates x,y / y,x between consecutive sub-steps, which makes both splitting
errors second order; `lie` is fct_lr's order, for the record. Everything else
-- the air mass carried in flux form and remapped back onto the fixed levels
at the end of the call (rho_reset), the open faces with the reservoir inflow,
the Zalesak limiter, the cap stirring, the continuity omega, the face-flux
diagnostic -- is fct_lr's, imported, not copied. fct_lr is not modified.

Zalesak at large Courant number: the limiter bounds the new value by the OLD
field's neighbours; with cells arriving from N cells upstream the old field at
the arrival point is irrelevant, so the bounds use the old field shifted by
the integer part of c (the donor neighbourhood; no shift for |c| < 1, so the
sweep stays fct_lr's there) and the low-order result.

THE CALLER'S CONTRACT (coupling.py)
-----------------------------------
Same signature as fct_lr.advect_step_batch; `cfl` is accepted and ignored.
`hourly_winds` (the list coupling.py builds under ADV_SL_HOURLY) is honoured:
sub-step k advects on the wind at its own midpoint, linearly interpolated
between the snapshots (exact with K = 6 hourly snapshots). Without it the
two bracketing snapshots are interpolated as fct_lr does. return_vflux and
return_rho as in fct_lr.

Env knobs (read once at import):
  ADV_FFSL_K      sub-steps per coupling step (default 24: within 0.2 % of lr on
                  the 30-day budgets; 6 = hourly carries an x/y splitting error)
  ADV_FFSL_SPLIT  strang | lie (default strang)
  ADV_FFSL_CMAX   largest |Courant| the whole-cell sum may span (default 64;
                  a guard, not a limit -- 64 cells is 90 deg of longitude)
"""
import os
import functools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from fct_fast import (RAD, DEG, grid_metric, _ppm_coeffs_per, _ppm_coeffs_nonper,
                      _zalesak, _mix_caps, _omega_continuity)
from fct_lr import _lr_vert

ADV_FFSL_K = int(os.environ.get('ADV_FFSL_K', '24'))
ADV_FFSL_SPLIT = os.environ.get('ADV_FFSL_SPLIT', 'strang').lower()
ADV_FFSL_CMAX = int(os.environ.get('ADV_FFSL_CMAX', '64'))
# How the FRACTIONAL part of the flux is measured on a bounded (latitude) row
# (2026-11-05). The whole cells carry their content a_k X_k exactly; the sliver
# of fraction f of the donor cell j needs an area, and the three choices differ
# by tan(phi) * dphi/2 ~ 4.5 % at the 80 deg cap edge:
#   cell   f * a_j            -- the 2026-10-14 convention: exact as f -> 1,
#                                 4.5 % high (poleward) / low (equatorward)
#                                 as f -> 0, so it does NOT converge to fct_lr
#                                 as K grows (the 10-day ladder of 2026-11-04:
#                                 polar-cap delivery +70 % at every K)
#   face   f * m_face          -- fct_lr's convention: exact as f -> 0, so the
#                                 K -> inf limit IS fct_lr; leaves a (1 - m/a)
#                                 residue in every cell a whole slab crosses
#   exact  area of the sliver  -- (sin(phi_face) - sin(phi_face -+ f dphi))/dphi:
#                                 exact at both limits, second order between,
#                                 whole-cell shifts stay exact. THE DEFAULT.
ADV_FFSL_FRAC = os.environ.get('ADV_FFSL_FRAC', 'exact').lower()
_FRACS = ('cell', 'face', 'exact')
if ADV_FFSL_FRAC not in _FRACS:
    raise ValueError(f"ADV_FFSL_FRAC={ADV_FFSL_FRAC!r}; expected one of {_FRACS}")
_SPLITS = ('strang', 'lie')
if ADV_FFSL_SPLIT not in _SPLITS:
    raise ValueError(f"ADV_FFSL_SPLIT={ADV_FFSL_SPLIT!r}; expected one of {_SPLITS}")


def describe():
    return (f"Lin-Rood flux form at large Courant number (ffsl_lr): K={ADV_FFSL_K} sub-steps, "
            f"FFSL PPM x/y sweeps + fct_lr vertical remap, {ADV_FFSL_SPLIT} ordering"
            + (" (z/2 - xy|yx - z/2)" if ADV_FFSL_SPLIT == 'strang' else " (x y z)")
            + f", Zalesak limiter, air mass carried, caps stirred, sliver={ADV_FFSL_FRAC}")


def _edges(lat):
    """Cell-edge latitudes [rad] of a uniform pole-to-pole grid: half-cells at
    the poles, midpoints between -- fct_fast.grid_metric's convention."""
    lat = np.asarray(lat, dtype=np.float64)
    edge = np.empty(lat.size + 1)
    edge[1:-1] = 0.5 * (lat[:-1] + lat[1:]) * DEG
    edge[0] = lat[0] * DEG; edge[-1] = lat[-1] * DEG
    return edge


# ---- the large-Courant sweep ------------------------------------------------
def _cum_at(S, idx, periodic, total):
    """S(idx) = sum of the row content up to and including cell idx, for idx
    outside [0, n): periodic rows add whole-row totals per wrap, bounded rows
    give 0 below the first cell and the total above the last."""
    n = S.shape[-1]
    if periodic:
        return jnp.take_along_axis(S, idx % n, -1) + total * (idx // n).astype(S.dtype)
    v = jnp.take_along_axis(S, jnp.clip(idx, 0, n - 1), -1)
    return jnp.where(idx < 0, jnp.zeros_like(v), v)


def _gather(A, j, periodic):
    """A at cell j (j may be outside the row: periodic wraps, bounded gives 0)."""
    n = A.shape[-1]
    if periodic:
        return jnp.take_along_axis(A, j % n, -1)
    v = jnp.take_along_axis(A, jnp.clip(j, 0, n - 1), -1)
    return jnp.where((j < 0) | (j >= n), jnp.zeros_like(v), v)


def _ffsl_sweep(rho, rhoq, c, periodic, mf=None, ac=None, cmax=ADV_FFSL_CMAX,
                edge=None, dphi=None, frac=None):
    """One flux-form Lin-Rood sweep along the last axis at ANY Courant number
    c (at the RIGHT faces). Returns (rho', rhoq'). Reduces to fct_lr._lr_sweep
    for |c| < 1 (the whole-cell sum is empty)."""
    n = rho.shape[-1]
    a = 1.0 if ac is None else ac
    if not periodic:
        c = c.at[..., -1].set(0.0)
    c = jnp.clip(c, -cmax, cmax)
    # mf is accepted for signature parity with fct_lr._lr_sweep and not used:
    # the fraction is measured in the donor cell's own area (below), and the
    # cap-internal faces that grid_metric(lat_freeze) closes (mf = 0) stay OPEN
    # here. Closing them would let a slab longer than one cell reach across a
    # closed face and drain the first cap row without refilling it. Open, the
    # cap rows are sheared by v(lon) and re-stirred after every sub-step; the
    # half-cell at the pole (ac = dphi/8) is safe because a cell can lose at
    # most |c| times its own content, never m/a = 4 times it.
    i = jnp.broadcast_to(jnp.arange(n, dtype=jnp.int32), c.shape)
    pos = c >= 0
    N = jnp.floor(jnp.abs(c)).astype(jnp.int32)
    f = jnp.abs(c) - N                                   # [0, 1)
    sgn = jnp.where(pos, 1.0, -1.0).astype(rho.dtype)
    j = jnp.where(pos, i - N, i + N + 1)                 # the fractional donor cell
    lo_idx = jnp.where(pos, i - N, i)                    # whole cells: (lo_idx, hi_idx]
    hi_idx = jnp.where(pos, i, i + N)

    def whole(X):                                        # sum of a X over the whole cells, f64
        S = jnp.cumsum((a * X).astype(jnp.float64), -1)
        tot = S[..., -1:]
        return (_cum_at(S, hi_idx, periodic, tot) - _cum_at(S, lo_idx, periodic, tot)).astype(X.dtype)

    def sliver(X):                                       # PPM mean of the fraction f of cell j
        qL, qR, dq, q6 = _ppm_coeffs_per(X) if periodic else _ppm_coeffs_nonper(X)
        g = lambda A: _gather(A, j, periodic)
        qL, qR, dq, q6, Xj = g(qL), g(qR), g(dq), g(q6), g(X)
        hi = jnp.where(pos, qR - 0.5 * f * (dq - q6 * (1 - (2 / 3) * f)),
                            qL + 0.5 * f * (dq + q6 * (1 - (2 / 3) * f)))
        return hi, Xj

    shl = ((lambda A: jnp.roll(A, 1, -1)) if periodic else
           (lambda A: jnp.concatenate([jnp.zeros_like(A[..., :1]), A[..., :-1]], -1)))

    # The fraction f of cell j carries f * a_j of its content, the same measure
    # as the whole cells (a_k X_k): on the sphere that is what keeps a whole-cell
    # shift exact (fct_lr's c * m_face * X convention for |c| < 1 is the same
    # divergence to O(dphi^2) but would leave a (1 - m/a) residue, ~2 % at 80
    # deg, in every cell a whole slab crosses). Periodic rows have a = m = 1.
    frac = ADV_FFSL_FRAC if frac is None else frac
    if ac is None or frac == 'cell' or edge is None:
        a_j = 1.0 if ac is None else _gather(jnp.broadcast_to(a, c.shape), j, periodic)
        f_a = f * a_j
    else:
        # edge: (n+1,) cell-edge latitudes [rad]; the sliver of cell j sits
        # against its RIGHT face (c > 0, edge[j+1]) or its LEFT face (c < 0,
        # edge[j]) and extends f*dphi into the cell (clipped to the cell for
        # the pole half-cells). Measured in the same units as ac
        # (= (sin(e_r) - sin(e_l)) / dphi, fct_fast.grid_metric).
        ej = jnp.clip(j, 0, n - 1)
        e_l = jnp.take(edge, ej); e_r = jnp.take(edge, ej + 1)
        if frac == 'face':
            f_a = f * jnp.where(pos, jnp.cos(e_r), jnp.cos(e_l))
        else:   # exact
            near = jnp.where(pos, e_r, e_l)
            far = jnp.where(pos, jnp.maximum(e_r - f * dphi, e_l), jnp.minimum(e_l + f * dphi, e_r))
            f_a = jnp.abs(jnp.sin(near) - jnp.sin(far)) / dphi
        f_a = jnp.where((j < 0) | (j >= n), 0.0, f_a).astype(rho.dtype)
    W = whole
    # air
    rhi, _ = sliver(rho)
    Fa_frac = f_a * rhi
    Fa = sgn * (W(rho) + Fa_frac)
    rho_n = rho - (Fa - shl(Fa)) / a
    # tracer: the whole cells carry their rho q, the sliver rides on the air sliver.
    # A cell drained by the previous sweep of this sub-step (rho = 0: a pole row
    # whose one open face had |c| >= 1; the caps are stirred afterwards) has no
    # mixing ratio; 0 keeps it out of the limiter bounds and out of NaN.
    q = jnp.where(rho > 1e-6, rhoq / jnp.maximum(rho, 1e-6), 0.0)
    qhi, qlo = sliver(q)
    Wq = W(rhoq)
    Ft_hi = sgn * (Wq + Fa_frac * qhi)
    Ft_lo = sgn * (Wq + Fa_frac * qlo)
    rhoq_lo = rhoq - (Ft_lo - shl(Ft_lo)) / a
    # a cell at a bounded row end with |c| >= 1 at its one open face is emptied
    # (rho' = 0 = rho q'); the mixing ratio there is arbitrary, not NaN
    rho_s = jnp.maximum(rho_n, 1e-6)
    jr = jnp.where(pos, i - N, i + N)                    # the donor neighbourhood (== i for |c| < 1)
    q_ref = _gather(q, jr, periodic)
    q_lo = jnp.where(rho_n > 1e-6, rhoq_lo / rho_s, q_ref)
    A = Ft_hi - Ft_lo
    if not periodic:   # a missing donor (beyond the row end) must not widen the bounds to 0
        q_ref = jnp.where((jr < 0) | (jr >= n), q, q_ref)
    dql = _zalesak(q_ref, q_lo, A, periodic=periodic, ac=a * rho_s)
    return rho_n, (q_lo + dql) * rho_n


# ---- one sub-step -----------------------------------------------------------
def _horizontal(rho, rhoq, u, v, dt, lat, polar, mf, ac, dlam, dphi, order, edge=None, frac=None):
    nlev, nlat, nlon = rhoq.shape
    def x(rho, rhoq):
        uf = 0.5 * (u + jnp.roll(u, -1, 2))
        cx = uf * dt / (RAD * jnp.cos(lat * DEG) * dlam)[None, :, None]
        cx = jnp.where(polar[None, :, None], 0.0, cx)
        sh = lambda A: A.reshape(nlev * nlat, nlon)
        r2, t2 = _ffsl_sweep(sh(rho), sh(rhoq), sh(cx), periodic=True)
        return r2.reshape(nlev, nlat, nlon), t2.reshape(nlev, nlat, nlon)
    def y(rho, rhoq):
        vf = 0.5 * (v + jnp.roll(v, -1, 1))
        cy = vf * dt / (RAD * dphi)
        tr = lambda A: A.transpose(0, 2, 1).reshape(nlev * nlon, nlat)
        r2, t2 = _ffsl_sweep(tr(rho), tr(rhoq), tr(cy), periodic=False, mf=mf, ac=ac,
                             edge=edge, dphi=dphi, frac=frac)
        back = lambda A: A.reshape(nlev, nlon, nlat).transpose(0, 2, 1)
        return back(r2), back(t2)
    for op in (order):
        rho, rhoq = (x if op == 'x' else y)(rho, rhoq)
    return rho, rhoq


def _vertical(rho, rhoq, dt, dp, w_face, qfroz):
    nlev, nlat, nlon = rhoq.shape
    col = lambda A: A.reshape(nlev, -1).T
    # _lr_vert extends the AIR outside the slab at the edge cell's density and
    # the tracer at the value given, so the inflow carries the reservoir
    # mixing ratio only if that value is rho_edge * q_res. fct_lr passes q_res
    # (a ~1 % inconsistency at its 100 s sub-steps, where rho stays within 1 %
    # of 1); at one-hour sub-steps rho_edge is off by several percent and a
    # uniform field would no longer be uniform at the edge levels.
    rho_c, rhoq_c, f_top, f_bot = _lr_vert(
        col(rho), col(rhoq), dp, dt, w_face.reshape(nlev + 1, -1).T,
        bc_top=(qfroz[0] * rho[0]).reshape(-1), bc_bot=(qfroz[-1] * rho[-1]).reshape(-1))
    return (rho_c.T.reshape(nlev, nlat, nlon), rhoq_c.T.reshape(nlev, nlat, nlon),
            jnp.stack([f_top.reshape(nlat, nlon), f_bot.reshape(nlat, nlon)]))


def _step_3d(rho, rhoq, u, v, dt, lat, dp, polar, qfroz, mf, ac, pol_mode, dlam, dphi,
             w_face, order, split, edge=None, frac=None):
    if pol_mode == 1:
        rho = _mix_caps(rho, polar, lat, ac)
        rhoq = _mix_caps(rhoq, polar, lat, ac)
    if split == 'strang':
        rho, rhoq, vf1 = _vertical(rho, rhoq, 0.5 * dt, dp, w_face, qfroz)
        rho, rhoq = _horizontal(rho, rhoq, u, v, dt, lat, polar, mf, ac, dlam, dphi, order, edge, frac)
        rho, rhoq, vf2 = _vertical(rho, rhoq, 0.5 * dt, dp, w_face, qfroz)
        vflux = vf1 + vf2
    else:
        rho, rhoq = _horizontal(rho, rhoq, u, v, dt, lat, polar, mf, ac, dlam, dphi, order, edge, frac)
        rho, rhoq, vflux = _vertical(rho, rhoq, dt, dp, w_face, qfroz)
    if pol_mode == 1:
        rho = _mix_caps(rho, polar, lat, ac)
        rhoq = _mix_caps(rhoq, polar, lat, ac)
    else:
        rhoq = jnp.where(polar[None, :, None], qfroz * rho, rhoq)
    return rho, rhoq, vflux


def _step_batch(rho, rhoqb, u, v, w, dt, lat, dp, polar, qfrozb, mf, ac, pol_mode,
                dlam, dphi, order, split, edge=None, frac=None):
    w_face = _omega_continuity(u, v, w, lat, dp, polar, mf, ac, dlam, dphi)
    one = lambda rq, qf: _step_3d(rho, rq, u, v, dt, lat, dp, polar, qf, mf, ac, pol_mode,
                                  dlam, dphi, w_face, order, split, edge, frac)
    rho_n, rhoq_n, vfl = jax.vmap(one, in_axes=(0, 0))(rhoqb, qfrozb)
    return rho_n[0], rhoq_n, vfl


@functools.partial(jax.jit, static_argnames=('K', 'pol_mode', 'split', 'frac'))
def _loop(rho, rhoqb, W, dt_sub, lat, dp, polar, qfrozb, mf, ac, dlam, dphi, edge, *, K, pol_mode, split, frac):
    """W (K, 3, nlev, nlat, nlon): [u, v, omega] at the sub-step midpoints."""
    def body(i, carry):
        rho_, rq_, vf_ = carry
        u, v, w = W[i, 0], W[i, 1], W[i, 2]
        args = (rho_, rq_, u, v, w)
        step = lambda order: (lambda a: _step_batch(*a, dt_sub, lat, dp, polar, qfrozb, mf, ac,
                                                    pol_mode, dlam, dphi, order, split, edge, frac))
        if split == 'strang':
            rho_, rq_, vfs = jax.lax.cond(i % 2 == 0, step('xy'), step('yx'), args)
        else:
            rho_, rq_, vfs = step('xy')(args)
        return rho_, rq_, vf_ + vfs
    vf0 = jnp.zeros((rhoqb.shape[0], 2) + rhoqb.shape[2:], rhoqb.dtype)
    return jax.lax.fori_loop(0, K, body, (rho, rhoqb, vf0))


def _midpoint_winds(hours, K, dtype):
    """hours: list of n >= 2 (u, v, w) snapshots equally spaced over the step.
    Returns (K, 3, nlev, nlat, nlon): the wind at each sub-step MIDPOINT,
    linearly interpolated in time (fct_lr's convention with n = 2)."""
    H = np.stack([np.stack(h) for h in hours]).astype(dtype, copy=False)
    n = len(hours)
    t_h = np.linspace(0.0, 1.0, n)
    t_k = (np.arange(K) + 0.5) / K
    idx = np.clip(np.searchsorted(t_h, t_k, side='right') - 1, 0, n - 2)
    a = ((t_k - t_h[idx]) / (t_h[idx + 1] - t_h[idx])).astype(dtype).reshape((K,) + (1,) * (H.ndim - 1))
    return (1 - a) * H[idx] + a * H[idx + 1]


def advect_step_batch(qb0, u0, v0, w0, u1, v1, w1, lat, dp, qfrozb, dt_total,
                      lat_freeze=80.0, cfl=None, dtype=jnp.float32, rho0=None,
                      polar_mode=None, return_vflux=False, return_rho=False, rho_reset=True,
                      hourly_winds=None, nsub=None, split=None, plev=None, frac=None):
    """fct_lr.advect_step_batch's contract at K = nsub (ADV_FFSL_K) sub-steps
    of any Courant number. `cfl` and `plev` are accepted and ignored."""
    K = int(ADV_FFSL_K if nsub is None else nsub)
    split = ADV_FFSL_SPLIT if split is None else split
    frac = ADV_FFSL_FRAC if frac is None else frac
    if frac not in _FRACS:
        raise ValueError(f"frac={frac!r}; expected one of {_FRACS}")
    if split not in _SPLITS:
        raise ValueError(f"split={split!r}; expected one of {_SPLITS}")
    lat_np = np.asarray(lat); dp_np = np.asarray(dp)
    nlon = np.asarray(qb0).shape[-1]
    dlam = 2.0 * np.pi / float(nlon)
    dphi = float(lat_np[1] - lat_np[0]) * DEG
    if polar_mode is None:
        polar_mode = os.environ.get('ADV_POLAR', 'zonal').lower()
    pol_mode = 1 if polar_mode == 'zonal' else 0
    dt_sub = float(dt_total) / K

    _mf, _ac = grid_metric(lat_np, lat_freeze if pol_mode == 1 else None)
    mf = jnp.asarray(_mf, dtype=dtype); ac = jnp.asarray(_ac, dtype=dtype)
    edge = jnp.asarray(_edges(lat_np), dtype=dtype)
    polar = jnp.asarray(np.abs(lat_np) > lat_freeze)
    lat_j = jnp.asarray(lat_np, dtype=dtype); dp_j = jnp.asarray(dp_np, dtype=dtype)
    qb = jnp.asarray(qb0, dtype=dtype)
    rho = (jnp.ones(qb.shape[1:], dtype) if rho0 is None else jnp.asarray(rho0, dtype=dtype))
    qfr = jnp.asarray(qfrozb, dtype=dtype)
    hours = hourly_winds if hourly_winds is not None else [(u0, v0, w0), (u1, v1, w1)]
    W = jnp.asarray(_midpoint_winds([tuple(np.asarray(x) for x in h) for h in hours], K,
                                    np.dtype(dtype).type))

    rho_n, rhoq_n, vfl = _loop(rho, qb * rho[None], W, dt_sub, lat_j, dp_j, polar, qfr, mf, ac,
                               dlam, dphi, edge, K=K, pol_mode=pol_mode, split=split, frac=frac)
    qb_n = (rhoq_n if rho_reset else rhoq_n / rho_n[None]).astype(jnp.float64)
    out = [qb_n, K]
    if return_vflux:
        out.append(vfl.astype(jnp.float64))
    if return_rho:
        out.append(rho_n.astype(jnp.float64))
    return tuple(out)
