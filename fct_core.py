"""Flux-corrected PPM transport core (extracted from ../advection/fct.py).

Contains ONLY the pure, side-effect-free transport functions plus a batched
wrapper `advect_hour_batch` that advects a stack of tracer fields
(shape (ntracer, nlev, nlat, nlon)) with a single shared CESM wind field.

The single-tracer functions are copied verbatim from fct.py so the coupled
model reproduces the validated advection scheme exactly.

SPEED (2026-07-15): the batched hour driver was restructured to match the
device-loop idea in ../advection/fct_openbc_6hsub_fast.py. The whole n-substep
loop -- including the linear-in-time wind interpolation -- now runs on-device
inside ONE jitted lax.fori_loop, batched over the tracer axis. Winds are
uploaded once per coupling step (the two bracketing snapshots) instead of once
per substep, and n/nv are traced scalars so the driver compiles exactly once.
These are speed-only changes: the per-substep math is the SAME float64 ops as
before (verified bit-identical on a 17-level/80-tracer test), and the transport
core functions are still the verbatim fct.py routines.

NOT ported from the fast file: (a) its raw netCDF wind cache -- coupling.py does
its own I/O, so it is irrelevant here; (b) the XLA CUDA-graph command-buffer
flag -- that helped the fast file's tiny single-tracer micro-kernels (which were
kernel-LAUNCH bound), but coupling advects an 80-tracer vmap whose kernels are
large and compute-bound, so graph capture there only ADDED overhead (measured
~1.5x slower) and, being a process-global XLA setting, would also perturb the
coagulation kernels. So it is deliberately left off.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np

RAD = 6.371e6
DEG = np.pi / 180.0

##if periodic (lon)
def _ppm_coeffs_per(q):
    #roll forward to get the right neighbor, roll backward to get the left neighbor
    qm1 = jnp.roll(q,  1, -1); qp1 = jnp.roll(q, -1, -1); qp2 = jnp.roll(q, -2, -1)
    #CW84 eq. (1.9) for the right face value, then roll to get the left face value
    qR = (7/12)*(q+qp1) - (1/12)*(qp2+qm1)
    qL = jnp.roll(qR, 1, -1)
    #q6 is the curvature, CW84 eq. (1.6). It measures how far the cell mean q sits above or below the average of the two edges
    dq = qR-qL; q6 = 6*(q-0.5*(qL+qR))
    ##rules section (eqs. 1.10)
    #when q is not between qL and qR, this is true
    ext = (qR-q)*(q-qL) <= 0
    #where ext is true (local extremum), set qL and qR to q
    qL = jnp.where(ext, q, qL); qR = jnp.where(ext, q, qR)
    #the parabols overshoots the left or right cell, respectively (vertex outside the cell)
    cl = dq*q6 > dq*dq; cr = dq*q6 < -dq*dq
    #put the vertex at the cell boundary if it overshoots
    qL = jnp.where(~ext & cl, 3*q-2*qR, qL)
    qR = jnp.where(~ext & cr, 3*q-2*qL, qR)
    #get corrected edges with the vertex at the cell boundary if it overshoots
    dq = qR-qL; q6 = 6*(q-0.5*(qL+qR))
    return qL, qR, dq, q6

##if not periodic (lat/alt)
def _ppm_coeffs_nonper(q):
    #for non-periodic, we just repeat the edge values to get the neighbors...can't roll here so this is the best we can do 
    qm1 = jnp.concatenate([q[...,:1], q[...,:-1]], -1)
    qp1 = jnp.concatenate([q[...,1:], q[...,-1:]], -1)
    qp2 = jnp.concatenate([q[...,2:], q[...,-1:], q[...,-1:]], -1)
    qR = (7/12)*(q+qp1) - (1/12)*(qp2+qm1)
    qL = jnp.concatenate([qR[...,:1], qR[...,:-1]], -1)
    dq = qR-qL; q6 = 6*(q-0.5*(qL+qR))
    ext = (qR-q)*(q-qL) <= 0
    qL = jnp.where(ext, q, qL); qR = jnp.where(ext, q, qR)
    cl = dq*q6 > dq*dq; cr = dq*q6 < -dq*dq
    qL = jnp.where(~ext & cl, 3*q-2*qR, qL)
    qR = jnp.where(~ext & cr, 3*q-2*qL, qR)
    dq = qR-qL; q6 = 6*(q-0.5*(qL+qR))
    return qL, qR, dq, q6

##flux-corrected transport from Zalesak 1979...helps stability for multidimensional fluid problems
## A is at the cell face (right face = A, left face = A_l_)
##q_lo and q are cell-centered values
def _zalesak(q, q_lo, A, periodic):
    """Given old field q, low-order updated field q_lo, and antidiffusive flux A
    at each RIGHT face (per unit cell width), return the limited net update to add
    to q_lo. Bounds q_new to [neighbor_min, neighbor_max] over q and q_lo."""
    #get neighboring values
    if periodic:
        qm1 = jnp.roll(q, 1, -1);  qP1 = jnp.roll(q, -1, -1)
        lm1 = jnp.roll(q_lo, 1, -1); lP1 = jnp.roll(q_lo, -1, -1)
        A_l = jnp.roll(A, 1, -1)
    else:
        qm1 = jnp.concatenate([q[...,:1], q[...,:-1]], -1)
        qP1 = jnp.concatenate([q[...,1:], q[...,-1:]], -1)
        lm1 = jnp.concatenate([q_lo[...,:1], q_lo[...,:-1]], -1)
        lP1 = jnp.concatenate([q_lo[...,1:], q_lo[...,-1:]], -1)
        A_l = jnp.concatenate([jnp.zeros_like(A[...,:1]), A[...,:-1]], -1)

    #get the max over each cell and its neighbors for higher and lower order
    #eq 17/18
    #qmax = max(q, qm1, qP1, q_lo, lm1, lP1)
    #qmin = min(q, qm1, qP1, q_lo, lm1, lP1)
    qmax = jnp.maximum(jnp.maximum(jnp.maximum(q, qm1), qP1),
                       jnp.maximum(jnp.maximum(q_lo, lm1), lP1))
    qmin = jnp.minimum(jnp.minimum(jnp.minimum(q, qm1), qP1),
                       jnp.minimum(jnp.minimum(q_lo, lm1), lP1))
    #total antidiffusive flux in/out of each cell (depending on wind direction)
    #eq 7
    P_plus  = jnp.maximum(0.0, A_l) - jnp.minimum(0.0, A)
    P_minus = jnp.maximum(0.0, A)   - jnp.minimum(0.0, A_l)
    #how much rise/fall is allowed in each cell (based on the max/min of neighbors)
    #eq 8
    Q_plus  = qmax - q_lo
    Q_minus = q_lo - qmin

    eps = 1e-300
    #eq 9
    R_plus  = jnp.where(P_plus  > 0.0, jnp.minimum(1.0, Q_plus  / (P_plus  + eps)), 0.0)
    R_minus = jnp.where(P_minus > 0.0, jnp.minimum(1.0, Q_minus / (P_minus + eps)), 0.0)

    if periodic:
        R_plus_p1  = jnp.roll(R_plus,  -1, -1)
        R_minus_p1 = jnp.roll(R_minus, -1, -1)
    else:
        R_plus_p1  = jnp.concatenate([R_plus[...,1:],  R_plus[...,-1:]],  -1)
        R_minus_p1 = jnp.concatenate([R_minus[...,1:], R_minus[...,-1:]], -1)
    #eq 13
    C_face = jnp.where(A >= 0,
                       jnp.minimum(R_plus_p1, R_minus),
                       jnp.minimum(R_plus,    R_minus_p1))
    #step 5
    A_lim = C_face * A
    #get the left face now
    if periodic:
        A_lim_l = jnp.roll(A_lim, 1, -1)
    else:
        A_lim_l = jnp.concatenate([jnp.zeros_like(A_lim[...,:1]), A_lim[...,:-1]], -1)
    #return is step 6..delta_x is bake into A
    return -(A_lim - A_lim_l)

#periodic
def _ppm_frac_step_per(q, cf):
    qL, qR, dq, q6 = _ppm_coeffs_per(q)
    ##flux for positive wind (traveling from our grid cell to next) (average flux in the grid cell)
    ##CW eq. 1.12
    f_pos = qR - 0.5*cf*(dq - q6*(1-(2/3)*cf))
    #getting the flux for negative wind...need right neighbor's distribution for flow coming from neighboring cell into ours
    qLp1 = jnp.roll(qL, -1, -1); dqp1 = jnp.roll(dq, -1, -1); q6p1 = jnp.roll(q6, -1, -1)
    #when cf < 0, material crossing the face froms from the right neighbor, so right face value comes from right neighbor
    qp1  = jnp.roll(q, -1, -1)
    #cf is the courant number at the cell face, for negative flow cf is negative so a=-cf makes it positive
    a = -cf
    #when cf is postitive, the flow is from left to right, so we use f_pos. When cf is negative, the flow is from right to left, so we use f_neg.
    #f_pos/f_neg is the averge tracer value and cf is the fraction of cell that crosses the face
    #this flux is the actual amount crossing the face (equations 1.13 a_bar)
    f_neg = qLp1 + 0.5*a*(dqp1 + q6p1*(1-(2/3)*a))
    #this flux is the extension term in 1.13...sort flux by wind direction (positive or negative) and multiply by the courant number to get the actual flux at right face
    flux_hi = jnp.where(cf >= 0, cf*f_pos, cf*f_neg)
    #low order flux (smears out cell gradients but it is stable)
    flux_lo = jnp.where(cf >= 0, cf*q,     cf*qp1)

    T_hi = flux_hi; T_lo = flux_lo; C = cf
    Tlo_l = jnp.roll(T_lo, 1, -1); Cl = jnp.roll(C, 1, -1)
    #wind divergence (needed mass consistency)
    div  = q*(C - Cl)
    #low-order wind update (net flux out of right face minus net flux out of left face) plus divergence term
    q_lo = q - (T_lo - Tlo_l) + div
    # antidiffusive flux (equation #3 from Zalesak 1979)
    A = T_hi - T_lo
    q_new = q_lo + _zalesak(q, q_lo, A, periodic=True)
    return q_new


#nonperiodic
def _ppm_frac_step_nonper(q, cf, dx=None):
    qL, qR, dq, q6 = _ppm_coeffs_nonper(q)
    f_pos = qR - 0.5*cf*(dq - q6*(1-(2/3)*cf))
    qLp1 = jnp.concatenate([qL[...,1:], qL[...,-1:]], -1)
    dqp1 = jnp.concatenate([dq[...,1:], dq[...,-1:]], -1)
    q6p1 = jnp.concatenate([q6[...,1:], q6[...,-1:]], -1)
    qp1  = jnp.concatenate([q[...,1:],  q[...,-1:]],  -1)
    a = -cf
    f_neg = qLp1 + 0.5*a*(dqp1 + q6p1*(1-(2/3)*a))
    flux_hi = jnp.where(cf >= 0, cf*f_pos, cf*f_neg)
    flux_lo = jnp.where(cf >= 0, cf*q,     cf*qp1)

    flux_hi = flux_hi.at[..., -1].set(0.0)
    flux_lo = flux_lo.at[..., -1].set(0.0)
    cR      = cf.at[..., -1].set(0.0)

    # useful for non-uniform vertical, not needed for lat/lon (equal spacing)..dx is different at different faces
    w = 1.0 if dx is None else dx
    T_hi = flux_hi*w; T_lo = flux_lo*w; C = cR*w
    #zeros the left boundary with this code, the np.zeros just applies a single zero and everything else is fed in from the right faces above
    #flux at left face carries the width of the neighboring cell, so using dx is a necessary normalization to get the flux
    Tlo_l = jnp.concatenate([jnp.zeros_like(T_lo[...,:1]), T_lo[...,:-1]], -1)
    Cl    = jnp.concatenate([jnp.zeros_like(C[...,:1]),    C[...,:-1]],    -1)

    div  = q*(C - Cl)/w
    q_lo = q - (T_lo - Tlo_l)/w + div
    ## A at the right face
    A = (T_hi - T_lo)/w
    q_new = q_lo + _zalesak(q, q_lo, A, periodic=False)
    return q_new

##for longitude only
def _ffsl_x(q, c):
    #split the courant number into interger parts (K) and fractional remainder (cf)
    #rows: (nlev*nlat, 1)    # all non-sweep dims flattened
    #cols: (1, N)            # N = nlon (sweep axis length)
    N = q.shape[-1]
    K = jnp.round(c).astype(jnp.int32)
    cf = c - K
    cols = jnp.arange(N)[None, :]
    #shift q by K cells in the direction of flow (positive or negative) and wrap around for periodicity
    idx = (cols - K) % N
    q = jnp.take_along_axis(q, idx, axis=-1)
    return _ppm_frac_step_per(q, cf)

## key routine where q is advected in 3D (lon, lat, alt) with sub-stepping in the vertical...q gets advected one call at a time (lon,lat, and lastly altitude)
def _step_3d(q, u, v, w, dt, nv, lat, dp, polar, qfroz):
    """Advance ONE tracer field q(nlev,nlat,nlon) by one sub-step."""
    nlev, nlat, nlon = q.shape
    # average u to the cell right face (uf) and compute the courant number cx for the longitude sweep (convert to meteres)
    # expand shape for cx to (1,nlat,1) for broadcasting as cx is used in ffsl_x which expects (nlev*nlat,nlon)
    uf = 0.5*(u + jnp.roll(u, -1, 2))
    cx = uf * dt / (RAD * jnp.cos(lat*DEG) * DEG)[None, :, None]
    #reshape q and cx to 2D for the longitude sweep, then reshape back to 3D after the sweep
    q  = _ffsl_x(q.reshape(nlev*nlat, nlon),
                 cx.reshape(nlev*nlat, nlon)).reshape(nlev, nlat, nlon)
    #average v to the cell front/back face (vf) and compute the courant number cy for the latitude sweep (convert to meters)
    vf = 0.5*(v + jnp.roll(v, -1, 1))
    cy = vf * dt / (RAD * DEG)
    #reshape as above..need to transpose to get the correct shape for the latitude sweep, switch lat and lon ordering
    qT  = q.transpose(0,2,1).reshape(nlev*nlon, nlat)
    cyT = cy.transpose(0,2,1).reshape(nlev*nlon, nlat)
    #call ppm with periodicity False for the latitude sweep, then reshape back to 3D and transpose back to original ordering
    q   = _ppm_frac_step_nonper(qT, cyT).reshape(nlev, nlon, nlat).transpose(0,2,1)
    #q.reshape(nlev,-1) → (nlev, nlat*nlon). Then .T → (nlat*nlon, nlev)
    qc = q.reshape(nlev, -1).T
    wc = w.reshape(nlev, -1).T
    #divide timestep by nv, cz is the vertical courant number for each level (broadcasted to all grid points)
    #need dv as model levels are not evenly spaced
    dtv = dt / nv
    cz  = wc * dtv / dp[None, :]
    #run the sub-stepping, cz incorporates the substepping, dp array shape = [nlat*nlon, nlev]
    def body(_, qc_):
        return _ppm_frac_step_nonper(qc_, cz, dx=dp[None, :])
    qc = jax.lax.fori_loop(0, nv, body, qc)
    #reshape back to 3D
    q = qc.T.reshape(nlev, nlat, nlon)
    #for latitudes above 80 deg, reset cells to initial state as small polar cells can cause numerical issues with the scheme. 
    #This is a simple fix to avoid instability in the polar regions for cx ~ 1/cos(lat) (at high latitudes blows up towards infinity))
    q = jnp.where(polar[None, :, None], qfroz, q)
    return q


# ---- Batched over a leading tracer axis; winds/geometry shared ----
def _step_3d_batch(qb, u, v, w, dt, nv, lat, dp, polar, qfrozb):
    """qb: (ntracer, nlev, nlat, nlon); qfrozb same. u,v,w: (nlev,nlat,nlon)."""
    step_one = lambda q, qfroz: _step_3d(q, u, v, w, dt, nv, lat, dp, polar, qfroz)
    return jax.vmap(step_one, in_axes=(0, 0))(qb, qfrozb)


# ---- FAST driver: whole substep loop + on-device wind interpolation ----
# The entire n-substep integration for one coupling step runs inside ONE jitted
# fori_loop, batched over the tracer axis. dt_sub, n and nv are passed as traced
# scalars (NOT static), so this compiles exactly once no matter how the CFL
# substep count varies hour to hour. The per-substep math below is identical to
# the old Python-loop driver -- speed only, numerics unchanged.
@jit
def _advect_step_dev_batch(qb, u0, v0, w0, u1, v1, w1, dt_sub, n, nv,
                           lat, dp, polar, qfrozb):
    def body(i, qb_):
        #midpoint of each substep (0.5 is the offset for the midpoint)
        a = (i.astype(jnp.float64) + 0.5) / n
        #weighting of u between the two points, at i = 0, u ~ u0 and at i=n-1, u ~ u1
        u = (1.0 - a)*u0 + a*u1
        v = (1.0 - a)*v0 + a*v1
        w = (1.0 - a)*w0 + a*w1
        #advance every tracer one substep with the shared, interpolated winds
        return _step_3d_batch(qb_, u, v, w, dt_sub, nv, lat, dp, polar, qfrozb)
    return jax.lax.fori_loop(0, n, body, qb)


def advect_hour_batch(qb0, u0, v0, w0, u1, v1, w1,
                      lat, dp, qfrozb, lat_freeze=80.0,
                      cfl=0.2, dt_total=3600.0):
    """Advect a STACK of tracers qb0 (ntracer,nlev,nlat,nlon) over one hour.

    Winds are linearly interpolated between the hour endpoints (u0->u1 etc.),
    exactly as in fct.advect_hour_jax. Sub-step count is set by the CFL of the
    (shared) wind field, so every tracer uses the same stepping.
    """
    lat_np = np.asarray(lat)
    dp_np  = np.asarray(dp)
    dy     = RAD * DEG

    u0n=np.asarray(u0); u1n=np.asarray(u1)
    v0n=np.asarray(v0); v1n=np.asarray(v1)
    w0n=np.asarray(w0); w1n=np.asarray(w1)
    #max wind speeds, compute substeps based on these speeds and the CFL condition
    vmax = max(float(np.abs(v0n).max()), float(np.abs(v1n).max()))
    wmax = max(float(np.abs(w0n).max()), float(np.abs(w1n).max()))
    #minimum timestep to hit this cfl threshold (min courant number) for horizontal sweeps and sets that as the dt_sub timestep
    dt_sub = cfl * dy / max(vmax, 1e-6)
    # compute the number of substeps needed (needs to be an integer as dt_sub above won't be exact)
    n      = int(np.ceil(dt_total / dt_sub)); dt_sub = dt_total / n
    #courant number evaluated at the worst-case values (wmax and dp min) using the substepping from above
    #0.5 ensures that cz_sub can't be greater than 0.5 (I have proof for this), 1 needed to ensure at least one vertical substep is taken
    nv = max(1, int(np.ceil(np.abs(wmax*dt_sub/dp_np.min())/0.5)))

    polar = jnp.asarray(np.abs(lat_np) > lat_freeze)
    lat_j = jnp.asarray(lat_np, dtype=jnp.float64)
    dp_j  = jnp.asarray(dp_np,  dtype=jnp.float64)
    qfr_j = qfrozb if isinstance(qfrozb, jnp.ndarray) else jnp.asarray(qfrozb, dtype=jnp.float64)

    qb = qb0 if isinstance(qb0, jnp.ndarray) else jnp.asarray(qb0, dtype=jnp.float64)

    # winds as float64 host arrays (uploaded to the device(s) below)
    winds_np = [np.asarray(a, dtype=np.float64)
                for a in (u0n, v0n, w0n, u1n, v1n, w1n)]

    devices = jax.devices()
    ndev = len(devices)
    if ndev == 1:
        # single GPU: upload the two bracketing wind snapshots + geometry ONCE
        # (not once per substep); the on-device fori_loop interpolates between
        # them and runs all n substeps for every tracer.
        w6 = [jnp.asarray(a) for a in winds_np]
        qb = _advect_step_dev_batch(qb, *w6, dt_sub, n, nv,
                                    lat_j, dp_j, polar, qfr_j)
        return qb, n

    # multi-GPU: the tracers advect INDEPENDENTLY under the SHARED winds, so
    # shard the tracer axis across all visible devices (contiguous group i ->
    # device i), advect each shard on its own GPU with its own copy of the
    # winds/geometry, then gather back to device 0. Per-tracer math is untouched,
    # so this is bit-identical to the single-GPU path (matches how micro and
    # radiation shard). n/nv/dt_sub come from the shared winds -> same for all.
    groups = np.array_split(np.arange(qb.shape[0]), ndev)
    parts = []
    for i, idx in enumerate(groups):
        if idx.size == 0:
            continue
        d = devices[i]
        a, b = int(idx[0]), int(idx[-1]) + 1
        w6 = [jax.device_put(x, d) for x in winds_np]
        parts.append(_advect_step_dev_batch(
            jax.device_put(qb[a:b], d), *w6, dt_sub, n, nv,
            jax.device_put(lat_j, d), jax.device_put(dp_j, d),
            jax.device_put(polar, d), jax.device_put(qfr_j[a:b], d)))
    d0 = devices[0]
    qb = jnp.concatenate([jax.device_put(o, d0) for o in parts], axis=0)

    return qb, n
