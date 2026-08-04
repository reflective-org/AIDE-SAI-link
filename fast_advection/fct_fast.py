"""EXPERIMENTAL fast advection sandbox (isolated copy of ../fct_core.py).

Goal: drive advection from ~10 s -> sub-0.5 s per 6h step (see ../README notes
and this folder's logs). This file is a SELF-CONTAINED copy so the production
../fct_core.py is never touched while we iterate. The transport core functions
below are copied VERBATIM from fct_core.py (same scheme, same comments); only
three things differ, all opt-in:

  1. `dtype` knob on advect_hour_batch (default float64 == baseline). float32
     roughly halves memory + bandwidth (~2x). The tracer field is cast to
     `dtype` for the sweeps and cast back to float64 on return so the rest of
     the coupled model still sees float64.
  2. `eps` in _zalesak is dtype-aware (1e-300 for float64 -> exact baseline;
     a representable tiny for float32 so the FCT ratio guard doesn't underflow
     to 0/0 -> NaN).
  3. `cfl` is already a parameter; raising it cuts the horizontal substep count
     `n` (the x-sweep is FFSL / unconditionally stable; only the latitude and
     vertical sweeps limit the step).

PLANNED NEXT (not yet here): FFSL in latitude so the y-sweep also handles
Courant > 1, collapsing the substep count toward the vertical limit.

Multi-GPU tracer sharding is carried over from fct_core.py unchanged.
"""
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np

RAD = 6.371e6
DEG = np.pi / 180.0


# ---- spherical grid metric for the latitude sweep -------------------------
# The lat-lon cell area is proportional to cos(phi), which varies ACROSS the
# y-sweep. A sweep with no metric conserves the unweighted sum(q) but not the
# physical burden sum(q * area): poleward transport moves mass from wide cells
# into narrow ones, so the area-weighted total silently decays (this was the
# dominant term in the coupling's mass leak). These two weights restore it:
#
#   mf[j] = cos(phi_{j+1/2})   metric at the RIGHT face of cell j
#   ac[j] = (sin phi_{j+1/2} - sin phi_{j-1/2}) / dphi
#           exact cell-mean of cos(phi) == cell area / (dphi * dlambda * R^2)
#
# ac is the weight the burden diagnostic MUST use for the budget to close (it
# equals cos(lat_j) to O(dphi^2) in the interior but stays strictly positive at
# the +-90 rows, where cos(lat_j) is exactly 0 and would divide by zero).
# mf at the +-90 faces is cos(90) = 0, so the poleward flux there vanishes
# identically -- the zero-flux pole condition falls out of the metric itself.
def grid_metric(lat, lat_freeze=None):
    """(mf, ac) for a uniform pole-to-pole latitude grid `lat` [degrees].

    lat_freeze: if given, the faces INTERNAL to each polar cap (|lat|>lat_freeze)
    get mf = 0, so the y-sweep moves no mass between cap rows -- only across the
    cap-EDGE face. Each cap is then re-mixed as one cell (see _step_3d pol_mode=1)
    and the pair behaves as a single big reservoir cell.

    Why this is necessary: ac falls off a cliff at the poles (the +-90 rows are
    half-cells, ac ~ dphi/8, an 8:1 volume ratio to their neighbour, and
    mf/ac = 4 there). A PPM reconstruction that assumes uniform cell volumes is
    simply invalid across such a jump, and the metric-weighted flux/volume
    division overshoots violently -- 1000x overshoots and negative mass in the
    +-89/90 rows, which is 99% of the residual mass drift. Treating the cap as
    one cell removes the bad volume ratio entirely; the cap-edge Courant number
    is ~0.02 because the cap is 1.5% of the globe's area, so it is very stable.
    """
    lat = np.asarray(lat, dtype=np.float64)
    dphi = (lat[1] - lat[0]) * DEG
    #cell edges: half-cells at the two poles, midpoints in between
    edge = np.empty(lat.size + 1)
    edge[1:-1] = 0.5 * (lat[:-1] + lat[1:]) * DEG
    edge[0] = lat[0] * DEG
    edge[-1] = lat[-1] * DEG
    mf = np.cos(edge[1:])                                  # right face of cell j
    ac = (np.sin(edge[1:]) - np.sin(edge[:-1])) / dphi      # cell-mean cos(phi)
    if lat_freeze is not None:
        pol = np.abs(lat) > lat_freeze
        #face j joins cells j and j+1; internal to a cap when both are cap rows
        mf[:-1] = np.where(pol[:-1] & pol[1:], 0.0, mf[:-1])
    return mf, ac


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
def _zalesak(q, q_lo, A, periodic, ac=1.0):
    """Given old field q, low-order updated field q_lo, and antidiffusive flux A
    at each RIGHT face (per unit cell width), return the limited net update to add
    to q_lo. Bounds q_new to [neighbor_min, neighbor_max] over q and q_lo.

    ac : cell area weight (see grid_metric). A is then in FACE-metric units and
    every conversion of a face flux to a cell increment divides by ac, so the
    limited correction telescopes and conserves sum(ac*q) exactly. ac=1.0 is the
    original unweighted behaviour (bit-identical)."""
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
    #eq 7 -- /ac converts the face fluxes to a cell increment so the ratios below
    #are compared against q bounds in the right units
    P_plus  = (jnp.maximum(0.0, A_l) - jnp.minimum(0.0, A)) / ac
    P_minus = (jnp.maximum(0.0, A)   - jnp.minimum(0.0, A_l)) / ac
    #how much rise/fall is allowed in each cell (based on the max/min of neighbors)
    #eq 8
    Q_plus  = qmax - q_lo
    Q_minus = q_lo - qmin

    # dtype-aware floor: 1e-300 keeps the float64 path bit-identical to the
    # baseline; float32 would underflow 1e-300 to 0 and turn the guarded 0/0
    # into a NaN, so use a representable tiny there instead.
    eps = 1e-300 if q.dtype == jnp.float64 else 1e-38
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
    return -(A_lim - A_lim_l) / ac

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
def _ppm_frac_step_nonper(q, cf, dx=None, mf=None, ac=None):
    """One non-periodic PPM/FCT sweep along the last axis.

    mf/ac : spherical metric from grid_metric() -- face metric and cell area.
    Supplying them makes the sweep conserve sum(ac*q) to roundoff apart from the
    advective-form divergence term q*(C - Cl)/ac, which is the physically
    required term and vanishes for a continuity-consistent wind. With mf/ac None
    the sweep is the original unweighted form (bit-identical)."""
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
    # SPHERICAL METRIC: mf weights each FACE flux by cos(phi_face) and ac divides
    # each cell increment by the cell area -- the two are DIFFERENT arrays, which
    # is what the old single-`w` form could not express. Without them the sweep
    # conserves sum(q) instead of the physical sum(area*q).
    if mf is not None:
        wf = mf; wc = ac
    else:
        wf = 1.0 if dx is None else dx
        wc = wf
    T_hi = flux_hi*wf; T_lo = flux_lo*wf; C = cR*wf
    #zeros the left boundary with this code, the np.zeros just applies a single zero and everything else is fed in from the right faces above
    #flux at left face carries the width of the neighboring cell, so using dx is a necessary normalization to get the flux
    Tlo_l = jnp.concatenate([jnp.zeros_like(T_lo[...,:1]), T_lo[...,:-1]], -1)
    Cl    = jnp.concatenate([jnp.zeros_like(C[...,:1]),    C[...,:-1]],    -1)

    div  = q*(C - Cl)/wc
    q_lo = q - (T_lo - Tlo_l)/wc + div
    ## A at the right face (FACE-metric units; _zalesak divides by the cell area)
    A = (T_hi - T_lo)
    if mf is None:
        A = A / wc
    q_new = q_lo + _zalesak(q, q_lo, A, periodic=False,
                            ac=(1.0 if mf is None else wc))
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

## VERTICAL: conservative PPM cell-integrated semi-Lagrangian remap (NEW)
def _vert_remap(qc, wc, dp, dt, w_face=None, bc_top=None, bc_bot=None):
    """Unconditionally-stable vertical transport -- replaces the nv substep loop.

    qc (ncol,nlev) cell-mean mixing ratio; wc (ncol,nlev) omega [Pa/s]
    cell-centred; dp (nlev,) layer thickness [Pa]; dt [s].

    w_face (ncol,nlev+1): omega AT THE LEVEL FACES, supplied by the caller from
    the discrete continuity equation. When given, the two domain faces carry
    their real (nonzero) omega and the slab is OPEN in the vertical: trajectories
    may leave through the bottom face (outflow, no BC needed -- the departure
    layer simply lies inside the domain) or enter through either face (inflow at
    the reservoir concentration bc_top / bc_bot, each (ncol,)). This is the
    physically right slab boundary AND the thing that makes the mass budget
    close: a slab cannot satisfy continuity with zero flux through its faces, so
    forcing omega=0 there (the old behaviour, w_face=None) leaves a divergence
    residual that the advective-form update converts directly into a mass leak.

    Method: reconstruct q with the same CW84/PPM limiters used elsewhere, build
    the cumulative-mass function M(p), trace each Eulerian interface back to its
    departure pressure p_dep = p - omega*dt, and set the new cell mean to the
    average of the reconstruction over the DEPARTURE layer:
        q_new[k] = (M(p_dep[k+1]) - M(p_dep[k])) / (p_dep[k+1] - p_dep[k])
    Dividing by the departure-layer thickness (not the Eulerian dp) gives the
    ADVECTIVE form dq/dt = -omega dq/dp -- i.e. a mixing ratio conserved along
    trajectories, matching the old flux-form+divergence vertical step. Handles
    any Courant number in one pass, so no nv substepping."""
    ncol, nlev = qc.shape
    dt_ = qc.dtype
    #PPM reconstruction (CW84 monotonicity limiters, same as the sweeps)
    qL, qR, dq, q6 = _ppm_coeffs_nonper(qc)
    #Eulerian interface coordinate = cumulative pressure thickness (shared cols)
    xe = jnp.concatenate([jnp.zeros(1, dt_), jnp.cumsum(dp)]).astype(dt_)     # (nlev+1,)
    #cumulative tracer mass at each interface, per column
    Mcum = jnp.concatenate([jnp.zeros((ncol, 1), dt_),
                            jnp.cumsum(qc * dp[None, :], axis=1)], axis=1)     # (ncol,nlev+1)
    if w_face is None:
        #LEGACY: omega interpolated to interfaces; zero at the two domain faces
        #(no flux out). Inconsistent with continuity -- see the docstring.
        w_e = jnp.concatenate([jnp.zeros((ncol, 1), dt_),
                               0.5 * (wc[:, :-1] + wc[:, 1:]),
                               jnp.zeros((ncol, 1), dt_)], axis=1)             # (ncol,nlev+1)
        #departure pressure of each interface, kept in-domain and monotone so
        #trajectories don't cross (=> departure layers stay non-negative width)
        xdep = jnp.clip(xe[None, :] - w_e * dt, xe[0], xe[-1])
    else:
        #OPEN vertical faces: do NOT clip the departure points back into the
        #domain -- a departure point outside the slab is exactly what inflow
        #through a face means, and M(x) is extended below/above with the
        #reservoir concentration to serve it.
        w_e = w_face
        xdep = xe[None, :] - w_e * dt
    xdep = jax.lax.cummax(xdep, axis=1)
    #locate the cell holding each departure point + within-cell fraction s in [0,1]
    m = jnp.clip(jnp.searchsorted(xe, xdep, side='right') - 1, 0, nlev - 1)    # (ncol,nlev+1)
    s = jnp.clip((xdep - xe[m]) / dp[m], 0.0, 1.0)
    #M(p_dep) = mass to the cell's top edge + integral of the parabola 0->s
    Mc  = jnp.take_along_axis(Mcum, m, axis=1)
    qLm = jnp.take_along_axis(qL, m, axis=1)
    dqm = jnp.take_along_axis(dq, m, axis=1)
    q6m = jnp.take_along_axis(q6, m, axis=1)
    M_at = Mc + dp[m] * (qLm*s + 0.5*dqm*s*s + q6m*(0.5*s*s - s*s*s/3.0))       # (ncol,nlev+1)
    if w_face is not None:
        #extend the cumulative-mass function OUTSIDE the slab at the reservoir
        #concentration, so a departure point above the top face (or below the
        #bottom face) draws inflow at that concentration instead of being
        #truncated. Outflow needs no BC: its departure layer lies inside.
        qt = qc[:, :1] if bc_top is None else bc_top[:, None]
        qb = qc[:, -1:] if bc_bot is None else bc_bot[:, None]
        M_at = jnp.where(xdep < xe[0],  Mcum[:, :1]  - qt * (xe[0]  - xdep), M_at)
        M_at = jnp.where(xdep > xe[-1], Mcum[:, -1:] + qb * (xdep - xe[-1]), M_at)
    #advective-form update: divide by DEPARTURE-layer thickness (floored to avoid
    #blow-up under strong convergence; monotone xdep keeps this non-negative)
    dep_thick = jnp.maximum(xdep[:, 1:] - xdep[:, :-1], 0.01 * dp[None, :])
    qnew = (M_at[:, 1:] - M_at[:, :-1]) / dep_thick
    if w_face is None:
        return qnew
    #face exchange, in q*Pa units per column: mass drawn IN through the top face
    #and mass carried OUT through the bottom face this substep. With the slab open
    #the gross exchange is large (|omega|~9e-3 Pa/s over a 7500 Pa slab is ~10% of
    #the column per day), so the budget MUST carry these terms explicitly instead
    #of assuming the faces are sealed.
    f_top = Mcum[:, 0] - M_at[:, 0]              # >0 = inflow from above
    f_bot = Mcum[:, -1] - M_at[:, -1]            # >0 = outflow through the bottom
    return qnew, f_top, f_bot


def _mix_caps(q, polar, lat, ac):
    """Stir each polar cap into ONE well-mixed cell per level, area-weighted.

    Exactly mass conserving: an area-weighted mean leaves sum(ac*q) over the cap
    unchanged. The two caps are handled separately so nothing teleports between
    poles. q: (nlev,nlat,nlon)."""
    wcap = (ac if ac is not None else jnp.cos(lat*DEG))[None, :, None]
    def mix(qq, mask):
        m = mask[None, :, None]
        tot = (qq * wcap * m).sum(axis=(1, 2), keepdims=True)
        den = (wcap * m).sum(axis=(1, 2), keepdims=True) * qq.shape[2]
        return jnp.where(m, tot / den, qq)
    q = mix(q, polar & (lat < 0.0))
    return mix(q, polar & (lat > 0.0))


def _omega_continuity(u, v, w, lat, dp, polar, mf, ac, dlam, dphi, dt=None):
    """omega at the level FACES from the discrete continuity equation.

    In pressure coordinates the atmosphere satisfies div_p(u) + d(omega)/dp = 0,
    which is exactly why the advective and flux forms of tracer transport agree
    and the burden is conserved. Archived CESM omega does NOT satisfy that
    discretely -- it was interpolated to fixed pressure levels, subsampled to 6 h
    and re-interpolated in time, and (worst) the slab's top/bottom faces were
    forced to zero. So we do what offline CTMs do: keep the horizontal winds and
    REDERIVE omega by integrating continuity downward from the top face,

        omega_{k+1/2} = omega_{k-1/2} - dp_k * (Sx + Sy)_k

    using the SAME discrete divergence operators the x- and y-sweeps apply. The
    result cancels their divergence terms to roundoff, which drops the interior
    divergence residual from ~5e-6 1/s to ~4e-22 1/s and with it the mass leak.
    The top face is anchored to the CESM omega there (a constant offset in the
    anchor shifts the through-slab flux but not the conservation property)."""
    nlev, nlat, nlon = u.shape
    #x-divergence as the x-sweep applies it (cos(lat) is constant along a row, so
    #the x-sweep needs no area metric and neither does this).
    #TRIED AND REJECTED: _ffsl_x splits cx into an integer cell shift (a pure
    #permutation, which applies no divergence) plus a fraction, so it is tempting
    #to build Sx from the FRACTION cx-round(cx) to match what the sweep really
    #applies. Measured: that made things ~40x WORSE at cfl 0.8 (-8.4e-3 vs
    #-2.0e-4 per 6h step), because cx-round(cx) jumps discontinuously wherever cx
    #crosses a half-integer, so the derived omega picks up grid-scale noise and
    #drives spurious vertical transport. The smooth full-cx divergence wins.
    uf = 0.5*(u + jnp.roll(u, -1, 2))
    cxr = uf / (RAD*jnp.cos(lat*DEG)*dlam)[None, :, None]     # cx/dt
    Sx = cxr - jnp.roll(cxr, 1, 2)
    #y-divergence exactly as the y-sweep applies it, metric included
    vf = 0.5*(v + jnp.roll(v, -1, 1))
    cyr = (vf / (RAD*dphi)).at[:, -1, :].set(0.0)      # zero flux at the +90 face
    if mf is not None:
        C  = cyr * mf[None, :, None]
        Cl = jnp.concatenate([jnp.zeros_like(C[:, :1]), C[:, :-1]], axis=1)
        Sy = (C - Cl) / ac[None, :, None]
    else:
        Cl = jnp.concatenate([jnp.zeros_like(cyr[:, :1]), cyr[:, :-1]], axis=1)
        Sy = cyr - Cl
    #polar caps: cx ~ 1/cos(lat) reaches ~1e16 at the +-90 rows, so Sx there is
    #meaningless. Zero it BEFORE the cumsum (otherwise it poisons every face
    #below) and fall back to the legacy omega in the caps, which their own
    #treatment (zonal mean / freeze) handles.
    S = jnp.where(polar[None, :, None], 0.0, Sx + Sy)
    wf = jnp.concatenate([jnp.zeros_like(w[:1]),
                          -jnp.cumsum(dp[:, None, None] * S, axis=0)], axis=0) + w[:1]
    wf_leg = jnp.concatenate([jnp.zeros_like(w[:1]), 0.5*(w[:-1] + w[1:]),
                              jnp.zeros_like(w[:1])], axis=0)
    return jnp.where(polar[None, :, None], wf_leg, wf)


## key routine where q is advected in 3D (lon, lat, alt) with sub-stepping in the vertical...q gets advected one call at a time (lon,lat, and lastly altitude)
def _step_3d(q, u, v, w, dt, nv, lat, dp, polar, qfroz, mf=None, ac=None,
             pol_mode=0, dlam=DEG, dphi=DEG, w_face=None):
    """Advance ONE tracer field q(nlev,nlat,nlon) by one sub-step.

    mf/ac    : spherical metric (grid_metric); None = legacy unweighted y-sweep.
    pol_mode : 0 = overwrite the polar caps with qfroz (legacy, DISCARDS the mass
               transport delivers there); 1 = replace each polar row by its own
               zonal mean (conserves mass exactly, see below).
    dlam/dphi: GRID SPACING in radians. These must be the real cell widths --
               the Courant number is a fraction of a CELL, which is what both the
               PPM parabola integration and the _ffsl_x integer shift assume.
               Passing DEG (1 degree) for both is the legacy behaviour and is
               wrong on any grid that isn't 1x1 degree: on the f09 0.94x1.25 grid
               it made cx 1.25x too large and cy 0.94x too small. Beyond the
               transport-speed error that scales the x and y DIVERGENCES by
               different factors, so div_x + div_y + div_z cannot cancel even for
               a perfectly non-divergent wind -- a permanent mass-budget leak."""
    nlev, nlat, nlon = q.shape
    if pol_mode == 1:
        # Mix the caps BEFORE the x-sweep, not after. cx ~ 1/cos(lat) reaches
        # ~1e18 at the +-90 rows, so _ffsl_x's K = round(cx).astype(int32)
        # overflows and cf becomes garbage. A zonally UNIFORM row survives that
        # exactly -- the low-order flux difference q*(cf_i - cf_i-1) and the
        # divergence term are the same floating-point expression and cancel
        # identically, and the antidiffusive flux is exactly 0 -- but a raw
        # non-uniform cap is destroyed by it. (The legacy freeze hid this by
        # overwriting the wreckage with qfroz every substep, which is precisely
        # the polar mass discard.) So the cap must enter every x-sweep stirred.
        q = _mix_caps(q, polar, lat, ac)
    # average u to the cell right face (uf) and compute the courant number cx for the longitude sweep (convert to meteres)
    # expand shape for cx to (1,nlat,1) for broadcasting as cx is used in ffsl_x which expects (nlev*nlat,nlon)
    uf = 0.5*(u + jnp.roll(u, -1, 2))
    cx = uf * dt / (RAD * jnp.cos(lat*DEG) * dlam)[None, :, None]
    #reshape q and cx to 2D for the longitude sweep, then reshape back to 3D after the sweep
    q  = _ffsl_x(q.reshape(nlev*nlat, nlon),
                 cx.reshape(nlev*nlat, nlon)).reshape(nlev, nlat, nlon)
    #average v to the cell front/back face (vf) and compute the courant number cy for the latitude sweep (convert to meters)
    vf = 0.5*(v + jnp.roll(v, -1, 1))
    cy = vf * dt / (RAD * dphi)
    #reshape as above..need to transpose to get the correct shape for the latitude sweep, switch lat and lon ordering
    qT  = q.transpose(0,2,1).reshape(nlev*nlon, nlat)
    cyT = cy.transpose(0,2,1).reshape(nlev*nlon, nlat)
    #call ppm with periodicity False for the latitude sweep, then reshape back to 3D and transpose back to original ordering
    #mf/ac carry the cos(phi) area metric so the sweep conserves the AREA-weighted
    #burden (what the coupling's Mbur measures), not the raw sum over rows
    q   = _ppm_frac_step_nonper(qT, cyT, mf=mf, ac=ac
                                ).reshape(nlev, nlon, nlat).transpose(0,2,1)
    #q.reshape(nlev,-1) → (nlev, nlat*nlon). Then .T → (nlat*nlon, nlev)
    qc = q.reshape(nlev, -1).T
    wc = w.reshape(nlev, -1).T
    #VERTICAL via conservative semi-Lagrangian remap: unconditionally stable, so
    #the old nv inner substep loop is gone (nv is now unused). See _vert_remap.
    if w_face is None:
        qc = _vert_remap(qc, wc, dp, dt)
        vflux = jnp.zeros((2, nlat, nlon), q.dtype)
    else:
        #continuity-consistent omega + OPEN slab faces. Inflow through either face
        #arrives at the reservoir concentration (qfroz's edge level) -- for the
        #bottom face that is tropical upwelling bringing aerosol-poor tropopause
        #air INTO the band, which is the physical inflow the frozen bottom clamp
        #was standing in for. Outflow needs no boundary value.
        qc, f_top, f_bot = _vert_remap(qc, wc, dp, dt,
                                       w_face=w_face.reshape(nlev + 1, -1).T,
                                       bc_top=qfroz[0].reshape(-1),
                                       bc_bot=qfroz[-1].reshape(-1))
        vflux = jnp.stack([f_top.reshape(nlat, nlon), f_bot.reshape(nlat, nlon)])
    #reshape back to 3D
    q = qc.T.reshape(nlev, nlat, nlon)
    #for latitudes above 80 deg, reset cells to initial state as small polar cells can cause numerical issues with the scheme.
    #This is a simple fix to avoid instability in the polar regions for cx ~ 1/cos(lat) (at high latitudes blows up towards infinity))
    if pol_mode == 1:
        # MASS-CONSERVING polar caps: mix each cap into ONE well-stirred cell per
        # level instead of overwriting it with a frozen reservoir. Rationale:
        #  * conserves mass exactly -- an area-weighted mean over the cap leaves
        #    sum(ac*q) over the cap unchanged, and the two caps are handled
        #    separately so nothing teleports between poles. The overwrite, by
        #    contrast, deletes everything the Brewer-Dobson circulation
        #    converges into the caps.
        #  * kills BOTH polar numerical problems at once. Zonally: a uniform row
        #    is an exact fixed point of the x-sweep at ANY Courant number
        #    (constant reconstruction => dq=q6=0 => flux_hi==flux_lo==cf*q, the
        #    antidiffusive flux is 0 and the div term cancels the flux term), so
        #    the cx ~ 1/cos(lat) blow-up cannot be excited. Meridionally: with
        #    the cap's internal faces closed (grid_metric(lat_freeze)) the only
        #    y-flux is across the cap edge, so the +-90 half-cells never see the
        #    8:1 volume ratio that made the metric sweep overshoot.
        #  * no external reservoir, so the polar CESM dependence disappears.
        # We re-mix every substep, so the caps always ENTER the next sweep well
        # stirred; the vertical remap is per-column and needs no polar care.
        q = _mix_caps(q, polar, lat, ac)
    else:
        q = jnp.where(polar[None, :, None], qfroz, q)
    return q, vflux


# ---- Batched over a leading tracer axis; winds/geometry shared ----
def _step_3d_batch(qb, u, v, w, dt, nv, lat, dp, polar, qfrozb,
                   mf=None, ac=None, pol_mode=0, dlam=DEG, dphi=DEG, wcont=False):
    """qb: (ntracer, nlev, nlat, nlon); qfrozb same. u,v,w: (nlev,nlat,nlon)."""
    #the continuity omega depends only on the winds and geometry, so build it ONCE
    #here rather than per tracer inside the vmap
    w_face = (_omega_continuity(u, v, w, lat, dp, polar, mf, ac, dlam, dphi, dt)
              if wcont else None)
    step_one = lambda q, qfroz: _step_3d(q, u, v, w, dt, nv, lat, dp, polar,
                                         qfroz, mf, ac, pol_mode, dlam, dphi,
                                         w_face)
    return jax.vmap(step_one, in_axes=(0, 0))(qb, qfrozb)   # (qb, vflux)


# ---- FAST driver: whole substep loop + on-device wind interpolation ----
# Same as fct_core, except the interpolation weight `a` is built in the winds'
# dtype (so a float32 run stays entirely float32 instead of being promoted back
# to float64 by an int/float mix).
@partial(jit, static_argnums=(16, 19))
def _advect_step_dev_batch(qb, u0, v0, w0, u1, v1, w1, dt_sub, n, nv,
                           lat, dp, polar, qfrozb, mf, ac, pol_mode,
                           dlam=DEG, dphi=DEG, wcont=False):
    def body(i, carry):
        qb_, vf_ = carry
        #midpoint of each substep (0.5 is the offset for the midpoint)
        a = (i.astype(u0.dtype) + 0.5) / n
        #weighting of u between the two points, at i = 0, u ~ u0 and at i=n-1, u ~ u1
        u = (1.0 - a)*u0 + a*u1
        v = (1.0 - a)*v0 + a*v1
        w = (1.0 - a)*w0 + a*w1
        #advance every tracer one substep with the shared, interpolated winds
        qb_, vfs = _step_3d_batch(qb_, u, v, w, dt_sub, nv, lat, dp, polar, qfrozb,
                                  mf, ac, pol_mode, dlam, dphi, wcont)
        #accumulate the vertical face exchange over the substeps (q*Pa units)
        return qb_, vf_ + vfs
    vf0 = jnp.zeros((qb.shape[0], 2) + qb.shape[2:], qb.dtype)
    return jax.lax.fori_loop(0, n, body, (qb, vf0))


def advect_hour_batch(qb0, u0, v0, w0, u1, v1, w1,
                      lat, dp, qfrozb, lat_freeze=80.0,
                      cfl=0.2, dt_total=3600.0, dtype=jnp.float64,
                      metric=None, polar_mode=None, dxfix=None, wcont=None,
                      return_vflux=False):
    """Advect a STACK of tracers qb0 (ntracer,nlev,nlat,nlon) over one hour.

    dtype: precision of the transport sweeps (jnp.float32 or jnp.float64).
    Everything is cast to `dtype` for the substep loop and the result cast back
    to float64. cfl sets the horizontal substep count. Multi-GPU sharding over
    the tracer axis is applied automatically when >1 device is visible.

    metric     : apply the cos(phi) area metric in the y-sweep so the AREA-weighted
                 burden is conserved (default on; env ADV_METRIC=0 to disable).
                 When on, the burden diagnostic must weight rows by grid_metric()'s
                 `ac`, NOT cos(lat), for the mass budget to close.
    polar_mode : 'zonal' = mass-conserving zonal-mean caps (default),
                 'freeze' = legacy overwrite with qfrozb (discards mass).
                 Env ADV_POLAR overrides the default.
    dxfix      : use the TRUE grid spacing in the Courant numbers (default on;
                 env ADV_DXFIX=0 restores the legacy 1-degree assumption).
    return_vflux: also return the accumulated vertical FACE exchange, shape
                 (ntracer, 2, nlat, nlon) in q*Pa units: [0] = mass drawn in
                 through the top face, [1] = mass carried out through the bottom
                 face, summed over the substeps. With open faces the gross
                 exchange is ~10%/day of the slab, so the caller must carry these
                 terms in its mass budget rather than assume sealed faces.
    wcont      : rederive omega from discrete continuity and OPEN the slab's
                 vertical faces (default on; env ADV_WCONT=0 restores the legacy
                 archived-omega / zero-flux-faces behaviour). This is the term
                 that actually zeroes the leak -- see _omega_continuity.
    """
    if metric is None:
        metric = os.environ.get('ADV_METRIC', '1') != '0'
    if polar_mode is None:
        polar_mode = os.environ.get('ADV_POLAR', 'zonal').lower()
    if dxfix is None:
        dxfix = os.environ.get('ADV_DXFIX', '1') != '0'
    if wcont is None:
        wcont = os.environ.get('ADV_WCONT', '1') != '0'
    pol_mode = 1 if polar_mode == 'zonal' else 0

    lat_np = np.asarray(lat)
    dp_np  = np.asarray(dp)
    #real cell widths in radians. The lon grid is global+periodic so dlam follows
    #from nlon exactly; dphi from the (uniform) lat spacing.
    if dxfix:
        dlam = 2.0 * np.pi / float(np.asarray(qb0).shape[-1])
        dphi = float(lat_np[1] - lat_np[0]) * DEG
    else:
        dlam = dphi = DEG          # legacy: pretend the grid is 1 x 1 degree
    dy     = RAD * dphi

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

    # geometry on the chosen dtype; polar is boolean (dtype-independent)
    polar = jnp.asarray(np.abs(lat_np) > lat_freeze)
    #lat_freeze closes the caps' internal faces so each cap acts as one cell
    _mf, _ac = grid_metric(lat_np, lat_freeze if pol_mode == 1 else None)
    #ac is always supplied: the y-sweep only applies the metric when mf is given,
    #but the polar cap mixing needs a correct area weight either way (cos(lat) is
    #exactly 0 at the +-90 rows, which would drop their mass from the cap mean).
    ac_j = jnp.asarray(_ac, dtype=dtype)
    mf_j = jnp.asarray(_mf, dtype=dtype) if metric else None
    lat_j = jnp.asarray(lat_np, dtype=dtype)
    dp_j  = jnp.asarray(dp_np,  dtype=dtype)
    qfr_j = jnp.asarray(np.asarray(qfrozb), dtype=dtype)
    qb    = jnp.asarray(np.asarray(qb0),    dtype=dtype)

    # winds as `dtype` host arrays (uploaded to the device(s) below)
    winds_np = [np.asarray(a, dtype=np.float64) for a in (u0n, v0n, w0n, u1n, v1n, w1n)]

    devices = jax.devices()
    ndev = len(devices)
    if ndev == 1:
        w6 = [jnp.asarray(a, dtype=dtype) for a in winds_np]
        qb, vfl = _advect_step_dev_batch(qb, *w6, dt_sub, n, nv,
                                         lat_j, dp_j, polar, qfr_j,
                                         mf_j, ac_j, pol_mode, dlam, dphi, wcont)
        if return_vflux:
            return qb.astype(jnp.float64), n, vfl.astype(jnp.float64)
        return qb.astype(jnp.float64), n

    # multi-GPU: shard the tracer axis across all visible devices (see fct_core)
    groups = np.array_split(np.arange(qb.shape[0]), ndev)
    parts = []
    for i, idx in enumerate(groups):
        if idx.size == 0:
            continue
        d = devices[i]
        a, b = int(idx[0]), int(idx[-1]) + 1
        w6 = [jax.device_put(jnp.asarray(x, dtype=dtype), d) for x in winds_np]
        parts.append(_advect_step_dev_batch(
            jax.device_put(qb[a:b], d), *w6, dt_sub, n, nv,
            jax.device_put(lat_j, d), jax.device_put(dp_j, d),
            jax.device_put(polar, d), jax.device_put(qfr_j[a:b], d),
            None if mf_j is None else jax.device_put(mf_j, d),
            None if ac_j is None else jax.device_put(ac_j, d), pol_mode,
            dlam, dphi, wcont))
    d0 = devices[0]
    qb = jnp.concatenate([jax.device_put(o[0], d0) for o in parts], axis=0)
    if return_vflux:
        vfl = jnp.concatenate([jax.device_put(o[1], d0) for o in parts], axis=0)
        return qb.astype(jnp.float64), n, vfl.astype(jnp.float64)
    return qb.astype(jnp.float64), n
