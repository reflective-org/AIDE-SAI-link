"""Lin-Rood FLUX-FORM transport with air-mass tracking.

WHY THIS EXISTS
---------------
fct_fast.py uses the ADVECTIVE form: each sweep applies a flux divergence plus a
`+q*div(c)` correction. That form is *consistent* (a constant field is preserved
exactly) but it is NOT conservative: the global burden changes by sum(q*div),
which vanishes only if the discrete 3-D wind is divergence-free. fct_fast's
ADV_WCONT fix rederives omega so that it very nearly is, which cut the leak from
-0.233%/day to +0.014%/day. What survives is operator-SPLITTING error -- the three
sweeps apply their divergence terms sequentially, to different intermediate
fields, so the cancellation is only O(dt) exact. With SAI injection on (sharp
plume + nucleation-burst gradients) that residual grows to ~+0.036%/day.

This module removes the residual by construction rather than by making the winds
good. Following Lin & Rood (1996): carry the AIR MASS as a prognostic field and
advect the pair (rho, rho*q) in pure FLUX form,

    rho'    = rho    - div(F_air)/area,        F_air = C * m_face * rho_face
    (rho q)'= (rho q)- div(F_trc)/area,        F_trc = F_air * q_face
    q'      = (rho q)' / rho'

Every flux appears twice with opposite signs, so sum(area*dp*rho*q) telescopes to
the boundary fluxes ALONE -- exactly, to roundoff, for ANY wind field, divergent
or not, and regardless of splitting order. The wind's inconsistency no longer
destroys tracer mass; it shows up as rho drifting away from 1, which is an
explicit, diagnosable field.

This is the property that matters for emulator winds: they will not satisfy
discrete continuity, and flux form is what makes that harmless. The cost is that
`rho` must persist across steps (the caller carries it) and the conserved
quantity becomes sum(area*dp*rho*q), not sum(area*dp*q).

WHAT IS REUSED
--------------
The PPM reconstruction, the Zalesak limiter, the spherical metric, the polar cap
stirring and the continuity-omega derivation are imported from fct_fast so there
is exactly one copy of each and this module cannot drift from the validated one.
fct_fast.py is not modified.
"""
import os

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as np

from fct_fast import (RAD, DEG, grid_metric, _ppm_coeffs_per, _ppm_coeffs_nonper,
                      _zalesak, _mix_caps, _omega_continuity)


# ---- PPM sliver means -----------------------------------------------------
# The mean of the reconstruction over the sliver that crosses a face in one step.
# flux = cf * sliver_mean (cf = courant fraction at each face). fct_fast computes this inline; here it is factored out
# because the air field and the tracer field both need it against the SAME cf.

""""hi = "PPM says the departing slab has this concentration" (3rd-order accurate, but can overshoot near sharp gradients like your injection plume), 
and lo = "just use the whole donor cell's mean" (1st-order upwind: diffusive, but guaranteed bounded). 
Multiply either by cf·m and you have a flux; Zalesak later blends toward hi as far as boundedness allows"""
def _sliver(q, cf, periodic):
    """(hi, lo) upwind sliver means at each RIGHT face. lo = donor-cell value."""
    if periodic:
        qL, qR, dq, q6 = _ppm_coeffs_per(q)
        sh = lambda a: jnp.roll(a, -1, -1)
    else:
        qL, qR, dq, q6 = _ppm_coeffs_nonper(q)
        sh = lambda a: jnp.concatenate([a[..., 1:], a[..., -1:]], -1)
    #outflow through the right face: mean over the last cf of THIS cell
    f_pos = qR - 0.5*cf*(dq - q6*(1 - (2/3)*cf))
    #inflow from the right neighbour: mean over its first |cf|
    a = -cf
    f_neg = sh(qL) + 0.5*a*(sh(dq) + sh(q6)*(1 - (2/3)*a))
    hi = jnp.where(cf >= 0, f_pos, f_neg)
    lo = jnp.where(cf >= 0, q, sh(q))
    #average mixing ratio inside the slab that will cross that face this step 
    return hi, lo


def _shift_left(a, periodic):
    """value at the LEFT face of each cell (i.e. the right face of cell i-1)."""
    if periodic:
        return jnp.roll(a, 1, -1)
    return jnp.concatenate([jnp.zeros_like(a[..., :1]), a[..., :-1]], -1)


def _lr_sweep(rho, rhoq, cf, periodic, mf=None, ac=None):
    """One flux-form Lin-Rood sweep along the last axis.

    Returns (rho', rhoq'). The tracer flux rides on the AIR flux, so a uniform
    mixing ratio is preserved exactly while sum(ac*rho*q) is conserved to
    roundoff -- independent of how divergent cf is."""
    #m is the length of the face and a is the cell area
    m = 1.0 if mf is None else mf
    a = 1.0 if ac is None else ac
    if not periodic:
        #hard zero-flux at the far domain face (the near face is zeroed by the
        #left-shift); on a latitude sweep mf is already cos(90)=0 there, this just
        #makes it exact
        cf = cf.at[..., -1].set(0.0) if hasattr(cf, 'at') else cf

    q = rhoq / rho
    #--- air flux (high order; rho is smooth and ~1 so PPM needs no limiting) ---
    rhi, _ = _sliver(rho, cf, periodic)
    # the amount of air moving through the face..cf = distance air moves in one step
    # Fa = (how far (in cell width)) x (how wide the face is) x (how much air is in the crossing slab) = cf * m * rhi
    Fa = cf * m * rhi
    Fa_l = _shift_left(Fa, periodic)
    #discrete continuity: rho' = rho - div(F_air)/area, where div(F_air) = Fa - Fa_l
    rho_n = rho - (Fa - Fa_l) / a

    #--- tracer flux: the SAME air flux, weighted by the tracer sliver mean -----
    #qhi = PPM sliver mean (3rd order, can overshoot at a plume edge);
    #qlo = donor-cell mean (1st-order upwind: diffusive but always bounded)
    qhi, qlo = _sliver(q, cf, periodic)
    # tracer flux is not computed independently, it's the air flux carrying the mixing ratio
    # Fa = kg air, q = kg tracer / kg air, so Fa * q = kg tracer
    Ft_hi = Fa * qhi
    Ft_lo = Fa * qlo
    Ftlo_l = _shift_left(Ft_lo, periodic)
    #Low-order (donor-cell) update
    #summing over cells cancels every interior flux and leaves only the two domain
    #edges: conservation, exact to roundoff, for any cf however divergent
    rhoq_lo = rhoq - (Ft_lo - Ftlo_l) / a
    # back to mixing ratio units for the Zalesak limiter, where rho_n is the air-mass field after the sweep
    q_lo = rhoq_lo / rho_n
    #--- FCT: limit the antidiffusive tracer flux so q stays bounded ------------
    # increments are converted to q units by dividing by (area * rho'), which is
    # what _zalesak's `ac` argument does.
    A = Ft_hi - Ft_lo
    dq = _zalesak(q, q_lo, A, periodic=periodic, ac=a * rho_n)
    #back to the conserved variable for the next sweep
    return rho_n, (q_lo + dq) * rho_n


# ---- vertical: conservative cell-integrated SL remap, FLUX form ------------
def _lr_vert(rho, rhoq, dp, dt, w_face, bc_top=None, bc_bot=None):
    """Flux-form version of fct_fast._vert_remap on the pair (rho, rho*q).

    Identical geometry (PPM reconstruction, departure points, out-of-domain
    extension for open faces) with ONE change: divide by the EULERIAN dp instead
    of the departure-layer thickness. That single substitution turns the
    advective-form remap into an exactly conservative one --
    sum(dp*rho*q) = M(xdep_last) - M(xdep_0), i.e. the interior total plus the two
    face fluxes and nothing else. Unconditional stability is unaffected (it comes
    from the semi-Lagrangian departure-point tracing, not from the normalisation).

    rho/rhoq: (ncol,nlev). Returns (rho', rhoq', f_top, f_bot) with the face
    exchange in rho*q*Pa units."""
    ncol, nlev = rhoq.shape
    dt_ = rhoq.dtype
    #Eulerian interface pressures: xe[k] = pressure at the top of cell k, so the
    #vertical coordinate here is PRESSURE, increasing downward.
    #say dp = [10,20,40] Pa, then xe = [0,10,30,70] Pa.
    xe = jnp.concatenate([jnp.zeros(1, dt_), jnp.cumsum(dp)]).astype(dt_)

    #departure pressure of each interface; open faces => do NOT clip in-domain
    #w_face is omega (Pa/s) so w*dt is a pressure displacement
    xdep = jax.lax.cummax(xe[None, :] - w_face * dt, axis=1)
    #m = which Eulerian cell each departure point lands in; s = fractional
    #position within that cell, so (m, s) locates xdep on the subgrid.
    #m = top of the cell and s = fraction of cell below the top
    m = jnp.clip(jnp.searchsorted(xe, xdep, side='right') - 1, 0, nlev - 1)
    s = jnp.clip((xdep - xe[m]) / dp[m], 0.0, 1.0)

    def M_at(fld, bc):
        Mc = jnp.concatenate([jnp.zeros((ncol, 1), dt_),
                              jnp.cumsum(fld * dp[None, :], axis=1)], axis=1)
        qL, qR, dqc, q6 = _ppm_coeffs_nonper(fld)
        #mass above the top of the cell + the partial but inside the cell
        #using CW parabola integration
        #cumulative tracer mass above departure point
        out = (jnp.take_along_axis(Mc, m, axis=1)
               + dp[m] * (jnp.take_along_axis(qL, m, axis=1) * s
                          + 0.5 * jnp.take_along_axis(dqc, m, axis=1) * s * s
                          + jnp.take_along_axis(q6, m, axis=1)
                          * (0.5 * s * s - s * s * s / 3.0)))
        #extend outside the slab at the reservoir value so inflow through a face
        #is served instead of truncated
        qt = fld[:, :1] if bc is None else bc[0][:, None]
        qb = fld[:, -1:] if bc is None else bc[1][:, None]
        out = jnp.where(xdep < xe[0],  Mc[:, :1]  - qt * (xe[0] - xdep), out)
        out = jnp.where(xdep > xe[-1], Mc[:, -1:] + qb * (xdep - xe[-1]), out)
        # out is the new state and Mc is the old state, 
        # so the difference is the net flux into the slab across each interface
        return Mc, out

    #air mass
    Mc_r, Mr = M_at(rho, None)                      # air inflow at rho=edge value
    #tracer mass
    Mc_t, Mt = M_at(rhoq, bc_top if bc_top is None else (bc_top, bc_bot))
    #mass in the slab arriving in each cell, divided by thickness
    rho_n = (Mr[:, 1:] - Mr[:, :-1]) / dp[None, :]
    rhoq_n = (Mt[:, 1:] - Mt[:, :-1]) / dp[None, :]
    f_top = Mc_t[:, 0] - Mt[:, 0]                   # >0 = inflow through the top
    f_bot = Mc_t[:, -1] - Mt[:, -1]                 # >0 = outflow through bottom

    if VPOS:
        rhoq_n, f_top, f_bot = _vert_positive(rhoq, rhoq_n, Mc_t, Mt, dp)
    return rho_n, rhoq_n, f_top, f_bot


# ---- vertical positivity limiter (ADV_VPOS=1) -------------------------------
# WHY THIS EXISTS. The remap above is exactly CONSERVATIVE but not POSITIVE, and
# those are different properties: it can put -d in one cell and +d in its
# neighbour while the sum is untouched. `_ppm_coeffs_nonper` carries the
# Colella-Woodward MONOTONICITY limiter, which is not a positivity constraint --
# the edge value qR = (7/12)(q+qp1) - (1/12)(qp2+qm1) goes negative when the
# 4-cell stencil spans a steep enough gradient, and the overshoot fallbacks
# 3q-2qR / 3q-2qL can too. coupling.py then clips the negatives to zero and
# CREATES number (mass has a budgeted `floor` term; number's was never budgeted).
#
# MEASURED 2026-07-29 on real states (scripts/validation/floor_anatomy.py):
#   * 100% of the negatives come from THIS operator. The horizontal x/y sweeps
#     produce exactly zero (0/40 bins) because _lr_sweep's Zalesak step bounds
#     the update to neighbour min/max, which is already positivity-preserving.
#   * The floor adds ~3.3e-3 of the standing number burden per 6 h step, of which
#     97.4% lands in bins below 10 nm (number-weighted mean Dp 3.3 nm) and
#     0.0025% in the optically active 150-1200 nm bins. 97.8% of it is at the
#     tropical injection ring, where the vertical number gradient is steepest.
#   * That distribution does NOT migrate to larger sizes as a run matures
#     (checked at day 90 vs hour 42), so mass/AOD/ARF are safe over multi-year
#     runs; what is NOT trustworthy is total number and anything under ~10 nm.
#
# WHY A FLUX LIMITER AND NOT A CLIP. Clipping is what causes the problem: the
# undershoot is a conservative dipole, so deleting the -d while keeping the +d is
# a one-sided source. Limiting the DRAINING FLUXES instead keeps the telescoping
# sum exact, so positivity is bought without breaking conservation.
#
# The update above is already in telescoping flux form. With F_k := Mc_t[:,k] -
# Mt[:,k] (the net flux INTO the slab across interface k -- note f_top is F_0 and
# f_bot is F_nlev by definition), the identity is
#       rhoq_n[k] = rhoq[k] + (F_k - F_{k+1}) / dp[k].
# So F_k drains cell k-1 when F_k > 0 and drains cell k when F_k < 0. Scale every
# interface flux by the min over the cells it drains and no cell can be emptied
# past zero:
#       outflow_k after scaling <= r_k * (|F_k| + F_{k+1}) = r_k * O_k <= avail_k.
# Each F_k is one number applied with opposite signs to the two cells it joins, so
# rescaling it preserves the telescoping exactly -- conservation is untouched and
# the two BOUNDARY fluxes are returned rescaled, so the caller's vface budget
# stays consistent with what actually moved.
#
# Inflow is never limited: F_0 > 0 (into the slab through the top) would "drain
# cell -1", which does not exist, and likewise F_nlev < 0 at the bottom.
#
# DEFAULT ON. The fallback deliberately does not depend on which checkout you are
# standing in: when it differed between copies, the same command meant different
# physics per directory and was only right by accident when a launcher set the
# variable explicitly (run_prod.sh does; nothing else has to).
# Why ON is the right default: `lr` is the DEFAULT ADV_SCHEME, so a default of OFF
# silently reintroduced the spurious number source for any other launcher.
# ADV_VPOS=0 IS NOT A SUPPORTED CONFIGURATION -- a forensic escape hatch only, kept
# so an old log can be re-derived when diagnosing one. It reproduces a run whose
# NUMBER FIELD WAS CORRUPT (floor manufactured ~3.3e-3 of the standing number burden
# per 6 h step, 97.4% below 10 nm). No research question wants it.
VPOS = os.environ.get('ADV_VPOS', '1') != '0'


def _vert_positive(rhoq, rhoq_n, Mc_t, Mt, dp):
    """Rescale the vertical interface fluxes so no cell goes negative.

    Conservative: only the shared face fluxes are scaled. Returns
    (rhoq_n_limited, f_top, f_bot) with the boundary fluxes consistent.
    
    Before:
    cell0: 1 + (0 − 3)    = −2   ← negative
    cell1: 2 + (3 − (−1)) =  6
    cell2: 5 + (−1 − 0)   =  4

    After:
    cell0: −2 + (0 − (−2)) = 0
    cell1:  6 + (−2 − 0)   = 4
    cell2:  4 + (0 − 0)    = 4



    """
    #Rebuild the interface fluxes
    F = Mc_t - Mt                                  # (ncol, nlev+1) net flux in
    avail = jnp.maximum(rhoq, 0.0) * dp[None, :]    # tracer available to drain
    # outflow demanded from cell k: F_k<0 drains it, F_{k+1}>0 drains it
    out = jnp.maximum(-F[:, :-1], 0.0) + jnp.maximum(F[:, 1:], 0.0)
    # NO epsilon floor here: the production driver runs this in FLOAT32
    # (driver_fast.py:119, ADV_F32=1 by default) where a guard like
    # jnp.maximum(out, 1e-300) UNDERFLOWS TO ZERO -- float32's smallest normal is
    # ~1.2e-38 -- and the division then returns inf/NaN on every cell with no
    # outflow, which is most of them. Mask the divide instead of flooring it.
    safe = out > 0.0
    r = jnp.clip(jnp.where(safe, avail / jnp.where(safe, out, 1.0), 1.0), 0.0, 1.0)
    # interface k is limited by r[k-1] when F_k>0, by r[k] when F_k<0.
    # Pad with 1.0 at the ends so genuine INFLOW across a domain face is untouched.
    r_up = jnp.concatenate([jnp.ones((r.shape[0], 1), r.dtype), r], axis=1)
    r_dn = jnp.concatenate([r, jnp.ones((r.shape[0], 1), r.dtype)], axis=1)
    s = jnp.where(F > 0.0, r_up, r_dn)
    # Apply only the CORRECTION dF = Fl - F, never a from-scratch recomputation.
    # Where nothing is limited, s == 1 so dF is EXACTLY 0.0 and rhoq_n is returned
    # BITWISE UNCHANGED. Recomputing as rhoq + (Fl_k - Fl_k+1)/dp instead is
    # algebraically identical but numerically worse: F is a difference of
    # cumulative integrals, so in the production FLOAT32 path the cancellation
    # cost 2.9e-5 relative on a SMOOTH field with no negatives present -- a price
    # paid on every tracer, mass and gases included, to fix a number-only defect.
    # (In float64 the same formulation cost only 1.2e-13, which is why this only
    # showed up once the validation was rerun in the production precision.)
    # Still exactly conservative: dF is one number per interface applied with
    # opposite signs to the two cells it joins, so it telescopes.
    dF = F * (s - 1.0)
    rhoq_l = rhoq_n + (dF[:, :-1] - dF[:, 1:]) / dp[None, :]
    # corrected tracers and new fluxes
    return rhoq_l, F[:, 0] + dF[:, 0], F[:, -1] + dF[:, -1]


# ---- one 3-D substep ------------------------------------------------------
def _lr_step_3d(rho, rhoq, u, v, w, dt, lat, dp, polar, qfroz, mf, ac,
                pol_mode, dlam, dphi, w_face):
    nlev, nlat, nlon = rhoq.shape
    if pol_mode == 1:
        #stir the caps in rho*q AND rho (each conservative), so q = rhoq/rho is
        #well mixed and no mass moves between caps. Must precede the x-sweep:
        #cx ~ 1/cos(lat) overflows int32 at the +-90 rows and only a zonally
        #uniform row survives that exactly (see fct_fast._step_3d).
        rho = _mix_caps(rho, polar, lat, ac)
        rhoq = _mix_caps(rhoq, polar, lat, ac)

    # ---- x (periodic). No integer-shift FFSL here: the shift is a permutation
    # of CELL CONTENTS, which is only mass-preserving for the advective form. In
    # flux form the fluxes must be the real ones, so we rely on substepping to
    # keep |cx| < 1 in the resolved rows and on the cap stirring at the poles.
    # average u to the cell right face (uf) and compute the courant number cx for the longitude sweep (convert to meteres)
    # expand shape for cx to (1,nlat,1) for broadcasting as cx is used in ffsl_x which expects (nlev*nlat,nlon)
    uf = 0.5*(u + jnp.roll(u, -1, 2))
    cx = uf * dt / (RAD * jnp.cos(lat*DEG) * dlam)[None, :, None]
    #inside the caps the row is uniform, so clamp the (meaningless, ~1e18) cx
    #there to keep the arithmetic finite; a uniform row is invariant anyway.
    cx = jnp.where(polar[None, :, None], 0.0, cx)
    sh = lambda A: A.reshape(nlev*nlat, nlon)
    r2, t2 = _lr_sweep(sh(rho), sh(rhoq), sh(cx), periodic=True)
    rho = r2.reshape(nlev, nlat, nlon); rhoq = t2.reshape(nlev, nlat, nlon)

    # ---- y (non-periodic, spherical metric)
    vf = 0.5*(v + jnp.roll(v, -1, 1))
    cy = vf * dt / (RAD * dphi)
    tr = lambda A: A.transpose(0, 2, 1).reshape(nlev*nlon, nlat)
    r2, t2 = _lr_sweep(tr(rho), tr(rhoq), tr(cy), periodic=False, mf=mf, ac=ac)
    back = lambda A: A.reshape(nlev, nlon, nlat).transpose(0, 2, 1)
    rho = back(r2); rhoq = back(t2)

    # ---- vertical
    col = lambda A: A.reshape(nlev, -1).T
    rho_c, rhoq_c, f_top, f_bot = _lr_vert(
        col(rho), col(rhoq), dp, dt, w_face.reshape(nlev + 1, -1).T,
        bc_top=qfroz[0].reshape(-1), bc_bot=qfroz[-1].reshape(-1))
    rho = rho_c.T.reshape(nlev, nlat, nlon)
    rhoq = rhoq_c.T.reshape(nlev, nlat, nlon)
    vflux = jnp.stack([f_top.reshape(nlat, nlon), f_bot.reshape(nlat, nlon)])

    if pol_mode == 1:
        rho = _mix_caps(rho, polar, lat, ac)
        rhoq = _mix_caps(rhoq, polar, lat, ac)
    else:
        rhoq = jnp.where(polar[None, :, None], qfroz * rho, rhoq)
    return rho, rhoq, vflux


def _lr_step_batch(rho, rhoqb, u, v, w, dt, lat, dp, polar, qfrozb,
                   mf, ac, pol_mode, dlam, dphi):
    """rho is SHARED by every tracer (it is the air), so it is advected once and
    the per-tracer sweeps reuse it. Air flux therefore cannot disagree between
    tracers -- which is what keeps every tracer's budget closed by the same
    amount.
    
 
    Computes the continuity-derived vertical velocity once, then vmaps the full 
    3-D step over the tracer stack with all tracers sharing the same air field 
    and winds, keeping one copy of the resulting rho
    
    omega_{k+1/2} = omega_{k-1/2} - dp_k * (Sx + Sy)_k

    Continuity holds in the atmosphere, and it held in CESM. It breaks in the handoff
    due to potential regridding, temporal sampling, and surface pressure tendencies.
    """


    w_face = _omega_continuity(u, v, w, lat, dp, polar, mf, ac, dlam, dphi)
    one = lambda rq, qf: _lr_step_3d(rho, rq, u, v, w, dt, lat, dp, polar, qf,
                                     mf, ac, pol_mode, dlam, dphi, w_face)
    rho_n, rhoq_n, vfl = jax.vmap(one, in_axes=(0, 0))(rhoqb, qfrozb)
    #every tracer advected the same air, so the rho copies are identical; keep one
    # air field, tracer fields, and the vertical fluxes returned
    return rho_n[0], rhoq_n, vfl


@partial(jit, static_argnums=(13,))
def _lr_loop(rho, rhoqb, u0, v0, w0, u1, v1, w1, dt_sub, n, lat, dp, polar,
             pol_mode, qfrozb, mf, ac, dlam, dphi):
    def body(i, carry):
        rho_, rq_, vf_ = carry
        a = (i.astype(u0.dtype) + 0.5) / n
        ## wind interpolation
        u = (1.0 - a)*u0 + a*u1
        v = (1.0 - a)*v0 + a*v1
        w = (1.0 - a)*w0 + a*w1
        #advance current air and tracer state by one substep
        rho_, rq_, vfs = _lr_step_batch(rho_, rq_, u, v, w, dt_sub, lat, dp,
                                        polar, qfrozb, mf, ac, pol_mode,
                                        dlam, dphi)
        return rho_, rq_, vf_ + vfs
    vf0 = jnp.zeros((rhoqb.shape[0], 2) + rhoqb.shape[2:], rhoqb.dtype)
    return jax.lax.fori_loop(0, n, body, (rho, rhoqb, vf0))


def advect_hour_batch(qb0, u0, v0, w0, u1, v1, w1, lat, dp, qfrozb,
                      lat_freeze=80.0, cfl=0.2, dt_total=3600.0,
                      dtype=jnp.float64, rho0=None, polar_mode=None,
                      return_vflux=False, return_rho=False, rho_reset=True):
    """Advect a tracer stack in Lin-Rood flux form.

    qb0  : (ntracer,nlev,nlat,nlon) MIXING RATIOS
    rho0 : (nlev,nlat,nlon) air-mass factor carried in; None starts it at 1.

    rho_reset=True (default) remaps the air mass back onto the fixed pressure
    levels at the end of the call -- the standard offline-CTM "pressure fixer",
    applied cell-locally:

        the transport says this cell holds rho*dp of air carrying mixing ratio q,
        but the grid insists the cell holds dp, so the mixing ratio that preserves
        the cell's TRACER MASS is q' = rho*q -- and since q = rho*q/rho, that is
        just the advected rho*q itself.

    Two consequences worth being explicit about:
      * sum(A*q') equals the exactly-conserved sum(A*rho*q), so the CALLER'S
        ORDINARY BURDEN DIAGNOSTIC IS EXACT. No rho needs to be carried between
        steps and no diagnostic has to change -- this is a drop-in replacement.
      * it is not free: q is rescaled by rho (~1%/step), which is a real
        redistribution of concentration. That is the honest price of insisting on
        fixed pressure levels while the winds do not conserve air mass on them.
        Resetting EVERY step keeps the rescaling small; letting rho run for days
        lets it reach tens of percent (measured: rms 2.5%, extremes +26% at 4 d).

    rho_reset=False returns the mixing ratio with rho still carried, in which case
    the conserved quantity is sum(A*rho*q) and the caller MUST persist rho.

    VALIDITY: rho must stay POSITIVE. The flux telescoping is exact regardless,
    but q = rho*q/rho loses all precision if rho approaches zero, so a wind so
    inconsistent that it drives the air mass through zero is a hard failure rather
    than a small error. Measured with real CESM winds + continuity omega, rho
    stays in [0.81, 1.26] over 4 days with rms(rho-1) saturating near 2.5%; with
    rho_reset (the default) it never accumulates past one step (~1%). Pass
    return_rho=True and watch rho.min() if you change the wind source -- this is
    the one number that says whether flux form is still trustworthy.

    Returns (qb, nsub[, vflux][, rho]).
    """
    lat_np = np.asarray(lat); dp_np = np.asarray(dp)
    nlon = np.asarray(qb0).shape[-1]
    dlam = 2.0 * np.pi / float(nlon)
    dphi = float(lat_np[1] - lat_np[0]) * DEG
    if polar_mode is None:
        polar_mode = os.environ.get('ADV_POLAR', 'zonal').lower()
    pol_mode = 1 if polar_mode == 'zonal' else 0

    #substep count: |cy| and, unlike fct_fast, also |cx| must stay < 1 because
    #flux form has no integer-shift FFSL. cx is evaluated on the RESOLVED rows
    #only -- the caps are stirred and their cx is zeroed.
    keep = np.abs(lat_np) <= lat_freeze
    umax = max(float(np.abs(np.asarray(u0)[:, keep]).max()),
               float(np.abs(np.asarray(u1)[:, keep]).max()))
    vmax = max(float(np.abs(np.asarray(v0)).max()), float(np.abs(np.asarray(v1)).max()))
    dx_min = float((RAD * np.cos(lat_np[keep] * DEG) * dlam).min())
    dt_x = cfl * dx_min / max(umax, 1e-6)
    dt_y = cfl * (RAD * dphi) / max(vmax, 1e-6)
    dt_sub = min(dt_x, dt_y)
    n = int(np.ceil(dt_total / dt_sub)); dt_sub = dt_total / n

    _mf, _ac = grid_metric(lat_np, lat_freeze if pol_mode == 1 else None)
    mf = jnp.asarray(_mf, dtype=dtype); ac = jnp.asarray(_ac, dtype=dtype)
    polar = jnp.asarray(np.abs(lat_np) > lat_freeze)
    lat_j = jnp.asarray(lat_np, dtype=dtype); dp_j = jnp.asarray(dp_np, dtype=dtype)
    qb = jnp.asarray(np.asarray(qb0), dtype=dtype)
    rho = (jnp.ones(qb.shape[1:], dtype) if rho0 is None
           else jnp.asarray(np.asarray(rho0), dtype=dtype))
    qfr = jnp.asarray(np.asarray(qfrozb), dtype=dtype)
    w6 = [jnp.asarray(np.asarray(x), dtype=dtype)
          for x in (u0, v0, w0, u1, v1, w1)]

    rho_n, rhoq_n, vfl = _lr_loop(rho, qb * rho[None], *w6, dt_sub, n, lat_j,
                                  dp_j, polar, pol_mode, qfr, mf, ac, dlam, dphi)
    #q' = rho*q == rhoq when the air mass is remapped back onto the fixed levels
    qb_n = (rhoq_n if rho_reset else rhoq_n / rho_n[None]).astype(jnp.float64)
    out = [qb_n, n]
    if return_vflux:
        out.append(vfl.astype(jnp.float64))
    if return_rho:
        out.append(rho_n.astype(jnp.float64))
    return tuple(out)
