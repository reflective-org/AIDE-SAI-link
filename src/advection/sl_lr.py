"""Semi-Lagrangian tracer transport for the coupling (ADV_SCHEME=sl).

WHY THIS EXISTS
---------------
fct_lr.py (Lin-Rood flux form, PPM + Zalesak) is CFL-limited in both
horizontal sweeps; the sub-step is set by the ~24 km longitude spacing at the
80 deg cap edge, so a 6-h coupling step takes ~180-270 sub-steps and 6.6 s of
the 21 s emulator step (docs/TRANSPORT_SL_HANDOFF.md). A semi-Lagrangian
remap has no CFL limit: per sub-step it is one departure-point solve per
level plus one gather per tracer, whatever the wind. The kernels are
dinosaur's (sl_core.py, vendored with a pole-row patch).

TWO VERTICAL TREATMENTS (ADV_SL_VERT)
-------------------------------------
flux (default, 2026-10-13 after the A/B below)  -- THE SPLIT OPERATOR
    per sub-step of dt_total/K:
      1. horizontal: 2-D departure points per level (2 fixed-point
         iterations, Cartesian), cubic 4x4 gather with the quasi-monotone
         limiter, per tracer. Each level is a CLOSED sphere, so the exact
         mass target of every (tracer, level) is its mass before the remap,
         sum(ac_j q); the fixer (global or Bermejo-Conde local) hits it to
         float64 roundoff.
      2. vertical: fct_lr._lr_vert on every column -- the production scheme's
         own flux-form PPM remap with its OPEN faces (inflow served at the
         reservoir value qfroz[0]/qfroz[-1], free outflow), its positivity
         limiter (ADV_VPOS) and its reported face fluxes -- run with air mass
         rho = 1 and taken in ADVECTIVE form, q2 = (rho q)'/rho', so a uniform
         mixing ratio stays uniform (the flux-form rescale q' = (rho q)' would
         imprint dt * div_h(v) per sub-step, because the SL horizontal step
         moves mixing ratio, not air). The column mass is then fixed to the
         flux-form target M1 + sum(ac (f_top - f_bot)), which is exactly what
         fct_lr conserves, so the coupling's B_vf_in/out diagnostic and its
         "advection numerical residual" mean what they mean under lr.
         Face omega: ADV_SL_OMEGA=cont (default) rederives it from discrete
         continuity with fct_fast._omega_continuity, the same operator and
         top-face anchor lr uses, so the boundary physics is lr's; cesm
         averages the CESM center omega to the faces instead.
    This is lr with its two CFL-limited horizontal sweeps replaced by one SL
    remap. ADV_WCONT=1 (open faces, BC_EDGE=open, reservoir refresh of qfroz,
    BC_BOT_AER) applies exactly as under lr.

sl  -- the handoff's 3-D design, kept for the record
    3-D departure points (omega at the centers, pressure clipped to the
    center range = closed extended column), cubic x linear 3-D gather, one
    fixer per tracer on the closed column, Dirichlet reset of the edge levels
    by the coupling (ADV_WCONT=0). FAILED the 30-day A/B: resetting the whole
    edge LEVEL each step is a ~3x stronger sink than the flux through its FACE
    (B_bc -0.23..-0.29 M0 vs lr's B_vf -0.085 M0 over 30 d; mass/SO2/AOD
    9-15 % low at day 30). docs/PROCESSES.md 1.8.

Both: the polar caps (|lat| > lat_freeze) follow ADV_POLAR as under lr:
'zonal' (default) stirs each cap into one well-mixed cell per level after
every sub-step (fct_fast._mix_caps, area-weighted, exactly conserving, so the
coupling's "advection numerical residual" stays at roundoff); 'freeze'
overwrites the cap rows from qfrozb after the step, which discards what
transport delivered there (~6e-5 M0 per step in the 1-day run, booked as the
residual). The SL operator itself has no polar singularity. The gathers
and the vertical sweep run in `dtype` (float32 in production); the stack is
carried between sub-steps in float64 and every fixer sum is float64.

THE CALLER'S CONTRACT (coupling.py)
-----------------------------------
Same positional signature as fct_lr.advect_step_batch; `cfl` is accepted and
ignored (the sub-step count is ADV_SL_K). `return_vflux=True` appends the
summed signed face fluxes (ntr, 2, nlat, nlon) in rho*q*Pa units, [0] = top
face (> 0 inflow), [1] = bottom face (> 0 outflow), exactly fct_lr's; zeros
under ADV_SL_VERT=sl (closed column).

Env knobs (read once at import, like every other module here):
  ADV_SL_K       sub-steps per coupling step (default 6 = hourly at 6 h)
  ADV_SL_FIXER   global | local | none         (default global)
  ADV_SL_VERT    flux | sl                     (default flux)
  ADV_SL_OMEGA   cont | cesm  face omega, flux mode (default cont)
  ADV_SL_VORDER  linear | cubic vertical gather, sl mode (default linear)
  ADV_SL_ITER    departure-point iterations    (default 2)
  ADV_SL_SPLIT   strang | lie  V/2-H-V/2 with the air mass carried (default,
                 2026-10-14; see _strang_substep) or the first-order H-V
  ADV_SL_RHO     0 | 1  lie only: carry the air mass through the horizontal
                 step (strang always does; see below)
  ADV_SL_JAC     1 | 0  the air mass after the horizontal step from the
                 Jacobian of the departure map and the bookkeeping remainder
                 returned through the faces (default; see _jacobian and
                 _remainder_sweep), or the Eulerian 1 - dt div_h v with the
                 remainder spread by the column fixer (first order in K)

AIR MASS IN THE HORIZONTAL STEP (ADV_SL_RHO / ADV_SL_SPLIT, 2026-10-14)
-----------------------------------------------------------------------
The SL remap moves MIXING RATIO with the air mass held at 1. Under a divergent
horizontal wind that is wrong by O(dt div_h v) per sub-step: where the air
converges, tracer mass per unit area must rise with the air density (the
vertical sweep then carries it away with the continuity omega, which is
exactly -div_h v integrated down the column). With rho frozen the remap
loses that mass where the flow converges and gains it where it diverges, and
the per-level fixer puts the difference back as ONE multiplicative rescale
of the whole level, every sub-step: a spurious horizontal transfer of ~0.4 %
of a level's mass per hour that does not shrink with K (first order in dt,
summed over the step). Over months it moved plume mass into the extratropical
downwelling: the one-year A/B exported 74 % more through the bottom face than
lr by day 172 (-13 % burden). The analytic tests (non-divergent flows) could
not see it. ADV_SL_RHO=1 does what lr's flux-form sweeps do implicitly:
    rho_h = 1 - dt div_h v = 1 + dt (omega_{k+1/2} - omega_{k-1/2}) / dp_k
from the SAME discrete divergence _omega_continuity integrates, so the
per-level fixer targets sum(ac rho_h q) = sum(ac q_before) (now an O(dt^2)
correction instead of O(dt)), and the vertical sweep starts from rho_h
instead of 1 and returns the air mass to 1 to O(dt^2).

That repaired the vertical distribution (30-day A/B: mid-levels -5 % of lr
instead of -25..-49 %) but moved the error to the open bottom face: with
H(dt) V(dt) the sweep sees the whole sub-step's convergence at once and the
face flux rides on rho_h ~ 1.3 for the hour (export +37 % vs lr at K = 6,
+9 % at K = 24: first order in dt). ADV_SL_SPLIT=strang (the default) centres
the horizontal step between two half vertical sweeps, which makes the face
term second order with still one SL gather per sub-step: 30-day A/B at K = 6
mass -1.2 %, export +16 %, settling and micro within 0.1 %, closed-column
3-D truth test L2 0.0064 vs lr 0.0089 (scripts/validation/
compare_3d_operators.py). The remaining export spread is lr's diffusion of
the plume's lower edge plus the hourly winds (docs/PROCESSES.md 1.8).
"""
import os
import functools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import sl_core as S
from fct_fast import grid_metric, _omega_continuity, _mix_caps, DEG
from fct_lr import _lr_vert

ADV_SL_K = int(os.environ.get('ADV_SL_K', '6'))
ADV_SL_FIXER = os.environ.get('ADV_SL_FIXER', 'global').lower()
ADV_SL_VERT = os.environ.get('ADV_SL_VERT', 'flux').lower()
ADV_SL_OMEGA = os.environ.get('ADV_SL_OMEGA', 'cont').lower()
ADV_SL_VORDER = os.environ.get('ADV_SL_VORDER', 'linear').lower()
ADV_SL_ITER = int(os.environ.get('ADV_SL_ITER', '2'))
ADV_SL_RHO = os.environ.get('ADV_SL_RHO', '0') != '0'
ADV_SL_SPLIT = os.environ.get('ADV_SL_SPLIT', 'strang').lower()
ADV_SL_JAC = os.environ.get('ADV_SL_JAC', '1') != '0'
_SPLITS = ('lie', 'strang')
if ADV_SL_SPLIT not in _SPLITS:
    raise ValueError(f"ADV_SL_SPLIT={ADV_SL_SPLIT!r}; expected one of {_SPLITS}")
_FIXERS = ('none', 'global', 'local')
_VERTS = ('flux', 'sl')
_OMEGAS = ('cont', 'cesm')
for _name, _val, _ok in (('ADV_SL_FIXER', ADV_SL_FIXER, _FIXERS),
                         ('ADV_SL_VERT', ADV_SL_VERT, _VERTS),
                         ('ADV_SL_OMEGA', ADV_SL_OMEGA, _OMEGAS)):
    if _val not in _ok:
        raise ValueError(f"{_name}={_val!r}; expected one of {_ok}")


def describe():
    vert = (f"vertical = fct_lr flux-form PPM on open faces ({ADV_SL_OMEGA} omega)"
            if ADV_SL_VERT == 'flux' else
            f"3-D gather ({ADV_SL_VORDER} vertical) on the closed column")
    return (f"semi-Lagrangian (sl_lr): K={ADV_SL_K} sub-steps, cubic horizontal gather + "
            f"quasi-monotone limiter, {vert}, {ADV_SL_ITER} departure iterations, "
            f"mass fixer={ADV_SL_FIXER}"
            + ("" if ADV_SL_VERT != 'flux' else
               ", Strang V/2-H-V/2 with the air mass carried (ADV_SL_SPLIT=strang"
               + (", Jacobian air mass ADV_SL_JAC=1)" if ADV_SL_JAC else ", Eulerian air mass ADV_SL_JAC=0)")
               if ADV_SL_SPLIT == 'strang' else
               ", Lie H-V with the air mass carried (ADV_SL_SPLIT=lie ADV_SL_RHO=1)"
               if ADV_SL_RHO else
               ", Lie H-V with the air mass frozen at 1 (ADV_SL_SPLIT=lie ADV_SL_RHO=0)"))


# ---- mass fixers (sums in float64) -------------------------------------------
def _fix_mass(qH, qL, m, M0, fixer):
    """Fix the field qH (any shape) to the mass target M0 under the weights m
    (broadcastable to qH). qL is the low-order (linear) answer at the same
    departure points for the local fixer. Returns float64."""
    qH = qH.astype(jnp.float64)
    if fixer == 'none':
        return qH
    M1 = jnp.sum(m * qH)
    if fixer == 'global':
        return qH * jnp.where(M1 > 0, M0 / jnp.where(M1 > 0, M1, 1.0), 1.0)
    # Bermejo-Conde: take the error only from cells where the high-order
    # answer overshoots the low-order one in the direction of the error.
    D = M1 - M0
    w = jnp.maximum(0.0, jnp.sign(D) * (qH - qL.astype(jnp.float64)))
    Sw = jnp.sum(m * w)
    # |lambda| <= 1 keeps q between q_L and q_H (positivity); the sign follows D
    lam = jnp.sign(D) * jnp.minimum(jnp.where(Sw > 0, jnp.abs(D) / jnp.where(Sw > 0, Sw, 1.0), 0.0), 1.0)
    q = qH - lam * w
    M2 = jnp.sum(m * q)
    return q * jnp.where(M2 > 0, M0 / jnp.where(M2 > 0, M2, 1.0), 1.0)


# ---- the two sub-step bodies ---------------------------------------------------
def _rho_h(w_face, dp):
    """Air mass after the horizontal step, 1 - dt div_h v, from the face omega
    the vertical sweep will apply. w_face: dt * omega at the faces (nlev+1,
    nlon, nlat) [Pa]; dp (nlev,) [Pa]. Returns (nlev, nlon, nlat) float64."""
    return 1.0 + (w_face[1:] - w_face[:-1]).astype(jnp.float64) / jnp.asarray(dp, jnp.float64)[:, None, None]


def _jacobian(lon_dep, sin_lat_dep, grid, cap):
    """Air-mass ratio of the horizontal step from the departure map itself:
    rho_arrival / rho_departure = |d(lambda_d, mu_d) / d(lambda, mu)|, the
    area Jacobian in the equal-area coordinates (lon, sin lat), by central
    differences over the arrival grid (lon periodic, the displacement
    unwrapped). WHY (2026-10-14): the Eulerian 1 - dt div_h v is the
    Jacobian to first order only, so the per-level fixer still moved
    O((dt div)^2) of a level's mass uniformly every sub-step (2e-4 at
    dt div = 0.03, i.e. ~2 % per hour at the 143 hPa face where dt div ~
    0.3) and the coupled run converged to lr at FIRST order in the sub-step
    (export excess 14.7 / 7.9 / 4.5 % at K = 6 / 12 / 24). The Jacobian of
    second-order departure points cuts that load ~30x at one hour. Inside
    the caps (`cap` (nlat,) bool, stirred afterwards anyway) and on the two
    pole rows the caller's Eulerian value is kept. lon_dep, sin_lat_dep
    (nlev, nlon, nlat); returns (nlev, nlon, nlat) f64 or None where cap."""
    lam = jnp.asarray(grid.longitudes, jnp.float64)                      # (nlon,)
    mu = jnp.asarray(np.sin(grid.latitudes), jnp.float64)                # (nlat,)
    dlam = lam[1] - lam[0]
    delta = jnp.mod(lon_dep.astype(jnp.float64) - lam[None, :, None] + jnp.pi, 2 * jnp.pi) - jnp.pi
    mud = sin_lat_dep.astype(jnp.float64)
    d_lam = lambda f: (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dlam)
    def d_mu(f):                                                          # non-uniform central, one-sided ends
        num = f[:, :, 2:] - f[:, :, :-2]
        den = (mu[2:] - mu[:-2])[None, None, :]
        lo = (f[:, :, 1:2] - f[:, :, :1]) / (mu[1] - mu[0])
        hi = (f[:, :, -1:] - f[:, :, -2:-1]) / (mu[-1] - mu[-2])
        return jnp.concatenate([lo, num / den, hi], axis=2)
    J = (1.0 + d_lam(delta)) * d_mu(mud) - d_mu(delta) * d_lam(mud)
    return jnp.clip(J, 0.1, 10.0), cap[None, None, :]


def _horizontal_substep(q, u, v, grid, dt, ac2, order, limiter, iterations, fixer, gdtype,
                        rho_in=None, rho_out=None, jac=False, cap=None):
    """SL remap of q (ntr, nlev, nlon, nlat) along 2-D trajectories, one set
    per level, with the per-(tracer, level) mass fixer. ac2: (1, nlat) f64.
    rho_in / rho_out (nlev, nlon, nlat) f64, optional: the air mass before and
    after the horizontal step (None = 1); the fixer targets
    sum(ac rho_out q_after) = sum(ac rho_in q_before) -- each level is a
    closed sphere, so that is exact for the tracer mass. With `jac` the air
    mass after the step is rho_in(departure) * Jacobian instead (_jacobian),
    with the given Eulerian rho_out kept inside the caps. Returns (q_after,
    rho_out as used)."""
    dep = S.horizontal_departure_points(u, v, grid, dt, iterations=iterations)
    lon, sin_lat = S.cartesian_to_lon_sin_lat(dep.cartesian)
    st = S.stencil_2d_levels(grid, lon, sin_lat, order)
    if jac and rho_out is not None:
        J, incap = _jacobian(lon, sin_lat, grid, cap)
        if rho_in is not None:
            stR = S.stencil_2d_levels(grid, lon, sin_lat, 'linear')
            J = J * S.gather_2d_levels(rho_in.astype(gdtype), grid, stR, None).astype(jnp.float64)
        rho_out = jnp.where(incap, rho_out, J)
    M0 = jnp.sum(ac2[None, None] * (q if rho_in is None else rho_in[None] * q), axis=(2, 3))   # (ntr, nlev)
    qg = q.astype(gdtype)
    qH = jax.vmap(lambda f: S.gather_2d_levels(f, grid, st, limiter))(qg)
    if fixer == 'local':
        stL = S.stencil_2d_levels(grid, lon, sin_lat, 'linear')
        qL = jax.vmap(lambda f: S.gather_2d_levels(f, grid, stL, None))(qg)
    else:
        qL = qH
    fix = functools.partial(_fix_mass, fixer=fixer)
    if rho_out is None:
        per_level = jax.vmap(lambda a, b, c: fix(a, b, ac2, c))        # over levels
        return jax.vmap(per_level)(qH, qL, M0), None                   # over tracers
    per_level = jax.vmap(lambda a, b, c, r: fix(a, b, ac2 * r, c))
    return jax.vmap(lambda a, b, c: per_level(a, b, c, rho_out))(qH, qL, M0), rho_out


def _vertical_sweep(q1, qfroz, w_face, dp_g, dt, gdtype, rho_in=None):
    """fct_lr._lr_vert on every column over dt. q1 (ntr, nlev, nlon, nlat) f64
    MIXING RATIO; rho_in (nlev, nlon, nlat) f64 the air mass it rides on (None
    = 1); qfroz like q1 (its edge levels are the open-face inflow values);
    w_face (nlev+1, nlon, nlat) omega [Pa/s]. The open-face inflow: _lr_vert
    extends the AIR outside the slab at the edge cell's density, so the
    TRACER inflow is that density times the reservoir mixing ratio (or a
    uniform field is no longer uniform at the edge levels once rho != 1).
    Returns (rho_n (nlev, nlon, nlat), q2 = (rho q)'/max(rho', 0.01),
    rhoq_n, vfl (ntr, 2, nlon, nlat)), all f64. rho' is the thickness of the
    departure layer over dp; it is 0 where trajectories cross (_lr_vert's
    cummax), which needs |d omega/dp| dt > 1 -- never in CESM winds, but
    reachable with synthetic ones; the floor turns 0/0 into the flux-form
    answer there (the cell is emptied), smoothly and positively."""
    ntr, nlev, nlon, nlat = q1.shape
    ncol = nlon * nlat
    col = lambda a: a.reshape(a.shape[0], nlev, ncol).swapaxes(1, 2)   # (ntr, ncol, nlev)
    uncol = lambda a: a.swapaxes(1, 2).reshape(ntr, nlev, nlon, nlat)
    q1c = col(q1)
    wf = w_face.reshape(nlev + 1, ncol).T.astype(gdtype)
    rho64 = (jnp.ones((ncol, nlev), jnp.float64) if rho_in is None
             else rho_in.reshape(nlev, ncol).T)
    rho = rho64.astype(gdtype)
    bct = (rho64[None, :, 0] * qfroz[:, 0].reshape(ntr, ncol)).astype(gdtype)
    bcb = (rho64[None, :, -1] * qfroz[:, -1].reshape(ntr, ncol)).astype(gdtype)
    rhoq_in = (rho64[None] * q1c).astype(gdtype)
    # rho' depends on no batched input, so out_axes=None keeps ONE copy of it
    rho_n, rhoq_n, f_top, f_bot = jax.vmap(
        lambda rq, bt, bb: _lr_vert(rho, rq, dp_g, dt, wf, bt, bb),
        out_axes=(None, 0, 0, 0))(rhoq_in, bct, bcb)
    rho_n = rho_n.astype(jnp.float64); rhoq_n = rhoq_n.astype(jnp.float64)
    q2 = rhoq_n / jnp.maximum(rho_n, 0.01)[None]
    vfl = jnp.stack([f_top, f_bot], axis=1).reshape(ntr, 2, nlon, nlat).astype(jnp.float64)
    return rho_n.T.reshape(nlev, nlon, nlat), uncol(q2), uncol(rhoq_n), vfl


def _remainder_sweep(rhoq_n, rho_n, qfroz, dp):
    """Return the air mass to exactly 1 by sending the sub-step's bookkeeping
    remainder rho_n - 1 (O(dt^2): the Lagrangian horizontal convergence J
    minus what the two half sweeps took out with the Eulerian omega) down the
    column as a face flux at the LOCAL upwind mixing ratio, the bottom face
    included (reservoir value on inflow). WHY (2026-10-14): dividing it out
    and closing the mass with the global column fixer spread an O(dt^2)
    local remainder uniformly over the globe every sub-step -- a first-order
    transfer that kept the coupled run converging to lr at first order in K
    (export excess 10 / 3.3 % at K = 6 / 24 with the Jacobian alone). Here
    the remainder stays in its column as the vertical transport it is, a
    uniform field stays exactly uniform, and dM == f_top - f_bot holds
    without a fixer. rhoq_n (ntr, nlev, nlon, nlat), rho_n (nlev, nlon,
    nlat), qfroz like rhoq_n, dp (nlev,). Returns (rhoq with rho = 1,
    vfl_add (ntr, 2, nlon, nlat) with the bottom-face flux, > 0 outflow)."""
    dp64 = jnp.asarray(dp, jnp.float64)[:, None, None]
    phi = jnp.cumsum((rho_n - 1.0) * dp64, axis=0)            # excess air through face k+1/2 [Pa], > 0 down
    q = rhoq_n / jnp.maximum(rho_n, 0.01)[None]
    phi_int = phi[:-1][None]
    F_int = phi_int * jnp.where(phi_int > 0, q[:, :-1], q[:, 1:])
    phi_bot = phi[-1][None]
    F_bot = phi_bot * jnp.where(phi_bot > 0, q[:, -1], qfroz[:, -1])
    F = jnp.concatenate([jnp.zeros_like(F_bot)[:, None], F_int, F_bot[:, None]], axis=1)
    rhoq = rhoq_n - (F[:, 1:] - F[:, :-1]) / dp64[None]
    vfl_add = jnp.stack([jnp.zeros_like(F_bot), F_bot], axis=1)
    return rhoq, vfl_add


def _column_fix(q2, M_ref, vfl, m, ac2):
    """Global per-tracer rescale of q2 (ntr, nlev, nlon, nlat) to the flux-form
    target M_ref + sum(ac (f_top - f_bot)) -- exactly what fct_lr conserves.
    m (nlev, 1, nlat) = dp * ac f64; ac2 (1, nlat) f64. The correction is the
    rho' != 1 remainder of the advective division: O(dt^2) once the air mass
    is carried (ADV_SL_RHO / strang), O(dt div) with rho frozen at 1."""
    target = M_ref + jnp.sum(ac2[None] * (vfl[:, 0] - vfl[:, 1]), axis=(1, 2))
    M2 = jnp.sum(m[None] * q2, axis=(1, 2, 3))
    return q2 * jnp.where(M2 > 0, target / jnp.where(M2 > 0, M2, 1.0), 1.0)[:, None, None, None]


def _lie_substep(q, qfroz, u, v, wf, grid, dt, dp, dp_g, m, ac2, order, limiter, iterations,
                 fixer, gdtype, jac, cap, rho_mode):
    """H(dt) then V(dt). With rho_mode the air mass 1 - dt div_h v the
    horizontal step leaves is carried into the vertical sweep (ADV_SL_RHO=1);
    otherwise rho is 1 throughout (the 2026-10-13 split)."""
    rho_h = _rho_h(dt * wf, dp) if rho_mode else None
    q1, rho_h = _horizontal_substep(q, u, v, grid, dt, ac2, order, limiter, iterations, fixer, gdtype,
                                    rho_out=rho_h, jac=jac, cap=cap)
    rho_n, q2, rhoq_n, vfl = _vertical_sweep(q1, qfroz, wf, dp_g, dt, gdtype, rho_in=rho_h)
    if jac and rho_h is not None:
        q2, vadd = _remainder_sweep(rhoq_n, rho_n, qfroz, dp)
        vfl = vfl + vadd
    if fixer != 'none':
        M_ref = jnp.sum(m[None] * (q1 if rho_h is None else rho_h[None] * q1), axis=(1, 2, 3))
        q2 = _column_fix(q2, M_ref, vfl, m, ac2)
    return q2, vfl


def _strang_substep(q, qfroz, u, v, wf, grid, dt, dp, dp_g, m, ac2, order, limiter, iterations,
                    fixer, gdtype, jac, cap):
    """V(dt/2) H(dt) V(dt/2) with the air mass carried throughout (ADV_SL_SPLIT=
    strang, 2026-10-14). WHY: with Lie splitting the horizontal convergence of
    a whole sub-step lands on the column before the vertical sweep sees any of
    it, so at the open bottom face the sweep exports dt * div_h v more tracer
    than the continuous dynamics (where the air mass never leaves 1): first
    order in dt, +37 % cumulative export vs lr at K = 6 in the 30-day A/B,
    +9 % at K = 24. Strang centres the convergence on the vertical motion: the
    face flux is second order and the sub-step keeps ONE SL gather (every
    extra gather smooths, so K cannot be raised instead). Vertical sweeps are
    cheap. The per-level fixer conserves sum(ac rho q) across the horizontal
    step and the global fixer closes the O(dt^2) remainder at the end."""
    rho_a, q_a, _, vf1 = _vertical_sweep(q, qfroz, wf, dp_g, 0.5 * dt, gdtype, rho_in=None)
    rho_b = rho_a * _rho_h(dt * wf, dp)                     # after the full horizontal convergence
    q_b, rho_b = _horizontal_substep(q_a, u, v, grid, dt, ac2, order, limiter, iterations, fixer, gdtype,
                                     rho_in=rho_a, rho_out=rho_b, jac=jac, cap=cap)
    rho_n, q2, rhoq_n, vf2 = _vertical_sweep(q_b, qfroz, wf, dp_g, 0.5 * dt, gdtype, rho_in=rho_b)
    vfl = vf1 + vf2
    if jac:
        q2, vadd = _remainder_sweep(rhoq_n, rho_n, qfroz, dp)
        vfl = vfl + vadd
    if fixer != 'none':
        q2 = _column_fix(q2, jnp.sum(m[None] * q, axis=(1, 2, 3)), vfl, m, ac2)
    return q2, vfl


def _vertical_sl_substep(q, w, grid, dt, p_nodes_g, m, order, limiter, vertical_order,
                         iterations, fixer, gdtype):
    """The handoff's 3-D design: one 3-D gather on the closed extended column
    and the column fixer. (The horizontal step is folded in: this replaces
    both.) q (ntr, nlev, nlon, nlat) f64; w (3, nlev, nlon, nlat)."""
    dep = S.departure_points_3d(w[0], w[1], w[2], grid, dt, p_nodes_g, iterations=iterations)
    lon, sin_lat = S.cartesian_to_lon_sin_lat(dep.cartesian)
    stH = S.stencil_3d(grid, lon, sin_lat, dep.p, p_nodes_g, order, vertical_order)
    M0 = jnp.sum(m[None] * q, axis=(1, 2, 3))
    qg = q.astype(gdtype)
    qH = jax.vmap(lambda f: S.gather_3d(f, grid, stH, limiter))(qg)
    if fixer == 'local':
        stL = S.stencil_3d(grid, lon, sin_lat, dep.p, p_nodes_g, 'linear', 'linear')
        qL = jax.vmap(lambda f: S.gather_3d(f, grid, stL, None))(qg)
    else:
        qL = qH
    return jax.vmap(lambda a, b, c: _fix_mass(a, b, m, c, fixer))(qH, qL, M0)


# ---- the jitted K-sub-step kernel --------------------------------------------
@functools.partial(jax.jit, static_argnames=('grid', 'K', 'order', 'limiter', 'vert', 'omega',
                                             'vertical_order', 'iterations', 'fixer',
                                             'gdtype', 'polar_mode', 'dphi', 'rho_mode', 'split',
                                             'jac'))
def _sl_kernel(qb, W, p_nodes, dp, lat_deg, mf, ac, m, polar_mask, qfrozb, dt_sub, *,
               grid, K, order, limiter, vert, omega, vertical_order, iterations, fixer,
               gdtype, polar_mode, dphi, rho_mode=False, split='lie', jac=False):
    """qb (ntr, nlev, nlon, nlat) f64; W (K+1, 3, nlev, nlon, nlat) [u, v, omega]
    at the sub-step nodes in the gather dtype; p_nodes/dp (nlev,); lat_deg
    (nlat,); mf/ac (nlat,) grid metric; m (nlev, 1, nlat) f64 cell weights;
    polar_mask (nlat,) bool; qfrozb like qb. Returns (q, vfl)."""
    ntr, nlev, nlon, nlat = qb.shape
    p_nodes_g = jnp.asarray(p_nodes, dtype=gdtype)
    dp_g = jnp.asarray(dp, dtype=gdtype)
    ac2 = jnp.asarray(ac, jnp.float64)[None, :]                        # (1, nlat)
    m = jnp.asarray(m, jnp.float64)                                    # (nlev, 1, nlat) dp*ac
    lat_g = jnp.asarray(lat_deg, dtype=gdtype)
    # _omega_continuity's x-divergence is u/cos(lat): ~1e17 on the +-90 rows,
    # which it zeroes only where `polar` is set. lr always has |lat| > 80 rows
    # there; with lat_freeze >= 90 (tests) the pole rows must still be excluded
    # (they take its legacy center-averaged omega), or the derived omega is
    # ~1e19 Pa/s and the vertical sweep is garbage.
    pol_cont = polar_mask | (jnp.abs(jnp.asarray(lat_deg)) >= 89.99)
    lat64 = jnp.asarray(lat_deg, jnp.float64)
    ac64 = jnp.asarray(ac, jnp.float64)

    def stir_caps(q):
        # fct_fast._mix_caps works in (lev, lat, lon) on one tracer, f64 here
        return jax.vmap(lambda f: swap(_mix_caps(swap(f), polar_mask, lat64, ac64)))(q)
    dlam = 2.0 * np.pi / nlon
    swap = lambda a: jnp.swapaxes(a, -1, -2)

    def body(k, carry):
        q, vfl = carry
        w = 0.5 * (W[k] + W[k + 1])                      # midpoint-in-time winds
        if vert == 'sl':
            q = _vertical_sl_substep(q, w, grid, dt_sub, p_nodes_g, m, order, limiter,
                                     vertical_order, iterations, fixer, gdtype)
            return (stir_caps(q) if polar_mode == 'zonal' else q), vfl
        if omega == 'cont':
            # the same face omega lr uses: continuity integrated from the top
            # face, in fct's (lev, lat, lon) layout
            wf = _omega_continuity(swap(w[0]), swap(w[1]), swap(w[2]), lat_g, dp_g,
                                   pol_cont, jnp.asarray(mf, gdtype),
                                   jnp.asarray(ac, gdtype), dlam, dphi)
            wf = swap(wf)
        else:
            wf = jnp.concatenate([w[2][:1], 0.5 * (w[2][:-1] + w[2][1:]), w[2][-1:]], axis=0)
        args = (qfrozb, w[0], w[1], wf, grid, dt_sub, dp, dp_g, m, ac2, order, limiter,
                iterations, fixer, gdtype, jac, pol_cont)
        if split == 'strang':
            q2, vf = _strang_substep(q, *args)
        else:
            q2, vf = _lie_substep(q, *args, rho_mode)
        return (stir_caps(q2) if polar_mode == 'zonal' else q2), vfl + vf

    vfl0 = jnp.zeros((ntr, 2, nlon, nlat), jnp.float64)
    q, vfl = jax.lax.fori_loop(0, K, body, (qb, vfl0))
    if polar_mode == 'freeze':
        q = jnp.where(polar_mask[None, None, None, :], qfrozb, q)
    return q, vfl


# ---- the coupling-facing driver ----------------------------------------------
def _to_sl(a, dtype):
    """(..., lat, lon) -> (..., lon, lat) in dtype. No np.asarray: the stack
    the coupling passes is already on the device (870 MB in f64), and a host
    round-trip costs ~0.5 s per call -- measured in the 1-day timing run."""
    return jnp.swapaxes(jnp.asarray(a, dtype=dtype), -1, -2)


def _resample_winds(hours, K, dtype):
    """hours: list of n >= 2 (u, v, w) tuples at equally spaced times spanning
    the step. Returns (K+1, 3, nlev, nlat, nlon) winds at the sub-step nodes,
    linearly interpolated in time (exact when K+1 == n)."""
    n = len(hours)
    # one pass: the CESM fields arrive as float32 already, so this is a single
    # copy (a per-array cast + a second stack was ~75 ms of host time per call)
    H = np.stack([np.stack(h) for h in hours]).astype(dtype, copy=False)      # (n,3,...)
    if n == K + 1:
        return H
    t_h = np.linspace(0.0, 1.0, n)
    t_k = np.linspace(0.0, 1.0, K + 1)
    idx = np.clip(np.searchsorted(t_h, t_k, side='right') - 1, 0, n - 2)
    a = ((t_k - t_h[idx]) / (t_h[idx + 1] - t_h[idx])).astype(dtype)
    a = a.reshape((K + 1,) + (1,) * (H.ndim - 1))
    return (1 - a) * H[idx] + a * H[idx + 1]


def advect_step_batch(qb0, u0, v0, w0, u1, v1, w1, lat, dp, qfrozb, dt_total,
                      lat_freeze=80.0, cfl=None, dtype=jnp.float32, *,
                      hourly_winds=None, nsub=None, fixer=None, vert=None, omega=None,
                      vertical_order=None, iterations=None, order='cubic',
                      limiter='quasi_monotone', polar_mode=None, radius=6.371e6,
                      plev=None, return_vflux=False, rho=None, split=None, jac=None):
    """Advect a tracer stack semi-Lagrangially (horizontal) over dt_total.

    qb0   : (ntracer, nlev, nlat, nlon) mixing ratios
    u0..w1: CESM U, V [m/s] and OMEGA [Pa/s] at the level CENTERS at the start
            and end of the interval, (nlev, nlat, nlon). Ignored (except for
            shape) when `hourly_winds` is given.
    lat   : (nlat,) degrees, increasing, poles allowed.  dp: (nlev,) Pa.
    plev  : (nlev,) level pressures [Pa], increasing; the vertical node
            coordinate of the vert='sl' gather. coupling.py passes PLEV_PA.
    qfrozb: (ntracer, nlev, nlat, nlon) reservoir/polar values. qfrozb[:, 0]
            and [:, -1] are the top/bottom inflow concentrations of the open
            faces (vert='flux'); under polar_mode='freeze' the rows with
            |lat| > lat_freeze are overwritten from it after the step.
    polar_mode : 'zonal' (stir the caps, conserving; the default from
            ADV_POLAR like fct_lr), 'freeze' (reset from qfrozb) or 'none'.
    dt_total : the coupling step [s].
    cfl   : accepted for signature compatibility with fct_lr; unused.
    dtype : gather/sweep precision (production float32). The stack is carried
            in float64 between sub-steps and returned in float64.
    hourly_winds : optional list of (u, v, w) tuples at n equally spaced times
            spanning the step, first = start, last = end (the coupling passes
            the STEP_HOURS+1 hourly CESM snapshots). Resampled linearly to the
            K+1 sub-step nodes; each sub-step uses the mean of its two nodes.
    nsub, fixer, vert, omega, vertical_order, iterations, rho : override the
            ADV_SL_* defaults (rho: ADV_SL_RHO, flux + lie only; split:
            ADV_SL_SPLIT, default strang; jac: ADV_SL_JAC, default 1).
    return_vflux : also return the summed face fluxes (ntracer, 2, nlat, nlon),
            [0] top face (> 0 inflow), [1] bottom face (> 0 outflow), in
            rho*q*Pa units like fct_lr; zeros under vert='sl'.

    Returns (qb float64 (ntracer, nlev, nlat, nlon), nsub[, vflux]).
    """
    K = int(ADV_SL_K if nsub is None else nsub)
    fixer = ADV_SL_FIXER if fixer is None else fixer
    vert = ADV_SL_VERT if vert is None else vert
    omega = ADV_SL_OMEGA if omega is None else omega
    vertical_order = ADV_SL_VORDER if vertical_order is None else vertical_order
    iterations = ADV_SL_ITER if iterations is None else int(iterations)
    rho_mode = bool(ADV_SL_RHO if rho is None else rho) and vert == 'flux'
    split = ADV_SL_SPLIT if split is None else split
    jac = bool(ADV_SL_JAC if jac is None else jac)
    if split not in _SPLITS:
        raise ValueError(f"split={split!r}; expected one of {_SPLITS}")
    if polar_mode is None:
        polar_mode = os.environ.get('ADV_POLAR', 'zonal').lower()
        polar_mode = 'zonal' if polar_mode == 'zonal' else 'freeze'
    if polar_mode not in ('zonal', 'freeze', 'none'):
        raise ValueError(f"polar_mode={polar_mode!r}")
    for name, val, ok in (('fixer', fixer, _FIXERS), ('vert', vert, _VERTS), ('omega', omega, _OMEGAS)):
        if val not in ok:
            raise ValueError(f"{name}={val!r}; expected one of {ok}")
    lat_np = np.asarray(lat, dtype=np.float64)
    dp_np = np.asarray(dp, dtype=np.float64)
    nlon = int(np.asarray(qb0).shape[-1])
    grid = S.SLGrid.from_latitudes(lat_np, nlon, radius)

    # vertical coordinate of the vert='sl' gather: the true level pressures
    # (omega is dp/dt in Pa/s). Without plev the nodes are rebuilt from the
    # cumulative dp (exact spacing only if dp is the true layer thickness --
    # np.gradient's central differences are ~10 % off on log-spaced levels).
    if plev is not None:
        p_nodes = np.asarray(plev, dtype=np.float64)
    else:
        p_nodes = np.concatenate([[0.0], np.cumsum(0.5 * (dp_np[:-1] + dp_np[1:]))])
    if p_nodes.shape != dp_np.shape or not np.all(np.diff(p_nodes) > 0):
        raise ValueError('level coordinate must be increasing top -> bottom, one per level')

    # cell weights for the mass targets: dp_k * exact cell-mean cos(phi)_j, the
    # coupling's own burden weight (ADV_METRIC=1). Longitude-uniform. mf/ac
    # are also what _omega_continuity's y-divergence uses (no cap closing:
    # the SL path does not stir the caps).
    mf, ac = grid_metric(lat_np, None)
    m = jnp.asarray(dp_np[:, None, None] * ac[None, None, :], dtype=jnp.float64)
    polar_mask = jnp.asarray(np.abs(lat_np) > lat_freeze)

    hours = hourly_winds if hourly_winds is not None else [(u0, v0, w0), (u1, v1, w1)]
    W = _resample_winds(hours, K, np.dtype(dtype).type)        # (K+1,3,nlev,nlat,nlon)
    W = jnp.swapaxes(jnp.asarray(W), -1, -2)                   # -> (...,nlon,nlat)

    qb = _to_sl(qb0, jnp.float64)
    qfr = _to_sl(qfrozb, jnp.float64)
    out, vfl = _sl_kernel(qb, W, jnp.asarray(p_nodes), jnp.asarray(dp_np), jnp.asarray(lat_np),
                          jnp.asarray(mf), jnp.asarray(ac), m, polar_mask, qfr,
                          float(dt_total) / K, grid=grid, K=K, order=order, limiter=limiter,
                          vert=vert, omega=omega, vertical_order=vertical_order,
                          iterations=iterations, fixer=fixer, gdtype=jnp.dtype(dtype),
                          polar_mode=polar_mode,
                          dphi=float(lat_np[1] - lat_np[0]) * DEG, rho_mode=rho_mode,
                          split=split, jac=jac)
    res = [jnp.swapaxes(out, -1, -2), K]
    if return_vflux:
        res.append(jnp.swapaxes(vfl, -1, -2))
    return tuple(res)
