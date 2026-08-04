"""Gravitational settling (sedimentation) of the TOMAS size bins.

Physics
-------
Terminal fall speed per bin from the Stokes drag law with the Cunningham
slip correction (Seinfeld & Pandis eq. 9.42):
https://www.sciencedirect.com/science/article/pii/S1352231006011897

    v_g = rho_p * g * Dp^2 * Cc / (18 * mu)          [m/s, downward]

    Cc  = 1 + Kn * (1.257 + 0.4 * exp(-1.1 / Kn))    (S&P eq. 9.34)
    Kn  = 2 * mfp / Dp

In the stratosphere the mean free path is large (Kn >> 1 for sub-micron
particles) so the slip correction dominates: Cc ~ 1.657*Kn and v_g grows
linearly with Dp rather than quadratically. This is why settling is a real
sink for SAI aerosol even at accumulation-mode sizes.

Air viscosity mu(T) and mean free path mfp(T,p) use the SAME formulas as
tomas-jax (physics/properties.py: power-law fit to Sutherland for mu, S&P
eq. 8.6 for mfp), so transport and microphysics see one consistent gas.

Both moments of a bin (number and mass) settle with a single per-bin
velocity evaluated at the bin's geometric-mean diameter. Using one velocity
per bin keeps each (Nk, Mk) pair consistent -- if number and mass settled at
different speeds a bin's mean particle mass would drift outside its bounds
and the two-moment clip in the microphysics would eat the difference.

Numerics
--------
The model's vertical coordinate is pressure, so the settling velocity is
converted to a pressure velocity (positive = downward = toward higher p):

    w_set = rho_air * g * v_g                        [Pa/s]

The column update is a flux-form, first-order upwind, BACKWARD-Euler sweep
solved top-down. Because settling only moves material downward, the implicit
system is lower-bidiagonal and one pass from the model top to the model
bottom solves it exactly (each level only needs the already-updated level
above it):

    q_new[k] = (q_old[k] * dp[k] + dt * w[k-1] * q_new[k-1])
               / (dp[k] + dt * w[k])

This is unconditionally stable -- important because the slip-corrected fall
speed of the largest bins at the top of the band gives Courant numbers >> 1
over a 6 h coupling step, which would blow up any explicit sweep without
heavy sub-stepping. The price is upwind diffusion; if the settling profile
ever needs to be sharp (e.g. a thin falling layer study) this is the place
to upgrade to the PPM machinery with an outflow bottom face.

Boundary conditions: zero flux through the TOP face (nothing settles in from
above the band) and OPEN OUTFLOW through the BOTTOM face -- mass crossing the
lowest face leaves the domain. Physically this is the aerosol sedimenting
across the tropopause into the troposphere where (unmodelled) wet removal
destroys it; numerically it is the model's one true aerosol sink. The
per-column outflow is returned so the driver can close the mass budget.

Conservation: interior transfers move q*dp between adjacent levels exactly,
so the burden measure used in coupling.py (sum of q * dp * cos(lat)) changes
ONLY by the bottom outflow term, to float64 roundoff.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

GRAV = 9.80665          # m/s2
RD = 287.05             # J/kg/K dry air
PI = np.pi
R_GAS = 8.314462618     # J/mol/K
MOLAR_MASS_AIR = 0.0289644  # kg/mol (matches tomas-jax core/config.py)


# =========================================================================
# WET aerosol size: H2SO4/H2O solution droplets
# =========================================================================
# TOMAS carries DRY SO4 mass, but a stratospheric sulfate particle is an
# H2SO4/H2O solution droplet. Both the settling velocity (which goes as Dp and
# rho_p) and the optics in radiation.py need the WET size, so the canonical
# implementation lives here -- this module is dependency-light (numpy + jax),
# whereas radiation.py drags in netCDF4 and rrtmgp.
#
# Composition comes from tomas_jax.fast.water.h2so4_equilibrium_wt (Tabazadeh
# et al. 1997) and density from the Tang (1997) polynomial below -- the SAME
# parameterizations the fast microphysics engine already applies internally
# every step, so settling, optics and coagulation now all see one droplet.

def tang_density(wt_pct):
    """Binary H2SO4/H2O solution density [kg/m3] at H2SO4 weight percent wt_pct.

    Tang (1997), the same polynomial as tomas_jax.fast.density.calc_density. It
    is reimplemented here rather than imported because that function carries the
    Fortran aerodens.f empty-bin fallback (`where(mtot < 1e-15, 1000.0, ...)`),
    which is calibrated for mass PER GRID CELL. Per-PARTICLE masses are ~1e-18 kg,
    so calling it directly silently returns 1000 kg/m3 for every bin and inflates
    the growth factor (1.09 -> 1.30 at 80 wt%). Taking the polynomial alone is
    exact for our use and cannot trip that guard.

    NB the argument convention is the Fortran one: the polynomial is evaluated at
    x = 100*m_H2SO4/(m_SO4 + m_H2O), i.e. H2SO4-basis solute over RAW SO4+H2O
    total, which is very slightly below the true H2SO4 weight fraction. wet_size()
    below reproduces that convention exactly so optics match microphysics.
    """
    x = jnp.asarray(wt_pct, dtype=jnp.float64)
    ds0 = 0.9971 + x * (7.367e-3 + x * (-4.934e-5
                        + x * (1.754e-6 + x * -1.104e-8)))
    return ds0 * 1000.0


def wet_size(mmid_kg, wt_pct, rho_dry, mw_h2so4=98.0, mw_so4=96.0):
    """Wet droplet diameter [m] and solution density for dry SO4 mass mmid_kg.

    Replicates tomas_jax.fast.coagulation._particle_properties_cell exactly:
        mp  = (m_SO4 + m_H2O) / N          <- RAW SO4 basis, not H2SO4 basis
        rho = calc_density(Mk)             <- Tang, evaluated at the H2SO4-basis x
        Dp  = cbrt(mp / rho * 6/pi)
    Water follows tomas_jax.fast.water.equilibrium_water:
        m_acid = m_SO4 * 98/96 ;  m_H2O = m_acid * (100/wt - 1)
    so the wet mass is m_acid*100/wt and wt_pct is on the H2SO4 mass basis.

    Returns (dp_wet_m, rho_sol) broadcast over the inputs. Also returns the DRY
    diameter's growth factor implicitly: dp_wet / cbrt(mmid/rho_dry*6/pi).
    """
    m = jnp.asarray(mmid_kg, dtype=jnp.float64)
    wt = jnp.asarray(wt_pct, dtype=jnp.float64)
    acid = m * (mw_h2so4 / mw_so4)
    m_h2o = acid * (100.0 / wt - 1.0)
    m_tot = m + m_h2o                                  # raw SO4 + H2O
    x = 100.0 * acid / m_tot                           # Tang's argument
    rho = tang_density(x)
    dp_wet = jnp.cbrt(m_tot / rho * (6.0 / PI))
    return dp_wet, rho


def settling_velocity(dp_bin_m, temp3d, pres3d, rho_p):
    """Slip-corrected Stokes terminal velocity per bin and grid cell.

    dp_bin_m : (NBINS,) dry bin diameter [m], OR (NBINS,nlev,nlat,nlon) wet
               diameters when the composition varies per cell
    temp3d   : (nlev,nlat,nlon) [K]
    pres3d   : (nlev,nlat,nlon) [Pa]
    rho_p    : particle density [kg/m3]; scalar for the dry case, or
               (NBINS,nlev,nlat,nlon) solution density for the wet case
    Returns v_g (NBINS,nlev,nlat,nlon) [m/s, positive downward].

    Both size and density must move together: a droplet taking up water grows
    (raising Dp^2) but dilutes (lowering rho_p), and v_g ~ rho_p*Dp^2 in the
    Stokes limit / ~rho_p*Dp in the slip-dominated stratospheric limit, so the
    two effects partly cancel. Passing a wet Dp with the dry 1770 kg/m3 density
    would overstate the fall speed.
    """
    # air viscosity: power-law fit to Sutherland (tomas-jax properties.py,
    # <0.1% error 200-350K -- stratospheric range is inside the fit window)
    mu = 2.5277e-7 * jnp.power(temp3d, 0.75302)
    # mean free path, S&P eq. 8.6 (same formula as tomas-jax)
    mfp = 2.0 * mu / (pres3d * jnp.sqrt(8.0 * MOLAR_MASS_AIR / (PI * R_GAS * temp3d)))
    # accept either a (NBINS,) dry grid or a full (NBINS,nlev,nlat,nlon) wet field
    dpb = jnp.asarray(dp_bin_m)
    if dpb.ndim == 1:
        dpb = dpb[:, None, None, None]
    rp = jnp.asarray(rho_p)
    if rp.ndim == 1:
        rp = rp[:, None, None, None]
    # Knudsen number per (bin, cell): broadcast bins against the 3-D fields
    Kn = 2.0 * mfp[None] / dpb
    # Cunningham slip correction, S&P eq. 9.34
    Cc = 1.0 + Kn * (1.257 + 0.4 * jnp.exp(-1.1 / Kn))
    # Stokes law with slip
    vg = rp * GRAV * (dpb ** 2) * Cc / (18.0 * mu[None])
    return vg


def equilibrium_wt_field(temp3d, rh3d):
    """Tabazadeh equilibrium H2SO4 weight percent, (nlev,nlat,nlon).

    rh3d is a FRACTION (0-1), matching coupling.py's relhum(). Thin wrapper so
    callers need not know where the parameterization lives.
    """
    from tomas_jax.fast.water import h2so4_equilibrium_wt
    return h2so4_equilibrium_wt(jnp.asarray(temp3d),
                                jnp.clip(jnp.asarray(rh3d), 0.0, 1.0) * 100.0)


def wet_size_field(mmid_kg, wt_pct3d, rho_dry):
    """Per-(bin,cell) wet diameter [m] and solution density [kg/m3].

    mmid_kg  : (NBINS,) dry SO4 mass per particle
    wt_pct3d : (nlev,nlat,nlon) equilibrium H2SO4 weight percent
    Returns (dp_wet, rho_sol), both (NBINS,nlev,nlat,nlon).

    wet_size is elementwise in (mass, wt%), so this is just the outer broadcast
    of the bin axis against the cell axes -- same formula, same convention.
    """
    m = jnp.asarray(mmid_kg)[:, None, None, None]
    w = jnp.asarray(wt_pct3d)[None]
    return wet_size(m, w, rho_dry)


@jax.jit
def settle_step(num, mas, temp3d, pres3d, dp, dt, dp_bin_m, rho_p):
    """Advance settling for all bins over dt with the implicit upwind sweep.

    num, mas : (NBINS, nlev, nlat, nlon) mixing ratios [#/kg, kg/kg]
               level index 0 = TOP of the band (smallest pressure)
    temp3d   : (nlev,nlat,nlon) [K]     pres3d : (nlev,nlat,nlon) [Pa]
    dp       : (nlev,) level thickness [Pa]     dt : [s]
    dp_bin_m : (NBINS,) bin diameters [m]       rho_p : [kg/m3]
    Returns num2, mas2, out_num, out_mas where out_* is the mass crossing the
    bottom face, per (bin,lat,lon), in mixing-ratio * Pa units -- i.e. the
    same q*dp units the burden diagnostic integrates, so the driver's
    'settle' budget stage should equal -sum(out_mas * cos(lat)) exactly.
    """
    vg = settling_velocity(dp_bin_m, temp3d, pres3d, rho_p)
    # pressure-coordinate settling velocity [Pa/s], positive downward
    rho_air = pres3d / (RD * temp3d)
    w = rho_air[None] * GRAV * vg                     # (NBINS,nlev,nlat,nlon)

    # stack the two moments: they share w per bin (single-velocity convention)
    qs = jnp.stack([num, mas], axis=0)                # (2,NBINS,nlev,nlat,nlon)

    # scan wants the swept axis leading: (nlev, 2, NBINS, nlat, nlon)
    qs_l = jnp.moveaxis(qs, 2, 0)
    w_l = jnp.moveaxis(w, 1, 0)                       # (nlev,NBINS,nlat,nlon)

    def body(F_in, xs):
        # F_in: settling flux arriving through the cell TOP face [q*Pa/s],
        # zero for the first (topmost) level. Backward-Euler upwind update:
        q_k, w_k, dp_k = xs
        q_new = (q_k * dp_k + dt * F_in) / (dp_k + dt * w_k[None])
        F_out = w_k[None] * q_new                     # flux through bottom face
        return F_out, q_new

    F_top = jnp.zeros_like(qs_l[0])                   # nothing falls in from above
    F_bot, qs_new_l = jax.lax.scan(body, F_top, (qs_l, w_l, dp))

    qs_new = jnp.moveaxis(qs_new_l, 0, 2)
    # bottom-face outflow over the step, in q*dp-compatible units (q * Pa)
    out = dt * F_bot                                  # (2,NBINS,nlat,nlon)
    return qs_new[0], qs_new[1], out[0], out[1]
