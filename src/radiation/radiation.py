"""Aerosol -> radiation -> heating driver coupling TOMAS bins to jax-rrtmgp.

Pipeline per coupling step:
    1. Per-bin aerosol optical properties from precomputed Mie tables
       (bhmie from tomas-jax + Palmer & Williams 1975 75 wt% H2SO4 refractive
       indices, band-averaged onto the 14 SW + 16 LW RRTMGP bands).
    2. jax-rrtmgp two-stream solve (clear-sky gases + our aerosol) on the
       full native CESM column (all levels with p >= 1.005 Pa, the RRTMGP
       lookup-table range), aerosol only on the coupled band levels.
    3. Heating rate [K/s] returned on the band levels in CESM level order.

Anomaly mode (the default use): heating is computed twice per step -- once
with the evolved TOMAS bins at T_eff = T_CESM + dT, once with reference
MAM4-binned aerosol at T_CESM -- and the difference drives d(dT)/dt. This
isolates the radiative effect of the aerosol *evolution* without the model
drifting toward radiative equilibrium (there is no other physics to balance
the full heating rate while the circulation is prescribed).

MVP approximations (each is a flagged refinement, not a hidden assumption):
    * clear sky (no clouds; hourly cloud water is not in the CESM archive)
    * refractive index clamped outside 0.36-25 um (Palmer & Williams range)
      and outside 25-95.6 wt% (their six tabulated solution strengths)
    * solar zenith from a standard declination formula (no equation of time)

(Until 2026-08-03 the optics also used DRY bin diameters at a fixed 75 wt%
composition. They now use the RH/T-dependent wet droplet -- see WET_OPTICS.)
"""
import os
import sys

# --- the two dependency repos, as submodules under models/ -------------------
# Duplicated from coupling.py on purpose: radiation.py is imported directly by
# validation/test_radiation.py and validate_radiation.py, which never load
# coupling.py, so it cannot rely on coupling having fixed up sys.path first.
# Keep the two copies in step. See coupling._dep_path for the full rationale.
# Bootstrap src/ (the parent of this module's own directory) onto sys.path so
# _paths imports. radiation.py is loaded directly by the validation harnesses,
# which never load coupling.py, so it cannot rely on coupling having done this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths                      # noqa: E402 -- needs the insert above
_HERE = _paths.SRC
_MODELS = _paths.MODELS
for _env, _name in (('TOMAS_JAX_PATH', 'tomas-jax'), ('RRTMGP_PATH', 'jax-rrtmgp')):
    for _cand in (os.environ.get(_env),
                  os.path.join(_MODELS, _name)):
        if _cand and os.path.isdir(_cand):
            if _cand not in sys.path:
                sys.path.insert(0, _cand)
            break

import numpy as np
import jax
import jax.numpy as jnp
import netCDF4

try:
    from rrtmgp.config import radiative_transfer
    from rrtmgp import rrtmgp as rrtmgp_lib
    from rrtmgp import rrtmgp_common
    from rrtmgp import constants as rrtmgp_constants
    from tomas_jax.physics.bhmie import bhmie_qsca_jax
except ImportError as _e:
    raise SystemExit(
        f"radiation.py: cannot import a dependency repo ({_e}).\n"
        "  jax-rrtmgp (radiation) and tomas-jax (Mie) are separate repos, NOT\n"
        "  vendored here. Run `git submodule update --init` to populate\n"
        "  models/jax-rrtmgp and models/tomas-jax, install them, or set\n"
        "  RRTMGP_PATH / TOMAS_JAX_PATH.\n"
        "  Run without radiation entirely with RAD=0.")

GRAV   = 9.80665
#stefan-boltzmann constant [W/m2/K4] for computing surface temperature from net upward flux
SIGMA_SB = 5.670374419e-8
RRTMGP_PKG = os.path.dirname(rrtmgp_lib.__file__)          # .../jax-rrtmgp/rrtmgp
RRTMGP_DATA = os.path.join(RRTMGP_PKG, 'optics/rrtmgp_data')
RRTMGP_TESTDATA = os.path.join(RRTMGP_PKG, 'optics/test_data')
# Repo-root-relative, not __file__-relative: the static input data lives in
# inputs/, which is a sibling of wherever this module sits, not a child of it.
RI_FILE = os.path.join(_paths.INPUTS, 'rad_data',
                       'palmer_williams_h2so4.dat')

CO2_PPM  = float(os.environ.get('CO2_PPM', '380.0'))   # ~2005 value for 1996-2014
N2O_PPB  = float(os.environ.get('N2O_PPB', '319.0'))
SFC_EMIS = 0.98
ALB_FALLBACK = 0.15            # night / FSDS~0 columns (SW is zeroed anyway)
TSI_DEFAULT = 1361.0           # W/m2; per-step irradiance override possible
# rows per rrtmgp call. Sweep (bench_rad2.py, 192 lat, 55-lvl column, 1 GPU,
# 1-pass heating): CHUNK 16->45.9s, 32->16.0s, 48->10.6s, 96->9.3s(best), 192->16.2s.
# 16 (the old rad_smoke value) is pathological: 12 tiny launches too small to
# saturate the H100, per-launch dispatch dominates. 96 (2 chunks) is ~5x faster,
# BIT-IDENTICAL (columns independent). Default 96; drop to 48 if radiation OOMs on
# a memory-tight card (per-band fields are ~3GB/globe, so fewer/bigger chunks cost
# more memory -- 96=2 chunks ~1.5GB each, 48=4 chunks ~0.75GB each).
RAD_LAT_CHUNK = int(os.environ.get('RAD_LAT_CHUNK', '96'))
# Size the optics on the WET H2SO4/H2O droplet rather than the dry SO4 core.
# ON by default as of 2026-08-03; 0 reproduces the legacy dry-diameter optics.
WET_OPTICS = os.environ.get('WET_OPTICS', '1') != '0'
N_QUAD_WL = 3                  # wavelength quadrature points per band


# =========================================================================
# Palmer & Williams (1975) 75 wt% H2SO4 refractive index
# =========================================================================
# =========================================================================
# WET aerosol size + composition (H2SO4/H2O solution droplets)
# =========================================================================
# TOMAS carries DRY SO4 mass. Optics must use the WET droplet, because that is
# the particle light actually sees. Before 2026-08-03 the Mie tables were built
# on the dry diameter DP_BIN (= cbrt(MMID/RHO_AER*6/pi), RHO_AER=1770) while the
# refractive index was that of a 75 wt% H2SO4 SOLUTION -- i.e. a droplet already
# 25% water by mass, sized as if it were dry. That mismatch alone underestimated
# 550 nm extinction by ~47% for this model's size distribution; with the real
# RH/T-dependent composition (which reaches 40-55 wt% near the moist 143 hPa base
# of the band) the error runs to a factor of 2 or more.
#
# The composition and density parameterizations are the SAME ones the fast
# microphysics engine already applies internally every step
# (tomas_jax.fast.water.h2so4_equilibrium_wt, Tabazadeh et al. 1997 GRL 24,
# 1931-1934; tomas_jax.fast.density.calc_density, Tang 1997 JGR 102, 1883-1893),
# so the optics now see the same droplet the coagulation kernel does. Previously
# the engine computed equilibrium water, used it for the microphysics, and then
# driver_fast.py read back only the dry SO4 column -- so the wet size existed
# inside the timestep and was discarded before radiation ever ran.
# tang_density/wet_size live in settling.py: gravitational settling needs the same
# wet diameter and solution density, and settling.py is dependency-light (numpy +
# jax only), so putting the canonical copy there lets both consumers share one
# implementation without radiation.py's netCDF4/rrtmgp imports leaking into it.
from settling import tang_density, wet_size

WT_GRID = np.arange(10.0, 80.0 + 1e-9, 5.0)      # wt% H2SO4, Tabazadeh Table 1 grid
_WT_DGRID = float(WT_GRID[1] - WT_GRID[0])       # uniform 5 wt% spacing

# Palmer & Williams (1975) tabulates n,k at these six solution strengths; the
# .dat columns are [cm-1, um, 25%, 38%, 50%, 75%, 84.5%, 95.6%].
_RI_WT = np.array([25.0, 38.0, 50.0, 75.0, 84.5, 95.6])
_RI_COL = np.array([2, 3, 4, 5, 6, 7])

##the water fraction is whatever the local thermodynamics says!!!!!!!!!!! same with settling, both a function of RH

def refindex_at_wt(wt_pct, path=RI_FILE):
    """(wl_um, n, k) for an H2SO4 solution of wt_pct, linearly interpolated
    across Palmer & Williams' six tabulated strengths.

    Composition changes the refractive index as well as the size -- a droplet
    diluting from 80 to 45 wt% moves n,k toward water's. Holding the RI at 75 wt%
    while growing the diameter would trade one inconsistency for another.
    Outside 25-95.6 wt% the nearest tabulated column is used (clamped), which
    only bites below 25 wt%; the Tabazadeh composition floor is 10 wt%.
    """
    wl, ncols, kcols = _load_refindex_all(path)
    w = float(np.clip(wt_pct, _RI_WT[0], _RI_WT[-1]))
    j = int(np.clip(np.searchsorted(_RI_WT, w), 1, len(_RI_WT) - 1))
    w0, w1 = _RI_WT[j - 1], _RI_WT[j]
    f = (w - w0) / (w1 - w0)
    n = (1.0 - f) * ncols[:, j - 1] + f * ncols[:, j]
    k = (1.0 - f) * kcols[:, j - 1] + f * kcols[:, j]
    return wl, n, k


def _load_refindex_all(path=RI_FILE):
    """(wl_um ascending, n[:, 6], k[:, 6]) -- all six solution strengths."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 8:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
    rows = np.asarray(rows)
    assert rows.shape[0] % 2 == 0 and rows.shape[0] > 0, \
        f'unexpected palmer_williams format ({rows.shape[0]} numeric rows)'
    half = rows.shape[0] // 2
    real, imag = rows[:half], rows[half:]
    assert np.allclose(real[:, 0], imag[:, 0]), 'real/imag wavenumber mismatch'
    wl = real[:, 1]
    order = np.argsort(wl)
    return wl[order], real[order][:, _RI_COL], imag[order][:, _RI_COL]


def load_h2so4_refindex(path=RI_FILE, wt_col=5):
    """
    The refractive index of sulfuric acid tells you how those droplets interact with light
    75% weight is the equilibrium composition of H2SO4 in the stratosphere, so we use that for the Mie calculations (75% of droplet weight is sulfuric acid, 25% is water).
    
    Parse the HITRAN palmer_williams_h2so4.dat file.
    https://hitran.org/data/Aerosols/Aerosols%20data%20from%20previous%20HITRAN/Aerosols-2016/ascii/single_files/palmer_williams_h2so4.dat
    wt_col: column index in [cm-1, um, 25%, 38%, 50%, 75%, 84.5%, 95.6%] (could pick any of these...);
    default 5 -> 75 wt%. Returns (wl_um ascending, n, k).
"""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 8:
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue
    rows = np.asarray(rows)
    assert rows.shape[0] % 2 == 0 and rows.shape[0] > 0, \
        f'unexpected palmer_williams format ({rows.shape[0]} numeric rows)'
    half = rows.shape[0] // 2
    real, imag = rows[:half], rows[half:]
    # both blocks are sorted by wavenumber; sanity-check they line up
    assert np.allclose(real[:, 0], imag[:, 0]), 'real/imag wavenumber mismatch'
    wl = real[:, 1]
    order = np.argsort(wl)
    return wl[order], real[order, wt_col], imag[order, wt_col]


def band_limits_um(nc_path):
    """RRTMGP band limits [wavenumber cm-1] -> (n_bnd, 2) wavelengths in um."""
    with netCDF4.Dataset(nc_path) as d:
        wn = np.asarray(d['bnd_limits_wavenumber'][:])   # (n_bnd, 2) cm-1
    wl = 1.0e4 / wn[:, ::-1]                             # cm to um, ascending per band by flipping axis
    return wl


# =========================================================================
# Mie tables: per-bin x per-band extinction cross-section, ssa, g
# =========================================================================
def build_mie_tables(dp_bin_m, nc_path, nq=N_QUAD_WL, ri_path=RI_FILE,
                     wt_pct=75.0):
    """Band-averaged Mie properties for each TOMAS bin.

    dp_bin_m : (NBINS,) bin diameters [m] -- WET diameters in normal use; the
               caller is responsible for having grown them (see wet_size).
    wt_pct   : H2SO4 weight percent selecting the refractive index, so size and
               composition stay consistent. 75.0 reproduces the pre-2026-08-03
               fixed-composition tables.
    Returns dict of np arrays: sigma_ext (NBINS, n_bnd) [m2/particle],
    ssa (NBINS, n_bnd), g (NBINS, n_bnd).
    Band average = mean over nq log-spaced wavelengths inside the band
    (flat weighting; bands are narrow enough that this beats band-center
    evaluation without needing solar/Planck weights).
    """
    #refractive index of the solution at THIS weight percent
    wl_tab, n_tab, k_tab = refindex_at_wt(wt_pct, ri_path)
    #get wavelength edges of each RRTMGP band in microns
    lims = band_limits_um(nc_path)                       # (nb, 2) um
    nb = lims.shape[0]
    nbin = dp_bin_m.size
    # quadrature wavelengths per band (um), log-spaced interior points
    tq = (np.arange(nq) + 0.5) / nq
    #log-linear interpolation between band edges to get quadrature points
    wlq = np.exp(np.log(lims[:, :1]) +
                 tq[None, :] * (np.log(lims[:, 1:]) - np.log(lims[:, :1])))  # (nb, nq)
    # refractive index at each quadrature wavelength (clamped to data range)
    wlq_c = np.clip(wlq, wl_tab[0], wl_tab[-1])
    #linearly interpolate real and imaginary refractive index to quadrature wavelengths
    n_q = np.interp(wlq_c, wl_tab, n_tab)
    k_q = np.interp(wlq_c, wl_tab, k_tab)
    # size parameters: (nb, nq, nbin) pi*diameter/wavelength
    # dimensionless ratio of a particle's circumference to the wavelength of light; determines scattering regime
    x = np.pi * dp_bin_m[None, None, :] / (wlq[:, :, None] * 1e-6)
    # bhmie's fixed series length caps x ~< 220; beyond that Mie efficiencies
    # have converged to the geometric-optics limit, so clamping is safe.
    x = np.minimum(x, 200.0)
    #complex refractive index: n + i*k
    m = (n_q + 1j * np.maximum(k_q, 1e-12))[:, :, None] * np.ones_like(x)
    #given a size parameter and complex refractive index, bhmie computes the scattering and extinction efficiencies, and what direction light is scattered in (g)
    qe, qs, gs = jax.vmap(bhmie_qsca_jax)(jnp.asarray(x.ravel()),
                                          jnp.asarray(m.ravel()))
    qe = np.asarray(qe).reshape(nb, nq, nbin)
    qs = np.asarray(qs).reshape(nb, nq, nbin)
    gs = np.asarray(gs).reshape(nb, nq, nbin)
    qs = np.minimum(qs, qe)                              # numerical guard...scattering efficiency cannot exceed extinction efficiency
    area = np.pi / 4.0 * dp_bin_m**2                     # geometric cross-sectional area of a sphere using diameter (pi*r^2 = pi*(d/2)^2 = pi*d^2/4)
    #average scattering and extinction efficiencies over quadrature points, then multiply by geometric area to get cross-sections
    sigma_ext = qe.mean(axis=1).T * area[:, None]        # (nbin, nb)
    sigma_sca = qs.mean(axis=1).T * area[:, None]
    #average asymmetry parameter over quadrature points (-1 (backwards) to 1 (forwards)), weighted by scattering efficiency (qs)
    #g values where scattering is high are weighted more, as they contribute more to the overall scattering behavior of the particle
    g_bin = ((gs * qs).mean(axis=1) / np.maximum(qs.mean(axis=1), 1e-30)).T
    # ratio of scattering to extinction cross-section, dimensionless (single scattering albedo)
    ssa = sigma_sca / np.maximum(sigma_ext, 1e-30)
    return {'sigma_ext': sigma_ext, 'ssa': ssa, 'g': g_bin}


def build_wet_mie_tables(mmid_kg, rho_dry, nc_path, nq=N_QUAD_WL,
                         ri_path=RI_FILE, wt_grid=WT_GRID):
    """Mie tables on the (wt%, bin, band) mesh for wet H2SO4/H2O droplets.

    For each solution strength in wt_grid the bins are grown to their wet
    diameter (wet_size) AND the refractive index is taken at that same strength,
    then the usual band-averaged Mie integration runs. heating() interpolates
    these in wt% per grid cell.

    Cost: len(wt_grid) Mie builds instead of one, paid once at init. The runtime
    cost is len(wt_grid) small einsums per radiation call, which is negligible
    next to the RRTMGP two-stream solve.

    Returns dict of (n_wt, NBINS, n_bnd) arrays plus 'dp_wet' (n_wt, NBINS) [m].
    """
    out = {k: [] for k in ('sigma_ext', 'ssa', 'g')}
    dp_wet_all = []
    for w in wt_grid:
        dp_wet, _ = wet_size(mmid_kg, w, rho_dry)
        dp_wet = np.asarray(dp_wet)        # wet_size is jnp-based; Mie build is numpy
        t = build_mie_tables(dp_wet, nc_path, nq=nq, ri_path=ri_path, wt_pct=w)
        for k in out:
            out[k].append(t[k])
        dp_wet_all.append(dp_wet)
    res = {k: np.stack(v, axis=0) for k, v in out.items()}
    res['dp_wet'] = np.stack(dp_wet_all, axis=0)
    return res


def wt_weights(wt_pct, wt_grid=WT_GRID, dwt=_WT_DGRID):
    """Linear-interpolation weights of wt_pct onto the uniform wt_grid.

    Returns (n_wt, ...) hat-function weights summing to 1 along axis 0. Used
    instead of an index gather because gathering per-cell Mie tables would
    materialize (cells, NBINS, n_bnd) -- tens of GB. With weights we instead
    accumulate n_wt cheap einsums, each the size of the original single-table
    one, and the result is identical to per-cell linear interpolation.
    """
    w = jnp.clip(jnp.asarray(wt_pct), wt_grid[0], wt_grid[-1])
    d = jnp.abs(w[None, ...] - jnp.asarray(wt_grid).reshape(
        (-1,) + (1,) * jnp.ndim(w)))
    return jnp.clip(1.0 - d / dwt, 0.0, 1.0)


def mie_at_wavelength(dp_bin_m, wl_um, ri_path=RI_FILE, wt_pct=75.0):
    """(sigma_ext, sigma_sca) per bin at a single wavelength (diagnostics).
    Similar to function above but only evaluates at one wavelength, so no quadrature or band averaging.
    Production path never uses this default wt_pct"""
    wl_tab, n_tab, k_tab = refindex_at_wt(wt_pct, ri_path)
    #interpolate n and k at one wavelength
    n = np.interp(wl_um, wl_tab, n_tab); k = np.interp(wl_um, wl_tab, k_tab)
    x = jnp.asarray(np.pi * dp_bin_m / (wl_um * 1e-6))
    m = jnp.full(x.shape, n + 1j * max(k, 1e-12))
    qe, qs, _ = jax.vmap(bhmie_qsca_jax)(x, m)
    area = np.pi / 4.0 * dp_bin_m**2
    # convert dimensionless efficiencies to cross-sections by multiplying by geometric area
    #used later to compte column AOD
    return np.asarray(qe) * area, np.asarray(qs) * area


# =========================================================================
# Aerosol optical fields from bin number mixing ratios
# =========================================================================
@jax.jit
def _aod550(num_mr, dp_pa, sig_ext_550):
    """Column AOD at 550 nm: (NBINS, nzb, nx, nlon) -> (nx, nlon).
    converting number mixing ratio to column number density by using layer pressure thickness and gravity
    from #/kg to #/m^2"""
    ncol = num_mr * (dp_pa[None, :, None, None] / GRAV)
    return jnp.einsum('k,kzyx->yx', sig_ext_550, ncol)


@jax.jit
def _aod550_wet(num_mr, dp_pa, sig550_wt, wwt):
    """As _aod550 but with per-cell composition. sig550_wt (n_wt, NBINS),
    wwt (n_wt, nzb, nx, nlon) the wt%-interpolation weights."""
    ncol = num_mr * (dp_pa[None, :, None, None] / GRAV)
    # sum_j sum_k sig[j,k] * (ncol[k,cell] * w[j,cell])
    return jnp.einsum('jk,jkzyx->yx', sig550_wt, ncol[None] * wwt[:, None])


@jax.jit
def _aerosol_props_wet(num_mr, dp_pa, sigma_ext, ssa, g, wwt):
    """Per-cell wet-composition aerosol optics.

    sigma_ext/ssa/g : (n_wt, NBINS, n_bnd) tables from build_wet_mie_tables
    wwt             : (n_wt, nzb, nlat, nlon) linear weights over WT_GRID
    *g_f             : (n_bnd, nzb, nlat, nlon) extinction-weighted asymmetry factor

    Mathematically identical to evaluating each cell's own interpolated Mie
    table, but accumulated as n_wt einsums so nothing of size
    (cells, NBINS, n_bnd) is ever materialized. tau/sca/ag are summed over the
    composition axis BEFORE the ssa/g ratios are formed, which is what keeps the
    result a proper extinction-weighted mixture rather than an average of ratios.
    """
    ncol = num_mr * (dp_pa[None, :, None, None] / GRAV)      # (k,z,y,x)
    nw = ncol[None] * wwt[:, None]                           # (j,k,z,y,x)
    tau = jnp.einsum('jkb,jkzyx->bzyx', sigma_ext, nw)
    sca = jnp.einsum('jkb,jkzyx->bzyx', sigma_ext * ssa, nw)
    ag = jnp.einsum('jkb,jkzyx->bzyx', sigma_ext * ssa * g, nw)
    ssa_f = sca / jnp.maximum(tau, 1e-30)
    g_f = ag / jnp.maximum(sca, 1e-30)
    return tau, ssa_f, g_f


@jax.jit
def _aerosol_props(num_mr, dp_pa, sigma_ext, ssa, g):
    """num_mr (NBINS, nzb, nlat, nlon) [#/kg]; dp_pa (nzb,) layer thickness.
    Returns tau, ssa, g each (n_bnd, nzb, nlat, nlon) in CESM level order."""
    ncol = num_mr * (dp_pa[None, :, None, None] / GRAV)          # #/m2 per layer
    #cross-section times number density gives optical depth per layer; sum over bins to get total optical depth
    tau = jnp.einsum('kb,kzyx->bzyx', sigma_ext, ncol)
    #scattering optical depth
    sca = jnp.einsum('kb,kzyx->bzyx', sigma_ext * ssa, ncol)
    #scattering cross-section weighted by asymmetry factor
    ag  = jnp.einsum('kb,kzyx->bzyx', sigma_ext * ssa * g, ncol)
    #extinction weighted single scattering albedo and asymmetry factor
    ssa_f = sca / jnp.maximum(tau, 1e-30)
    g_f   = ag / jnp.maximum(sca, 1e-30)
    return tau, ssa_f, g_f


# =========================================================================
# Solar geometry (noleap calendar)
# =========================================================================
"""https://www.ncei.noaa.gov/sites/default/files/2021-07/GOES-R_ABI_solar_zenith_angle_description_12.docx"""
def solar_zenith(time_val, lat_deg, lon_deg):
    """Solar zenith angle [rad], shape (nlat, nlon). time_val: cftime noleap."""
    #fractional day of year
    doy = time_val.dayofyr + time_val.hour / 24.0
    #solar declination (ranging from -23.44 to +23.44 degrees over the year)
    #+10 shifts to solstice (doy=172) for max declination, and to equinox (doy=80) for zero declination
    dec = -23.44 * np.pi / 180.0 * np.cos(2 * np.pi * (doy + 10.0) / 365.0)
    #how far the sun has rotated from local noon (ranging from -pi to +pi over the day)
    hang = 2 * np.pi * (time_val.hour / 24.0 - 0.5) + np.deg2rad(lon_deg)
    phi = np.deg2rad(lat_deg)
    #
    cosz = (np.sin(phi)[:, None] * np.sin(dec) +
            np.cos(phi)[:, None] * np.cos(dec) * np.cos(hang)[None, :])
    #convert to zenith angle, clamp to [-1, 1] to avoid NaN from arccos
    return np.arccos(np.clip(cosz, -1.0, 1.0))


# =========================================================================
# Radiation driver
# =========================================================================
class RadiationDriver:
    """Owns the RRTMGP object, CESM inputs, Mie tables, and the column layout.

    open_var : callable(str) -> xr.Dataset (coupling.py's opener, reused so
               file conventions live in one place)
    klevs    : native level indices of the coupled aerosol band
    """

    def __init__(self, open_var, klevs, lat, lon, dp_bin_m,
                 g_lw='rrtmgp-gas-lw-g128.nc', g_sw='rrtmgp-gas-sw-g112.nc',
                 mmid_kg=None, rho_dry=None):
        self.lat, self.lon = np.asarray(lat), np.asarray(lon)
        self.nlat, self.nlon = self.lat.size, self.lon.size

        # ---- CESM inputs ----
        self.dT = open_var('T'); self.dQ = open_var('Q')
        self.dO3 = open_var('O3'); self.dCH4 = open_var('CH4')
        self.dFLDS = open_var('FLDS'); self.dFLNS = open_var('FLNS')
        self.dFSDS = open_var('FSDS'); self.dFSNS = open_var('FSNS')
        self.time = self.dT['time'].values

        # ---- vertical layout: full native column within the LUT p-range ----
        ds = self.dT
        #pressure levels and layer interfaces
        plev = ds['hyam'].values * ds['P0'].values + ds['hybm'].values * 1.0e5
        pilev = ds['hyai'].values * ds['P0'].values + ds['hybi'].values * 1.0e5
        #find which native CESM levels are within the RRTMGP lookup-table range (>= 1.005 Pa)
        self.krad = np.where(plev >= 1.005)[0]           # native indices, top->bot
        self.p_lay = plev[self.krad]                     # (nzc,) Pa, top->bot
        self.dp_lay = np.diff(pilev)[self.krad]          # interface thickness, Pa
        self.nzc = self.krad.size
        #dictionary mapping native level index to its position in the band-level array (for slicing)
        kmap = {k: i for i, k in enumerate(self.krad)}
        #translates the coupled aerosol level indices into their positions within the radiation column subset**** (very important)
        self.band_pos = np.array([kmap[k] for k in klevs])   # band rows in column
        self.dp_band = self.dp_lay[self.band_pos]
        self.klevs = list(klevs)

        # ---- RRTMGP ----
        cfg = radiative_transfer.RadiativeTransfer(
            optics=radiative_transfer.OpticsParameters(
                optics=radiative_transfer.RRTMOptics(
                    longwave_nc_filepath=os.path.join(RRTMGP_DATA, g_lw),
                    shortwave_nc_filepath=os.path.join(RRTMGP_DATA, g_sw),
                    cloud_longwave_nc_filepath=os.path.join(RRTMGP_DATA, 'cloudysky_lw.nc'),
                    cloud_shortwave_nc_filepath=os.path.join(RRTMGP_DATA, 'cloudysky_sw.nc'),
                )),
            atmospheric_state_cfg=radiative_transfer.AtmosphericStateCfg(
                sfc_emis=SFC_EMIS, sfc_alb=ALB_FALLBACK,
                zenith=0.0, irrad=TSI_DEFAULT, toa_flux_lw=0.0,
                vmr_global_mean_filepath=os.path.join(
                    RRTMGP_TESTDATA, 'rcemip_global_mean_vmr.json'),
                vmr_sounding_filepath=os.path.join(
                    RRTMGP_TESTDATA, 'rcemip_vmr_sounding.csv'),
            ))
        # dz only feeds cloud paths; we are clear-sky, so any scalar works (for LWP/IWP calcs).
        self.rt = rrtmgp_lib.RRTMGP(cfg, dz=500.0)
        self.nb_sw = self.rt.optics_lib.gas_optics_sw.n_bnd
        self.nb_lw = self.rt.optics_lib.gas_optics_lw.n_bnd

        # ---- Mie tables ----
        # WET optics (default): tables on the (wt%, bin, band) mesh, with both the
        # diameter and the refractive index taken at each solution strength. The
        # per-cell equilibrium wt% comes from Tabazadeh via _wt_field() below.
        # WET_OPTICS=0 restores the legacy dry-diameter/fixed-75wt% tables, which
        # underestimate 550 nm extinction by ~47% for this size distribution --
        # kept only for reproducing pre-2026-08-03 runs.
        self.wet_optics = WET_OPTICS and mmid_kg is not None
        if self.wet_optics:
            self.dRHq = None                       # RH derived from Q, see _wt_field
            wet_sw = build_wet_mie_tables(mmid_kg, rho_dry,
                                          os.path.join(RRTMGP_DATA, g_sw))
            wet_lw = build_wet_mie_tables(mmid_kg, rho_dry,
                                          os.path.join(RRTMGP_DATA, g_lw))
            self.dp_wet = wet_sw.pop('dp_wet')     # (n_wt, NBINS) [m]
            wet_lw.pop('dp_wet')
            self.mie_sw = {k: jnp.asarray(v) for k, v in wet_sw.items()}
            self.mie_lw = {k: jnp.asarray(v) for k, v in wet_lw.items()}
            # 550 nm cross-section per (wt%, bin), same wet diameters + RI
            self.sig_ext_550 = np.stack(
                [mie_at_wavelength(self.dp_wet[j], 0.550, wt_pct=w)[0]
                 for j, w in enumerate(WT_GRID)], axis=0)
            gf = self.dp_wet / np.asarray(dp_bin_m)[None, :]
            print(f"  optics: WET (Tabazadeh 1997 wt%, Tang 1997 density, "
                  f"P&W composition-dependent RI); {len(WT_GRID)} wt% nodes "
                  f"{WT_GRID[0]:.0f}-{WT_GRID[-1]:.0f}%, "
                  f"D_wet/D_dry {gf.min():.3f}-{gf.max():.3f}", flush=True)
        else:
            mie_sw = build_mie_tables(dp_bin_m, os.path.join(RRTMGP_DATA, g_sw))
            mie_lw = build_mie_tables(dp_bin_m, os.path.join(RRTMGP_DATA, g_lw))
            self.mie_sw = {k: jnp.asarray(v) for k, v in mie_sw.items()}
            self.mie_lw = {k: jnp.asarray(v) for k, v in mie_lw.items()}
            #compute 550 nm extinction cross-section per bin, scatter not needed for AOD
            self.sig_ext_550, _ = mie_at_wavelength(dp_bin_m, 0.550)
            print("  optics: DRY diameters, fixed 75 wt% RI (LEGACY -- "
                  "underestimates extinction ~47%; set WET_OPTICS=1)", flush=True)

        self._jit_hr = jax.jit(self._hr_impl)

        # ---- multi-GPU: the lat-chunk loop in heating() shards across all
        # visible devices (rrtmgp columns are independent, so this is exact).
        # Pre-place the small per-bin constants on every device so each chunk's
        # optics run entirely on its own GPU; rrtmgp's own tables are placed by
        # JAX automatically on first use per device.
        self.devices = jax.devices()
        self.ndev = len(self.devices)
        self._mie_sw_dev = [jax.device_put(self.mie_sw, d) for d in self.devices]
        self._mie_lw_dev = [jax.device_put(self.mie_lw, d) for d in self.devices]
        self._dp_band_dev = [jax.device_put(jnp.asarray(self.dp_band), d)
                             for d in self.devices]
        self._sig550_dev = [jax.device_put(jnp.asarray(self.sig_ext_550), d)
                            for d in self.devices]

    # ------------------------------------------------------------------
    # per-hour CESM fields
    # ------------------------------------------------------------------
    def _read3d(self, ds, var, t):
        """(nzc, nlat, nlon) native full-column field at absolute hour t."""
        return ds[var].isel(time=t, lev=self.krad).values

    """Surface fluxes"""
    def surface(self, t):
        flds = self.dFLDS['FLDS'].isel(time=t).values
        flns = self.dFLNS['FLNS'].isel(time=t).values
        fsds = self.dFSDS['FSDS'].isel(time=t).values
        fsns = self.dFSNS['FSNS'].isel(time=t).values
        # FLNS = up - down at surface  =>  sigma*Ts^4 = FLNS + FLDS (emis ~ 1)
        # getting the upward flux ... F = sigma*T^4 => T = (F/sigma)^(1/4)
        ts = ((np.maximum(flns + flds, 1.0)) / SIGMA_SB) ** 0.25
        # SURFACE albedo = reflected/incoming = (flux down - flux net)/ flux down = (FSDS - FSNS)/FSDS = 1 - FSNS/FSDS
        alb = np.where(fsds > 5.0,
                       np.clip(1.0 - fsns / np.maximum(fsds, 1e-3), 0.02, 0.95),
                       ALB_FALLBACK)
        return ts, alb

    # ------------------------------------------------------------------
    # column assembly: CESM (lev top->bot, lat, lon) -> rrtmgp (lat, lon, z-up)
    # with one halo cell at each z end
    # ------------------------------------------------------------------
    @staticmethod
    def _to_xyz(a, floor=None):
        """(nz, nlat, nlon) top->bottom  ->  (nlat, nlon, nz+2) bottom->top.

        Halos are linearly extrapolated; `floor` clamps them (linear
        extrapolation of small positive tracers can go negative, which the
        gas-optics lookups turn into NaN)."""
        #flip z-axis to bottom-up
        a = jnp.moveaxis(jnp.asarray(a), 0, -1)[..., ::-1]      # z now bottom-up
        #radiation needs a layer above and below the column; we linearly extrapolate the first and last layers to get these halo layers
        lo = 2.0 * a[..., :1] - a[..., 1:2]                     # linear halo
        hi = 2.0 * a[..., -1:] - a[..., -2:-1]
        #clamps the halo values to a minimum value (floor) to avoid negative values which can cause issues in the gas-optics lookups
        if floor is not None:
            lo = jnp.maximum(lo, floor); hi = jnp.maximum(hi, floor)
        return jnp.concatenate([lo, a, hi], axis=-1)

    def _band_to_xyz(self, prop):
        """(nb, nzb, nx, nlon) band-level aerosol -> (nb, nx, nlon, nz+2)."""
        nb, _, nx, _ = prop.shape
        full = jnp.zeros((nb, self.nzc, nx, self.nlon), prop.dtype)
        full = full.at[:, self.band_pos].set(prop)
        full = jnp.moveaxis(full, 1, -1)[..., ::-1]
        #instead of linear extrapolation, we just pad with zeros for the halo layers
        #may need to think about whether zeroing the halo layers is physically reasonable (rather than fixing to boundary conditions), 
        #but for now it avoids NaN issues in the gas-optics lookups
        pad = jnp.zeros(full.shape[:-1] + (1,), prop.dtype)
        return jnp.concatenate([pad, full, pad], axis=-1)

    def _hr_impl(self, T, q, o3, ch4, ts, alb, zen, aer_sw, aer_lw):
        """One rrtmgp call on already-chunked (x=lat rows) xyz fields."""
        p = self._p_xyz_chunk(T.shape[0])
        rho = p / (rrtmgp_constants.R_D * T)
        zero = jnp.zeros_like(T)
        #constants
        co2 = jnp.full_like(T, CO2_PPM * 1e-6)
        n2o = jnp.full_like(T, N2O_PPB * 1e-9)
        #this is the core RRTMGP call
        #5 zero arrays for cloud fields and empty dict for cloud optics
        out = self.rt.compute_heating_rate(
            rho, q, zero, zero, zero, zero, zero, T, ts, p, {},
            zenith=zen, irrad=TSI_DEFAULT, sfc_alb=alb, sfc_emis=SFC_EMIS,
            vmr_fields={'o3': o3, 'ch4': ch4, 'co2': co2, 'n2o': n2o},
            aerosol_optics_sw={'optical_depth': aer_sw[0], 'ssa': aer_sw[1],
                               'asymmetry_factor': aer_sw[2]},
            aerosol_optics_lw={'optical_depth': aer_lw[0], 'ssa': aer_lw[1],
                               'asymmetry_factor': aer_lw[2]})
        #extracts the heating from from the output dictionary ()
        hr = out[rrtmgp_common.KEY_STORED_RADIATION]            # K/s, (x,y,nz)
        diags = jnp.stack([out['sw_flux_down_full'][..., -1],   # TOA SW down
                           out['sw_flux_up_full'][..., -1],     # TOA SW up
                           out['lw_flux_up_full'][..., -1],     # OLR
                           out['sw_flux_down_full'][..., 0],    # sfc SW down
                           out['lw_flux_down_full'][..., 0]])   # sfc LW down
        #return the heating rate and diagnosti fluxes 
        return hr, diags

    def _wt_field(self, T_band, q_band):
        """Equilibrium H2SO4 weight percent on the band levels, (nzb,nlat,nlon).

        RH is derived from the CONSERVED water vapour mixing ratio q and the
        temperature actually being evaluated, NOT from CESM's RELHUM field. That
        matters in anomaly mode: the perturbed call runs at T_eff = T_CESM + dT,
        and CESM's RELHUM was computed at T_CESM, so reusing it would hold RH
        fixed while the temperature moved -- physically the vapour is what is
        conserved and RH must fall as the layer warms. Doing it this way makes
        aerosol warming shrink the droplet (warmer -> lower RH -> more
        concentrated -> smaller -> less extinction), a real negative feedback
        that the fixed-RH version would have suppressed.

        Uses the same saturation-vapour-pressure form as the Tabazadeh fit, so
        the RH we form here and the P_sat it divides back out cancel exactly.
        """
        from tomas_jax.fast.water import h2so4_equilibrium_wt, _TABAZADEH_EQ1
        p = self.p_lay[self.band_pos][:, None, None]         # Pa
        # vapour partial pressure from mixing ratio (kg/kg, dry-air basis eps)
        eps = 0.621981
        qq = np.maximum(np.asarray(q_band), 1e-12)
        p_h2o = qq * p / (eps + (1.0 - eps) * qq)            # Pa
        c0, c1, c2, c3 = _TABAZADEH_EQ1
        Tb = np.maximum(np.asarray(T_band), 150.0)
        ln_p_sat = c0 + c1 / Tb + c2 / Tb**2 + c3 / Tb**3    # ln(mbar)
        rh_pct = 100.0 * (p_h2o / 100.0) / np.exp(ln_p_sat)  # Pa->mbar
        return np.asarray(h2so4_equilibrium_wt(jnp.asarray(Tb),
                                               jnp.asarray(rh_pct)))

    def _p_xyz_chunk(self, nx):
        p = np.broadcast_to(self.p_lay[:, None, None],
                            (self.nzc, nx, self.nlon))
        px = self._to_xyz(p)
        # keep halo pressures inside the gas-optics lookup range
        return jnp.clip(px, 1.006, 1.09e5)

    # ------------------------------------------------------------------
    def heating(self, t, num_band, T_band_override=None):
        """Heating rate for absolute hour index t with aerosol num_band.

        num_band : (NBINS, nzb, nlat, nlon) [#/kg] on the coupled band levels
        T_band_override : optional (nzb, nlat, nlon) replacing CESM T on the
                          band levels (T_eff = T_CESM + dT for the perturbed call)
        Returns hr_band (nzb, nlat, nlon) [K/s] in CESM level order, and a
        diag dict (TOA/sfc fluxes, band AOD at 550 nm).
        """
        T = self._read3d(self.dT, 'T', t).astype(np.float64)
        #replaces the temperature on the coupled band levels to emulate the aerosol direct effect on heating rate (T_eff = T_CESM + dT)
        if T_band_override is not None:
            T[self.band_pos] = np.asarray(T_band_override)
        q = self._read3d(self.dQ, 'Q', t)
        o3 = self._read3d(self.dO3, 'O3', t)
        ch4 = self._read3d(self.dCH4, 'CH4', t)
        ts, alb = self.surface(t)
        zen = solar_zenith(self.time[t], self.lat, self.lon)
        #converts 3D fields to bottom-up xyz coordinates with halos
        Tx = self._to_xyz(T)
        qx = self._to_xyz(np.maximum(q, 1e-12), floor=1e-12)
        o3x = self._to_xyz(np.maximum(o3, 0.0), floor=0.0)
        ch4x = self._to_xyz(np.maximum(ch4, 0.0), floor=0.0)
        #converts fields to jax arrays
        tsj = jnp.asarray(ts); albj = jnp.asarray(alb)
        zenj = jnp.asarray(zen)[:, :, None]                     # (nlat,nlon,1)

        # aerosol optics are computed PER lat-chunk: the full-globe per-band
        # fields are ~3 GB and were the OOM driver on memory-tight GPUs.
        hr_rows, dg_rows, aod_rows = [], [], []
        C = RAD_LAT_CHUNK
        num_j = jnp.asarray(num_band)
        # per-cell composition -> interpolation weights over WT_GRID. Built from
        # the SAME T array the radiation sees (incl. any T_band_override), so the
        # droplet composition tracks the perturbed temperature.
        if self.wet_optics:
            wwt_all = wt_weights(self._wt_field(T[self.band_pos],
                                                q[self.band_pos]))
        # MULTI-GPU: keep each chunk <= RAD_LAT_CHUNK rows (memory bound) but make
        # the chunk count a multiple of ndev and send chunk i to device i%ndev, so
        # the chunks run concurrently across GPUs. Columns are independent in
        # rrtmgp, so the result is bit-identical to the serial single-GPU loop.
        nchunk = self.ndev * int(np.ceil(self.nlat / (self.ndev * C)))
        C = int(np.ceil(self.nlat / nchunk))
        for i, a in enumerate(range(0, self.nlat, C)):
            b = min(a + C, self.nlat)
            k = i % self.ndev                       # target device for this chunk
            put = lambda x: jax.device_put(x, self.devices[k])
            dp_d = self._dp_band_dev[k]
            nc = put(num_j[:, :, a:b])
            #compuets aerosol optical properties (tau, ssa, g) for both SW and LW bands
            #using the precomputed Mie tables and the number mixing ratios for each bin
            if self.wet_optics:
                ww = put(wwt_all[:, :, a:b])
                tau_s, ssa_s, g_s = _aerosol_props_wet(
                    nc, dp_d, self._mie_sw_dev[k]['sigma_ext'],
                    self._mie_sw_dev[k]['ssa'], self._mie_sw_dev[k]['g'], ww)
                tau_l, ssa_l, g_l = _aerosol_props_wet(
                    nc, dp_d, self._mie_lw_dev[k]['sigma_ext'],
                    self._mie_lw_dev[k]['ssa'], self._mie_lw_dev[k]['g'], ww)
            else:
                tau_s, ssa_s, g_s = _aerosol_props(nc, dp_d,
                                               self._mie_sw_dev[k]['sigma_ext'],
                                               self._mie_sw_dev[k]['ssa'], self._mie_sw_dev[k]['g'])
                tau_l, ssa_l, g_l = _aerosol_props(nc, dp_d,
                                               self._mie_lw_dev[k]['sigma_ext'],
                                               self._mie_lw_dev[k]['ssa'], self._mie_lw_dev[k]['g'])
            #converts the aerosol optical properties to bottom-up xyz coordinates with zero halos
            aer_sw = tuple(self._band_to_xyz(v) for v in (tau_s, ssa_s, g_s))
            aer_lw = tuple(self._band_to_xyz(v) for v in (tau_l, ssa_l, g_l))
            hr, dg = self._jit_hr(
                put(Tx[a:b]), put(qx[a:b]), put(o3x[a:b]), put(ch4x[a:b]),
                put(tsj[a:b]), put(albj[a:b]), put(zenj[a:b]), aer_sw, aer_lw)
            #550 nm AOD
            if self.wet_optics:
                aod_rows.append(_aod550_wet(nc, dp_d, self._sig550_dev[k], ww))
            else:
                aod_rows.append(_aod550(nc, dp_d, self._sig550_dev[k]))
            hr_rows.append(hr); dg_rows.append(dg)
        # gather every chunk back to the primary device before concatenating
        # (no-op when ndev==1, so single-GPU behaviour is unchanged)
        d0 = self.devices[0]
        hr = jnp.concatenate([jax.device_put(x, d0) for x in hr_rows], axis=0)     # (nlat,nlon,nz)
        diags = jnp.concatenate([jax.device_put(x, d0) for x in dg_rows], axis=1)  # (5,nlat,nlon)
        aod550 = jnp.concatenate([jax.device_put(x, d0) for x in aod_rows], axis=0)# (nlat,nlon)

        # back to CESM order, band levels only (strip halos, flip z)
        hr_cesm = jnp.moveaxis(hr[..., 1:-1][..., ::-1], -1, 0)  # (nzc,nlat,nlon)
        #extracts the heating rate for the coupled band levels only, in CESM level order
        hr_band = hr_cesm[self.band_pos]

        dnames = ('sw_dn_toa', 'sw_up_toa', 'olr', 'sw_dn_sfc', 'lw_dn_sfc')
        #creates a dictionary mapping diagnostic names to their corresponding arrays
        diag = {n: diags[i] for i, n in enumerate(dnames)}
        diag['aod550'] = aod550
        #returns the heating rate for the coupled band levels and a dictionary of diagnostic fluxes and AOD
        return hr_band, diag

    def heating_anomaly(self, t, num_evolved, num_ref, dT_band):
        """(HR(evolved bins, T_CESM+dT) - HR(reference bins, T_CESM)) [K/s].

        The perturbed call sees T_eff so LW cooling responds to accumulated
        warming (negative Planck feedback bounds dT); the reference call is
        pure CESM. Returns (hr_anom, diag_perturbed, diag_reference).
        """
        #CESM temp on just aerosol levels
        T_cesm_band = self._read3d(self.dT, 'T', t)[self.band_pos]
        #evolved tomas bins and temperature 
        #dt_rad contains the accumulated warming from all previous steps
        hr1, d1 = self.heating(t, num_evolved,
                               T_band_override=T_cesm_band + np.asarray(dT_band))
        #background reference bins and pure CESM temperature
        hr0, d0 = self.heating(t, num_ref)
        #heating rate anomoly
        return hr1 - hr0, d1, d0
