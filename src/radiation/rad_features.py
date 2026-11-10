"""Per-column features and targets of the radiation emulator -- ONE definition
shared by the label writer (scripts/radiation/label_columns.py), the trainer
(scripts/radiation/train_rad_mlp.py, which stores the layout it was trained
on) and the adapter (rad_emulator.py). docs/RADIATION_EMULATOR_SCOPE.md §4.3.

A column is what RRTMGP sees and nothing else: the gas column on the native
levels, the surface, the sun, the aerosol band optics on the band levels, and
the temperature increment the evolved call runs at. No latitude, longitude or
time: the solve is column-local, so the network is.

  features (NF = 4*nzc + 3 + nzb + 3*(nb_sw+nb_lw)*nzb + (nb_sw+nb_lw)):
    T[nzc]  log q[nzc]  log o3[nzc]  log ch4[nzc]        native levels, top->bottom
    ts  alb  cos(zenith)+                                  surface, sun (cosz clipped at 0)
    dT[nzb]                                                 T_eff - T_CESM on the band levels
    log1p(tau/TAU0)  ssa  g   per band (SW then LW) x band level
    log1p(sum_z tau / TAU0_COL)  per band                   column optical depth
  targets (NT = 2*nzb + 2), what the network must add to the cached clear-sky
  fields (rad_cache.py) to reproduce the evolved call:
    y_sw[nzb]  = 86400 * (hr1_sw - hr_clear_sw - hr_dT_sw * dT)          [K/day]
    y_lw[nzb]  = 86400 * (hr1_lw - hr_clear_lw - hr_dT_lw * dT)          [K/day]
    y_toa[2]   = (sw_up_toa1 - dc_sw_up_toa - ddT_sw_up_toa * <dT>,
                  olr1       - dc_olr       - ddT_olr * <dT>)             [W/m2]
  where hr_dT is the cached finite-difference response to +1 K on every band
  level (a Jacobian row sum) applied to the LOCAL dT -- the linearisation
  scripts/radiation/check_planck_linearization.py verifies -- and <dT> is
  the dT_WEIGHT-weighted mean of dT over the band (the flux response to the
  uniform +1 K probe, scaled). Both residuals of these approximations are in
  the target, so the network corrects them; at zero aerosol and zero dT the
  target is exactly zero.
"""
import numpy as np
import jax
import jax.numpy as jnp

TAU0 = 1e-7          # per-layer optical depth scale of the log1p feature
TAU0_COL = 1e-6      # column optical depth scale
Q_FLOOR = 1e-12      # kg/kg (heating() itself floors q at 1e-12)
VMR_FLOOR = 1e-15    # o3 / ch4 vmr floor before the log
SEC_PER_DAY = 86400.0


def layout(nzc, nzb, nb_sw, nb_lw):
    """Feature/target slices. Returns dict name -> (start, stop) and sizes."""
    nb = nb_sw + nb_lw
    L = {}; i = 0
    def add(name, n):
        nonlocal i
        L[name] = (i, i + n); i += n
    add('T', nzc); add('logq', nzc); add('logo3', nzc); add('logch4', nzc)
    add('sfc', 3); add('dT', nzb)
    add('tau', nb * nzb); add('ssa', nb * nzb); add('g', nb * nzb)
    add('tau_col', nb)
    L['NF'] = i
    T = {}; j = 0
    def addt(name, n):
        nonlocal j
        T[name] = (j, j + n); j += n
    addt('y_sw', nzb); addt('y_lw', nzb); addt('y_toa', 2)
    L['NT'] = j
    L['nzc'] = nzc; L['nzb'] = nzb; L['nb_sw'] = nb_sw; L['nb_lw'] = nb_lw
    L['targets'] = T
    return L


def _flat(a):
    """(..., nx, nlon) -> (nx*nlon, ...)  with the leading axes moved last."""
    a = jnp.asarray(a)
    nd = a.ndim
    a = jnp.moveaxis(a, (-2, -1), (0, 1))
    return a.reshape((a.shape[0] * a.shape[1], -1)) if nd > 2 else a.reshape((-1, 1))


@jax.jit
def column_features(T, q, o3, ch4, ts, alb, zen, dT,
                    tau_s, ssa_s, g_s, tau_l, ssa_l, g_l):
    """Assemble (ncol, NF) float32 features from one lat-chunk of fields.

    T, q, o3, ch4 : (nzc, nx, nlon) native levels, CESM order (top->bottom)
    ts, alb, zen  : (nx, nlon)   surface T [K], albedo, solar zenith [rad]
    dT            : (nzb, nx, nlon)  T_eff - T_CESM on the band levels [K]
    tau_s..g_l    : (nb, nzb, nx, nlon)  band optics on the band levels
    """
    f = [_flat(T), _flat(jnp.log(jnp.maximum(q, Q_FLOOR))),
         _flat(jnp.log(jnp.maximum(o3, VMR_FLOOR))),
         _flat(jnp.log(jnp.maximum(ch4, VMR_FLOOR))),
         _flat(ts), _flat(alb), _flat(jnp.maximum(jnp.cos(zen), 0.0)),
         _flat(dT)]
    tau = jnp.concatenate([tau_s, tau_l], axis=0)       # (nb, nzb, nx, nlon)
    ssa = jnp.concatenate([ssa_s, ssa_l], axis=0)
    g = jnp.concatenate([g_s, g_l], axis=0)
    f += [_flat(jnp.log1p(tau / TAU0)), _flat(ssa), _flat(g),
          _flat(jnp.log1p(tau.sum(axis=1) / TAU0_COL))]
    return jnp.concatenate(f, axis=1).astype(jnp.float32)


def dT_mean_weights(nzb):
    """Weights of the band-mean dT that scales the cached flux sensitivity
    (uniform: the probe was a uniform +1 K)."""
    return np.full(nzb, 1.0 / nzb)


def targets_from_fields(hr1, hr1_sw, d1, clear, dT, nzb):
    """(ncol, NT) float32 residual targets. hr1/hr1_sw (nzb, nx, nlon) K/s
    on the band levels; d1 the evolved call's diag dict; clear the cached
    CLEAR_KEYS arrays (full globe, pre-sliced to the chunk by the caller);
    dT (nzb, nx, nlon)."""
    hr1 = np.asarray(hr1); hr1_sw = np.asarray(hr1_sw); dT = np.asarray(dT)
    hr1_lw = hr1 - hr1_sw
    hc_sw = clear['hr_clear_sw']; hc_lw = clear['hr_clear'] - clear['hr_clear_sw']
    s_sw = clear['hr_dT_sw']; s_lw = clear['hr_dT'] - clear['hr_dT_sw']
    y_sw = SEC_PER_DAY * (hr1_sw - hc_sw - s_sw * dT)
    y_lw = SEC_PER_DAY * (hr1_lw - hc_lw - s_lw * dT)
    w = dT_mean_weights(nzb)
    dTm = np.tensordot(w, dT, axes=(0, 0))                # (nx, nlon)
    y_up = np.asarray(d1['sw_up_toa']) - clear['dc_sw_up_toa'] - clear['ddT_sw_up_toa'] * dTm
    y_olr = np.asarray(d1['olr']) - clear['dc_olr'] - clear['ddT_olr'] * dTm
    cols = [np.asarray(_flat(y_sw)), np.asarray(_flat(y_lw)),
            np.asarray(_flat(y_up)), np.asarray(_flat(y_olr))]
    return np.concatenate(cols, axis=1).astype(np.float32)


def prep_hour(drv, t):
    """The host fields heating() reads for absolute hour t (same reads, same
    floors): T, q, o3, ch4 (nzc, nlat, nlon); ts, alb, zen (nlat, nlon)."""
    import radiation as R
    T = drv._read3d(drv.dT, 'T', t).astype(np.float64)
    q = drv._read3d(drv.dQ, 'Q', t); o3 = drv._read3d(drv.dO3, 'O3', t)
    ch4 = drv._read3d(drv.dCH4, 'CH4', t)
    ts, alb = drv.surface(t)
    zen = R.solar_zenith(drv.time[t], drv.lat, drv.lon)
    return dict(T=T, q=q, o3=o3, ch4=ch4, ts=ts, alb=alb, zen=zen)


def optics_chunk(drv, num_chunk, wwt_chunk, k=0):
    """Band optics of one lat-chunk exactly as heating() computes them:
    (tau, ssa, g) for SW and LW, each (nb, nzb, nx, nlon), on device k."""
    import radiation as R
    dp_d = drv._dp_band_dev[k]
    if drv.wet_optics:
        sw = R._aerosol_props_wet(num_chunk, dp_d, drv._mie_sw_dev[k]['sigma_ext'],
                                  drv._mie_sw_dev[k]['ssa'], drv._mie_sw_dev[k]['g'], wwt_chunk)
        lw = R._aerosol_props_wet(num_chunk, dp_d, drv._mie_lw_dev[k]['sigma_ext'],
                                  drv._mie_lw_dev[k]['ssa'], drv._mie_lw_dev[k]['g'], wwt_chunk)
    else:
        sw = R._aerosol_props(num_chunk, dp_d, drv._mie_sw_dev[k]['sigma_ext'],
                              drv._mie_sw_dev[k]['ssa'], drv._mie_sw_dev[k]['g'])
        lw = R._aerosol_props(num_chunk, dp_d, drv._mie_lw_dev[k]['sigma_ext'],
                              drv._mie_lw_dev[k]['ssa'], drv._mie_lw_dev[k]['g'])
    return sw + lw


def wt_weights_for(drv, T_full, q_full):
    """Composition weights (n_wt, nzb, nlat, nlon) from the FULL-column T (with
    any band override applied) and q, as heating() builds them; None when
    the optics are dry."""
    import radiation as R
    if not drv.wet_optics:
        return None
    return R.wt_weights(drv._wt_field(T_full[drv.band_pos], q_full[drv.band_pos]))
