"""Shared helpers for the coupled-system (no feedback) figures. Read-only on the run dirs."""
import json
import numpy as np

import os
# Paths are overridable from the environment (see /data/AIDE-SAI-link/linking_plots/GUIDE.md).
RUNS = os.environ.get('LP_RUNS', '/data/AIDE-SAI-link/runs/emu_campaign_1997')            # year 1 (steps 0-1459)
RUNS_Y2 = os.environ.get('LP_RUNS_Y2', '/data/AIDE-SAI-link/runs/emu_campaign_1997_y2')   # year 2 (steps 1460-2919)
MET = os.environ.get('LP_MET', '/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997/met')    # 2 years: .../emu_winds_P2s200_1997_1998/met
NDAYS = int(os.environ.get('LP_NDAYS', '365'))                                             # 730 for two years


def frame_file(run, s):
    """3-D frame of step s: year-1 directory for s < 1460, year-2 directory after."""
    return f'{RUNS if s < 1460 else RUNS_Y2}/{run}/frames3d/step_{s:04d}.npz'
CLIM = '/data/AIDE-SAI-link/runs/bc_clim/bc_clim.npz'
OUT = os.environ.get('LP_COUPLED', '/data/AIDE-SAI-link/linking_plots/coupled')
CACHE = f'{OUT}/cache'
PAIRS = {'emu': ('r1_tomas_emu', 'r1c_tomas_emu_ctl'),
         'num': ('r2_tomas_fast', 'r2c_tomas_fast_ctl'),
         'mam': ('r3_mam4', 'r3c_mam4_ctl')}
G = 9.80616
RD = 287.04
H0 = 8766            # run start hour (absolute); frame step s: hour_abs = H0 + 6(s+1); T3d used met index s
RHO_AER = 1770.0

_TAB1 = np.array([
    [10, 19.726, -4364.8, -147620], [15, 19.747, -4390.9, -144690], [20, 19.761, -4414.7, -142940],
    [25, 19.794, -4451.1, -140870], [30, 19.883, -4519.2, -136500], [35, 20.078, -4644.0, -127240],
    [40, 20.379, -4828.5, -112550], [45, 20.637, -5011.5, -98811], [50, 20.682, -5121.3, -94033],
    [55, 20.555, -5177.6, -96984], [60, 20.405, -5252.1, -100840], [65, 20.383, -5422.4, -97966],
    [70, 20.585, -5743.8, -83701], [75, 21.169, -6310.6, -48396], [80, 21.808, -6985.9, -12170]])
_EQ1 = (18.452406985, -3505.1578801, -330918.55082, 12725068.262)


def h2so4_equilibrium_wt(temp, rh_percent):
    """numpy port of tomas_jax.fast.water.h2so4_equilibrium_wt (Tabazadeh et al. 1997)."""
    temp = np.asarray(temp, np.float64)
    rh = np.clip(np.asarray(rh_percent, np.float64), 1e-3, 99.0)
    c0, c1, c2, c3 = _EQ1
    lnps = c0 + c1 / temp + c2 / temp**2 + c3 / temp**3
    x = -(np.log(rh / 100.0) + lnps)
    T = temp[..., None]
    xp = -(_TAB1[:, 1] + _TAB1[:, 2] / T + _TAB1[:, 3] / T**2)
    n = 15
    i = np.clip((xp < x[..., None]).sum(-1), 1, n - 1)
    x0 = np.take_along_axis(xp, (i - 1)[..., None], -1)[..., 0]
    x1 = np.take_along_axis(xp, i[..., None], -1)[..., 0]
    fr = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    wt = _TAB1[:, 0]
    return wt[i - 1] + fr * (wt[i] - wt[i - 1])


def tang_density(x):
    ds0 = 0.9971 + x * (7.367e-3 + x * (-4.934e-5 + x * (1.754e-6 + x * -1.104e-8)))
    return ds0 * 1000.0


def wet_d3(mmid_kg, wt):
    """settling.wet_size: returns wet diameter CUBED [m^3] for dry SO4 mass mmid (bins) x wt (cells)."""
    acid = mmid_kg * (98.0 / 96.0)
    m_h2o = acid * (100.0 / wt - 1.0)
    mt = mmid_kg + m_h2o
    x = 100.0 * acid / mt
    return mt / tang_density(x) * (6.0 / np.pi)


def month_weights(t, doy_c):
    doy = (t / 24.0) % 365.0 + 0.5 / 24.0
    c = doy_c
    if doy < c[0]:
        return 11, 0, (doy + 365.0 - c[11]) / (c[0] + 365.0 - c[11])
    if doy >= c[11]:
        return 11, 0, (doy - c[11]) / (c[0] + 365.0 - c[11])
    m1 = int(np.searchsorted(c, doy, side='right')); m0 = m1 - 1
    return m0, m1, (doy - c[m0]) / (c[m1] - c[m0])


def meta():
    with open(f'{MET}/meta.json') as f:
        return json.load(f)


def wmo_tropopause(Tf, pf):
    """WMO lapse-rate tropopause. Tf, pf: (nlev, ncol) top->bottom (K, Pa).
    Lowest level (searching upward from 500 hPa, up to 50 hPa) where the lapse rate to the next level is
    <= 2 K/km and the average lapse rate from it to every level within the next 2 km stays <= 2 K/km.
    Returns p_trop (ncol,) [Pa], NaN where none found."""
    T = Tf[::-1].astype(np.float64); p = pf[::-1].astype(np.float64)     # bottom-up
    nl, nc = T.shape
    dz = RD * 0.5 * (T[1:] + T[:-1]) / G * np.log(p[:-1] / p[1:])       # m
    z = np.concatenate([np.zeros((1, nc)), np.cumsum(dz, 0)], 0)
    lapse = -(T[1:] - T[:-1]) / dz * 1000.0                              # K/km, half levels
    ptrop = np.full(nc, np.nan)
    found = np.zeros(nc, bool)
    for k in range(nl - 1):
        cand = (~found) & (p[k] <= 50000.0) & (p[k] >= 5000.0) & (lapse[k] <= 2.0)
        if not cand.any():
            continue
        ok = cand.copy()
        for j in range(k + 1, nl):
            dzj = z[j] - z[k]
            within = dzj <= 2000.0
            if not (within & ok).any():
                break
            avg = (T[k] - T[j]) / np.maximum(dzj, 1e-6) * 1000.0
            ok &= ~(within & (avg > 2.0))
        ptrop[ok] = p[k][ok]
        found |= ok
        if found.all():
            break
    return ptrop


MAM_SIGMA_G = (1.8, 1.6, 1.8, 1.6)    # mam4_adapter.SIGMA_G (accum, aitken, coarse, primary carbon)


def mam_modes_to_bins(num_m, so4_m, dp_bin_nm):
    """numpy copy of microphysics/mam4_adapter.modes_to_bins: each lognormal mode (fixed sigma_g) is cut at
    the TOMAS bin-boundary masses, conserving the in-range number and mass. Bin edges: xk_k = MMID_0/sqrt2 * 2^k."""
    from scipy.special import erf
    mmid = (np.asarray(dp_bin_nm) * 1e-9) ** 3 * np.pi / 6 * RHO_AER
    lnxk = np.log(mmid[0] / np.sqrt(2.0) * 2.0 ** np.arange(len(mmid) + 1))
    phi = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    shape = num_m.shape[1:]
    nb = np.zeros((len(mmid),) + shape, np.float32); mb = np.zeros_like(nb)
    for m in range(num_m.shape[0]):
        n = np.asarray(num_m[m], np.float64); s_ = np.asarray(so4_m[m], np.float64)
        valid = (n > 0) & (s_ > 0)
        if not valid.any():
            continue
        s = 3.0 * np.log(MAM_SIGMA_G[m])
        amm = np.where(valid, s_ / np.maximum(n, 1e-300), 1.0)
        lnmg = np.log(amm) - 0.5 * s * s
        for k in range(len(mmid)):
            fn = phi((lnxk[k + 1] - lnmg) / s) - phi((lnxk[k] - lnmg) / s)
            fm = phi((lnxk[k + 1] - lnmg - s * s) / s) - phi((lnxk[k] - lnmg - s * s) / s)
            nb[k] += np.where(valid, fn * n, 0.0); mb[k] += np.where(valid, fm * s_, 0.0)
    return nb, mb
