"""Builders for the per-hour cache entries (rad_cache.py), shared by
scripts/radiation/build_ref_cache.py, label_columns.py and the emulator's
own miss path. All take the production RadiationDriver."""
import numpy as np
import jax.numpy as jnp

import rad_cache


def clear_fields(drv, t, dT_probe=1.0):
    """The archive-only clear-sky fields of hour t (rad_cache.CLEAR_KEYS):
    zero-aerosol heating at T_CESM (total + SW part), its finite-difference
    sensitivity to +dT_probe K on the band levels (per K), and the same for
    the diag fluxes. Two solves."""
    T_band = drv._read3d(drv.dT, 'T', t)[drv.band_pos]
    hr_c, d_c = drv.heating(t, None, split=True)
    hr_p, d_p = drv.heating(t, None, T_band_override=T_band + dT_probe, split=True)
    hr_c = np.asarray(hr_c); hr_p = np.asarray(hr_p)
    sw_c = np.asarray(d_c['hr_sw_band']); sw_p = np.asarray(d_p['hr_sw_band'])
    out = dict(hr_clear=hr_c.astype(np.float32),
               hr_clear_sw=sw_c.astype(np.float32),
               hr_dT=((hr_p - hr_c) / dT_probe).astype(np.float32),
               hr_dT_sw=((sw_p - sw_c) / dT_probe).astype(np.float32))
    for n in rad_cache.DIAG_NAMES[:5]:
        out[f'dc_{n}'] = np.asarray(d_c[n]).astype(np.float32)
        out[f'ddT_{n}'] = ((np.asarray(d_p[n]) - np.asarray(d_c[n])) / dT_probe).astype(np.float32)
    return out


def ensure_clear(drv, t):
    """Clear-sky fields of hour t from the driver's cache, computing and
    storing them on a miss (or computing without storing if no cache)."""
    c = drv.ref_cache
    if c is not None:
        e = c.get(t, rad_cache.CLEAR_KEYS)
        if e is not None:
            return e
    e = clear_fields(drv, t)
    if c is not None:
        c.put(t, **e)
    return e


def ensure_ref(drv, t, num_ref_fn):
    """Reference solve of hour t (rad_cache.REF_KEYS), via drv.reference()
    which stores on a miss. num_ref_fn(t) -> (NBINS, nzb, nlat, nlon) #/kg."""
    if drv.ref_ready(t):
        hr0, d0 = drv.reference(t, None)
    else:
        hr0, d0 = drv.reference(t, jnp.asarray(num_ref_fn(t)))
    return hr0, d0
