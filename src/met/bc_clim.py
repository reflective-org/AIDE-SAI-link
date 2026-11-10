"""BC_SRC=clim: the CESM archive replaced by a climatology file + the met file.

coupling.py reads every boundary field through `open_var(name)`, which returns
an xarray Dataset the code then indexes as `ds[name].isel(time=..., lev=...,
lat=...).values`, plus `ds['lat'].values`, `ds['hyam'].values`, `ds['P0']`,
`ds['time'].values[t]` (radiation's solar zenith), `ds.sizes['time']` and
`ds.encoding['source']` (the reference-cache key). `ClimArchive.open_var`
returns an object with exactly that surface, so switching the boundary source
is one rebinding in coupling.main and no physics code changes.

What it serves
  * climatology fields (scripts/met/make_bc_climatology.py: 12 calendar
    months of the 1996-2014 archive): Q, O3, CH4, SO2, H2SO4, OH, RELHUM, T,
    the MAM4 modes, FLDS/FLNS/FSDS/FSNS/PS/TS. At absolute hour t (the h1
    hour counter, noleap hours since 1996-01-01) the field is the LINEAR
    interpolation in day-of-year between the two bracketing month centres,
    cyclic over the year. No diurnal cycle: OH is only read under OH_SZA=0,
    and the surface fluxes enter as an albedo ratio and an upward LW.
  * the met file (scripts/met/emu_to_met.py, required): U, V, OMEGA on the
    band levels and T on the band levels OR on the radiation column (T_full,
    every native level the radiation uses), at the step-end hours. The
    archive's temperature is never served: under this source T IS the
    emulator's. A level selection other than those two is refused rather
    than silently filled from the climatology.

Why hour-indexed: the whole loop counts hours from H0 on the archive axis
and the met file's meta.json places its frame 0 on the same axis, so the
calendar (day-of-year, hour-of-day, solar zenith) is one integer arithmetic
away and nothing upstream had to learn a date type.
"""
import datetime
import json
import os

import cftime
import numpy as np

ORIGIN = cftime.DatetimeNoLeap(1996, 1, 1)   # hour index 0 of the archive axis
HOURS_PER_YEAR = 365 * 24


class _TimeAxis:
    """`ds['time'].values` stand-in: cftime noleap datetimes by hour index."""
    def __init__(self, n):
        self.size = int(n)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        i = int(i)
        if not 0 <= i < self.size:
            raise IndexError(f'hour {i} outside 0..{self.size - 1}')
        return ORIGIN + datetime.timedelta(hours=i)


class _Values:
    def __init__(self, values):
        self.values = values


class ClimVar:
    """A DataArray stand-in: chained .isel(time, lev, lat) then .values."""
    def __init__(self, arch, var, time=None, lev=None, lat=None):
        self.arch, self.var, self.time, self.lev, self.lat = arch, var, time, lev, lat

    def isel(self, time=None, lev=None, lat=None):
        return ClimVar(self.arch, self.var,
                       self.time if time is None else time,
                       self.lev if lev is None else lev,
                       self.lat if lat is None else lat)

    @property
    def values(self):
        return self.arch.read(self.var, self.time, self.lev, self.lat)


class ClimDataset:
    """The Dataset stand-in `open_var(var)` returns."""
    def __init__(self, arch, var):
        self.arch, self.var = arch, var
        self.sizes = dict(time=arch.ntime, lev=arch.lev.size, lat=arch.lat.size, lon=arch.lon.size)
        self.encoding = dict(source=f'bc_clim:{os.path.basename(arch.clim_file)}')

    def __getitem__(self, key):
        if key == self.var:
            return ClimVar(self.arch, self.var)
        if key in ('lat', 'lon', 'hyam', 'hybm', 'hyai', 'hybi', 'P0', 'lev'):
            return _Values(getattr(self.arch, key))
        if key == 'time':
            return _Values(_TimeAxis(self.arch.ntime))
        raise KeyError(f'{key}: not a coordinate and not the variable this dataset holds ({self.var})')

    def close(self):
        pass


class ClimArchive:
    def __init__(self, clim_file, met_dir):
        self.clim_file = clim_file
        z = np.load(clim_file, mmap_mode='r')
        self.z = z
        for k in ('lat', 'lon', 'hyam', 'hybm', 'hyai', 'hybi', 'lev'):
            setattr(self, k, np.asarray(z[k], np.float64))
        self.P0 = np.float64(z['P0'])
        self.doy_c = np.asarray(z['month_centre_doy'], np.float64)       # (12,)
        self.fields = set(str(v) for v in z['fields_3d']) | set(str(v) for v in z['fields_2d'])
        # the met file: winds and temperature
        with open(os.path.join(met_dir, 'meta.json')) as f:
            self.meta = json.load(f)
        self.met = {k: np.load(os.path.join(met_dir, f'{k}.npy'), mmap_mode='r')
                    for k in ('u', 'v', 'omega', 'T', 'T_full')}
        self.klevs = [int(k) for k in self.meta['klevs']]
        self.krad = [int(k) for k in self.meta['krad']]
        self.hour0 = int(self.meta['hour0']); self.step = int(self.meta['step_hours'])
        self.nframe = int(self.meta['nframe'])
        if not (np.allclose(self.meta['lat'], self.lat) and np.allclose(self.meta['lon'], self.lon)):
            raise SystemExit('bc_clim: the met file and the climatology are on different grids')
        # the hour axis ends where the met file ends: the loop's own range check
        # (coupling.main, "run needs time index up to ...") then says so
        self.ntime = self.hour0 + self.step * (self.nframe - 1) + 1
        self.met_vars = {'U': 'u', 'V': 'v', 'OMEGA': 'omega', 'T': 'T'}

    def open_var(self, var):
        if var not in self.fields and var not in self.met_vars:
            raise SystemExit(f'bc_clim: {var!r} is neither in the climatology file nor the met file')
        return ClimDataset(self, var)

    # ------------------------------------------------------------------
    def describe(self):
        return (f"climatology {os.path.basename(self.clim_file)} "
                f"({str(self.z['provenance'])}); U/V/OMEGA/T from the met file "
                f"({self.nframe} frames x {self.step} h from hour {self.hour0}, {self.meta.get('source')})")

    def _frame(self, t):
        if (t - self.hour0) % self.step:
            raise SystemExit(f'bc_clim: hour {t} is not on the met file\'s {self.step}-h grid')
        i = (t - self.hour0) // self.step
        if not 0 <= i < self.nframe:
            raise SystemExit(f'bc_clim: hour {t} = met frame {i}, outside 0..{self.nframe - 1}')
        return i

    def _month_weights(self, t):
        doy = (t / 24.0) % 365.0 + 0.5 / 24.0      # mid-hour, day-of-year in [0, 365)
        c = self.doy_c
        if doy < c[0]:                              # before the January centre: Dec -> Jan
            m0, m1 = 11, 0; w = (doy + 365.0 - c[11]) / (c[0] + 365.0 - c[11])
        elif doy >= c[11]:
            m0, m1 = 11, 0; w = (doy - c[11]) / (c[0] + 365.0 - c[11])
        else:
            m1 = int(np.searchsorted(c, doy, side='right')); m0 = m1 - 1
            w = (doy - c[m0]) / (c[m1] - c[m0])
        return m0, m1, float(w)

    @staticmethod
    def _as_index(sel):
        if sel is None:
            return slice(None)
        if isinstance(sel, slice):
            return sel
        if np.isscalar(sel):
            return int(sel)
        return np.asarray(sel, int)

    def read(self, var, time, lev, lat):
        if time is None:
            raise SystemExit(f'bc_clim: {var} read without a time index')
        if isinstance(time, slice) or not np.isscalar(time):
            raise SystemExit(f'bc_clim: {var}: only one hour per read is served (got {time!r})')
        t = int(time)
        if var in self.met_vars:
            i = self._frame(t)
            lv = None if lev is None else [int(k) for k in np.atleast_1d(np.asarray(lev))]
            if lv == self.klevs:
                a = self.met[self.met_vars[var]][i]
            elif var == 'T' and lv == self.krad:
                a = self.met['T_full'][i]
            else:
                raise SystemExit(f'bc_clim: {var} is only available on the band levels'
                                 + (' or the radiation column' if var == 'T' else '')
                                 + f', not on levels {lv}')
            a = np.array(a)
            if lat is not None:
                a = a[:, self._as_index(lat)]
            return a
        # .npz members cannot be memory-mapped: every self.z[var] re-reads the whole
        # (12, 70, 192, 288) member from disk (0.37 s on /data). Read each once and
        # keep it (3.2 GB for all fields; numerically identical).
        if not hasattr(self, '_cache'):
            self._cache = {}
        if var not in self._cache:
            self._cache[var] = np.asarray(self.z[var])
        f = self._cache[var]                         # (12, [lev,] lat, lon)
        m0, m1, w = self._month_weights(t)
        if f.ndim == 4:
            li = self._as_index(lev); la = self._as_index(lat)
            a0 = f[m0][li][..., la, :] if lat is not None else f[m0][li]
            a1 = f[m1][li][..., la, :] if lat is not None else f[m1][li]
        else:
            if lev is not None:
                raise SystemExit(f'bc_clim: {var} is 2-D, no lev selection')
            la = self._as_index(lat)
            a0 = f[m0][la] if lat is not None else f[m0]
            a1 = f[m1][la] if lat is not None else f[m1]
        return (np.asarray(a0, np.float32) * np.float32(1.0 - w)
                + np.asarray(a1, np.float32) * np.float32(w))
