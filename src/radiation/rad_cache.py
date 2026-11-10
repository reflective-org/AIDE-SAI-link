"""Per-archive-hour cache of the ARCHIVE-ONLY radiation fields.

docs/RADIATION_EMULATOR_SCOPE.md §3: in anomaly mode half of the radiation
stage (the reference-aerosol solve and the MAM4 re-binning that feeds it)
depends only on the CESM archive hour, never on the model state. Computed
once and stored, those fields take the stage from 8.5 s to ~2.9 s per step
without touching the physics: the cached arrays ARE the arrays the loop would
have computed (same code, same inputs, float64 on disk), so a run with the
cache is bit-identical to one without.

The same store holds what the EMULATOR (src/radiation/rad_emulator.py) needs
per hour and cannot compute itself: the clear-sky (zero-aerosol) heating at
T_CESM, its Planck sensitivity to +1 K on the band levels, and the clear-sky
TOA fluxes. The network predicts only the aerosol-induced residual on top of
those (§4.2).

One `.npz` per absolute archive hour under
    <root>/<key>/h<hour:06d>.npz          + <root>/<key>/key.json
where <key> is a hash of everything that determines the fields: the CESM
case, the band levels, the optics flags/tables, the gases, the reference
aerosol source and binning, and the jax-rrtmgp commit. A change to any of
those gets a fresh directory rather than silently stale fields.

Entries can be PARTIAL. The in-loop miss path (coupling.py) stores just the
reference solve (`hr0`, `d_*0`); `scripts/radiation/build_ref_cache.py` fills
the clear-sky fields as well. `get()` returns None unless every requested
field is present, so a partial entry is a miss for what it lacks and a hit
for what it has.

Fields (all (nzb, nlat, nlon) unless noted; hr in K/s, fluxes in W/m2):
  hr0            float64  heating with the reference aerosol at T_CESM
  d0_<name>      float64  (nlat, nlon) the 6 diag fields of that call
                          (sw_dn_toa, sw_up_toa, olr, sw_dn_sfc, lw_dn_sfc, aod550)
  hr_clear       float32  zero-aerosol heating at T_CESM (total)
  hr_clear_sw    float32  its shortwave part
  hr_dT          float32  [hr_clear(T+1K on band levels) - hr_clear(T)] per K
  hr_dT_sw       float32  shortwave part of that
  dc_<name>      float32  (nlat, nlon) clear-sky diag fields (5 fluxes)
  ddT_<name>     float32  (nlat, nlon) their +1 K sensitivities
  net_<tag>      float32  (n_out, nlat, nlon) the emulator <tag>'s raw output
                          on the reference aerosol (RAD_EMU_ANOM=net)
"""
import hashlib
import json
import os

import numpy as np

REF_KEYS = ('hr0', 'd0_sw_dn_toa', 'd0_sw_up_toa', 'd0_olr', 'd0_sw_dn_sfc',
            'd0_lw_dn_sfc', 'd0_aod550')
CLEAR_KEYS = ('hr_clear', 'hr_clear_sw', 'hr_dT', 'hr_dT_sw',
              'dc_sw_dn_toa', 'dc_sw_up_toa', 'dc_olr', 'dc_sw_dn_sfc', 'dc_lw_dn_sfc',
              'ddT_sw_dn_toa', 'ddT_sw_up_toa', 'ddT_olr', 'ddT_sw_dn_sfc', 'ddT_lw_dn_sfc')
DIAG_NAMES = ('sw_dn_toa', 'sw_up_toa', 'olr', 'sw_dn_sfc', 'lw_dn_sfc', 'aod550')


def _submodule_commit(repo_root, name):
    """HEAD of a submodule without invoking git (the checkout is on a
    root-squashed NFS mount where git refuses 'dubious ownership')."""
    try:
        gitfile = os.path.join(repo_root, 'models', name, '.git')
        if os.path.isdir(gitfile):
            gitdir = gitfile
        else:
            with open(gitfile) as f:
                line = f.read().strip()
            gitdir = os.path.normpath(os.path.join(repo_root, 'models', name,
                                                   line.split('gitdir:', 1)[1].strip()))
        with open(os.path.join(gitdir, 'HEAD')) as f:
            head = f.read().strip()
        if head.startswith('ref:'):
            with open(os.path.join(gitdir, head.split(' ', 1)[1])) as f:
                head = f.read().strip()
        return head
    except OSError:
        return 'unknown'


def make_key(key_dict):
    """Stable short hash of the dict that defines the cached fields."""
    s = json.dumps(key_dict, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


class RefCache:
    def __init__(self, root, key_dict, label='rad_ref'):
        self.root = root
        self.key_dict = dict(key_dict)
        self.key = make_key(self.key_dict)
        self.dir = os.path.join(root, f'{label}_{self.key}')
        self.n_hit = 0; self.n_miss = 0; self.n_put = 0
        os.makedirs(self.dir, exist_ok=True)
        side = os.path.join(self.dir, 'key.json')
        if not os.path.exists(side):
            tmp = side + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(self.key_dict, f, indent=1, sort_keys=True, default=str)
            os.replace(tmp, side)

    def path(self, t):
        return os.path.join(self.dir, f'h{int(t):06d}.npz')

    def has(self, t, keys=REF_KEYS):
        p = self.path(t)
        if not os.path.exists(p):
            return False
        try:
            with np.load(p) as z:
                return all(k in z.files for k in keys)
        except (OSError, ValueError):
            return False

    def get(self, t, keys=REF_KEYS):
        """dict of the requested arrays, or None if any is missing."""
        p = self.path(t)
        if not os.path.exists(p):
            self.n_miss += 1
            return None
        try:
            with np.load(p) as z:
                if not all(k in z.files for k in keys):
                    self.n_miss += 1
                    return None
                out = {k: z[k] for k in keys}
        except (OSError, ValueError):
            self.n_miss += 1
            return None
        self.n_hit += 1
        return out

    def put(self, t, **arrays):
        """Merge `arrays` into the hour's entry (atomic replace)."""
        p = self.path(t)
        data = {}
        if os.path.exists(p):
            try:
                with np.load(p) as z:
                    data = {k: z[k] for k in z.files}
            except (OSError, ValueError):
                data = {}
        data.update({k: np.asarray(v) for k, v in arrays.items()})
        tmp = p[:-4] + f'.tmp{os.getpid()}.npz'
        np.savez(tmp, **data)
        os.replace(tmp, p)
        self.n_put += 1

    def stats(self):
        return f"hit {self.n_hit} miss {self.n_miss} put {self.n_put} ({self.dir})"


def diag_to_arrays(prefix, diag, dtype):
    """Flatten a heating() diag dict into cache fields."""
    return {f'{prefix}_{n}': np.asarray(diag[n]).astype(dtype)
            for n in DIAG_NAMES if n in diag}


def arrays_to_diag(prefix, arrays):
    return {n: arrays[f'{prefix}_{n}'] for n in DIAG_NAMES
            if f'{prefix}_{n}' in arrays}
