# inputs/

Static input data that the coupler reads, plus a pointer to the one dataset that
is far too large to live here.

## `rad_data/palmer_williams_h2so4.dat` (tracked)

Refractive index of aqueous sulfuric acid: Palmer & Williams (1975), as
distributed in HITRAN Aerosols-2016, tabulated over wavelength at 25, 38, 50,
75, 84.5 and 95.6 wt% H2SO4. `src/radiation/radiation.py` reads it
as `RI_FILE` to build the Mie lookup tables that turn TOMAS bins into RRTMGP
optical properties. 33 KB, so it is tracked in git rather than fetched.

## The CESM forcing archive (NOT here)

The hourly `h1` time series is the single input the model cannot run without,
and it is ~23 TB. It is never copied into this repo. On the machine these
results were produced on it lives at:

```
/data/cesm2.1.5_output/histSST/
  f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001/archive/atm/proc/tseries
```

`src/coupling.py` expects CESM's own tseries convention, one file per variable:

```
$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF
```

Point a clone elsewhere at its own archive with three environment variables
(defaults are the FWHIST run above):

| variable | default |
|---|---|
| `CESM_DIR` | `/data/cesm2.1.5_output/histSST/f.e21.FWHIST.../archive/atm/proc/tseries` |
| `CESM_PREFIX` | `f.e21.FWHIST.f09_f09_mg17.atmos-scale_fixedSST_1996-2014.001.cam` |
| `CESM_SUF` | `.1996010100-2014123100.nc` |

## RRTMGP spectral data (NOT here either)

The `rrtmgp-gas-{sw,lw}-g*.nc` k-distribution and cloud optics files ship inside
the jax-rrtmgp package and are resolved from it at import
(`RRTMGP_DATA = <models/jax-rrtmgp>/rrtmgp/optics/rrtmgp_data`). Nothing to
install separately — `git submodule update --init` brings them.
