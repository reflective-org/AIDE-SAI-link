# aide_sai_core

One-way coupled **CESM → TOMAS-JAX** sectional aerosol model for stratospheric
aerosol injection (SAI), in JAX on GPU.

> [!WARNING]
> This project is under active development, there is no guarantee that it will
> operate as expected. In particular the model is **not calibrated for absolute
> radiative forcing** — see the standing caveats in [MANIFEST.md](./MANIFEST.md)
> before quoting any number out of it.

CESM meteorology forces the aerosol; the aerosol does not feed back on the
circulation. Within the stratospheric band the aerosol is internally generated
and internally removed:

```
CESM h1 hourly ──► U, V, OMEGA ──────► transport (Lin-Rood flux-form)
               ├─► T, p, RELHUM ─────► microphysics (TOMAS, 40 bins)
               ├─► MAM4 num/so4 ─────► IC, open-BC reservoir, polar caps
               └─► SO2, H2SO4 ───────► gas-phase IC/BC

  SOURCE  continuous SO2 injection    SO2 ─OH─► H2SO4 ─► nucleation/condensation
  SINK    gravitational settling out of the band bottom
  OPTICS  RRTMGP + Mie on the wet H2SO4/H2O droplet ──► heating, AOD550, ARF_toa
```

- **State**: 40 TOMAS bins × (number, dry SO4 mass) + 2 gas tracers = 82
  advected 3-D fields, on the native CESM f09 grid (192 × 288) over a
  1–150 hPa band (24 levels), 6 h coupling step.
- **Scale**: a 90-day run is ~33 h on one H100; microphysics is ~94% of it.

## Installation

```bash
git clone https://github.com/reflective-org/aide_sai_core.git
cd aide_sai_core
pip install -r requirements.txt
pip install --upgrade "jax[cuda12]>=0.6.2"   # GPU wheel -- the CPU one cannot do production
```

Two dependencies are separate repos, not on PyPI. Clone them **beside** this one
and they are found automatically; otherwise set `TOMAS_JAX_PATH` / `RRTMGP_PATH`:

| repo | role |
|---|---|
| [`reflective-org/tomas-jax`](https://github.com/reflective-org/tomas-jax) | sectional aerosol microphysics |
| [`climate-analytics-lab/jax-rrtmgp`](https://github.com/climate-analytics-lab/jax-rrtmgp) | radiative transfer (`RAD=0` skips it) |

**The CESM forcing is not included and cannot be.** The model reads 14 hourly
`h1` variables (`U V OMEGA T num_a{1,2,3} so4_a{1,2,3} SO2 H2SO4 OH RELHUM`); as
archived they total ~23 TB. Point `CESM_DIR` at an archive laid out as
`$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF`. Reads are one hour × band
levels at a time (~5.3 MB per variable-hour), so a subset is enough: ~3.6 GB for
a 2-day test, ~160 GB for a 90-day run. Full variable list and units:
[docs/COUPLING_VARIABLES.md](./docs/COUPLING_VARIABLES.md).

## Quick Start

```bash
# 1-step smoke test (~6 min on an H100), then the three figures
N_HOURS=6 OUT_TAG=smoke INJ_SO2_TG_YR=10 ./run_prod.sh
python3 plot_run.py smoke

# the production scenario: 10 Tg SO2/yr equatorial ring, 90 days
INJ_SO2_TG_YR=10 OUT_TAG=prod90d ./run_prod.sh
RESUME=1 INJ_SO2_TG_YR=10 OUT_TAG=prod90d ./run_prod.sh   # continue after a kill
```

`run_prod.sh` sets the whole production environment and execs `driver_fast.py`.
**A bare `./run_prod.sh` is a no-injection control** — `INJ_SO2_TG_YR` defaults
to 0 so a forgotten flag gives an obviously unforced baseline rather than a
silent SAI result. Always pair a scenario with its own `OUT_TAG`; outputs and
checkpoints are keyed by it.

Every knob, and the reasoning behind each default, is in
**[MANIFEST.md](./MANIFEST.md) — the canonical reference for this tree.** Where
`docs/` disagrees with it, MANIFEST is current.

## Running Tests

```bash
python3 validation/test_conservation.py                        # advection conservation
ADV_F32=1 ADV_CFL=0.5 python3 validation/validate_vpos_f32.py  # positivity limiter
```

Run them from the repo root. Validate advection changes at `ADV_F32=1
ADV_CFL=0.5`, the production precision — **not** the f64/cfl=0.2 module
defaults: two positivity-limiter bugs were invisible in f64 and fatal in f32.

## Citations

The parameterizations this model composes are other people's work:

- Lin & Rood (1996), *Mon. Weather Rev.* 124, 2046 — flux-form advection
- Adams & Seinfeld (2002), *J. Geophys. Res.* 107, 4370 — TOMAS sectional aerosol
- Tabazadeh et al. (1997), *Geophys. Res. Lett.* 24, 1931 — H2SO4/H2O equilibrium composition
- Tang (1997), *J. Geophys. Res.* 102, 1883 — solution density
- Palmer & Williams (1975), *Appl. Opt.* 14, 208 — H2SO4 refractive indices (via HITRAN Aerosols-2016)
- Pincus et al. (2019), *J. Adv. Model. Earth Syst.* 11, 3074 — RRTMGP
- Hanisco et al. (2001), *Geophys. Res. Lett.* 28, 4423 — the diurnal OH curve

## License

This project is released under the Apache 2.0 License - see the [LICENSE](./LICENSE) file for details.
