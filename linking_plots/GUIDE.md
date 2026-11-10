# Linked-system plots: code, data and how to rerun them

All plotting code, intermediate data and figures for the page **AIDE-SAI Linked System 1997**
(https://claude.ai/artifact/QD9nQpThJMo6QCoa6MNN6h) live here: `/data/AIDE-SAI-link/linking_plots/`.
Moved into this repository on 2026-11-09, with every path rewritten to `/data`. The originals are untouched.

```
linking_plots/
  GUIDE.md                  this file
  run_coupled_2yr.sh        one-command rebuild of the coupled figures for two years
  collect_presentation.py   copies every page figure into presentation/<section>/ (GIFs also as last-frame PNG)
  presentation/             slide-ready copies of the 42 figures on the page (made 2026-11-09)
  site/index.html, site/img/  the page source and the exact published images (42)
  climate_transport/        section 1 Climate state + Age of air, section 2 Advection   (NOTES.md there)
  micro_box/                section 3 Microphysics box test                            (NOTES.md there)
  coupled/                  section 4 Coupled system                                   (NOTES.md there)
  assets/ne_110m/           Natural Earth coastlines used by the map plots
```
Each section's `NOTES.md` holds the method details, the captions and the numbers quoted on the page.

## Ground rules
- **Run as `ubuntu`** (`sudo -u ubuntu ...`). `/data` is NFS with root-squash; files here are owned by ubuntu.
- For long jobs, detach: `sudo -u ubuntu setsid nohup bash job.sh > job.log 2>&1 < /dev/null &`.
- **Python environments:**
  - `/data/aide_emu_venv/bin/python` (`PY`): numpy, jax, matplotlib. Use it for processing, radiation, the micro box test and non-map plots.
  - `/data/strat-dycore-compare/.venv/bin/python` (`PYC`): cartopy and xarray. Use it for map figures (`plots_v2.py`) and the climate/ERA5 scripts.
  - Both venvs symlink to `/usr/bin/python3` (3.10).
- **Set `MPLCONFIGDIR=/tmp/mpl_ubuntu` and `PYTHONDONTWRITEBYTECODE=1`.** Without them, the scripts try to write a cache into
  your home directory or `__pycache__` and fail with a permission error.
- **On k8s GPU nodes** (like g265), GPU jobs also need `LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu` and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **Run directories are read-only inputs.** Never write into `runs/emu_campaign_1997*`.

## Path settings (environment variables, coupled scripts)
| variable | default | set to, for two years |
|---|---|---|
| `LP_RUNS` | `/data/AIDE-SAI-link/runs/emu_campaign_1997` (year 1: steps 0–1459) | unchanged |
| `LP_RUNS_Y2` | `/data/AIDE-SAI-link/runs/emu_campaign_1997_y2` (year 2: steps 1460–2919) | unchanged |
| `LP_MET` | `/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997/met` | `/data/AIDE-SAI-link/runs/emu_winds_P2s200_1997_1998/met` |
| `LP_NDAYS` | `365` | `730` |
| `LP_COUPLED` | `/data/AIDE-SAI-link/linking_plots/coupled` | — |

`common.frame_file(run, s)` picks the year-1 or year-2 directory from the step number. The per-day scripts (`process.py`,
`wet_dist.py`, `rad_diurnal.py`) and `agg2.py`/`plot_wet_dist.py` follow `LP_NDAYS`. The monthly means in `agg2.py` pool the same
calendar month across years. With `LP_NDAYS=365` everything reproduces the published year-1 output: `agg2.py` was re-run and
matches the original bit for bit, and `plots_v2.py`, `plot_wet_dist.py` and `heating_figs.py` were re-run from here.

## Which script makes which figure

### 1. Climate state (`climate_transport/`, PYC)
| page figure | script | input |
|---|---|---|
| T/u zonal-mean GIF (emulator) | `plot_climate.py` (writes `climate_T_u_zonalmean_1997.gif` only if missing) | `data/climate_zm.npz` |
| CESM / emulator / difference GIF | `plot_climate_gif_cmp.py` | `data/climate_zm.npz` |
| 3-row annual mean (CESM, emulator, diff) | `plot_climate_3row.py` | `data/climate_zm.npz` |
| ERA5 table, 4-row PNG, ERA5 GIF | `plot_climate_era5.py` | `data/climate_zm.npz`, `data/era5_zm.npz` |
| `data/climate_zm.npz` (daily zonal means, emulator + CESM, 1.2–143 hPa) | `climate_zm.py` (CPU, reads met + CESM archive) | met dir, CESM h1 archive |
| `data/era5_zm.npz` | `era5_zm.py` (streams WeatherBench2 ERA5 over HTTPS), check `era5_validate.py` | network |
| Age of air (5 figures + GIF) | `plot_transport.py` | `data/age_emu.npz`, `data/age_cesm.npz`, `data/aoa_run.npz` |
| `data/age_*.npz`, `data/aoa_run*.npz` | `aoa_transport.py` (GPU, about 6.8 min per model year): `--mode age` with `ADV_SL_FIXER=none` for age of air; `--mode clock` for the injection tracer and the fixer artifact | met dir; CESM winds from `build_cesm_met.py` (deleted to save 23 GB; rebuild if needed) |

`plotlib.save_gif` writes GIFs with one palette shared by all frames, so colour bars stay constant over time. Use `ms=130`.

### 2. Advection (`climate_transport/`)
The injection tracer GIFs, day panels, budget and CESM-wind comparison all come from `plot_transport.py`, using `data/aoa_run*.npz`.
The page uses `inj_*_final.gif`: the same frames re-timed to 130 ms with a shared palette. Re-time with the snippet in
`collect_presentation.py`, or with `plotlib.save_gif(frames, path, ms=130)`.

### 3. Microphysics (`micro_box/`, PY, GPU)
This is the 10k-cell, 24 h box test, emulator vs numerical, on 1997 frames. **It does not depend on year 2, so there is no need to rerun it.**
Recipe: `scripts/reproduce.sh` (cells, numerical, bare network), then `run_routed.py` (deployed routed emulator, 2 GPUs), then
`finish_routed.sh` (merge and figures, via `plot_box.py`). The page uses `fig1/2/3_*.png`; the bare-network figures are in `supp/bare/`.

### 4. Coupled system (`coupled/`)
Pipeline: `process.py` → `rad_diurnal.py` → `wet_dist.py` → `agg2.py` → figures.
| page figure | script |
|---|---|
| `v2_fig1_*` forcing maps, `v2_fig2` forcing time series, `v2_fig3_a/b/c` heating lat-p / monthly / 50 hPa map, `v2_fig4_a–e` size maps and distributions, `v2_fig5` sulfate GIF | `scripts/plots_v2.py` (PYC) |
| `v2_fig4_f` wet size distribution | `scripts/plot_wet_dist.py` (PY) |
| `fig3_d` heating time series, `fig3_e` latitude × season, `fig3_f` latitude bands | `extra/heating_figs.py` (PY). Rebuilt on 2026-11-09 from the code that made the published figures; reproduces the published numbers (net, last 30 days: emulator 27.48, numerical 28.04, MAM4 27.66 mK/day). `heating_timeseries_LEGACY_v1.py` is the older 2-scheme version. |
| `v2_fig0` tropopause | `scripts/plots.py` (v1, uses `cache/agg.npz` from `agg.py`). This is the only v1 figure still on the page; `plots.py` also reads a local `sai_dashboard` forcing CSV (path from `$FORCING_CSV`) for a v1 panel. |

Cache (`coupled/cache/`, 13 GB, all deletable and rebuildable):
- `day_DDD.npz`: emulator + numerical. Made by `process.py`.
- `day_mam_DDD.npz`: MAM4. Made by `PAIRS_SEL=mam PFX=mam_ process.py`.
- `diurnal/`: SW re-solved at 16 solar times per day, every 5th day. Made by `rad_diurnal.py <i> <n>`, one process per GPU.
- `wetdist/`: per-day wet-distribution sums.
- Aggregates: `agg_v2.npz`, `timeseries_v2.npz`, `daily_gif_v2.npz`.

Year-1 timings:
- `process.py`: about 8 min (emulator + numerical) and 13 min (MAM4) with 24 workers.
- `rad_diurnal.py`: about 75 min on 5 GPUs.
- `plots_v2.py`: about 4 min.

## Two-year rerun (after the year-2 runs finish)
The year-2 runs are set up in `/data/AIDE-SAI-link/runs/emu_campaign_1997_y2/INSTRUCTIONS.md`. When all six coupled runs show `Complete in`:

```
cd /data/AIDE-SAI-link/linking_plots
sudo -u ubuntu setsid nohup env GPUS="0 1 2 3 4 5 6 7" NW=24 bash run_coupled_2yr.sh > coupled_2yr.log 2>&1 < /dev/null &
```
This computes only the year-2 days (365–729), re-solves the SW diurnal cycle for them, re-aggregates and redraws the coupled
figures. Expected time is roughly year 1's: about 30 min of CPU processing and 40–60 min on 8 GPUs.

**Still one-year in the plotting code. Edit before or after the rerun:**
- `coupled/scripts/plots_v2.py`:
  - Time axes: line 100 `day = np.arange(1, 366)` and the axis label `'day of 1997'` (line 117).
  - The last-30-days label `days 336-365` (line 68). The window itself already follows `LP_NDAYS`.
  - The monthly GIF title `{mon[m]} 1997` (line 161). Months are now pooled over both years.
  - The sulfate GIF loop `for d in range(0, 365, 2)` and its title (lines 210–215). With 730 days, keep the GIF under about 10 MB with step 4 or `GIF5_DPI=72`.
- **"Annual mean" panels** become two-year means automatically (mean over all days). Rename the titles to "2-year mean".
- **DJF/JJA panels:** not written yet. Select days with `(day % 365)` in Dec–Feb or Jun–Aug from the per-day arrays (`cache/timeseries_v2.npz` or the `day_*.npz` caches).
- **`coupled/scripts/plot_wet_dist.py`:** the caption text says "all 365 days".
- **`coupled/extra/heating_figs.py`:** already follows `LP_NDAYS`. The x-label becomes "day since 1997-01-01".
- **`v2_fig0` tropopause** (`agg.py`/`plots.py`): `agg.py` still asserts 365 day files. Fine to leave as year 1, since the tropopause comes from the meteorology.

**Climate state for two years** (zonal-mean T, DJF/JJA averaged over 1997–1998, down to 300 hPa):
- `climate_zm.py`: set `MET` to the two-year met and `idx = np.arange(0, 2921, 4)`. The CESM archive covers 1998, so the CESM comparison works as is. Write the result to a new file such as `data/climate_zm_2yr.npz`.
- **To go to 300 hPa:** the met `u/v/omega/T` files only hold the 24 band levels (1.2–143 hPa).
  - **Temperature:** `T_full.npy` in the met dir has the emulator T on every CESM level down to the surface. Use its levels down to 300 hPa; the pressures are in `meta.json` `krad`, through the CESM hybrid coefficients.
  - **Winds below 143 hPa:** regrid from the emulator's T63L95 frames (`runs/emu_winds_P2s200_1997/frames_1997-01-01/`, and `..._1998/frames_1998-01-01/` once made) with the same log-p and bilinear method as `scripts/met/emu_to_met.py`, run with `--p-hi 300`.
  - **CESM:** read `lev` indices down to 300 hPa instead of `meta['klevs']`.
- `era5_zm.py`: dates start at 1997-01-01 for `len(d['hours'])` days. With the two-year `climate_zm` it covers 1998 automatically. ERA5 has 300 hPa on its 37 levels; extend the level selection.
- Aerosol and heating plots stay at 143 hPa: the coupled model has no aerosol below its band.

**Age of air / advection:** these are transport-only runs on cycled 1997 winds and are independent of the coupled year 2. To rerun them on the two-year winds, pass the two-year met to `aoa_transport.py --met`.

## Updating the page and the slides
- After figures change, run `sudo -u ubuntu /data/strat-dycore-compare/.venv/bin/python collect_presentation.py` to refresh
  `presentation/`. This copies from `site/img/`, so copy the new figure into `site/img/` under its page name first. The page
  names are the figure names with `v2_` or `micro_routed_` prefixes; see `site/index.html`.
- Republishing the page is done from a Claude Code session with the Artifact tool. Point it at
  `/data/AIDE-SAI-link/linking_plots/site/index.html` and the new images, and ask it to update
  https://claude.ai/artifact/QD9nQpThJMo6QCoa6MNN6h. The page text, including captions and numbers, is in `index.html`;
  numbers that change with two years need editing there.
