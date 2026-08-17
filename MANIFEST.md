# MANIFEST — CESM → TOMAS-JAX SAI model

The canonical reference for this tree: what each file is, every knob, and **why
each default is what it is**. `README.md` is the orientation; where `docs/`
disagrees with this file, this file is current.

## Scope

**This tree is CESM + TOMAS + RRTMGP**, and every default, measurement and
caveat recorded here refers to that combination. The intended end state is a
model-agnostic dycore–aerosol–radiation coupler: swappable meteorology (including
emulated dycores, which would close the circulation feedback the current setup
lacks) and swappable aerosol schemes (CARMA, MAM, GLOMAP).

Where that seam already exists, it is deliberate and worth preserving:

| slot | how it is already swappable | how far |
|---|---|---|
| aerosol IC/BC source | `AER_SRC=mam4\|carma\|fixed` | real, all three paths run. `fixed` is a prescribed uniform PSD with no CESM aerosol at all — the advection-only comparison |
| microphysics engine | `driver_fast.py` monkeypatches `tomas_jax.fast` into `coupling.py` at import instead of branching inside it | real, and why the driver is a separate file |
| radiation | `RAD=0` drops it and the `jax-rrtmgp` dependency entirely | on/off only |
| meteorology | none — the reader expects the CESM `h1` layout and variable names | a new source is work, not configuration |

## Layout

Core modules are **flat on purpose** — `coupling.py` does bare
`import settling` / `radiation`, and both `coupling.py` and `driver_fast.py` insert
`fast_advection/` on `sys.path` themselves. Moving them into subdirectories breaks
those imports.

```
coupling.py                 the model: grid, IC/BC, budget, diagnostics, main loop
driver_fast.py              THE production entry point. Imports coupling.py and swaps in
                            two pieces before running it: the tomas_jax.fast batched
                            microphysics engine (replacing coupling's per-cell chain) and
                            fct_lr advection. Separate file, not a flag, because the fast
                            engine has its own API and is hard-fixed at 40 bins.
fct_core.py                 legacy sealed-face advection. NOT on the run path --
                            coupling.py imports fct_lr directly, so standalone and
                            production use the same transport. Kept solely for the
                            bit-identical legacy check in validation/test_conservation.py.
settling.py                 gravitational settling; also the canonical wet-droplet
                            sizing (tang_density/wet_size), which radiation.py imports
radiation.py                RRTMGP + Mie optics
fast_advection/fct_lr.py    Lin-Rood flux-form advection (the production scheme)
fast_advection/fct_fast.py  PPM/Zalesak primitives that fct_lr imports
rad_data/                   palmer_williams_h2so4.dat (HITRAN Aerosols-2016)
run_prod.sh                 the production launcher (self-documenting header)
plot_run.py                 the post-run figures (dashboard, filmstrip, size dist,
                            zonal-mean cross-section, drainage -- the last is the
                            advection-only comparison's figure)
gif_run.py                  animated versions of the filmstrip and cross-section panels
run_pulse_bdc.sh            launcher for the tagged-pulse Brewer-Dobson experiment
pulse_progress_abs.py       tagged-pulse figure: one lat-pressure panel per year on one
                            shared absolute scale -- the decay
pulse_deep_branch.py        tagged-pulse figure: the ascent/descent Hovmoller, tropics and
                            both polar caps. --log for the log-colour variant
docs/                       CONFIGURATION (every env var), VALIDATION (the harnesses),
                            PROCESSES, COUPLING_VARIABLES, BOUNDARY_CONDITIONS, README
validation/                 harnesses -- see below
.github/workflows/ci.yml    CI: the two self-contained tests + a ruff errors-only gate
```

**Analysis scripts are flat and read `coupled_*_<TAG>.npz` from the current working
directory**, so run them from wherever the outputs are. Outputs land in whatever
directory you launch from, and `.gitignore` excludes `*.npz/*.png/*.gif/*.log`
— which hides a misplaced run rather than preventing it, so `run_prod.sh`
refuses to start with the repo as `$PWD`. Keep a runs directory outside the tree.

## Setup from a fresh clone

```bash
git clone https://github.com/reflective-org/aide_sai_core.git
git clone -b gpu-fast https://github.com/reflective-org/tomas-jax     # beside it -- found automatically
git clone https://github.com/climate-analytics-lab/jax-rrtmgp         # ditto
git -C jax-rrtmgp apply ../aide_sai_core/patches/jax-rrtmgp-zenith.patch  # required for RAD=1
pip install -r aide_sai_core/requirements.txt
pip install --upgrade "jax[cuda12]>=0.6.2"     # GPU wheel; the CPU one is unusable for production
mkdir -p runs && cd runs                                     # outputs go OUTSIDE the repo
N_HOURS=6 OUT_TAG=smoke ../aide_sai_core/run_prod.sh         # smoke test first
```

For the **advection-only** experiment, skip the `jax-rrtmgp` clone and its patch
entirely (`RAD=0` drops that dependency) and make the smoke test the transport-only
one — `../README.md` on the `advection-mip` branch has the exact command.
`tomas-jax` on `gpu-fast` is still required even at `MICRO=off`: `coupling.py`
imports it at module scope for the bin grid.

**Neither sibling repo is usable as a plain clone of its default branch**, and
both failures are at import or in the first radiation call, not subtle:

| repo | required state | why | symptom if skipped |
|---|---|---|---|
| `tomas-jax` | branch `gpu-fast` | `tomas_jax.fast`, the batched reduced engine `driver_fast.py` monkeypatches in, exists only there — `main` has the per-cell chain only | `ModuleNotFoundError: tomas_jax.fast` at `driver_fast.py:111` |
| `jax-rrtmgp` | `main` + `patches/jax-rrtmgp-zenith.patch` | `radiation.py` passes zenith per column, `(nlat, nlon, 1)`; upstream `sw_cell_source` assumes a scalar and broadcasts it against the `(nlat, nlon)` TOA flux | shape blow-up in the SW solve on the first `RAD=1` step |

The patch is one hunk against jax-rrtmgp v0.2.1 (`d7abe2e`) and is the only
carried modification to either dependency; keep it that way, and if upstream
takes the fix, delete the patch rather than let the two drift. `git apply`
refuses a partial application, so a failure means pin `d7abe2e` and retry.

| dependency | not shipped because | how it is found |
|---|---|---|
| `tomas-jax` | separate repo (microphysics) | `$TOMAS_JAX_PATH`, else `../tomas-jax`, else the normal import path |
| `jax-rrtmgp` | separate repo (radiation) | `$RRTMGP_PATH`, same order. `RAD=0` skips radiation and this dependency |
| CESM `h1` hourly meteorology + MAM4 | ~23 TB as archived | **On the shared H100 box, nothing to set** — the `coupling.py` defaults resolve to the FWHIST archive under `/data`, world-readable. Elsewhere: `$CESM_DIR` (+ `CESM_PREFIX`, `CESM_SUF`). Layout: `$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF`. Variables: `docs/COUPLING_VARIABLES.md`. Reads are one hour × band levels at a time (~5.3 MB per variable-hour), so a subset suffices: ~3.6 GB for 2 days, ~160 GB for 90 days |
| CUDA driver library | site-specific | `run_prod.sh` puts `$CUDA_DRIVER_LIB` on `LD_LIBRARY_PATH`. Where libcuda.so.1 is not on the loader path JAX **silently falls back to CPU**; set the variable, or empty it for a normal CUDA install |

An env var that is set but points nowhere is an error, not a silent fallback; a
missing CESM file names the variable that built the path.

Do not activate the tomas-jax `.venv` — it is CPU-only jaxlib with no xarray. The
working combo is **system python3** with GPU jax.

## Running

**`run_prod.sh` sets every knob except the SCENARIO.** A bare run is *not* the
production config: `INJ_SO2_TG_YR` defaults to **0 = no injection**, so a forgotten
flag gives an obviously unforced baseline rather than a silent 10 Tg/yr SAI result.
State the amount and give the scenario its own tag.

**Launch from a runs directory, never the repo.** Every output path is relative
to the working directory, so that directory is where the run lands.
`run_prod.sh` **refuses to start** with the repo as `$PWD` — the mistake is
otherwise invisible, since `.npz`/`.png` are gitignored and a misplaced run looks
entirely normal while filling the tree (29 GB accumulated that way before
2026-08-12). The script still execs `driver_fast.py` from its own tree, so it
runs the code you edited regardless of where you launched it.

```bash
cd <runs dir>                                     # NOT the repo; outputs land here
REPO=/path/to/aide_sai_core
INJ_SO2_TG_YR=10 OUT_TAG=prod90d $REPO/run_prod.sh   # the 10 Tg/yr scenario, 90 days, ~33 h
$REPO/run_prod.sh                                    # NO-INJECTION control (INJ_SO2_TG_YR=0)
RESUME=1 INJ_SO2_TG_YR=10 OUT_TAG=prod90d $REPO/run_prod.sh   # continue that scenario
```

Two checkouts launched from the *same* runs directory with the same `OUT_TAG`
overwrite each other. `OUT_TAG` discipline is what keeps runs apart.

`run_prod.sh` pins a **single** GPU (`CUDA_VISIBLE_DEVICES=${GPU:-0}`) rather than
auto-selecting one, so the card a long run lands on is never a function of what
else happened to be idle at launch. Microphysics shards across all *visible*
devices, so exporting more than one card is what makes it multi-GPU.

**The RESUME env must match the run you are resuming.** `INJ_*` mismatches are
refused outright (the scenario is stamped in the state ckpt); `WET_SETTLING` /
`WET_OPTICS` / `SETTLE` / `ADV_VPOS` mismatches only WARN, because they change
physics mid-trajectory rather than invalidating the state — a seam, not a
corruption. Repeat the scenario flags on the resume command line; they are not read
back from the checkpoint.

Checkpoints are written **atomically** (temp file + `os.replace`) and in a fixed
order — frames, timeseries, then state LAST — so the physics state is never newer
than the diagnostics. A resume trims frames that ran past the state, warns about a
frames history that stops short of it (the filmstrip would splice across the hole
without showing it), and tolerates an unreadable timeseries ckpt by continuing with
an empty plotted history: every cumulative counter the run needs comes from the
state ckpt, not from there.

The full effective config is printed in the run's own header, so any log is
self-describing.

### The advection-only inter-model comparison

The experiment the `advection-mip` branch exists for, and what `../README.md` on
that branch documents end to end. Transport (and, optionally, settling) of a
**prescribed uniform PSD** with no microphysics, no radiation and no injection, so
that re-running the identical code against another model's winds isolates
inter-model spread in transport. Both faces are aerosol-free, so the band drains
and nothing refills it.

```bash
cd <runs dir>
DRIVER=coupling.py MICRO=off RAD=0 AER_SRC=fixed \
  BC_TOP_AER=0 BC_BOT_AER=0 P_LO_HPA=0.03 P_HI_HPA=150 \
  N_HOURS=2160 FRAME_EVERY=24 OUT_TAG=mip_settle $REPO/run_prod.sh
#   ... SETTLE=0 ...            OUT_TAG=mip_nosettle   pure advection / age of air
#   ... SETTLE=0 N_BINS=1 ...   OUT_TAG=mip_2yr        multi-year, FRAME_EVERY=120
python3 $REPO/plot_run.py mip_settle        # -> mip_settle_drain.png, the figure
```

`DRIVER=coupling.py` is required: `driver_fast.py` exists only to swap in the
batched microphysics engine and **exits at `MICRO=off`**. Both entry points share
the advection and settling code, so the two agree where both can run.

**Why each setting is what it is.**

* **Uniform mixing ratio, not uniform concentration.** A constant q is an exact
  steady state of flux-form advection alone (it moves ρ·q, so constant q stays
  constant to roundoff for any wind field, divergent or not). Every departure is
  then attributable — settling, the open faces, or scheme error — and at
  `SETTLE=0` the run is a pure advection ACCURACY test with no analytic solution
  needed. Uniform concentration would impose a ~100× vertical mixing-ratio
  gradient across the band and the two effects would be inseparable.
* **`FIXED_PSD=lognormal`, `FIXED_N=1e8` #/kg, `FIXED_DG_NM=200`,
  `FIXED_SIGMA=1.6`.** The absolute scale is irrelevant — with no microphysics
  the system is linear in the aerosol, so every drained fraction is scale-free;
  1e8 #/kg is ~8 #/cm³ at 50 hPa/210 K so the printed concentrations are
  recognisable. `FIXED_DG_NM` is the load-bearing knob (settling ∝ D²), and the
  distribution is deliberately broad rather than tuned, because under `MICRO=off`
  the 40 bins are INDEPENDENT passive tracers differing only in fall speed: one
  run resolves the whole size-dependence of drainage, with the small bins acting
  as an age-of-air tracer and the large bins settling-dominated.
* **`P_LO_HPA=0.03` (33 levels) — the lid was the contaminant at 1 hPa.** There,
  6.9% of the domain's air descends through the top face per 90 days carrying
  q = 0 under `BC_TOP_AER=0`, which **dilutes without appearing in the
  `vface_top` mass term at all** — that term counts only aerosol *leaving*. At
  0.03 hPa it is ~0.8%, so the domain drains through its base and `BC_TOP_AER=1`
  (which would stop it draining at all) is not needed. Tropical zonal-mean ascent
  turns poleward at ~2.1 hPa in this forcing, so the old 1 hPa lid was already
  above the branch; the forcing is 70-level to 4.5e-6 hPa, so the lid is a free
  choice. Cost of the taller domain: nsub ~270 vs ~170, ~12 s/step vs ~5 s/step.
* **`P_HI_HPA=150` → a floor at 143 hPa**, 1–3 km above the real extratropical
  tropopause, so tracer leaves somewhat early: a measured decay rate is a LOWER
  bound on the true stratospheric lifetime, and the honest wording is "left the
  143 hPa domain", not "removed". Stated explicitly on the command line even
  though it is the launcher default, because where the faces sit governs the
  whole result.
* **`SETTLE=1` (default) vs `SETTLE=0`.** On: the drainage experiment, and the
  settling/advective split is the point — settling depends on the model's own T
  and nothing else and should be nearly model-independent, while the advective
  flux through the same face IS the residual circulation and is where two dycores
  are expected to disagree. `plot_run.py` therefore never merges the two channels
  into one curve. Off: the tracer is passive and the IC is an exact steady state.
* **`N_BINS=1` only with `SETTLE=0`.** With no fall speed and no microphysics the
  bins are identical, so 40 of them is 40 copies of one answer; with `SETTLE=1` it
  collapses the fall-speed spectrum the drainage run measures. Side effect: one
  bin spanning the whole mass grid leaves no bin-bound tying num to mas, so
  `Dp(M/N)` in the log is meaningless — read `frames_zm_mas`, ignore the number
  moment.
* **`FRAME_EVERY`** is overridable as of 2026-08-14 and must be raised beyond
  ~90 days: the whole frames history is REWRITTEN at every frame, so I/O cost
  grows as (frames)². 24 h for 90 days; 120 h (5-day) for multi-year.

**Measured, and worth knowing before designing a run.** On the 90-day `mip_cesm`
run (at the old 1 hPa lid): 46.8% of the mass left the band, but the fraction of
ORIGINAL air remaining at day 90 is 0.99–1.00 through 8.6–43 hPa — the
stratosphere's interior is untouched, and that mass loss is almost entirely the
100–143 hPa layer exchanging with the troposphere, since that is where the air
mass is. **90 days shows nothing of the Brewer-Dobson drainage; it needs years.**

**Do not read tracer descent off Eulerian zonal-mean omega** — in the winter
polar stratosphere it has the opposite sign to the residual circulation that
actually moves tracer.

### The tagged-pulse Brewer-Dobson experiment

`run_pulse_bdc.sh` is the **seeded** variant of the section above — the same
transport-only machinery (`AER_SRC=fixed MICRO=off RAD=0`, both faces at 0,
`SETTLE=0`, `P_LO_HPA=0.03`), differing only in where the tracer starts. It seeds
ONLY the tropical lower stratosphere — `FIXED_LAT_MAX_DEG=15`
with `FIXED_P_LO_HPA=40 / FIXED_P_HI_HPA=90`, landing on 5 levels (43.2–87.8 hPa,
≈18–24 km) at 4.77e-10 kg/kg, 0.036 Tg of SO4 — and watches where the blob goes.
`SETTLE=0` and `N_BINS=1` make the tracer passive; the launcher's header states why
each of those is load-bearing.

Needs `tomas-jax` (`gpu-fast`) but NOT `jax-rrtmgp` (`RAD=0`), and the h1 files for
`U V OMEGA T num_a1..3 so4_a1..3 SO2 H2SO4 OH RELHUM` — all 14 are opened at
startup even though transport-only reads few of them, so a missing one stops the
run. The two figure scripts import only numpy and matplotlib, so the figures can be
regenerated from `coupled_frames_<TAG>.npz` alone, with no GPU and no CESM.

```bash
cd <runs dir> && $REPO/run_pulse_bdc.sh 15          # ~12 h on one H100, 3.8 GB of frames
python3 $REPO/pulse_progress_abs.py pulse_15yr
python3 $REPO/pulse_deep_branch.py  pulse_15yr          # and --log
python3 $REPO/gif_run.py pulse_15yr --log --massdens --decades 5 --stride 4
```

Measured on the 15-year run (1996–2011, 365-day calendar, so a frame index is a date):

| quantity | value |
|---|---|
| tropical ascent of the arrival front, 43 → 10 hPa | 0.48 mm/s |
| ditto, 10 → 0.4 hPa | 0.92 mm/s |
| polar arrival, lowermost strat / upper strat (60–80 N) | 0.15 yr / 1.45 yr |
| ditto (60–80 S) | 0.55 yr / 1.75 yr |
| polar descent from the annual-cycle phase lag (N / S) | −1.46 / −0.89 mm/s |
| burden remaining after 15 yr | 0.097% of initial |

**Standing caveats for this experiment.**

* The bottom face is at 143 hPa, 1–3 km above the real extratropical tropopause,
  and 80% of the drainage happens poleward of 35°. Tracer is therefore removed
  somewhat early: the ~800 d e-folding is a LOWER bound on the true stratospheric
  lifetime, and the figure legends say "left the 143 hPa domain", not "removed".
* The descent rate is fitted from the ANNUAL-CYCLE PHASE LAG. Two other
  conventions were tried and both fail — per-level day-of-maximum hops between
  winters (+0.49 / −0.49 / −2.49 mm/s for N / S / combined), and first-arrival
  after a fixed cutoff measures the cutoff itself and comes out with the wrong
  sign. `pulse_deep_branch.py` documents both; do not reinstate them.
* `pulse_deep_branch.py` normalises each level by its own maximum. That is what
  makes the ascent legible, but it also means colour is a RATIO: the late fade at
  the bottom of the panels is the denominator, not drainage — the late e-folding
  is 2.13–2.26 yr at every level in both caps.

## Validation harnesses

Two kinds, and the difference matters — `docs/VALIDATION.md` has the full
inventory. **Automated tests** are self-contained (fixed seed, no GPU, no CESM,
no sibling repos), exit non-zero on failure, and run in CI on every push:

```bash
# advection conservation (the LR benchmark)
python3 $REPO/validation/test_conservation.py

# closed-form settling physics: Tang density, wet growth factor, fall speed,
# and that the implicit sweep changes the burden only by its bottom outflow
python3 $REPO/validation/test_physics_math.py
```

**Investigations** load a real run and print diagnostics for a human to read;
there is no pass/fail. Run them from the directory holding the output, since
that is where they look for a checkpoint and where they write any figure:

```bash
# where the number floor comes from, per bin / level / latitude
STATE=path/to/coupled_state_<tag>_ckpt.npz python3 $REPO/validation/floor_anatomy.py

# the ADV_VPOS positivity limiter: positivity, conservation, smooth-field inactivity
python3 $REPO/validation/validate_vpos_f32.py
```

**A change to advection confirmed only in float64 is not confirmed.** Two bugs
in the positivity limiter were invisible in f64 and fatal in f32: a `1e-300`
guard that underflows to zero below float32's smallest normal (~1.2e-38),
producing inf/NaN; and a reformulation that cost 2.9e-5 relative accuracy on
smooth fields where f64 showed only 1.2e-13.

In practice you set nothing: `validate_vpos_f32.py` already defaults to
`ADV_CFL=0.5` / `ADV_F32=1`, and `coupling.py` and `driver_fast.py` pin the same
values explicitly, so the `ADV_F32=1 ADV_CFL=0.5` prefix these commands once
carried is a no-op. The rule binds in exactly one place: calling
`fct_lr.advect_hour_batch` **directly**, e.g. from a new test. Its signature
defaults (`cfl=0.2`, `float64`) are a configuration no run uses, so a bare call
passes for reasons that do not transfer. See `docs/VALIDATION.md`.

**Run the harnesses by absolute path from wherever the output is, not from the
repo.** Python puts the *script's* directory on `sys.path`, never the cwd, so
each one inserts the repo root and `fast_advection/` itself and works from any
directory. What they take from the cwd is the checkpoint they read
(`validate_vpos_f32.py`, `floor_anatomy.py`; override with `STATE=<path>`) and
where `validate_radiation.py` drops its figure.

## Code state — why the defaults are what they are

- **`BC_GAS` defaults to `flux`** whenever the aerosol faces are open, derived
  from the resolved `BC_EDGE` so the gas and aerosol boundaries cannot desync — not
  even under an explicit `BC_EDGE=clamp`. An unconditional `clamp` default put the
  production config in the one incoherent corner of the four combinations: an
  unbounded Dirichlet gas source at a level whose particles are free to leave.
  Measured cost at 13.3 hPa over 24 h — that single level went from 0.3% to ~50% of
  the model's **total** number as a 6–8 nm mode, continuous nucleation fed by
  clamped H2SO4 rather than transport. What `flux` costs: the gases are no longer
  pinned to CESM. `BC_GAS=clamp` still works and is required to reproduce runs made
  under the old default.
- **`WET_SETTLING` / `WET_OPTICS` default ON.** TOMAS carries dry SO4 mass, but a
  stratospheric sulfate particle is an H2SO4/H2O solution droplet, and both the
  settling velocity (∝ Dp, ρ_p) and the optics need the wet size. Sizing the Mie
  tables on the dry core while using a 75 wt% *solution* refractive index — a
  droplet already 25% water by mass, sized as if dry — underestimates 550 nm
  extinction by ~47% for this size distribution; with the real RH/T-dependent
  composition (40–55 wt% near the moist 143 hPa band base) the error reaches a
  factor of 2+. Composition and density come from the same parameterizations the
  microphysics engine applies internally (Tabazadeh 1997, Tang 1997), so settling,
  optics and coagulation see one droplet. `=0` restores the dry sizing.
- `INIT_BIN` defaults to `so4`. The old `dgnum` path binned MAM4 *number* by
  `dgnumwet` and set `mas = num*MMID` without reading `so4_a*`, inflating sulfate
  mass **4.29×** (coarse mode 6.68×).
- `INJ_ZONAL` defaults to **1** (zonal ring). The point source is 5.6× slower and
  drives runaway nucleation. `INJ_MIRROR=1` releases at both ±`INJ_LAT`, splitting
  the same total 50/50 — the total is what you asked for, not doubled.
- Diagnostic core window is **symmetric**, `-1` per end → 1.6–121.5 hPa, 22/24
  levels, excluding exactly the two levels the reservoir is written into.
- **`ADV_VPOS`** vertical positivity limiter in `fct_lr.py`, default **ON** and set
  explicitly by `run_prod.sh` so the log records it. `ADV_VPOS=0` is a forensic
  escape hatch, not a supported configuration. Lin-Rood is exactly conservative but
  not positive; its vertical remap undershot on the steep ultrafine gradient at the
  injection ring and the clip turned that into a spurious number source. 100% of
  the negatives came from that one operator — the horizontal sweeps produce exactly
  zero, being Zalesak-bounded. Measured: the floor injected ~3.3e-3 of the standing
  number burden per 6 h step and accumulated to 35% of a day-90 standing N; 97.4%
  of it landed below 10 nm and 0.0025% in the optically active 150–1200 nm bins, so
  mass/AOD/ARF were always safe but total number and anything under ~10 nm were
  not. Over 8 steps of pure advection the unlimited scheme grew the number burden
  0.94% out of nothing; limited it sits at 0.99987.
- `vface` diagnostic labels are `top`/`bot`. **They are FACE labels, signed
  `+ = into slab` — not gross inflow/outflow**, and the per-column/per-substep
  cancellation means gross inflow is not recoverable from them.

## Reading the output — standing caveats

- `init N burden(sum num)` and `M burden(sum mas)` in the header are **unweighted
  sums**. Every ratio in the log normalises by the dp·cos(lat)-**weighted** `N0`
  and `M0`, which are ~260× and ~585× larger. Never divide a logged quantity by a
  header value.
- First ~week of any cold start is spin-up; 90 days is a **transient**, not a
  steady state (the sink is not burden-proportional, loss ~ M^0.35).
- `RAD_MODE=anomaly` baseline is **time-varying** under `AER_SRC=mam4`, so reported
  forcing is a difference against a moving target.
- No wet removal anywhere; settling and transport out of the band are the only
  sinks, so aerosol settling into the 100–150 hPa layer lingers.
- `AOD550` is stratospheric-only and sulfate-only over the band. It is
  **uncalibrated** — fine for relative work, and to be settled before quoting
  absolute forcing.
- The budget printout closes to roundoff (`sum` vs `M/M0-1`). If a change breaks
  that closure, the change is wrong — that line is the model's own audit.
