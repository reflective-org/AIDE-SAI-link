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
| aerosol IC/BC source | `AER_SRC=mam4\|carma` | real, both paths run |
| microphysics engine | `driver_fast.py` monkeypatches `tomas_jax.fast` into `coupling.py` at import instead of branching inside it | real, and why the driver is a separate file |
| radiation | `RAD=0` drops it and the `jax-rrtmgp` dependency entirely; `RAD_ENGINE=rrtmgp\|emulator` (2026-10-13) picks what solves the evolved call, `RAD_REF_CACHE` serves the archive-only half of the stage from disk | real: both engines run, same call sites |
| meteorology | none — the reader expects the CESM `h1` layout and variable names | a new source is work, not configuration |

## Layout

Source lives under `src/`, grouped by process, but the **module names stay
flat on purpose** — `coupling.py` does bare `import settling` / `radiation`, and
`src/_paths.py` puts each subdirectory on `sys.path` so those imports keep
resolving. `src/` is not a package; do not add `__init__.py` to a subdirectory
whose name collides with a module (`src/settling/` vs `settling.py`) without
converting every import at the same time. See `docs/REPO_LAYOUT.md`.

```
src/_paths.py               import-path setup; REPO_ROOT / MODELS / INPUTS anchors
src/coupling.py             the model: grid, IC/BC, budget, diagnostics, main loop
src/driver_fast.py          THE production entry point. Imports coupling.py and swaps in
                            two pieces before running it: the tomas_jax.fast batched
                            microphysics engine (replacing coupling's per-cell chain) and
                            fct_lr advection. Separate file, not a flag, because the fast
                            engine has its own API and is hard-fixed at 40 bins.
                            MICRO_ENGINE=fast|emulator picks which batched engine fills
                            that seam (2026-10-09); MICRO_REF=<other> runs both and logs
                            the per-step comparison (the run advances with MICRO_ENGINE).
src/advection/fct_core.py   legacy sealed-face advection. NOT on the run path --
                            coupling.py imports fct_lr directly, so standalone and
                            production use the same transport. Kept solely for the
                            bit-identical legacy check in scripts/validation/test_conservation.py.
src/settling/settling.py    gravitational settling; also the canonical wet-droplet
                            sizing (tang_density/wet_size), which radiation.py imports
src/radiation/radiation.py  RRTMGP + Mie optics (RAD_ENGINE=rrtmgp, the default)
src/radiation/rad_cache.py  per-archive-hour cache of the reference-aerosol solve and the
                            clear-sky fields (RAD_REF_CACHE; bit-identical on a hit).
                            rad_cache_build.py fills entries; scripts/radiation/build_ref_cache.py
                            batch-builds them
src/radiation/rad_emulator.py  RAD_ENGINE=emulator: two column networks (Planck response to
                            dT_rad, aerosol heating) on the cached clear sky, replacing the
                            two-stream solve of the EVOLVED call; Mie optics, AOD, archive
                            reads unchanged. rad_features.py is the one feature definition
                            shared with the label writer. docs/RADIATION.md
src/advection/ffsl_lr.py    the default ADV_SCHEME (=ffsl) since 2026-10-15, NO LONGER
                            RECOMMENDED (2026-11-05): fct_lr at large Courant number --
                            Lin & Rood's flux-form semi-Lagrangian extension of both
                            horizontal sweeps (whole upwind cells + a PPM sliver),
                            ADV_FFSL_K sub-steps, fct_lr's vertical remap, faces, budgets
                            and limiter imported. Its sliver measure was wrong at the cap
                            edge until 2026-11-05 (ADV_FFSL_FRAC=cell; now `exact`, and
                            `face` is fct_lr bit for bit at |c| < 1): it did not converge
                            to lr, the one-year gap was -4.6 % mass / +5 % forcing, and
                            against CESM's own CH4 it was 4.6x lr's error (fixed: 3.0x at
                            K=24, 1.2x at K=96 where it costs as much as lr).
                            docs/PROCESSES.md 1.9-1.10
src/advection/fct_lr.py     Lin-Rood flux-form advection at CFL sub-steps (ADV_SCHEME=lr: the
                            production scheme until 2026-10-15, byte for byte unchanged, and
                            the reference the budgets are measured against)
src/advection/fct_fast.py   PPM/Zalesak primitives that fct_lr imports
src/advection/sl_lr.py      semi-Lagrangian advection, ADV_SCHEME=sl (2026-10-13), THE
                            RECOMMENDED TRANSPORT since 2026-11-05: the SPLIT operator --
                            SL horizontal remap per level (K sub-steps, per-level mass
                            fixer, Jacobian air mass + remainder sweep) + fct_lr's own
                            flux-form vertical sweep on open faces (ADV_SL_VERT=flux);
                            lr's boundary physics, B_vf and residual unchanged. advect
                            0.5 s/step vs 1.5 (ffsl) / 7.0 (lr). Against CESM's own
                            transported CH4 over a year it is the MOST ACCURATE scheme in
                            the tree: 0.87x lr's RMSE at K=6, 0.75x at K=12 (lr itself is
                            biased high at 40-75 hPa). docs/PROCESSES.md 1.8, 1.10
src/advection/sl_core.py    the SL kernels: dinosaur's semi_lagrangian module (Google
                            NeuralGCM, Apache 2.0) vendored without its dependencies and
                            patched to accept rows AT the poles (the CESM grid)
src/microphysics/tomas_fast.py  the tomas_jax.fast adapter (coupling state <-> FastState)
src/microphysics/tomas_emulator.py  the learned-6h-operator adapter (coupling state <-> the
                            bin-token transformer's log-state); MICRO_ENGINE=emulator. docs/EMULATOR.md
src/microphysics/micro_bench.py  in-loop engine A/B: MICRO_REF=<engine> runs it on the same
                            input, compares, writes micro_bench_<TAG>.npz; the run is unchanged
src/microphysics/micro_sampler.py  MICRO_SAMPLE_EVERY=N: stratified samples of the engine's input
                            (training data for the emulator; scripts/emulator/ labels and trains)
scripts/emulator/           label_chains.py (sampled cells -> 48 h physics chains) and
                            finetune_bintf.py (fine-tune the emulator on them). Refuted as a
                            lever on 2026-10-13 (docs/EMULATOR.md); kept for the record
inputs/rad_data/            palmer_williams_h2so4.dat (Palmer & Williams 1975,
                            as distributed in HITRAN Aerosols-2016)
src/run_prod.py             the production launcher (self-documenting header)
scripts/utils/plot_run.py   the four post-run figures (dashboard, filmstrip, size dist, zonal)
scripts/utils/gif_run.py    animated versions of the filmstrip panels
docs/                       CONFIGURATION (every env var), VALIDATION (the harnesses),
                            PROCESSES, COUPLING_VARIABLES, BOUNDARY_CONDITIONS, README
scripts/validation/         harnesses -- see below
.github/workflows/ci.yml    CI: the two self-contained tests + a ruff errors-only gate
models/tomas-jax            SUBMODULE, pinned. Sectional microphysics (branch gpu-fast)
models/jax-rrtmgp           SUBMODULE, pinned. Radiative transfer (needs patches/)
inputs/                     static input data + provenance of the external CESM archive
runs/                       raw model output, one directory per OUT_TAG. Gitignored
outputs/                    derived figures and tables. Gitignored
```

**Analysis scripts are flat and read `coupled_*_<TAG>.npz` from the current working
directory**, so run them from wherever the outputs are. Outputs land in whatever
directory you launch from, and `.gitignore` excludes `*.npz/*.png/*.gif/*.log`
— which hides a misplaced run rather than preventing it, so `run_prod.py`
refuses to start with the repo as `$PWD`. Keep a runs directory outside the tree.

## Setup from a fresh clone

```bash
git clone https://github.com/reflective-org/AIDE-SAI-link.git
git clone -b gpu-fast https://github.com/reflective-org/tomas-jax     # beside it -- found automatically
git clone https://github.com/climate-analytics-lab/jax-rrtmgp         # ditto
git -C jax-rrtmgp apply ../AIDE-SAI-link/patches/jax-rrtmgp-zenith.patch  # required for RAD=1
pip install -r AIDE-SAI-link/requirements.txt
pip install --upgrade "jax[cuda12]>=0.6.2"     # GPU wheel; the CPU one is unusable for production
mkdir -p runs && cd runs                                     # outputs go OUTSIDE the repo
N_HOURS=6 OUT_TAG=smoke ../AIDE-SAI-link/src/run_prod.py         # smoke test first
```

**Neither submodule is usable as a plain checkout of its default branch**, and
both failures are at import or in the first radiation call, not subtle:

| repo | required state | why | symptom if skipped |
|---|---|---|---|
| `models/tomas-jax` | pinned `787c991`, branch `gpu-fast` | `tomas_jax.fast`, the batched reduced engine `driver_fast.py` monkeypatches in, exists only there — `main` has the per-cell chain only | `ModuleNotFoundError: tomas_jax.fast` at `driver_fast.py:111` |
| `models/jax-rrtmgp` | pinned `99d2d71` + `patches/jax-rrtmgp-zenith.patch` | `radiation.py` passes zenith per column, `(nlat, nlon, 1)`; upstream `sw_cell_source` assumes a scalar and broadcasts it against the `(nlat, nlon)` TOA flux | shape blow-up in the SW solve on the first `RAD=1` step |

The patch is one hunk against the pinned commit `99d2d71` and is the only
carried modification to either dependency; keep it that way, and if upstream
takes the fix, delete the patch rather than let the two drift. `git apply`
refuses a partial application, so a failure means the submodule was moved
forward and the patch needs regenerating against the new commit.

Because a gitlink records only a commit, the patch CANNOT live as a submodule
working-tree edit -- that is exactly where it was found before the submodule
conversion, uncommitted and reproducible only by whoever had applied it.

| dependency | not shipped because | how it is found |
|---|---|---|
| `tomas-jax` | separate repo (microphysics) | `$TOMAS_JAX_PATH`, else `models/tomas-jax`, else the normal import path |
| `jax-rrtmgp` | separate repo (radiation) | `$RRTMGP_PATH`, same order. `RAD=0` skips radiation and this dependency |
| CESM `h1` hourly meteorology + MAM4 | ~23 TB as archived | **On the shared H100 box, nothing to set** — the `coupling.py` defaults resolve to the FWHIST archive under `/data`, world-readable. Elsewhere: `$CESM_DIR` (+ `CESM_PREFIX`, `CESM_SUF`). Layout: `$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF`. Variables: `docs/COUPLING_VARIABLES.md`. Reads are one hour × band levels at a time (~5.3 MB per variable-hour), so a subset suffices: ~3.6 GB for 2 days, ~160 GB for 90 days |
| CUDA driver library | site-specific | `run_prod.py` puts `$CUDA_DRIVER_LIB` on `LD_LIBRARY_PATH`. Where libcuda.so.1 is not on the loader path JAX **silently falls back to CPU**; set the variable, or empty it for a normal CUDA install |

An env var that is set but points nowhere is an error, not a silent fallback; a
missing CESM file names the variable that built the path.

Do not activate the tomas-jax `.venv` — it is CPU-only jaxlib with no xarray. The
working combo is **system python3** with GPU jax.

## Running

**`run_prod.py` sets every knob except the SCENARIO.** A bare run is *not* the
production config: `INJ_SO2_TG_YR` defaults to **0 = no injection**, so a forgotten
flag gives an obviously unforced baseline rather than a silent 10 Tg/yr SAI result.
State the amount and give the scenario its own tag.

**Launch from a runs directory, never the repo.** Every output path is relative
to the working directory, so that directory is where the run lands.
`run_prod.py` **refuses to start** with the repo as `$PWD` — the mistake is
otherwise invisible, since `.npz`/`.png` are gitignored and a misplaced run looks
entirely normal while filling the tree (29 GB accumulated that way before
2026-08-12). The script still execs `driver_fast.py` from its own tree, so it
runs the code you edited regardless of where you launched it.

```bash
cd <runs dir>                                     # NOT the repo; outputs land here
REPO=/path/to/AIDE-SAI-link
INJ_SO2_TG_YR=10 OUT_TAG=prod90d $REPO/src/run_prod.py   # the 10 Tg/yr scenario, 90 days, ~3.5 h
$REPO/src/run_prod.py                                    # NO-INJECTION control (INJ_SO2_TG_YR=0)
RESUME=1 INJ_SO2_TG_YR=10 OUT_TAG=prod90d $REPO/src/run_prod.py   # continue that scenario
```

Two checkouts launched from the *same* runs directory with the same `OUT_TAG`
overwrite each other. `OUT_TAG` discipline is what keeps runs apart.

**Narrowing a run to one process.** Each coupling step walks a fixed stage list
(`docs/PROCESSES.md` §0), and each stage has an env flag that switches it off, so
a run can be reduced to the process being measured. Transport only:

```bash
MICRO=off RAD=0 SETTLE=0 STATE_CKPT=0 FRAME_EVERY=48 N_HOURS=48 \
  OUT_TAG=speed-benchmark $REPO/src/run_prod.py          # 8 steps, ~100 s
```

`MICRO=off` (2026-08-27) disables microphysics outright — a benchmark/diagnostic
mode, **not** a physics configuration: the aerosol is advected but never evolves.
It goes through the production step loop rather than a separate harness, so what
is measured is the advection the model actually runs. `MICRO` is validated at
import (`full|coag|off`); it previously fell through to the legacy `coag` path on
a typo. `driver_fast.py` accepts `full` and `off` only — pairing the fast engine
with the pre-SAI `coag` model would silently be neither.

Set `FRAME_EVERY` to the run **length**, not 0: `FRAME_EVERY_STEPS` is
`max(1, FRAME_EVERY/STEP_HOURS)`, so 0 means a frame every step. `STATE_CKPT=0`
suppresses the restart cycle; `coupled_final_<TAG>.npz` is still written
unconditionally and is the bulk of the output.

Measured on one H100 at the production grid (192×288, 24 levels, 1,327,104 cells)
and production transport settings (`fct_lr`, `ADV_F32=1`, `ADV_CFL=0.5`,
`ADV_VPOS=1`): **7.2 s/step of advection** in steady state (10.7 s on step 0,
which carries JIT), at ~168 CFL substeps → **43 ms/substep**, plus 1.5 s of
`bc+polar`. That is ~43 min of advection over a 360-step 90-day run, i.e. ~2% of
a full step. Full-physics cost, measured over a complete 1-year run
(`runs/prod1yr/`, 2026-08-28): **34.9 s/step** — micro 16.5 s (47%), radiation
9.4 s (27%), advection 7.4 s (21%), rest 1.7 s — and **flat over the year**
(quarter means 35.8/34.1/35.5/34.2 s, no stage drifting >15%). Burden growth does
not make steps more expensive. With the radiation reference cache on (default since 2026-10-13) the
radiation stage is one solve, 3.1 s/step on a quiet GPU; with
`RAD_ENGINE=emulator` (2026-10-13): the radiation stage is 0.7 s/step (30-day
A/B `runs/bench_rademu30d_v2phys`, MEASURED; 3.8 s on the first pass through a
set of archive hours in `RAD_EMU_ANOM=net`, while the per-hour network reference
is still being stored) -- `docs/RADIATION.md`. The 340–370 s/step and "microphysics ~94%" figures
that this repo carried until 2026-08-28 described the pre-2026-08-13 tomas-jax
engine and are superseded. Per-step cost tracks `nsub`, so quote the per-substep figure. As a check on
the mode itself, `N/N0` finished at 0.9998 over 8 steps of pure advection, matching
the 0.99987 recorded for the `ADV_VPOS` limiter in `run_prod.py`.

`run_prod.py` pins a **single** GPU (`CUDA_VISIBLE_DEVICES=${GPU:-0}`) rather than
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

## Validation harnesses

Two kinds, and the difference matters — `docs/VALIDATION.md` has the full
inventory. **Automated tests** are self-contained (fixed seed, no GPU, no CESM,
no submodules), exit non-zero on failure, and run in CI on every push:

```bash
# advection conservation (the LR benchmark)
python3 $REPO/scripts/validation/test_conservation.py

# semi-Lagrangian transport (ADV_SCHEME=sl): pole halo, constant field,
# Williamson case 1 over the poles, mass-fixer closure, bin consistency, driver
python3 $REPO/scripts/validation/test_sl_transport.py
# which horizontal operator is nearer the truth (analytic cases, GPU, ~1 min):
# SL K=6 beats lr 2-12x in L2; more SL sub-steps = more smoothing (K=24 worse)
CUDA_VISIBLE_DEVICES=4 python3 $REPO/scripts/validation/compare_hz_operators.py
# the same on a closed-column 3-D DIVERGENT reversing flow (exact = initial):
# the split (Strang K=6) is at or ahead of lr, so the coupled-run gap is the
# open bottom face, not the operators (KDEF=2|5 sets the deformation strength)
CUDA_VISIBLE_DEVICES=4 python3 $REPO/scripts/validation/compare_3d_operators.py

# lr at large Courant number (ADV_SCHEME=ffsl): sweep == fct_lr at |c| < 1,
# conservation/uniform field at |c| up to 8, exact whole-cell shifts, the 3-D
# driver's budget closure, Williamson 1 at hourly sub-steps vs lr (CPU, ~100 s)
python3 $REPO/scripts/validation/test_ffsl_transport.py

# closed-form settling physics: Tang density, wet growth factor, fall speed,
# and that the implicit sweep changes the burden only by its bottom outflow
python3 $REPO/scripts/validation/test_physics_math.py
```

**Investigations** load a real run and print diagnostics for a human to read;
there is no pass/fail. Run them from the directory holding the output, since
that is where they look for a checkpoint and where they write any figure:

```bash
# where the number floor comes from, per bin / level / latitude
STATE=path/to/coupled_state_<tag>_ckpt.npz python3 $REPO/scripts/validation/floor_anatomy.py

# the ADV_VPOS positivity limiter: positivity, conservation, smooth-field inactivity
python3 $REPO/scripts/validation/validate_vpos_f32.py

# any runs' burdens, r_eff, AOD and TOA forcing against a reference run (how the
# microphysics emulator is benchmarked over a month; docs/EMULATOR.md)
python3 $REPO/scripts/validation/compare_runs.py <REF_TAG> <TAG>... --out <name>

# the one-page figure of the production emulator vs the physics over 30 days
# (outputs/emulator_bench/emulator_summary.png)
python3 $REPO/scripts/validation/plot_emulator_summary.py

# cost of dinosaur's semi-Lagrangian transport at this model's size, GPU (the
# offline number behind ADV_SCHEME=sl) -- docs/TRANSPORT_SL_HANDOFF.md
PYTHONPATH=/data/dinosaur-sl python3 $REPO/scripts/validation/bench_sl_transport.py

# where one radiation call spends its time (solve vs optics vs prep), GPU + CESM
# archive; docs/RADIATION_EMULATOR_SCOPE.md
CUDA_VISIBLE_DEVICES=<free> python3 $REPO/scripts/validation/bench_radiation_split.py

# radiation reference cache: a served hour is bit-identical to a computed one, the
# SW/LW-split solve equals the default one; GPU + archive (docs/RADIATION.md)
CUDA_VISIBLE_DEVICES=<free> python3 $REPO/scripts/radiation/test_rad_cache.py
# radiation emulator adapter: contract, structural zeros, both anomaly modes, timing
# and (with real checkpoints) the offline error vs the physics on the day-30 state
CUDA_VISIBLE_DEVICES=<free> P_LO_HPA=1 P_HI_HPA=150 INIT_BIN=so4 python3 $REPO/scripts/radiation/test_rad_emulator.py
# why the Planck term is a network and not a Jacobian (WENO5 face temperatures)
CUDA_VISIBLE_DEVICES=<free> P_LO_HPA=1 P_HI_HPA=150 INIT_BIN=so4 python3 $REPO/scripts/radiation/check_planck_jacobian.py
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
`fct_lr.advect_step_batch` **directly**, e.g. from a new test. Its signature
defaults (`cfl=0.2`, `float64`) are a configuration no run uses, so a bare call
passes for reasons that do not transfer. `dt_total` has no default at all — it
is the coupling step (`STEP_SEC`), and the 1 h default it used to carry made a
caller who omitted it advect a sixth of the interval, silently. See `docs/VALIDATION.md`.

**Run the harnesses by absolute path from wherever the output is, not from the
repo.** Python puts the *script's* directory on `sys.path`, never the cwd, so
each one inserts the repo root and `src/advection/` itself and works from any
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
  explicitly by `run_prod.py` so the log records it. `ADV_VPOS=0` is a forensic
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
- Under `RAD_ENGINE=emulator`, the reported TOA forcing is only trustworthy with
  `RAD_EMU_ARF=exact` (the default when `RAD_REF=rrtmgp`): the networks' ±1 %
  bias on the aerosol's TOA effect is 10–30 % of the forcing early in a run. The
  prognostic `dT_rad` is the emulator's; its 30-day error against the physics is
  in `docs/RADIATION.md`. The networks' outputs are clamped (`RAD_EMU_CLIP`,
  20 K/day, logged when it triggers): a residual MLP can blow up on an input
  far outside its training set, and one did (674 K/day, NaN two steps later).
- No wet removal anywhere; settling and transport out of the band are the only
  sinks, so aerosol settling into the 100–150 hPa layer lingers.
- `AOD550` is stratospheric-only and sulfate-only over the band. It is
  **uncalibrated** — fine for relative work, and to be settled before quoting
  absolute forcing.
- The budget printout closes to roundoff (`sum` vs `M/M0-1`). If a change breaks
  that closure, the change is wrong — that line is the model's own audit.
- `ADV_SCHEME=sl` is **opt-in** (2026-10-13). Its default form is the split
  operator (`ADV_SL_VERT=flux`: SL horizontal + `fct_lr`'s flux-form vertical on
  open faces), advect 0.38 s vs 6.6 s, boundary physics and diagnostics as under
  `lr`. 30-day A/B against `fct_lr` (docs/PROCESSES.md §1.8): burdens within
  ~1 % to day 10; at day 30 mass −2 %, AOD −1.6 %, TOA forcing +7 %, from a
  20 % larger bottom-face export (a physical difference between the two
  horizontal operators, budgets close to roundoff). Superseded: its one-year run
  drifts a few %/year (mass −2.2 % at day 138, growing), and `ADV_SCHEME=ffsl`
  K = 24 -- `fct_lr` itself at large Courant number, within 0.2 % of `lr` on the
  30-day budgets at 1.5 s/step -- became the production default on 2026-10-15
  (docs/PROCESSES.md §1.9). The
  handoff's closed-column design (`ADV_SL_VERT=sl`) FAILED its A/B — at day 30
  mass/SO2/AOD 9–15 % low, entirely from the Dirichlet edge-level reset being
  ~3× the flux-face exchange — and is kept only for the record.
