Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[Unreleased]

Added

- **CH4 truth test for transport, `scripts/validation/ch4_truth_test.py`**
  (2026-11-05, `docs/PROCESSES.md` 1.10). Advects CESM's own hourly CH4 field
  (inert below ~30 hPa on a one-year scale) with one of this model's schemes
  on the same winds, exactly as the coupler calls it, and scores it against
  CESM's CH4 every step; persistence is the baseline. The first transport test
  whose answer does not come from `fct_lr`. Result over 1996: `sl` K=6/12 is
  0.87x/0.75x `lr`'s error and the cheapest scheme; `ffsl` K=24 (the default
  since 2026-10-15) was 4.6x `lr`'s error, 3.0x after the fix below, and only
  matches `lr` at K=96 where it costs the same. `ADV_SCHEME=sl` is the
  recommended transport; `lr` is biased high by ~3 % at 40-75 hPa against CESM,
  so its lower bottom export was `lr`'s error, not `sl`'s.
- `ADV_FFSL_FRAC = cell | face | exact` in `src/advection/ffsl_lr.py`
  (2026-11-05): how the fractional FFSL flux is measured on a latitude row.
  `cell` (the old `f*a_j`) is 4.5 % off at the 80 deg cap edge and never
  converges to `lr` (+70 % polar-cap delivery at every K, the cause of the
  -4.6 % one-year mass gap that the never-written-up `runs/ffsl1yr_K24`
  acceptance test showed); `face` is `fct_lr`'s `c*m_face`, bit-identical at
  |c| < 1; `exact` (default) is the true spherical sliver area. With it ffsl
  converges to `lr` in K (export -0.1 %, polar +0.2 % at K=192).

- **Radiation reference cache and emulator, `RAD_REF_CACHE` / `RAD_ENGINE=emulator`**
  (2026-10-13, `docs/RADIATION.md`, scoped in `docs/RADIATION_EMULATOR_SCOPE.md`).
  `src/radiation/rad_cache.py` stores, per archive hour, the reference-aerosol
  solve of `RAD_MODE=anomaly` (float64: a hit is bit-identical to computing it,
  `scripts/radiation/test_rad_cache.py`) and the clear-sky fields the emulator
  builds on; on a hit the loop skips the reference solve AND its MAM4 re-binning.
  Default ON (`inputs/cache/rad_ref`, gitignored; `0` disables; batch builder
  `scripts/radiation/build_ref_cache.py`). `RAD_ENGINE=emulator`
  (`src/radiation/rad_emulator.py`) replaces the two-stream solve of the EVOLVED
  call with two column networks on the cached clear sky -- a Planck network for
  the response to the prognostic `dT_rad` and an aerosol network whose output is a
  per-band efficiency contracted with the exact Mie optical depths (exactly zero
  without aerosol) -- leaving the Mie optics, the AOD and the archive reads as
  they are. `RAD_REF=rrtmgp` A/Bs it against the physics on the same input
  (`rad_bench_<TAG>.npz`), `RAD_SAMPLE_EVERY` writes training samples,
  `scripts/radiation/{label_columns,train_rad_mlp,test_rad_emulator}.py` make
  and gate the networks (`rad_features.py` is the one feature definition).
  Finding on the way: the solver's response to `dT_rad` is NOT linearisable --
  exactly linear for a uniform warming, non-linear at the 0.5 K scale for a
  sharp profile, because RRTMGP's face temperatures are a WENO5 reconstruction
  (`scripts/radiation/check_planck_jacobian.py`: a full 24x24 FD Jacobian leaves
  35 % rms); hence a network for that term, not a cached Jacobian. `PHYS_CFG`
  gains `RAD_ENGINE=emulator` (append-only). `RAD_ENGINE=rrtmgp` with the cache
  off is byte-for-byte the previous code path.
  **Measured:** radiation stage 8.4 → 0.7 s/step; 30-day A/B
  (`runs/bench_rademu30d_v2phys` vs the cached physics `bench_phys30d_rc`):
  `dT_rad` error 0.0018 K rms / 0.006 K p99 / 0.010 K zonal-mean max against a
  0.8 K field after 30 days (v3), microphysics identical; one column reaches
  0.08 K through a Planck-network feedback on its own `dT` shape (0.64 K before
  the noisy-profile retraining);
  the TOA forcing is reported from the physics samples (`RAD_EMU_ARF=exact`)
  because the networks' ±1 % TOA bias is 10–30 % of the forcing early in a run.
- **Radiation emulator v5 and an output clamp** (2026-10-14, `docs/RADIATION.md`
  "v4/v5"). Labels from trajectory states of both regimes the emulator runs
  in — the A/B scenario sampled through its month (`runs/raddata_30NS_jan`)
  and the year-long run resumed at day 177 with sampling
  (`runs/lr1yr_d177_sample`: AOD 0.07, `dT_rad` to 4.6 K) — plus a big-plume
  held-out set (`labels/val_yr_d207`). Trainer options `--x-cpu` (inputs in
  pinned host memory, bit-identical; the 8 M-column set no longer fits the
  GPU), `--max-aod` (the ×10-scaled year columns above AOD 1 blew the batch
  loss up to 10¹²) and `--lr` lowered to 3e-4 for both nets. Finding: a net
  whose training loss spiked is damaged even when its held-out score is good
  — the first aerosol net (`lr 1e-3`) passed every 30-day gate and then
  returned 674 K/day in one dry-upper-troposphere column of the day-207
  state, NaN two steps later. Hence `RAD_EMU_CLIP` (`--rad-emu-clip`, default
  20 K/day): the networks' heating outputs are clamped, NaN → 0, and every
  touched cell is counted and logged; a no-op on a healthy run. v5
  (`inputs/rad_emulator/v5`, held-out err/signal): aerosol SW 3.3 % / LW 1.4 %
  on the day-30 state (v3 7.3 / 3.1), 1.9 % / 1.1 % on the day-207 state
  (v3 20 / 5.1), Planck LW 7.0 % (v3 8.9) and 3.4 % (v3 8.5); global TOA SW
  bias +0.0013 W m⁻² (v3 +0.0074). In the loop: 30-day A/B `dT_rad` error
  0.00086 K rms / 0.012 K max (v3 0.0018 / 0.079), heating-anomaly error
  7.9 % (v3 15 %), stage 0.67 s; resumed from day 207 of the year-long run
  (AOD 0.07, `dT_rad` to 5 K) the anomaly error is a flat 5.6 % where v3's
  grew from 21 to 29 %, and `dT_rad` tracks the physics to 0.1 % (v3 2 %).
  `current -> v5`.
- **Semi-Lagrangian transport, `ADV_SCHEME=sl` (`--adv-scheme sl`)** (2026-10-13),
  opt-in. `src/advection/sl_lr.py` drives the kernels in `src/advection/sl_core.py`
  — dinosaur's `semi_lagrangian` module (Google NeuralGCM, Apache 2.0) vendored
  without its dependencies and patched to accept the CESM grid's rows at the
  poles. Same call signature as `fct_lr.advect_step_batch`; no CFL limit:
  `ADV_SL_K` (6) fixed sub-steps, each on its own hourly CESM wind (the five
  snapshots between the step ends are now read under `sl`), 3-D departure points
  in Cartesian coordinates with CESM omega as the vertical velocity in pressure,
  cubic × linear gather with the quasi-monotone limiter, and a per-tracer mass
  fixer (`ADV_SL_FIXER=global|local|none`) on the closed extended column that
  conserves to float64 roundoff. `coupling.py` defaults `ADV_WCONT=0` under `sl`
  (Dirichlet edge levels; the exchange is in `B_bc`, `B_vf_*` off) and stamps
  `ADV_SCHEME=sl` into `PHYS_CFG` (append-only). New CPU gate
  `scripts/validation/test_sl_transport.py` (pole halo, constant field,
  Williamson case 1 over the poles at L2 2.5 %, fixer closure ≤ 1e-15, bin
  consistency, driver contract); `test_conservation.py` untouched. Design,
  measurements and the A/B plan: `docs/TRANSPORT_SL_HANDOFF.md`;
  `docs/PROCESSES.md` §1.8. `--micro-engine fast` with the default `lr` is
  byte-for-byte unchanged. The handoff's closed-column 3-D design (now
  `ADV_SL_VERT=sl`) ran 6.6 → 0.35 s/step but FAILED its 30-day A/B
  (mass/SO2/AOD 9–15 % low at day 30: the Dirichlet edge-level reset is ~3×
  the flux-face exchange). **The default is therefore the split operator,
  `ADV_SL_VERT=flux`** (same day): SL horizontal remap per level with an exact
  per-level fixer + `fct_lr._lr_vert`'s flux-form vertical sweep on the open
  faces in advective form with the column fixed to the flux-form target, caps
  stirred (`ADV_POLAR=zonal`), `lr`'s continuity omega (`ADV_SL_OMEGA`). `lr`'s
  boundary physics, `B_vf` and the advection residual (roundoff) are unchanged;
  advect 0.38 s/step; day-1 burdens within 0.1 % of `lr`; 30-day A/B: within
  ~1 % to day 10, at day 30 mass −2 %, AOD −1.6 %, forcing +7 % (20 % more
  bottom-face export; PROCESSES.md §1.8). Stays opt-in.
- **`ADV_SCHEME=ffsl`: `lr` at large Courant number** (2026-10-14; the
  production default since 2026-10-15, see Changed).
  `src/advection/ffsl_lr.py` is `fct_lr` with Lin & Rood's flux-form
  semi-Lagrangian extension in both horizontal sweeps (whole upwind cells +
  a PPM sliver, exact flux form at any Courant number), so the CFL sub-stepping
  that cost 7 s/step goes: `ADV_FFSL_K` sub-steps (6 = hourly; default 24
  since 2026-10-15), Strang
  ordering of the vertical remap (`ADV_FFSL_SPLIT`). Everything else is
  `fct_lr`'s, imported; `fct_lr` is unchanged. Gate
  `scripts/validation/test_ffsl_transport.py`; both truth tests gained an
  `ffsl` entry. 10-day A/B vs `lr`: K = 24 mass −0.05 %, export +2.5 % at
  1.6 s/step (lr 7.0 s); K = 6 (0.6 s) carries an hourly x/y splitting error
  (export +22 %) that Lin & Rood's inner/outer cross terms would remove (not
  implemented). The one-year semi-Lagrangian pair meanwhile showed the split
  converging TO `lr` at the face, which is why `lr` is the reference for the
  budget (PROCESSES.md §1.8–1.9). 30-day A/B on `lr`'s own six-hourly
  bracketing winds (`ADV_SL_HOURLY=0`, the production wind cadence): K = 24
  is within 2 % of `lr` on every quantity (mass −0.16 %, export +0.8 %, AOD
  −0.3 %, number −0.4 %, TOA +2.0 %) at advect 1.5 s + read 0.2 s; the
  number/forcing residuals of the hourly-wind runs were the wind cadence,
  not the operator.
- **`sl`: Strang splitting at the open face, `ADV_SL_SPLIT=strang`, now the
  default under `sl`** (2026-10-14). The one-year A/B of the 2026-10-13 split
  did not saturate (mass −13 %, bottom export +74 % vs `lr` by day 172) and the
  day-30 per-level mass showed a downward redistribution uniform in latitude.
  Cause: Lie splitting puts a whole sub-step's horizontal convergence
  (`dt·div ≈ 0.3` at 143 hPa, 1 h) on the column before the vertical sweep sees
  it; with the air mass frozen at 1 the per-level fixer turned that into an
  O(dt·div) uniform transfer every sub-step, and with it carried
  (`ADV_SL_RHO=1`, new) the open-face flux rode on ρ ≈ 1.3 for the hour
  (export +37 % at K=6, +9 % at K=24: first order). Strang (half vertical
  sweep, horizontal remap with the air mass carried, half sweep) makes the
  face term second order with still one SL gather per sub-step: 30-day A/B
  mass −1.2 %, export +16 %, settling and microphysics within 0.1 %, number
  +2.7 %, forcing +8 %, and the gap's growth is slowing instead of
  accelerating. New closed-column 3-D divergent truth test
  `scripts/validation/compare_3d_operators.py` (Strang K=6 L2 0.0064 vs `lr`
  0.0089): the interior operators are fine, the whole A/B gap is the open
  face. A K = 6/12/24 sweep then showed the Strang run still converging to
  `lr` at first order (export excess 14.7 / 7.9 / 4.5 % at day 10) and found
  two local O(dt²) remainders that global fixers were spreading uniformly
  every sub-step: the Eulerian air mass (now the Jacobian of the departure
  map, `_jacobian`) and the post-sweep air-mass remainder (now returned
  through the column's faces, `_remainder_sweep`), both `ADV_SL_JAC=1`
  (default): day-10 export excess 14.7 → 8.4 %, mass −0.41 → −0.22 %.
  `--adv-sl-rho`, `--adv-sl-split`, `--adv-sl-jac`; `test_sl_transport.py`
  gains the air-mass, Strang and Jacobian checks. `lr` remains the production
  default; the one-year pair (`runs/lr1yr`, `runs/slrem1yr_K6`) is running.
- **A second microphysics engine at the `driver_fast.py` seam** (2026-10-09).
  `MICRO_ENGINE=emulator` (`--micro-engine emulator`) replaces `tomas_jax.fast`
  with the learned 6-hour operator from `/data/tomas-emulator`
  (`EMULATOR_HANDOFF.md`: a bin-token transformer trained on `tomas_jax.fast`
  trajectories), one forward pass per cell per step, through the new adapter
  `src/microphysics/tomas_emulator.py` — same signature, same mixing-ratio ↔
  per-box conversion and two-moment clip as `tomas_fast.py`. The fast physics
  stays the default and is byte-for-byte unchanged. `docs/EMULATOR.md` has the
  interface contract, the input-floor conventions the adapter had to learn
  (night-time OH of exactly zero is 17 σ outside the training set and produced
  spurious 1e8 cm⁻³ bursts), the benchmark and the standing caveats.
- **In-loop engine A/B** — `MICRO_REF=<other engine>` (`--micro-ref`) runs the
  reference engine on the same input every `MICRO_REF_EVERY` steps, compares
  the two cell by cell (`src/microphysics/micro_bench.py`) and writes
  `micro_bench_<TAG>.npz`: one-step error percentiles over all / active / plume
  cells, burden-weighted global error, per-bin and per-level error, zonal
  means and the wall time of BOTH engines on the same cells and GPU. The run
  advances with `MICRO_ENGINE` only; the reference output is discarded. This is
  how the emulator is benchmarked with the fast physics as ground truth.
- `scripts/validation/micro_ab_offline.py` — the same A/B on a `DUMP_PREMICRO`
  state, with the worst cells and their inputs printed (how the OH floor was
  found); `scripts/validation/plot_micro_bench.py` — the figures;
  `scripts/validation/test_emulator_adapter.py` — the adapter reproduces the
  handoff's reference accuracy on the emulator's own cached test cells (GPU).
- `PHYS_CFG` gains a fifth, appended field, `MICRO_ENGINE=emulator`, so a
  `RESUME` that switches engines warns like any other physics-mode change.
- **Emulator stage 7.8 s → 2.8 s** (2026-10-13). The adapter's bookkeeping
  now runs on the GPU — the state arrives as jax device arrays and goes to
  torch zero-copy through DLPack, the mixing-ratio ↔ per-box conversion,
  clip and diagnostics are torch float64, and the answer goes back the same
  way (4.3 s); and QUIET cells (`EMU_TRIAGE=1`, ~30 % of the grid holding
  80 % of the mass, which the physics itself moves by 0.1 % per step) get an
  exact pseudo-first-order SO2 + OH step with surface-area condensation
  instead of the network (2.8 s). The physics stage is 14.4 s, so 5.2× on
  microphysics; the full step goes 30.8 → ~18.5 s. Accuracy is unchanged on
  the network's cells and better on the quiet ones (global mass error
  +0.27 % → +0.06 % per step).
- **Training data from the coupled model's own states.** `MICRO_SAMPLE_EVERY=N`
  (`src/microphysics/micro_sampler.py`) writes a stratified sample of the
  engine's input every N steps; `scripts/emulator/label_chains.py` integrates
  each sampled cell with the production physics as a box model and records
  30-minute chains in the emulator's dataset format;
  `scripts/emulator/finetune_bintf.py` fine-tunes the handoff checkpoint on
  them (plus the original populations, so nothing is unlearned) and
  evaluates on held-out AIDE chains by stratum.
- **Emulator second pass: hybrid routing, the sulfur closure, and the
  rollout benchmark** (2026-10-13). Three adapter knobs, now the defaults for
  `--micro-engine emulator` (`--micro-engine fast` is byte-for-byte
  unchanged):
  `EMU_PHYS_TOP=f` (`--emu-phys-top`) hands the top fraction of cells by
  total sulfur per box — the injection ring, 80 % of all particles — to
  `tomas_jax.fast` through `tomas_fast` with the production knobs, padded to
  a stable jit shape (0.22 s per step at 1 %); `EMU_PHYS_NUM=f`
  (`--emu-phys-num`) does the same for the top fraction of the remaining
  cells by particle number — the dense fine populations the diffuse plume
  becomes, whose coagulation the network under-does by half (1.1 s at 4 %,
  ≈2 s at 8 %); and `EMU_PROJECT=chem` (`--emu-project chem`) replaces the
  network's gas answer with the exact pseudo-first-order SO2 loss and a
  production / condensation-sink H2SO4 using the physics' own
  Fuchs–Sutugin sink evaluated on the network's output, and gives the
  aerosol exactly the sulfur the gas lost (0.3 s). The network keeps number
  and shape, the things it is good at; the sulfur budget closes to the digit
  where the network alone under-oxidised SO2 by half and fabricated the
  mass back. 30-day production scenario against the physics run:
  number burden −1.2 %, mass +0.1 %, SO2 0.0 %, r_eff −1.1 %, AOD 0.0 % at
  day 30 (the network alone: −35 %, −0.3 %, +28 %, +16 %, +10 %).
  Microphysics stage 4.9 s (median of 120 steps) against 14.5 s for the
  physics, 3.0× (the first pass was 7.8 s); the full step is radiation and
  advection bound.
  `scripts/validation/compare_runs.py` is the trajectory benchmark (burdens,
  r_eff, AOD, TOA forcing of any runs against a reference run, table and
  figure). `docs/EMULATOR.md` has the attribution, the sweeps and the two
  negative results: fine-tuning the network on this model's own states
  (`scripts/emulator/finetune_bintf.py`, two checkpoints) improves every
  held-out chain score 3–6× and worsens every 30-day rollout, and a
  per-cell sulfur projection that trusts the network's gas answer is worse
  than none.
- **The figure page and the transport cost measurement** (2026-10-13).
  `scripts/validation/plot_emulator_summary.py` renders
  `outputs/emulator_bench/emulator_summary.png`: the production emulator
  against the fast physics over the 30-day scenario (burdens, AOD,
  Hovmöller and day-30 maps at the probe level, size distributions, the
  one-step error through the month, where the physics is routed, per-stage
  cost, and the accuracy-vs-cost of every configuration tried). The
  whole coupled step is 21.3 s with the emulator against 31.0 s (1.46×);
  advection (6.6 s, CFL-bound at ~200 sub-steps) and radiation (8.5 s) now
  hold two thirds of it. `scripts/validation/bench_sl_transport.py` times
  JCM's transport operator (dinosaur `semi_lagrangian`, JAX) at this
  model's size: 82 tracers, 3-D, cubic + quasi-monotone, 6 sub-steps,
  **49 ms per 6-h step** on an H100, 135× less than `fct_lr`. It is not
  wired in; `docs/TRANSPORT_SL_HANDOFF.md` has the interface it must meet,
  the four blockers (pole nodes, pressure levels, open faces, mass
  conservation) with a fix for each, a conservation design (closed extended
  column + Bermejo–Conde local fixer), the validation gates and the task
  list.
- **Radiation stage measured and a radiation emulator scoped** (2026-10-13).
  `scripts/validation/bench_radiation_split.py`: one `heating()` call is
  2.70 s, of which the RRTMGP two-stream solve is 2.59 s (55,296 columns ×
  55 native levels), the Mie optics 0.03 s; the in-loop 8.5 s is two such
  calls plus ~3 s of binning the MAM4 reference aerosol. Both the reference
  solve and that binning depend only on the archive hour.
  `docs/RADIATION_EMULATOR_SCOPE.md` lays out the ladder: cache the
  reference call (exact, 8.5 → ~2.9 s), then a column-local emulator of the
  evolved solve targeting the aerosol heating perturbation rather than the
  full heating rate (the model consumes an anomaly that is ~10 % of the
  field), with the data plan, gates and the per-step accounting
  (21.3 s today → ~6.7 s with transport, cache and emulator). Nothing in the
  radiation physics was changed.
- A run environment that carries both `jax[cuda12]` and `torch`:
  `/data/aide_emu_venv` (the system python has neither; `tjax_venv` has no
  torch). `run_prod.py` execs whatever python launched it, so
  `/data/aide_emu_venv/bin/python src/run_prod.py ...` is the whole change.

- `models/` — `tomas-jax` (pinned `787c991`, branch `gpu-fast`) and `jax-rrtmgp`
  (pinned `99d2d71`) are now git submodules, so a clone reconstructs the exact
  code behind a result instead of relying on two unpinned sibling clones.
- `inputs/` — small tracked static input data (`rad_data/`) plus a README
  recording the provenance of all three input datasets, including the external
  CESM archive and its `CESM_DIR` / `CESM_PREFIX` / `CESM_SUF` overrides.
- `runs/<TAG>/` for raw model output and `outputs/` for derived products. Both
  gitignored, each with a tracked README.
- `docs/REPO_LAYOUT.md` — the tree, and the rule for where a new file goes.
- `docs/DEFERRED.md` — work deliberately not done, with reasons.
- `src/run_prod.py` — the launcher, converted from `run_prod.sh`. Every one of
  the 83 environment variables the model reads is now a documented `--flag`;
  `--help` is the inventory and `--dry-run` prints the resolved environment with
  the source of each value. Verified to hand `driver_fast.py` a byte-identical
  environment to the shell version it replaces.
- `FRAME_LEVELS` — `probe` (default, unchanged) or `all`, which adds a
  zonal-mean frame at every band level (`(nf, NBINS, nlev, nlat)`, +1.1 GB/year)
  beside the probe-level slabs. Without it the frames have a size axis but no
  vertical axis, so a Hovmöller at any altitude but the probe level, and any
  latitude–height animation, were not constructible from a run's own output.
- `scripts/utils/plot_run.py --level HPA` and a fourth figure,
  `<TAG>_zonal.png`; `scripts/utils/gif_run.py` gains `<TAG>_zonal_so4.gif` and
  `<TAG>_zonal_dTrad.gif` when the run carries zonal frames.
- `scripts/utils/run_summary.py` — parses a run log's `[prof]` lines into
  `<TAG>_summary.md`, quarter by quarter so a cost trend is visible rather than
  averaged away.

Changed

- **`ffsl` K = 24 on the step-snapshot winds is the production default**
  (2026-10-15). `ADV_SCHEME` defaults to `ffsl` (`coupling.py`,
  `driver_fast.py`), `ADV_FFSL_K` to 24 (`ffsl_lr.py`) and `ADV_SL_HOURLY` to 0
  (the two step snapshots interpolated in time, as `lr` always did and as
  production winds arrive), on the 30-day A/B on those winds
  (`runs/ffsl30d_K24brack` vs `runs/bench_phys30d`): mass −0.16 %, bottom-face
  export +0.8 %, AOD −0.3 %, number −0.4 %, TOA forcing +2.0 %, at advect 1.5 s
  + read 0.2 s per step instead of 7.0 + 0.2 s (PROCESSES.md §1.9).
  `ADV_SCHEME=lr` reproduces the previous default byte for byte (`fct_lr.py`
  untouched); `sl` stays opt-in (its one-year run drifts a few %/year).
  `--adv-scheme`, `--adv-ffsl-k`, `--adv-sl-hourly` help text and
  CONFIGURATION / PROCESSES / MANIFEST / REPO_LAYOUT updated.
- `coupling._dep_path` and its deliberate duplicate in `radiation.py` resolve
  the dependencies from `models/<name>` rather than `../<name>`. The
  `TOMAS_JAX_PATH` / `RRTMGP_PATH` overrides are unchanged and still win.
- `patches/jax-rrtmgp-zenith.patch` regenerated against the pinned commit; the
  old one targeted v0.2.1 (`d7abe2e`) and no longer applied. Applying it is now
  documented as required setup, because a gitlink cannot carry the fix.
- `radiation.RI_FILE` resolves from the repo root rather than `__file__`.
- Documentation that told the reader to clone the dependencies as siblings —
  README, MANIFEST, `docs/README.md`, `docs/CONFIGURATION.md`,
  `docs/VALIDATION.md` and two `ImportError` messages — now describes the
  submodule layout. Following the old instructions would have produced a tree
  the resolver does not look at.

- `src/` — all source needed to advance the coupled state, grouped by process:
  `coupling.py`, `driver_fast.py`, `run_prod.sh`, `advection/`, `radiation/`,
  `settling/`, `microphysics/`. Module names stay flat and **no import statement
  changed**; `src/_paths.py` puts the subdirectories on `sys.path`.
- `src/microphysics/tomas_fast.py` — the `tomas_jax.fast` adapter, extracted from
  `driver_fast.py` so every coupled process has a home under `src/`.
- `run_prod.sh` refuses to run from anywhere inside the checkout except under
  `runs/`. The old test was `$PWD == <script dir>`; from `src/` that would no
  longer have protected the repo root.

- `scripts/` — everything that reads a run rather than producing one:
  `scripts/validation/` (the six physics harnesses) and `scripts/utils/`
  (`plot_run.py`, `gif_run.py`). `src/` now holds only what advances the coupled
  state; a run depends on nothing in `scripts/`.

Removed

- `src/run_prod.sh` — replaced by `src/run_prod.py`. It survived briefly as a
  forwarding shim so that in-flight runs whose `run_chain.sh` called it by name
  could still `RESUME`; those runs finished, their `run_chain.sh` now call the
  Python launcher, and the shim is gone.
- `rad_data/` at the repo root (moved to `inputs/rad_data/`).
- `validation/`, `plot_run.py`, `gif_run.py` at the repo root (moved to `scripts/`).
- `fast_advection/` (moved to `src/advection/`).
- `sai_runs/` beside the repo, and the loose `coupled_*prod1d*` artifacts in its
  parent directory (moved to `runs/smoke/` and `runs/prod1d/`; nothing deleted).