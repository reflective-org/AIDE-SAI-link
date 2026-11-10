# Configuration reference

Every knob in this model is an environment variable, and since 2026-09-01 every
one of them also has a `run_prod.py --flag`. This page is the complete list;
[`../README.md`](../README.md) carries the five you need for a first run, and
`python3 src/run_prod.py --help` is the live inventory — generated from the same
table the launcher resolves, so it cannot drift from the code.

> **[`../MANIFEST.md`](../MANIFEST.md) is canonical for this tree.** It records
> every default *and why it is what it is*. This page tells you what the knobs
> are; MANIFEST tells you why. Where the two disagree, MANIFEST is current.

Each variable is read **once at module import** and echoed in the run header, so
any log is self-describing. Set them by prefixing the launcher:

```bash
REPO=$PWD/AIDE-SAI-link                    # wherever this clone lives -- must be ABSOLUTE, the next line cd's away
cd "$PWD/sai_runs"                         # launch from a runs dir, NOT the repo
OUT_TAG=inj20_30N INJ_SO2_TG_YR=20 INJ_LAT=30 INJ_MIRROR=1 $REPO/src/run_prod.py
```

or by passing the equivalent flag, which is usually easier to read:

```bash
python3 $REPO/src/run_prod.py --out-tag inj20_30N --inj-so2 20 --inj-lat 30 --inj-mirror 1
```

Anything the launcher does not preset passes through from your environment —
`AER_SRC=carma $REPO/src/run_prod.py` works, and so does `--aer-src carma`.

**Precedence, highest first:** `--flag` > an already-set environment variable >
`run_prod.py`'s production preset > `coupling.py`'s own default. `--dry-run`
prints the fully resolved environment, with the source of each value, and exits
without launching.

> [!NOTE]
> Until 2026-09-01 the shell launcher **hard-set** `INIT_BIN`, `STATE_CKPT`,
> `FRAME_EVERY`, `FAST_CELL_CAP`, `ADV_VPOS`, `DEBUG` and `PROFILE` as bare
> literals, so passing them was accepted by the shell and then *silently
> discarded* — the same failure that had already bitten `N_HOURS`, `FAST_SORT`
> and `INJ_ZONAL` one at a time. All of them are overridable now: presets and
> pass-throughs go through one table, so that class of bug cannot recur.

The **Default** column below is the *effective* production default, i.e. what
you get from `run_prod.py`; where the bare `coupling.py` module default differs
it is noted.

## Which flags actually apply to my run?

Most of this page is inert for any given run. Two things decide which part is
live: **which entry point you launched** and **which options you switched on**.

`run_prod.py` execs `driver_fast.py`, which imports `coupling.py` and then
*replaces* its microphysics with the batched `tomas_jax.fast` engine. So the
per-cell chain's own microphysics knobs are never reached on the production
path — the `FAST_*` ones take their place. Everything else (injection, run
length, domain, IC/BC, transport, settling, radiation) is shared, because both
paths run the same code for it.

| you launched | microphysics knobs that are live | the ones that do nothing |
|---|---|---|
| `run_prod.py` or `python3 src/driver_fast.py` — **the normal case** | `ALPHA_COND`, `FAST_DT`, `FAST_CELL_CAP`, `FAST_SORT`, `FAST_FN_SCALE`, `FAST_COAG_SUB_CAP`, `FAST_COND_SUB_CAP`, `FAST_COAG_CMAX` | `MICRO_SUBSTEPS`, `N_COAG_SUBSTEPS`, `COAG_MAX_SUBSTEPS`, `CELL_CHUNK`, all `NUC_*` |
| `python3 src/coupling.py` — standalone/dev only | `MICRO_SUBSTEPS`, `COAG_MAX_SUBSTEPS`, `CELL_CHUNK`, `ALPHA_COND`, `NUC_ORG`, `NUC_NH3`, `NUC_FION`, `NUC_FN_MAX` | every `FAST_*` |

The two engines also differ *physically*, not just in speed: the fast engine's
nucleation is **binary** (H2SO4–H2O only), which is why it has no `NUC_ORG` /
`NUC_NH3` / `NUC_FION` to set — `FAST_FN_SCALE` is its only nucleation dial.

Rows below marked **[chain]** are read only by the standalone `coupling.py`
path and are ignored under `run_prod.py`.

Beyond the entry point, four switches gate whole blocks of knobs:

| switch | at this setting | these stop mattering |
|---|---|---|
| `AER_SRC` | `mam4` (the default) | all `CARMA_*`, and `CARMA_FILE` |
| `RAD` | `0` | the whole Radiation section, and `RRTMGP_PATH` |
| `SETTLE` | `0` | `WET_SETTLING` |
| `INJ_SO2_TG_YR` / `INJ_H2SO4_TG_YR` | both `0` (a control run) | the geometry knobs `INJ_HPA`, `INJ_LAT`, `INJ_LON`, `INJ_ZONAL`, `INJ_MIRROR` |

## Injection scenario — the knobs meant to change run to run

| variable | default | meaning |
|---|---|---|
| `INJ_SO2_TG_YR` | `0.0` | continuous SO2 release, Tg(SO2)/yr. **0 = no-injection control.** Pass `10` to reproduce `prod90d` |
| `INJ_H2SO4_TG_YR` | `0.0` | direct gas-phase H2SO4 release at the same geometry. S-equivalent of *X* Tg SO2/yr is `X*1.531` |
| `INJ_HPA` | `55.0` | target altitude [hPa], snapped to the nearest model level |
| `INJ_LAT` | `0.0` | target latitude [deg], snapped to the nearest row |
| `INJ_LON` | `180.0` | target longitude [deg E]; **ignored** when `INJ_ZONAL=1` |
| `INJ_ZONAL` | `1` | `1` = spread over the whole latitude ring; `0` = single cell (5.6× slower, drives runaway nucleation) |
| `INJ_MIRROR` | `0` | `1` = release at **both** ±`INJ_LAT`, splitting the *same* total 50/50 (not doubled). No-op at `INJ_LAT=0` |

`INJ_*` is stamped into the state checkpoint, and a `RESUME` onto a mismatched
checkpoint is **refused** — repeat the scenario flags on the resume command
line, they are not read back.

## Run length, output and restart

| variable | default | meaning |
|---|---|---|
| `N_HOURS` | `2160` | forcing hours to integrate (module: `24*N_DAYS` = 48) |
| `N_DAYS` | `2` | fallback that supplies `N_HOURS` when it is unset |
| `H0` | `0` | start hour index into the CESM h1 series |
| `STEP_HOURS` | `6` | coupling step [h]. `driver_fast.py` **requires 6** and exits otherwise |
| `OUT_TAG` | `prod90d` | names every output file (module: `<N_DAYS>day`; `driver_fast.py` alone: `tomas_fast`) |
| `RESUME` | `0` | `1` = continue from `coupled_state_<TAG>_ckpt.npz` |
| `STATE_CKPT` | `1` | write the restart checkpoint (~400 MB, atomic, overwritten in place) |
| `FRAME_EVERY` | `24` | hours between frame + checkpoint writes |
| `FRAME_LEVELS` | `probe` | what a frame COVERS vertically. `probe` = full lat-lon at `PROBE_HPA` only. `all` adds a zonal-mean frame at **every** band level (+1.1 GB/yr) and is what the lat-height figures and `*_zonal_*.gif` need |
| `LOG_EVERY` | `1` | progress line every N coupling steps (N x `STEP_HOURS` hours) |
| `PROBE_HPA` | `50` | level [hPa] used for frames and probe diagnostics |
| `DIAG_CORE_HPA` | *(unset)* | `lo,hi` diagnostic core window; unset = symmetric −1 level per end (1.6–121.5 hPa) |
| `DEBUG` | `1` | verbose per-step budget/diagnostic output |
| `PROFILE` | `1` | per-phase timing breakdown |

Checkpoints are written frames → timeseries → **state last**, so the physics
state is never newer than the diagnostics.

## Input data and dependency paths

| variable | default | meaning |
|---|---|---|
| `CESM_DIR` | *(the FWHIST archive these results were made from)* | root of the CESM tseries archive; layout `$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF` |
| `CESM_PREFIX` | `f.e21.FWHIST.f09_f09_mg17...cam` | filename prefix inside that archive |
| `CESM_SUF` | `.1996010100-2014123100.nc` | filename suffix (the date range) |
| `TOMAS_JAX_PATH` | `models/tomas-jax` | microphysics submodule, else the normal import path |
| `RRTMGP_PATH` | `models/jax-rrtmgp` | radiation submodule, same order. Unused when `RAD=0` |
| `CARMA_FILE` | *(a site path)* | CARMA history file, only read when `AER_SRC=carma` |

A variable that is *set but points nowhere* is an error, not a silent fallback.

## Domain and resolution

| variable | default | meaning |
|---|---|---|
| `P_LO_HPA` | `1.0` | top of the band [hPa] |
| `P_HI_HPA` | `150.0` | bottom of the band [hPa] (module: `100.0`; `driver_fast.py` sets 150). 1–150 hPa = 24 native levels, 1,327,104 cells |
| `N_LEV` | `0` | `0` = every level in the band; else sub-sample to ~N levels |
| `N_BINS` | `0` | `0` = tomas-jax default (40). `driver_fast.py` accepts only `0` or `40` — the fast engine is hard-fixed at 40 |

## Initial and boundary conditions

| variable | default | meaning |
|---|---|---|
| `AER_SRC` | `mam4` | aerosol IC/BC/reservoir source: `mam4` (per-step dynamic from the hourly h1) or `carma` (static, only ~1 week of output exists). Gases are always CESM-forced |
| `INIT_BIN` | `so4` | bin MAM4 by `so4_a*` mass. `dgnum` is the legacy path that inflated sulfate mass 4.29× |
| `INIT_SIGMA` | *(unset)* | `s1,s2,s3` mode widths; unset = physical MAM4 widths |
| `BC_EDGE` | `open` | vertical faces: real flux boundaries. Derived — `clamp` when `ADV_WCONT=0` |
| `BC_GAS` | `flux` | SO2/H2SO4 edge treatment. **Derived from the resolved `BC_EDGE`** so gas and aerosol cannot desync. `clamp` reproduces pre-2026-07-30 runs |
| `BC_BOT_AER` | `1.0` | scale on the aerosol concentration flowing in through the bottom face (`0.0` = aerosol-free upwelling). Gases unaffected |
| `N_BC_TOP` | `1` | top band levels pinned to hourly MAM4 |
| `N_BC_BOT` | `1` | bottom band levels pinned to hourly MAM4 |
| `CARMA_FRAME` | `0` | time index into the CARMA file (48 frames) |
| `CARMA_RHO` | `1923.0` | CARMA sulfate mass-grid density [kg/m³] |
| `CARMA_SUBBIN` | `1` | sub-bin CARMA onto the TOMAS grid; `0` reproduces the pre-2026-07-26 "comb" |

More detail: [BOUNDARY_CONDITIONS.md](./BOUNDARY_CONDITIONS.md).

## Transport

| variable | default | meaning |
|---|---|---|
| `ADV_SCHEME` | `ffsl` | `ffsl` (default since 2026-10-15) = `lr` itself at large Courant number (`src/advection/ffsl_lr.py`, Lin & Rood's flux-form semi-Lagrangian extension): the same flux-form operators, faces, budgets and limiter at `ADV_FFSL_K` sub-steps instead of 70–250; at K = 24 on the step snapshots (`ADV_SL_HOURLY=0`) within 0.2 % of `lr` on the 30-day budgets and 2 % on every quantity, advect 1.5 s/step vs 7 s (PROCESSES.md §1.9). `lr` = Lin-Rood flux form at CFL sub-steps, conservative to roundoff: the default until 2026-10-15, byte for byte unchanged, and the reference the budgets are measured against. `fast` = advective form, ~3e-4/day residual, cheaper. `sl` = the split operator (`src/advection/sl_lr.py`): dinosaur's semi-Lagrangian remap per level (no CFL limit, ~0.4 s/step instead of ~6.6 s) Strang-composed with `lr`'s own flux-form vertical sweep on the open faces, so `ADV_WCONT=1`, `BC_EDGE=open`, `B_vf` and the budgets mean what they mean under `lr`. More accurate than `lr` on the analytic and closed-column truth tests; 30 days vs `lr`: mass −1.2 %, bottom-face export +16 %, forcing +8 %. Opt-in pending the one-year pair — [PROCESSES.md §1.8](./PROCESSES.md), [TRANSPORT_SL_HANDOFF.md](./TRANSPORT_SL_HANDOFF.md) |
| `ADV_SL_K` | `6` | `sl` only: sub-steps per coupling step (6 = one per hourly CESM wind at `STEP_HOURS=6`). Keep it at the wind cadence: every SL gather smooths, so K > 6 is LESS accurate on analytic tests (PROCESSES.md 1.8), and K < 6 under-resolves the vertical PPM sweep |
| `ADV_SL_FIXER` | `global` | `sl` only: per-tracer mass fixer on the closed extended column. `global` = proportional rescale (JCM's); `local` = Bermejo–Conde, corrects only where the cubic gather overshoots its linear counterpart (+1 linear gather); `none` = raw operator (diagnostics) |
| `ADV_SL_HOURLY` | `0` | `sl` and `ffsl`: `0` (default since 2026-10-15) interpolates the two step snapshots in time at every sub-step, as `lr` does and as production winds arrive (six-hourly); `1` reads the hourly CESM winds between them so each sub-step advects on its own hour (~1.3 s/step of I/O; worth ≈3 % in number and forcing at day 30, PROCESSES.md §1.9) |
| `ADV_SL_VERT` | `flux` | `sl` only: `flux` = the split operator — SL horizontal remap per level + `fct_lr`'s flux-form vertical sweep with its open faces, reservoir inflow, positivity limiter and face-flux diagnostic (`ADV_WCONT=1` semantics, `B_vf` on). `sl` = the handoff's closed-column 3-D gather with Dirichlet edge levels (sets `ADV_WCONT=0`), which failed the 30-day A/B |
| `ADV_SL_OMEGA` | `cont` | `sl`+`flux` only: face omega rederived from discrete continuity with `lr`'s own operator (default) or CESM center omega averaged to the faces |
| `ADV_SL_SPLIT` | `strang` | `sl`+`flux` only: how the horizontal remap and the vertical sweep are composed per sub-step. `strang` = half vertical sweep, horizontal remap with the air mass `1 − dt·div(v)` carried, half vertical sweep: the open-face flux is second order in the sub-step with still one SL gather. `lie` = the 2026-10-13 composition, horizontal then vertical, first order at the open face (30-day export +21 % vs `lr` with the air mass frozen, +37 % with it carried; PROCESSES.md 1.8) |
| `ADV_SL_JAC` | `1` | `sl`+`flux` only, with the air mass carried: `1` takes the air mass after the horizontal remap from the Jacobian of the departure map (second-order, like the departure points) and returns the sub-step's bookkeeping remainder `ρ − 1` through the column's own faces at the local mixing ratio; `0` uses the Eulerian `1 − dt·div(v)` and the global column fixer, which spread a local O(dt²) remainder over the globe every sub-step (first order in K; PROCESSES.md 1.8) |
| `ADV_SL_RHO` | `0` | `sl`+`flux`+`lie` only: `1` carries the air mass through the horizontal step so the per-level fixer is an O(dt²) correction instead of an O(dt·div) uniform transfer every sub-step (the downward redistribution behind the one-year export excess). `strang` always carries it; the knob is kept for the diagnostics |
| `ADV_SL_VORDER` | `linear` | `sl` only: vertical gather order (`cubic` degrades to linear in the end cells) |
| `ADV_SL_ITER` | `2` | `sl` only: departure-point fixed-point iterations |
| `ADV_CFL` | `0.5` | CFL target for internal sub-stepping |
| `ADV_F32` | `1` | single precision. **Validate advection changes at `ADV_F32=1 ADV_CFL=0.5`** |
| `ADV_VPOS` | `1` | vertical positivity limiter. `0` is a forensic escape hatch, **not a supported configuration** — it reproduces a corrupt number field |
| `ADV_FFSL_K` | `24` | `ffsl` only: sub-steps per coupling step (24 = the production operating point since 2026-10-15; 6 = one per hourly CESM wind). A convergence knob with no smoothing penalty: each sub-step costs ~1/200 of a production `lr` step, and K = 24 reproduces `lr`'s 10-day budget to 0.05 % in mass at 4× lower cost; K = 6 carries the x/y splitting error of hourly sub-steps at the face (PROCESSES.md 1.9) |
| `ADV_FFSL_SPLIT` | `strang` | `ffsl` only: `strang` = z/2, horizontal (x,y alternating with y,x), z/2; `lie` = `fct_lr`'s x, y, z |
| `ADV_WCONT` | `1` | rederive omega from discrete continuity and open the vertical faces. Default `0` only under `ADV_SCHEME=sl ADV_SL_VERT=sl` (that column is closed; `1` there leaves the edge levels free with no exchange at all, and the header warns) |
| `ADV_METRIC` | `1` | carry the cos(φ) area metric in the y-sweep |
| `ADV_DXFIX` | `1` | true grid spacing in the x-sweep |
| `ADV_POLAR` | `zonal` | mass-conserving well-mixed polar caps above \|lat\| 80°; anything else = legacy reservoir overwrite, which discards mass. Honoured by `sl` too (stirred after every sub-step) |
| `TRACER_CHUNK` | `0` | tracers advected per batch (`0` = all 82 at once). Lower it if advection OOMs |

## Microphysics and settling

Under `run_prod.py` only `MICRO`, `ALPHA_COND`, `SETTLE` and `WET_SETTLING` are
live here; the **[chain]** rows are the standalone `coupling.py` engine's and
are ignored. Their fast-engine counterparts are in the next section.

| variable | default | meaning |
|---|---|---|
| `MICRO` | `full` | `full` = chemistry + nucleation + coagulation + condensation. `coag` = legacy coagulation only. `off` = no microphysics at all, a transport-only benchmark mode and **not** a physics configuration (see `PROCESSES.md` §0). Validated at import. `driver_fast.py` accepts `full` and `off` |
| `ALPHA_COND` | `1.0` | H2SO4 accommodation (sticking) coefficient — the fraction of vapour–particle collisions that actually condense. `1.0` = every collision sticks, the fastest condensation physically allowed. **Read by both engines** |
| `SETTLE` | `1` | gravitational settling, the model's only true aerosol sink |
| `WET_SETTLING` | `1` | size the settling particle as the wet H2SO4/H2O droplet; `0` restores dry-core sizing. Ignored when `SETTLE=0` |
| `MICRO_SUBSTEPS` | `6` | **[chain]** substeps per coupling step (1 h each); an accuracy dial, not a stability requirement. The fast engine uses `FAST_DT` instead |
| `COAG_MAX_SUBSTEPS` | `256` | **[chain]** ceiling on the *adaptive* coagulation solver's substeps. A speed knob: the vmapped `while_loop` runs every lane to the slowest one. A cell that hits the cap is returned **partially integrated** and logs a `coag substep cap hit` warning |
| `CELL_CHUNK` | `300000` | **[chain]** cells per micro vmap batch — peak GPU memory vs. call overhead only. Micro is per-cell independent, so results are identical at any value |
| `N_COAG_SUBSTEPS` | `3` | **[chain]** fixed forward-Euler substeps for the legacy `MICRO=coag` path only; dead under `MICRO=full` |
| `NUC_ORG` | `1e7` | **[chain]** background organic vapour [molec/cm³] fed to the ternary rate. A boundary-layer value inherited from the tomas-jax SAI box model — retune before reading anything into it |
| `NUC_NH3` | `1e9` | **[chain]** NH3 [pptv]; same caveat |
| `NUC_FION` | `3.0` | **[chain]** ion-pair production [cm⁻³ s⁻¹] |
| `NUC_FN_MAX` | `1e6` | **[chain]** cap on total nucleation rate [cm⁻³ s⁻¹]; a numerical guard against gas-clamped ultrafine bursts |

## Fast microphysics engine (`driver_fast.py`) — the production microphysics

**This is the section that applies to any run started with `run_prod.py`.**
Apart from `FAST_FN_SCALE`, these are memory/scheduling knobs: micro is per-cell
independent, so how cells are grouped into chunks cannot change results.

| variable | default | meaning |
|---|---|---|
| `FAST_DT` | `360` | inner step [s]; 60 inner steps fill one 6 h coupling step. The fast engine's counterpart to `MICRO_SUBSTEPS` |
| `FAST_CELL_CAP` | `50000` | max cells per chunk (module: `250000`). Lower is **faster** here, and fits alongside another job. Counterpart to `CELL_CHUNK` |
| `FAST_SORT` | `1` | stiffness-sort chunks, worth ~27% of micro. Does one unchunked ~8 GB allocation — **first thing to set to `0` if the card is loaded** |
| `FAST_FN_SCALE` | `1.0` | nucleation rate scale — the **only** nucleation knob on this path, since the scheme is binary (no organics/NH3/ions to set) |
| `FAST_COAG_SUB_CAP` | `256` | coagulation substep cap. Counterpart to `COAG_MAX_SUBSTEPS` |
| `FAST_COND_SUB_CAP` | `40` | condensation substep cap |
| `FAST_COAG_CMAX` | `0.05` | coagulation adaptive-step tolerance — how much error per step is accepted, i.e. how many substeps you actually spend under the cap |

## Microphysics engine selection and the emulator (`driver_fast.py`)

Which batched engine fills the microphysics seam, and an optional second one
run alongside it for comparison. See `docs/EMULATOR.md`.

| variable | default | meaning |
|---|---|---|
| `MICRO_ENGINE` | `fast` | `fast` = `tomas_jax.fast.run_fast` (the production physics, above). `emulator` = the learned 6 h operator from `/data/tomas-emulator`, one forward pass per cell per step (`src/microphysics/tomas_emulator.py`). Requires torch in the run environment (`/data/aide_emu_venv`) |
| `MICRO_REF` | *(unset)* | name the OTHER engine to run it on the same input too, purely for comparison: the run advances with `MICRO_ENGINE`, the reference output is compared cell by cell and discarded, and the per-step error + both engines' timings go to `micro_bench_<TAG>.npz` |
| `MICRO_REF_EVERY` | `1` | compare every N coupling steps |
| `EMU_CKPT` | `$EMU_ROOT/frontier_bintf_kchain_big_rebal/avg_top3.pt` | emulator checkpoint. The lighter `frontier_bintf_kchain_rebal/avg_top3.pt` needs `EMU_D=256 EMU_NL=4` |
| `EMU_D`, `EMU_NL` | `384`, `6` | emulator width and depth; must match the checkpoint |
| `EMU_BS` | `16384` | cells per forward pass. Fixed per compiled CUDA graph; the last batch is padded |
| `EMU_DTYPE` | `fp16` | autocast precision, `fp16` or `fp32`. `bf16` is refused (widens the 14-day tails, handoff §2) |
| `EMU_COMPILE` | `1` | `torch.compile(mode='reduce-overhead')`; `0` = eager (2–3× slower, same numbers) |
| `EMU_OH_FLOOR` | `100` | floor on the OH samples handed to the emulator [molec/cm³]. The SZA parabola is exactly zero at night, which the emulator never saw; see `docs/EMULATOR.md` for the sweep behind the value |
| `EMU_SO2_FLOOR` | `1e-17` | floor on the per-box SO2 handed to the emulator [kg/box]. The training set's SO2 never fell below this; an advected-to-zero SO2 is 7 σ outside it and was answered with 1660× the physics' mass |
| `EMU_PROJECT` | `clip` | what is enforced on the emulator's answer: `clip` = two-moment consistency (as TOMAS's mnfix); `chem` (2026-10-13) = that plus the exact sulfur budget done from the chemistry side: SO2 decays by the physics' own k₁(T,P,RH)·OH, H2SO4 relaxes to its production / condensation-sink balance with the physics' Fuchs–Sutugin sink on the network's output distribution, and the aerosol gets exactly the sulfur the gas lost (the network's per-bin change kept, the residual deposited by sink fraction, re-clipped) — the network then only supplies number and the shape of the distribution; `sulfur` = conservation taken from the network's own gas answer — **worse, see `docs/EMULATOR.md`**; `0` = nothing |
| `EMU_CODE`, `EMU_ROOT` | `/data/tomas-emulator`, `/data/tomas-jax-output/emulator_v1` | the emulator code tree and the checkpoints / `norm_stats.npz` root |
| `EMU_TRIAGE` | `1` | skip the network on QUIET cells (2026-10-13): cells whose gas sulfur cannot move the aerosol mass by `EMU_TRIAGE_POT` (1e-3), with fewer than `EMU_TRIAGE_N` (300) cm⁻³, a sustainable H2SO4 below `EMU_TRIAGE_HSS` (1e5 cm⁻³) and less than `EMU_TRIAGE_FINE` (0.1) of their number below 10 nm. They get the exact pseudo-first-order SO2 + OH step with the acid condensed by surface area and no change in number. ~30 % of the grid, 80 % of the mass; the physics itself moves them by 0.13 % (number, median) per step. `0` sends every cell to the network |
| `EMU_PHYS_TOP` | `0` | hybrid (2026-10-13): hand this fraction of cells — the top ones by total sulfur per box (SO4 + SO2 + H2SO4) — to `tomas_jax.fast` with the production knobs instead of the network. The top 1 % of cells hold 80 % of all particles and carry ~90 % of the emulator's signed global number error (the injection-ring bursts); the physics on them costs 0.2 s/step (`0.01`) to 0.5 s (`0.03`) after a one-off 20 s jit compile, and the network / quiet step take the rest. `EMU_PHYS_PAD` (4096) pads the subset to a fixed shape for the jit cache; `EMU_PHYS_CHUNKS` (1) stiffness-sorts it into that many sequential chunks |
| `EMU_PHYS_NUM` | `0` | hybrid, second rule (2026-10-13): also hand this fraction of the *remaining* cells, ranked by total particle number per box, to the fast physics. Needed with `EMU_PROJECT=chem`: along the closure run's trajectory the network under-coagulates the dense fine populations (~1e4 cm⁻³) that the diffuse plume becomes — the physics halves their number per step, the network removes a quarter — and the sulfur ranking does not see them (little mass). The top 4 % by number carry 89 % of that error, the top 8 % 98 %; 1.1 s / ~2 s per step. |
| `MICRO_SAMPLE_EVERY` | `0` | every N steps write a stratified sample of the cells the engine is about to see to `micro_samples_<TAG>_s<step>.npz` (`src/microphysics/micro_sampler.py`): the emulator's training data from this model's own states. `MICRO_SAMPLE_N` (5000) cells per sample, plus the plume stratum capped at half of it. Needs torch (the triage test) even under `MICRO_ENGINE=fast` |

## OH chemistry

| variable | default | meaning |
|---|---|---|
| `OH_SZA` | `1` | diurnal OH parabola in cos(SZA) (Hanisco 2001). **Replaces** CESM's OH field — the step-mean magnitude changes too, not just the time resolution. `0` restores CESM OH held constant across the step |
| `OH_PEAK` | `2.3e6` | noon peak OH [molec/cm³] (the fit is least-squares, so peak OH is 4% above this) |
| `OH_SUBSTEPS` | *(per driver)* | samples of the curve per coupling step. `driver_fast.py` sets it to its inner-step count (60 at `FAST_DT=360`); the per-cell chain uses `MICRO_SUBSTEPS` |

## Radiation

| variable | default | meaning |
|---|---|---|
| `RAD` | `1` | `0` skips radiation entirely — and with it the `jax-rrtmgp` dependency |
| `RAD_MODE` | `anomaly` | `anomaly` = heating relative to the reference MAM4 bins; `full` = absolute (drifts toward radiative equilibrium). Under `AER_SRC=mam4` the anomaly baseline is **time-varying** |
| `RAD_EVERY` | `1` | coupling steps between radiation calls |
| `ARF_AVG_H` | `24` | trailing window [h] for the *reported* forcing; 24 h = 4 uniformly spaced local times = a proper diurnal mean. `0` disables (the raw instantaneous value is always stored) |
| `WET_OPTICS` | `1` | Mie on the wet droplet. Dry sizing underestimates 550 nm extinction by ~47%. Keep in step with `WET_SETTLING` |
| `RAD_LAT_CHUNK` | `96` | latitude rows per RRTMGP chunk; bit-identical either way. Drop to 48 if radiation OOMs |
| `CO2_PPM` | `380.0` | background CO2 (~2005, for 1996–2014 forcing) |
| `N2O_PPB` | `319.0` | background N2O |
| `RAD_REF_CACHE` | `inputs/cache/rad_ref` | per-archive-hour cache of the reference-aerosol solve (`RAD_MODE=anomaly`) and of the clear-sky fields the emulator needs (`src/radiation/rad_cache.py`). A hit is **bit-identical** to computing the field (float64 on disk, same code) and skips both the reference solve and its MAM4 re-binning (~5.6 of the 8.5 s stage); a miss computes and stores. `0` disables. Batch build: `scripts/radiation/build_ref_cache.py` |
| `RAD_ENGINE` | `rrtmgp` | which engine solves the **evolved** radiation call: `rrtmgp` = the physics (byte-for-byte unchanged), `emulator` = two column networks on the cached clear-sky fields (`src/radiation/rad_emulator.py`, `docs/RADIATION.md`). Stamped into `PHYS_CFG` |
| `RAD_REF` | — | with `RAD_ENGINE=emulator`: `rrtmgp` also runs the physics on the same input every `RAD_REF_EVERY` radiation steps and logs the comparison to `rad_bench_<TAG>.npz`; the run advances with the emulator. Mirrors `MICRO_REF` |
| `RAD_REF_EVERY` | `1` | compare every N radiation steps |
| `RAD_EMU_CKPT` | `inputs/rad_emulator/current` | directory holding `aer.npz` + `planck.npz` (exported by `scripts/radiation/train_rad_mlp.py`) |
| `RAD_EMU_ANOM` | `net` | emulator's reference term in anomaly mode: `net` = the networks on the reference aerosol too (errors correlated with the evolved call, cancel in the background; cached per hour and checkpoint), `phys` = the exact cached physics reference |
| `RAD_EMU_ARF` | `emu` (was `exact` when `RAD_REF=rrtmgp`, until 2026-10-17) | where the emulated run's reported TOA forcing (`arf_toa`) comes from: `exact` = the physics A/B samples (the forcing is the global mean of a ~3 % difference of the aerosol's TOA effect, below the networks' ±1 % bias early in a run; with `RAD_REF_EVERY=3` the samples cycle through the four local times, so the trailing window stays a diurnal mean), `emu` = the networks |
| `RAD_EMU_CLIP` | `20` | sanity clamp on the networks' heating outputs [K/day] (TOA deltas at 25× that in W m⁻²), NaN → 0, every clamped cell counted and logged. A residual MLP without normalisation can blow up on an input far outside its training set (an aerosol net trained at `lr 1e-3` returned 674 K/day in a column whose upper-troposphere humidity sat at −16 σ, and `dT_rad` was NaN two steps later); the physics never exceeds ~8 K/day even in the ×10-augmented labels, so 20 is a no-op on a healthy run. `0` disables it |
| `RAD_SAMPLE_EVERY` | `0` | write `rad_sample_<TAG>_s<step>.npz` (evolved bins, `dT_rad`, hour) every N radiation steps: emulator training data for `scripts/radiation/label_columns.py` |

## GPU and process environment (`run_prod.py`)

| variable | default | meaning |
|---|---|---|
| `GPU` | `0` | which card to pin (`CUDA_VISIBLE_DEVICES`). Microphysics shards across all *visible* devices, so exporting more than one is what makes it multi-GPU |
| `CUDA_DRIVER_LIB` | `/run/nvidia/driver/usr/lib/x86_64-linux-gnu` | where `libcuda.so.1` lives. Without it **JAX silently falls back to CPU**. Set it empty for a normal CUDA install |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | grow GPU memory on demand instead of preallocating 75% of the card |

## Debug-only

| variable | default | meaning |
|---|---|---|
| `DUMP_PREMICRO` | *(unset)* | dump the pre-microphysics state for one step |
| `DUMP_PREMICRO_STEP` | `0` | which step to dump |

## Resuming across a changed configuration

`INJ_*` mismatches against the checkpoint are **refused**. The physics-mode flags
`WET_SETTLING` / `WET_OPTICS` / `SETTLE` / `ADV_VPOS` only **warn** — the state is
still valid, the model integrating it forward just changes at the seam. Both
arrays are append-only, so adding a field never locks out an older checkpoint.

## Flag ↔ environment variable cross-reference

Generated from the `KNOBS` table in `src/run_prod.py` — the same table
the launcher resolves and `--help` prints, so this cannot drift from the
code. **prod** is `run_prod.py`'s production preset; blank means the
launcher sets nothing and `coupling.py`'s own default stands.

### run control

| flag | variable | prod | meaning |
|---|---|---|---|
| `--n-hours` | `N_HOURS` | `2160` | run length in hours (2160 = 90 d; 8760 = 1 yr) |
| `--n-days` | `N_DAYS` | — | run length in days (N_HOURS wins if both given) |
| `--out-tag` | `OUT_TAG` | `prod90d` | output tag; ALL outputs and checkpoints are keyed by it |
| `--resume` | `RESUME` | `0` | resume from coupled_state_<TAG>_ckpt.npz |
| `--h0` | `H0` | — | start hour index into the CESM h1 series (0 = 1996-01-01 00Z) |
| `--step-hours` | `STEP_HOURS` | — | coupling step [h]; one advect+coag per step (default 6) |
| `--state-ckpt` | `STATE_CKPT` | `1` | write the 3-D restart checkpoint (needed for --resume) |
| `--frame-every` | `FRAME_EVERY` | `24` | spatial frame every N hours (>= N_HOURS = final frame only) |
| `--frame-levels` | `FRAME_LEVELS` | — | 'probe' (default) = one PROBE_HPA level; 'all' adds a zonal-mean frame at every level (+1.1 GB/yr) and unlocks the lat-height figures and gifs |
| `--log-every` | `LOG_EVERY` | — | progress line every N coupling steps |
| `--probe-hpa` | `PROBE_HPA` | — | diagnostic/frame level (default 50) |
| `--debug` | `DEBUG` | `1` | verbose per-step diagnostics |
| `--profile` | `PROFILE` | `1` | per-stage timing lines |

### domain and grid

| flag | variable | prod | meaning |
|---|---|---|---|
| `--p-lo-hpa` | `P_LO_HPA` | — | top of the band (default 1) |
| `--p-hi-hpa` | `P_HI_HPA` | — | bottom of the band (default 100) |
| `--n-lev` | `N_LEV` | — | sub-sample the band to ~N levels (0 = full contiguous band) |
| `--n-bins` | `N_BINS` | — | TOMAS bin count (0 = native 40) |
| `--diag-core-hpa` | `DIAG_CORE_HPA` | — | core window for the 'int' burden diagnostics |

### injection scenario (the knobs meant to change run to run)

| flag | variable | prod | meaning |
|---|---|---|---|
| `--inj-so2` | `INJ_SO2_TG_YR` | `0.0` | SO2 injection rate. DEFAULT 0 = no-injection control |
| `--inj-h2so4` | `INJ_H2SO4_TG_YR` | `0.0` | direct H2SO4(g) injection rate |
| `--inj-hpa` | `INJ_HPA` | `55.0` | injection altitude (snapped to the nearest model level) |
| `--inj-lat` | `INJ_LAT` | `0.0` | injection latitude (snapped to the nearest row) |
| `--inj-lon` | `INJ_LON` | `180.0` | injection longitude (ignored when --inj-zonal 1) |
| `--inj-zonal` | `INJ_ZONAL` | `1` | 1 = spread around the whole latitude ring, 0 = single cell |
| `--inj-mirror` | `INJ_MIRROR` | `0` | release at BOTH +lat and -lat, total split 50/50 (not doubled) |

### aerosol source / initial condition

| flag | variable | prod | meaning |
|---|---|---|---|
| `--aer-src` | `AER_SRC` | — | IC/BC reservoir source (default mam4) |
| `--init-bin` | `INIT_BIN` | `so4` | how MAM4 modes are binned onto TOMAS |
| `--init-sigma` | `INIT_SIGMA` | — | override the log-normal mode width used for binning |
| `--carma-file` | `CARMA_FILE` | — | CARMA reservoir file (AER_SRC=carma) |
| `--carma-frame` | `CARMA_FRAME` | — | time index into the CARMA file |
| `--carma-rho` | `CARMA_RHO` | — | CARMA particle density |
| `--carma-subbin` | `CARMA_SUBBIN` | — | CARMA sub-binning factor |

### boundaries

| flag | variable | prod | meaning |
|---|---|---|---|
| `--n-bc-top` | `N_BC_TOP` | — | band levels pinned/served at the top face (default 1) |
| `--n-bc-bot` | `N_BC_BOT` | — | band levels pinned/served at the bottom face (default 1) |
| `--bc-edge` | `BC_EDGE` | — | aerosol edge treatment (derived from ADV_WCONT) |
| `--bc-gas` | `BC_GAS` | — | gas edge treatment (derived from BC_EDGE) |
| `--bc-bot-aer` | `BC_BOT_AER` | — | scale the bottom-face aerosol inflow (0 = clean upwelling) |

### physics stages

| flag | variable | prod | meaning |
|---|---|---|---|
| `--micro` | `MICRO` | — | microphysics mode; 'off' is transport-only |
| `--micro-substeps` | `MICRO_SUBSTEPS` | — | micro substeps per coupling step (default 6) |
| `--settle` | `SETTLE` | — | gravitational settling (default on) |
| `--wet-settling` | `WET_SETTLING` | — | hygroscopic growth in the settling velocity |
| `--wet-optics` | `WET_OPTICS` | — | hygroscopic growth in the optics |
| `--rad` | `RAD` | — | radiation on/off |
| `--rad-every` | `RAD_EVERY` | — | coupling steps between radiation calls |
| `--rad-mode` | `RAD_MODE` | — | heating-rate definition |
| `--arf-avg-h` | `ARF_AVG_H` | — | trailing window for the reported TOA forcing |
| `--alpha-cond` | `ALPHA_COND` | — | H2SO4 accommodation coefficient |

### chemistry, nucleation and coagulation

| flag | variable | prod | meaning |
|---|---|---|---|
| `--oh-sza` | `OH_SZA` | — | diurnal OH parabola in cos(SZA) |
| `--oh-peak` | `OH_PEAK` | — | noon OH peak (default 2.3e6) |
| `--oh-substeps` | `OH_SUBSTEPS` | — | OH samples per coupling step |
| `--nuc-org` | `NUC_ORG` | — | organic concentration for Riccobono nucleation |
| `--nuc-nh3` | `NUC_NH3` | — | ammonia concentration |
| `--nuc-fion` | `NUC_FION` | — | ion-induced nucleation factor |
| `--nuc-fn-max` | `NUC_FN_MAX` | — | cap on the total nucleation rate |
| `--n-coag-substeps` | `N_COAG_SUBSTEPS` | — | forward-Euler coag substeps (legacy MICRO=coag path) |
| `--coag-max-substeps` | `COAG_MAX_SUBSTEPS` | — | ceiling on the adaptive coag substeps (speed-critical) |

### microphysics engine selection and the emulator

| flag | variable | prod | meaning |
|---|---|---|---|
| `--micro-engine` | `MICRO_ENGINE` | — | which batched engine advances the run (default fast = tomas_jax.fast) |
| `--micro-ref` | `MICRO_REF` | — | ALSO run this engine on the same input and log the per-step comparison to micro_bench_<TAG>.npz |
| `--micro-ref-every` | `MICRO_REF_EVERY` | — | compare every N coupling steps (default 1) |
| `--emu-ckpt` | `EMU_CKPT` | — | emulator checkpoint (default: bintf_big_rebal avg_top3.pt) |
| `--emu-d` | `EMU_D` | — | emulator width (384 for the default ckpt, 256 for the light one) |
| `--emu-nl` | `EMU_NL` | — | emulator depth (6 for the default ckpt, 4 for the light one) |
| `--emu-bs` | `EMU_BS` | — | cells per emulator forward pass (default 16384) |
| `--emu-dtype` | `EMU_DTYPE` | — | emulator autocast precision (default fp16; bf16 refused) |
| `--emu-compile` | `EMU_COMPILE` | — | torch.compile(reduce-overhead) the emulator (default 1) |
| `--emu-oh-floor` | `EMU_OH_FLOOR` | — | OH floor handed to the emulator [molec/cm3] (default 100) |
| `--emu-so2-floor` | `EMU_SO2_FLOOR` | — | SO2 floor handed to the emulator [kg/box] (default 1e-17) |
| `--emu-project` | `EMU_PROJECT` | — | output projection: clip (default) / chem / sulfur / 0 |
| `--emu-triage` | `EMU_TRIAGE` | — | 1 = analytic step for quiet cells, network for the rest (default 1); 0 = network everywhere |
| `--emu-phys-top` | `EMU_PHYS_TOP` | — | hybrid: fraction of cells (top by sulfur) integrated by the fast physics instead of the network (0 = off) |
| `--emu-phys-chunks` | `EMU_PHYS_CHUNKS` | — | hybrid: stiffness-sorted chunks for the physics subset (default 1) |
| `--emu-phys-num` | `EMU_PHYS_NUM` | — | hybrid: fraction of the remaining cells (top by particle number) integrated by the fast physics (0 = off) |
| `--micro-sample-every` | `MICRO_SAMPLE_EVERY` | — | write a training-data sample of the engine's input every N steps (0 = off) |
| `--micro-sample-n` | `MICRO_SAMPLE_N` | — | cells per sample (default 5000; plus the plume, capped at n/2) |
| `--emu-code` | `EMU_CODE` | — | tomas-emulator code tree (default /data/tomas-emulator) |
| `--emu-root` | `EMU_ROOT` | — | emulator checkpoints/norm_stats root |

### fast microphysics engine

| flag | variable | prod | meaning |
|---|---|---|---|
| `--fast-dt` | `FAST_DT` | — | inner timestep of run_fast (default 360) |
| `--fast-cell-cap` | `FAST_CELL_CAP` | `50000` | cells per run_fast chunk; smaller is often FASTER |
| `--fast-sort` | `FAST_SORT` | `1` | stiffness sort (~27% of micro; one ~8 GB alloc -- first to disable on a loaded card) |
| `--fast-fn-scale` | `FAST_FN_SCALE` | — | nucleation-rate scale factor |
| `--fast-coag-sub-cap` | `FAST_COAG_SUB_CAP` | — | coag substep cap inside run_fast |
| `--fast-cond-sub-cap` | `FAST_COND_SUB_CAP` | — | condensation substep cap inside run_fast |
| `--fast-coag-cmax` | `FAST_COAG_CMAX` | — | coag Courant-like limit inside run_fast |

### advection

| flag | variable | prod | meaning |
|---|---|---|---|
| `--adv-scheme` | `ADV_SCHEME` | — | transport form (default ffsl = lr at large Courant number, the production scheme since 2026-10-15; lr = the previous default; sl = semi-Lagrangian, docs/TRANSPORT_SL_HANDOFF.md) |
| `--adv-sl-k` | `ADV_SL_K` | — | SL: sub-steps per coupling step (default 6 = one per hourly wind) |
| `--adv-sl-fixer` | `ADV_SL_FIXER` | — | SL: per-tracer mass fixer on the closed column (default global; local = Bermejo-Conde) |
| `--adv-sl-hourly` | `ADV_SL_HOURLY` | — | sl and ffsl: 1 = read the hourly CESM winds between the step snapshots; 0 (default) = the two snapshots interpolated in time |
| `--adv-sl-vorder` | `ADV_SL_VORDER` | — | SL: vertical gather order (default linear) |
| `--adv-sl-vert` | `ADV_SL_VERT` | — | SL: vertical treatment (default flux = fct_lr's open-face flux sweep; sl = closed-column 3-D gather, A/B-failed) |
| `--adv-sl-omega` | `ADV_SL_OMEGA` | — | SL flux vertical: face omega from continuity (lr's, default) or CESM centers averaged |
| `--adv-sl-split` | `ADV_SL_SPLIT` | — | SL flux vertical: strang (default) = V/2 H V/2 with the air mass carried; lie = H then V |
| `--adv-sl-rho` | `ADV_SL_RHO` | — | SL flux vertical, lie split only: carry the air mass through the horizontal step (0 / 1) |
| `--adv-sl-jac` | `ADV_SL_JAC` | — | SL flux vertical: Jacobian air mass + remainder through the faces (1, default) or Eulerian + column fixer (0) |
| `--adv-ffsl-k` | `ADV_FFSL_K` | — | ffsl: sub-steps per coupling step (default 24, the production operating point; 6 = hourly) |
| `--adv-ffsl-split` | `ADV_FFSL_SPLIT` | — | ffsl: sweep order (strang default; lie = fct_lr's) |
| `--adv-cfl` | `ADV_CFL` | — | horizontal CFL target (production 0.5) |
| `--adv-f32` | `ADV_F32` | — | float32 transport sweeps (production 1) |
| `--adv-vpos` | `ADV_VPOS` | `1` | VERTICAL POSITIVITY LIMITER -- see the header; do not turn off |
| `--adv-wcont` | `ADV_WCONT` | — | rederive omega from discrete continuity (opens the faces) |
| `--adv-metric` | `ADV_METRIC` | — | cos(phi) area metric in the y-sweep |
| `--adv-dxfix` | `ADV_DXFIX` | — | true grid spacing in the x-sweep |
| `--adv-polar` | `ADV_POLAR` | — | polar cap treatment (zonal = stirred, mass-conserving) |

### radiation gases and chunking

| flag | variable | prod | meaning |
|---|---|---|---|
| `--co2-ppm` | `CO2_PPM` | — | background CO2 for RRTMGP |
| `--n2o-ppb` | `N2O_PPB` | — | background N2O for RRTMGP |
| `--rad-lat-chunk` | `RAD_LAT_CHUNK` | — | latitude rows per radiation chunk (GPU memory) |
| `--rad-engine` | `RAD_ENGINE` | — | `rrtmgp` (physics, default) or `emulator` (column networks; `docs/RADIATION.md`) |
| `--rad-ref` | `RAD_REF` | — | `rrtmgp`: A/B the emulator against the physics on the same input, `rad_bench_<TAG>.npz` |
| `--rad-ref-every` | `RAD_REF_EVERY` | — | compare every N radiation steps (default 1) |
| `--rad-ref-cache` | `RAD_REF_CACHE` | — | reference/clear-sky cache directory (default `inputs/cache/rad_ref`; `0` = off) |
| `--rad-sample-every` | `RAD_SAMPLE_EVERY` | — | write emulator training samples every N radiation steps (0 = off) |
| `--rad-emu-ckpt` | `RAD_EMU_CKPT` | — | emulator checkpoint directory (`aer.npz` + `planck.npz`) |
| `--rad-emu-anom` | `RAD_EMU_ANOM` | — | `net` (default) or `phys` reference term for the emulated anomaly |
| `--rad-emu-arf` | `RAD_EMU_ARF` | — | `exact` (physics A/B samples; the default with `--rad-ref rrtmgp` until 2026-10-17 -- its 4-finite-sample window spans 72 h at RAD_REF_EVERY=3, not comparable with a physics run's arf_toa_avg) or `emu` TOA forcing diagnostic |
| `--rad-emu-clip` | `RAD_EMU_CLIP` | — | sanity clamp on the emulator's heating outputs, K/day (default 20; 0 = off) |

### memory / performance (cannot change results)

| flag | variable | prod | meaning |
|---|---|---|---|
| `--cell-chunk` | `CELL_CHUNK` | — | cells per microphysics vmap batch |
| `--tracer-chunk` | `TRACER_CHUNK` | — | tracers per advection batch (0 = all 82 at once) |

### input data locations

| flag | variable | prod | meaning |
|---|---|---|---|
| `--cesm-dir` | `CESM_DIR` | — | root of the CESM h1 tseries archive |
| `--cesm-prefix` | `CESM_PREFIX` | — | CESM case-name prefix of the h1 files |
| `--cesm-suf` | `CESM_SUF` | — | date-range suffix of the h1 files |
| `--tomas-jax-path` | `TOMAS_JAX_PATH` | — | tomas-jax checkout (default models/tomas-jax) |
| `--rrtmgp-path` | `RRTMGP_PATH` | — | jax-rrtmgp checkout (default models/jax-rrtmgp) |

### debug dumps

| flag | variable | prod | meaning |
|---|---|---|---|
| `--dump-premicro` | `DUMP_PREMICRO` | — | dump the pre-microphysics state to this file |
| `--dump-premicro-step` | `DUMP_PREMICRO_STEP` | — | step at which to take that dump |

### launcher-only

| flag | variable | default | meaning |
|---|---|---|---|
| `--gpu` | `GPU` → `CUDA_VISIBLE_DEVICES` | `0` | card to pin the run to |
| `--cuda-driver-lib` | `CUDA_DRIVER_LIB` | *(auto-search)* | directory holding `libcuda.so.1`; empty string disables the search |
| `--preallocate` | `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | grow GPU memory on demand |
| `--dry-run` | — | — | print the resolved environment and exit |
| `--allow-source-tree` | — | — | override the refusal to write into the checkout |
