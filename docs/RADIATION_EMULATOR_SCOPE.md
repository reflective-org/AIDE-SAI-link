# A radiation emulator for AIDE-SAI-link — what it would be, what it would buy, and what comes first

> **2026-10-13, same day:** built. `docs/RADIATION.md` records what was built,
> what was measured and where this note's plan had to change (the Planck term
> is not linearisable; the forcing diagnostic comes from the physics).

Scoping note written 2026-10-13, meant to be read on its own. Numbers tagged
MEASURED come from a script named here with its log; the rest is design.
The radiation *physics* in this repo has not been modified in any of the
work this note builds on, and this note does not propose changing it — it
proposes a second engine beside it, selected by a flag, exactly as
`MICRO_ENGINE=emulator` sits beside the fast microphysics.

---

## 0. The short version

- **Is radiation the main remaining bottleneck?** Not yet, and not for long.
  Today the step is 21.3 s: advection 6.6, microphysics 4.9, **radiation
  8.5**, everything else 1.4. Once the semi-Lagrangian transport is in
  (`docs/TRANSPORT_SL_HANDOFF.md`, 6.6 s → ~0.1 s) the step is ~15 s and
  radiation is 57 % of it. So yes, after transport, radiation is the
  bottleneck — but **more than half of it goes away without any machine
  learning** (§3), and after that the microphysics is the largest stage
  again. The honest ladder is in §6.
- **What the 8.5 s is** (MEASURED, §2): two RRTMGP solves per step
  (2.6 s each — the evolved aerosol and a reference aerosol, their
  difference is the heating anomaly that drives the model's temperature),
  plus ~3 s of binning the reference aerosol from MAM4. The Mie optics
  (bins → band τ, ω, g) cost 0.03 s. Everything an emulator could replace
  is the two-stream solve.
- **Step zero, exact, no ML (§3): cache the reference call.** The reference
  solve and the reference aerosol depend only on the CESM archive hour, not
  on the model state. Computed once per archive hour and stored, they take
  radiation from 8.5 s to ~2.9 s. That is a one-day engineering task.
- **The emulator (§4–5)** then replaces the remaining 2.6 s solve with a
  per-column network: RRTMGP is column-independent, the column is 55 native
  levels × (T, q, O₃, CH₄) plus 24 band levels × 30 bands × (τ, ω, g) of
  aerosol optics in, 24 heating rates + 2 TOA fluxes out, 55,296 columns
  per call. Labels are free: the physics labels 20,000 columns per second.
  The one design decision that matters is **what to emulate**: the
  quantity the model uses is a heating *anomaly* ~10 % of the total
  heating, so the network must target the aerosol-induced perturbation,
  not the full heating rate (§4.2). Target cost ~0.3 s; target accuracy
  set by the 30-day temperature increment and the TOA forcing (§5.3).
- **Order of work:** cache first (§3), then the emulator (§7). The cache
  changes nothing in the physics and is required anyway to make the
  emulator's residual target well-defined.

---

## 1. What the radiation stage does

`src/radiation/radiation.py` (775 lines; read its module docstring first).
Per coupling step (6 h), `coupling.py:2499–2545`:

1. **Aerosol optics.** The 40 TOMAS bins on the 24 band levels (1.2–143 hPa)
   → per-band extinction, single-scatter albedo and asymmetry factor on the
   14 shortwave + 16 longwave RRTMGP bands, from precomputed Mie tables for
   the wet H₂SO₄/H₂O droplet (composition from T and the conserved water
   vapour, 15 wt% nodes; `WET_OPTICS=1`). Also the 550 nm AOD. All of this
   is table lookups and sums — linear in the bin numbers.
2. **RRTMGP two-stream solve** (`models/jax-rrtmgp`, 112 SW + 128 LW
   g-points, clear sky, no clouds) on the **full native CESM column**: the
   55 native levels with p ≥ 1.005 Pa, gases T/q/O₃/CH₄ from the hourly
   archive, CO₂ and N₂O constant, surface temperature and albedo from the
   archive's surface fluxes, solar zenith from the date. Aerosol only on
   the 24 band levels. Returns heating rate [K/s] on the band levels and
   TOA/surface fluxes.
3. **Anomaly mode** (`RAD_MODE=anomaly`, the default): the solve runs
   **twice** — once with the evolved TOMAS bins at T_eff = T_CESM + dT,
   once with the *reference* aerosol (the MAM4 sulfate of the same archive
   hour, binned onto the TOMAS grid) at T_CESM — and the difference
   `hr_anom = hr1 − hr0` integrates the prognostic temperature increment
   `dT_rad`, which the microphysics (`T3d = temp + dT_rad`,
   coupling.py:2300) and the next radiation call see. The TOA aerosol
   forcing diagnostic `arf_toa` is the difference of the two calls' upward
   TOA fluxes, area-averaged and then averaged over a diurnal window
   (`ARF_AVG_H`). `aod550` comes from the evolved call.

Column count 192 × 288 = **55,296**, done in two latitude chunks of 96
rows (`RAD_LAT_CHUNK`), chunks round-robin across GPUs if several are
visible (bit-identical to serial).

Magnitudes on the day-30 production state (MEASURED, `rad_timing.log`):
total heating |hr| median 1.4 K/day, max 35 K/day (the stratopause); the
accumulated anomaly `dT_rad` after 30 days at 10 Tg/yr is 0.5 K zonal-mean
max, 0.8 K point max (`frames_dT` of `bench_chem_hyb1num4_30d`); the TOA
forcing −0.06 W m⁻² at AOD 0.0077 (uncalibrated, see `MANIFEST.md`
standing caveats). So **the signal the model consumes is ~10 % of the
field the solver computes.**

## 2. Where the 8.5 s goes (MEASURED)

`scripts/validation/bench_radiation_split.py`, log
`outputs/emulator_bench/bench_radiation_split.log`: the production driver
built exactly as `coupling.main()` builds it, fed the day-30 state, GPU 4,
median of 3 warm calls.

| piece of ONE `heating()` call | time |
|---|---|
| host prep (archive reads, transposes, wt% field) | 0.07 s |
| aerosol optics, bins → 30 bands × (τ, ω, g), 2 chunks | 0.03 s |
| **RRTMGP two-stream solve, SW + LW, 55,296 columns × 55 levels** | **2.59 s** |
| AOD 550 | 0.00 s |
| whole call | 2.70 s |

In the coupled loop the stage is 8.5 s = 2 × 2.7 s + **~3 s that is not
the solver**: `aer_fill(it1, klevs)` bins the MAM4 reference aerosol onto
the TOMAS grid every step (`bin_mam4`, 4.4 s when timed on CPU alone,
somewhat less on the GPU path) and the numpy diagnostics. That ~3 s and
one of the two solves depend only on the archive hour.

Per column the solve is 47 µs. For reference, the microphysics emulator
network costs 3 µs per cell and the fast microphysics 11 µs per cell.

## 3. Step zero: cache the reference call (exact, no ML)

The reference call is `heating(t, num_ref)` with `num_ref = bin_mam4(t)`:
a pure function of the archive hour `t`. Nothing the model does changes it.
So:

- Precompute, per 6-h archive hour of the years a run can touch, the
  reference heating on the band levels (24 × 192 × 288 float32 = 5.3 MB)
  and the two TOA fluxes it contributes to the forcing diagnostic
  (`sw_up_toa`, `olr`, 2 × 0.2 MB). One year = 1460 samples ≈ 8 GB; store
  under `inputs/cache/rad_ref/<cesm-case>/` (gitignored, documented in
  `inputs/README.md`), keyed by the things that define the reference:
  CESM case, band levels, `WET_OPTICS`, Mie-table version, `g_lw/g_sw`
  files, `AER_SRC`. Build lazily on first use, or in one batch job
  (1460 × 2.7 s ≈ 1.1 h per year per GPU).
- In anomaly mode, read `hr0, d0` from the cache instead of computing them;
  fall back to computing (and writing) on a miss. `RAD_REF_CACHE=path|0`.
- Nothing else changes: `hr1` is still the physics, `hr_anom`, `arf_toa`,
  `aod550` are bit-identical to today (the cached arrays are the same
  arrays the loop would have computed).

Effect: radiation 8.5 → ~2.9 s per step (one solve + optics + prep). This
is also what makes the emulator's target well-posed (§4.2): with `hr0`
exact, every error the emulator makes in `hr1` lands directly in the
anomaly, so the emulator has to be accurate on the anomaly's scale, which
is the point of the residual formulation.

Two other existing knobs, for completeness, both with an accuracy
question attached rather than an engineering one: `RAD_EVERY=2` halves the
cost again by calling radiation every 12 h (the diurnal sampling of the SW
anomaly and the `ARF_AVG_H` window would need checking); and `RAD_LAT_CHUNK`
with more GPUs visible spreads the solve across cards (already supported,
linear speedup, bit-identical).

## 4. The emulator: what it replaces and what it must learn

### 4.1 Scope

Replace the **two-stream solve** of the evolved call — nothing else. Keep
the Mie optics exact (0.03 s, linear, and the emulator stays valid if the
tables or the composition model change), keep the AOD exact, keep the
cached reference exact, keep the TOA diagnostics coming from the same
place. The emulator is a function

```
(gas column, surface, zenith, aerosol band optics on the band levels, dT)  →  (hr on 24 band levels, sw_up_toa, olr)
```

evaluated on 55,296 independent columns. RRTMGP is column-local, so the
network is column-local: no spatial structure to learn, no latitude or
longitude as inputs, and the 55k columns are a batch, not an image.

### 4.2 What to emulate — the decision that matters

The model consumes `hr1 − hr0` (≈ 0.1–1 K/day in the plume) and
`up0 − up1` at TOA (≈ 0.1 W m⁻² against 240). An emulator of the full
`hr1` accurate to 1 % of a 1.4 K/day median field puts 0.014 K/day of
noise into a 0.1 K/day anomaly: 15 %, and it accumulates in `dT_rad`. So
the network must predict the small quantity directly. Decompose the
evolved call as

```
hr1 = hr_clear(T_eff)  +  Δ_aer(optics_evolved; T_eff, gases, zenith)
```

- `hr_clear(T_eff)` is the clear-sky heating at the perturbed
  temperature. `hr_clear(T_CESM)` is archive-only → **cached** (one more
  field in the §3 cache). The dependence on dT (Planck response, |dT| ≲ 1 K)
  is linear to high accuracy: cache the diagonal finite-difference
  sensitivity `∂hr_clear/∂T` per archive hour as well (one extra solve per
  hour with T + 1 K, offline), so `hr_clear(T_eff) ≈ hr_clear(T) + (∂hr/∂T)·dT`
  exactly to first order. **Verify the linearization offline** on the
  day-30 state (one script, minutes); if it is not good enough, make dT an
  input of the network instead.
- `Δ_aer` is the **aerosol radiative heating perturbation**, the thing
  the network learns: zero when the optics are zero (enforce it
  structurally: multiply the output by a smooth function of the column τ,
  or feed the network τ-weighted features so an empty column gives zero),
  SW part proportional to the incoming solar (gate by cos(zenith) and set
  it to zero at night, as the prior radiation-module work did), LW part
  nearly zenith-independent. Two heads.
- TOA fluxes the same way: `sw_up_toa = sw_up_clear + Δ`, `olr = olr_clear + Δ`,
  with the clear parts cached.

The reference call's anomaly then is `hr1 − hr0 = [hr_clear(T_eff) − hr_clear(T)] + [Δ_aer(evolved) − Δ_aer(ref)]`,
where the first bracket is the cached Planck term and `Δ_aer(ref)` is
cached exactly (physics). Only `Δ_aer(evolved)` is emulated.

### 4.3 Inputs and outputs, concretely

Per column:

| input | size | source |
|---|---|---|
| T, log q, log O₃, log CH₄ on the 55 native levels | 220 | archive, known in advance |
| surface T, albedo, cos(zenith), p (fixed, can be dropped) | 3 | archive / date |
| aerosol optics on the 24 band levels: τ, ω, g × (14 SW + 16 LW) bands | 2,160 | exact Mie step |
| dT on the band levels (if the linearization is not used) | 24 | model state |

Outputs: `Δ_aer` heating on 24 band levels [K/day, scaled by the layer
mass], Δ sw_up_toa, Δ olr. ≈ 2,400 → 26.

Cost budget: 0.3 s for 55k columns is 5 µs per column — a 4-layer MLP of
width 512 (≈ 2.5 M parameters, 5 MFLOP per column) is ~0.3 TFLOP per
call, well under 0.1 s on an H100 in bf16 with `torch.compile` or a jitted
JAX MLP. The microphysics emulator's adapter (`src/microphysics/tomas_emulator.py`)
already has the DLPack zero-copy path and the fp16 compiled forward; copy
its pattern.

An alternative input is the 40 bin numbers instead of the 2,160 optics
numbers (the network then learns Mie). Not recommended: more to learn for
no cost saving (the optics are 0.03 s), and it bakes the composition
model into the weights.

### 4.4 Training data

Free and unlimited. The physics labels **20,000 columns per second**
(2.6 s / 55k), and every column it labels is exact. Sources:

1. **Every step of a coupled run**: 55k evolved columns per step; a 30-day
   run gives 6.6 M columns of realistic plume optics at every stage of its
   evolution. A `RAD_SAMPLE_EVERY=N` writer (mirror `MICRO_SAMPLE_EVERY` in
   `src/microphysics/micro_sampler.py`) stores the inputs of §4.3 and the
   physics outputs for a stratified sample (by column τ) of the columns.
2. **Offline augmentation**: the solve is cheap offline, so widen the
   optics distribution beyond what one scenario produces — scale τ by
   0.1–10, vary injection rate and latitude, seasons and years for the gas
   columns and zenith angles. 100 M labeled columns is ~1.4 GPU-hours.
   The microphysics emulator's lesson applies: the failures were
   **coverage** (states the network never saw), not capacity.
3. **Held-out test**: whole steps of a run the training never saw
   (different year, different injection), judged in-loop (§5).

### 4.5 Architecture and training

Start with the simplest thing that can hit the cost: a per-column MLP,
inputs standardised per feature (log for q/O₃/CH₄/τ), outputs the three
residual quantities scaled by their training-set rms, loss = MSE on
`Δ_aer` heating weighted by the layer mass (so it is an energy error) +
MSE on the two TOA deltas. A 1-D convolution over the 24 band levels with
bands as channels is the next step if the MLP's vertical structure is
noisy. Training on 10–50 M columns is minutes to an hour on one GPU.
Metric-aligned loss, tail losses, bigger models: all refuted levers in the
microphysics work (`/data/tomas-emulator/FRONTIER_RESULTS.md`); do not
start there.

## 5. How it is judged

### 5.1 Offline

On held-out columns: rms of the `Δ_aer` heating error relative to the rms
of `Δ_aer` itself (target < 5 %), per level; the same for the TOA deltas;
and the error as a function of column τ (the plume core is where it
matters and where the data are fewest).

### 5.2 In-loop A/B

Mirror `MICRO_REF`: `RAD_ENGINE=emulator`, `RAD_REF=rrtmgp` runs the
physics on the same input every N steps, compares `hr1` cell by cell and
the global `arf_toa`, `aod550`, and writes `rad_bench_<TAG>.npz` with the
wall time of both. The run advances with the engine only. This is how the
microphysics emulator was benchmarked and it found every one of its
problems (`docs/EMULATOR.md`).

### 5.3 The gates that matter

1. **`dT_rad` after 30 days** within 0.05 K of the physics run point-wise
   at the probe level and 0.02 K zonal-mean (the physics anomaly is 0.5–0.8
   K; the microphysics emulator's own effect on `dT` is the reference
   scale: compare `frames_dT` of `bench_chem_hyb1num4_30d` vs `bench_phys30d`).
2. **Global `arf_toa_avg`** within 2 % of the physics run day 10–30.
3. **Microphysics unaffected**: number/mass/r_eff/AOD at day 30 within the
   noise of the microphysics A/B (`compare_runs.py` table).
4. Cost: radiation stage ≤ 0.5 s with the cache and the emulator together.

Note the existing unexplained offset: the production microphysics emulator
run has TOA forcing 7 % weaker than the physics run at the same AOD
(`docs/EMULATOR.md`, standing caveats). That is a microphysics-side
difference in the size distribution, not a radiation error, and the
radiation A/B must be done with the *physics* microphysics so the two
questions do not mix.

## 6. The ladder: what each step buys

Whole-step cost on one H100, seconds per 6-h step (MEASURED for the first two rows; the rest are
the measured stage times with the named stage replaced by its target):

| configuration | advection | micro | radiation | other | step | vs physics (31.0 s) |
|---|---|---|---|---|---|---|
| fast physics (today's reference) | 6.6 | 14.6 | 8.5 | 1.4 | 31.0 | 1.0× |
| **microphysics engine = emulator (today; stage 14.6 → 4.9 s)** | 6.6 | 4.9 | 8.5 | 1.4 | **21.3** | 1.46× |
| + semi-Lagrangian transport (`TRANSPORT_SL_HANDOFF.md`) | ~0.1 | 4.9 | 8.5 | 1.4 | ~14.9 | 2.1× |
| + cached reference radiation (§3, exact) | ~0.1 | 4.9 | ~2.9 | 1.4 | ~9.3 | 3.3× |
| + radiation emulator (§4) | ~0.1 | 4.9 | ~0.3 | 1.4 | ~6.7 | 4.6× |
| + microphysics network forward 2.6 → ~1 s (distil / bf16 / smaller d) | ~0.1 | ~3.3 | ~0.3 | 1.4 | ~5.1 | 6× |

So: radiation becomes the largest stage once transport is fixed, the
exact cache takes most of it, the emulator takes the rest, and then the
microphysics (4.9 s: 2.6 s network forward, 1.2 s routed physics, 1.1 s
bookkeeping) is the bottleneck again, followed by the 1.4 s of "other"
(the BC/polar refresh at 1.2 s is `bin_mam4` on the polar rows and BC
levels — also archive-only and cacheable). A year of simulation at 5 s a
step is 2 h on one GPU.

## 7. Work plan for the agent

1. **Reference cache** (§3). `RAD_REF_CACHE` env + `--rad-ref-cache` in
   `run_prod.py`; a builder script `scripts/radiation/build_ref_cache.py
   --years ... --gpu N`; cache key in a JSON sidecar; `coupling.py` reads
   `hr0, d0` from it in anomaly mode. Gate: a 1-day run with the cache is
   bit-identical to one without (`compare_runs.py` all zeros, `[prof]`
   radiation ≈ 2.9 s). Also cache `hr_clear(T)` and the Planck
   sensitivity while at it (one extra solve each per archive hour), and
   verify the dT linearization on the day-30 state. ~1 day.
2. **Label writer** `RAD_SAMPLE_EVERY=N` + offline augmentation generator
   (`scripts/radiation/label_columns.py`): inputs of §4.3, outputs
   `Δ_aer` heating and TOA deltas, stratified by column τ. ~1 day.
3. **Train** the MLP (`scripts/radiation/train_rad_mlp.py`), offline
   metrics of §5.1. ~1–2 days including the coverage sweep.
4. **Adapter** `src/radiation/rad_emulator.py` with the `heating()`
   signature (returns `hr_band, diag` like `RadiationDriver.heating`),
   `RAD_ENGINE=rrtmgp|emulator`, `RAD_REF` A/B, `rad_bench_<TAG>.npz`,
   figures. ~1–2 days.
5. **30-day A/B** with physics microphysics against `bench_phys30d`
   (gates §5.3); then the full production combination. ~1 day of GPU time.
6. Docs: `docs/RADIATION.md` (new; the engine contract, floors, A/B
   results, caveats — the shape of `docs/EMULATOR.md`), `CONFIGURATION.md`
   rows, `CHANGELOG.md`, `MANIFEST.md`.

Standing constraints: `RAD_ENGINE=rrtmgp` (the default) must stay
byte-for-byte; `models/jax-rrtmgp` is a pinned submodule and is not edited;
the uncalibrated-forcing caveat in `MANIFEST.md` stays until someone
calibrates it, which is a different task.

## 8. Environment and files

- Repo `/data/AIDE-SAI-link` (branch `feat/frame-levels`), python
  `/data/aide_emu_venv` (jax 0.6.2 + torch), files owned by `ali:ubuntu` on
  root-squashed NFS: write via `cat f | sudo -n -u ubuntu tee dest`, GPU
  jobs as `sudo -n -u ubuntu env CUDA_VISIBLE_DEVICES=<n> LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu XLA_PYTHON_CLIENT_PREALLOCATE=false /data/aide_emu_venv/bin/python ...`;
  GPUs 1–3 reserved, use 0, 4–7.
- `src/radiation/radiation.py` — the driver; `RadiationDriver.heating()`
  (line 661) and `heating_anomaly()` (759) are the two entry points;
  `_hr_impl` (596) is the jitted solve; `_aerosol_props_wet` (364) the
  optics. `coupling.py:2499–2545` the call site; `aer_fill` (1525) the
  reference aerosol.
- `scripts/validation/bench_radiation_split.py` — the §2 measurement
  (builds the driver like `validate_radiation.py`, needs the CESM archive
  and a GPU, ~2 min); `test_radiation.py`, `validate_radiation.py` — the
  existing smoke test and the comparison against CESM's own heating and
  extinction (use them on the emulator too).
- Day-30 state for offline work:
  `runs/chemdump/debug_nucleation/premicro_state_step120.npz`
  (`num (40,24,192,288)` #/kg, `T3d, pres3d, rh3d, plev, lat, lon`); hour
  index `C.H0 + 720`.
- Prior art on this box: `/data/RADIATION_MODULE_SPEC.md` and
  `_RESULTS.md` — a *different* problem (identify CESM's radiation from
  archive data with weak constraints, no per-sample labels). Its findings
  carry over as warnings: stratospheric SW was the part that could not be
  identified weakly, and clouds were the LW limit. Here both are moot —
  labels are exact and unlimited, and the solve is clear-sky — but the
  SW/LW split and the zenith gate are worth copying.
- The microphysics emulator's documents are the template for everything
  procedural here: `docs/EMULATOR.md` (engine contract, A/B, refuted
  levers), `src/microphysics/tomas_emulator.py` (adapter), `micro_bench.py`
  (A/B), `micro_sampler.py` (label writer), `scripts/validation/compare_runs.py`
  (30-day table).
