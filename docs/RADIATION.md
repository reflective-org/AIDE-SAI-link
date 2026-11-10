# The radiation seam: reference cache and emulator

What `RAD_REF_CACHE` and `RAD_ENGINE=emulator` are, how they attach to the
model, what was measured, and what to watch. Written 2026-10-13 from the
scoping note `docs/RADIATION_EMULATOR_SCOPE.md`, which has the cost ladder and
the reasoning this document assumes. Numbers tagged MEASURED name the script
and log that produced them. Nothing in the radiation *physics* changed:
`RAD_ENGINE=rrtmgp` with `RAD_REF_CACHE=0` is byte-for-byte the previous code.

## What it is

Per 6-h step the radiation stage runs RRTMGP twice (the evolved TOMAS bins at
`T_CESM + dT_rad`, and the MAM4 reference aerosol at `T_CESM`) and integrates
their difference into the prognostic `dT_rad`. Measured at 8.5 s per step
(`scripts/validation/bench_radiation_split.py`): 2 × 2.7 s of two-stream
solve on 55,296 columns × 55 native levels, plus ~3 s of re-binning the MAM4
reference aerosol onto the TOMAS grid. Two things follow from the fact that
the reference half depends only on the archive hour:

1. **The reference cache** (`src/radiation/rad_cache.py`). The reference solve
   and the re-binning are computed once per archive hour, stored in float64,
   and served from disk afterwards. A hit is the same array the loop would have
   computed, so the run is bit-identical, and the stage drops to one solve.
2. **The emulator** (`src/radiation/rad_emulator.py`). The remaining solve,
   the evolved call, is replaced by two per-column networks evaluated on top of
   the cached clear-sky fields of the hour. Everything around the solve stays
   the physics: the Mie optics (bins → band τ/ω/g on the 24 band levels, 0.03 s),
   the 550 nm AOD, the archive reads, the solar geometry, the diagnostics.

## How it is attached

- `coupling.py` builds the `RadiationDriver` as before, attaches a
  `rad_cache.RefCache` when `RAD_REF_CACHE != 0` (default on), and wraps the
  driver in `rad_emulator.RadEmulator` when `RAD_ENGINE=emulator`. The call
  sites are unchanged: `heating()`, `heating_anomaly()` keep their signatures
  and return contract; the only new call is `rad.ref_ready(t)`, which lets the
  loop skip `aer_fill` (the MAM4 re-binning) when the hour's reference is
  stored. `RAD_ENGINE` is stamped into `PHYS_CFG` (append-only) so checkpoints
  say which engine made them.
- `RAD_REF=rrtmgp` with the emulator runs the physics on the same input every
  `RAD_REF_EVERY` radiation steps and logs the comparison to
  `rad_bench_<TAG>.npz` (`scripts/radiation/plot_rad_bench.py` draws it). The run
  advances with the emulator; this is the in-loop A/B that `MICRO_REF` is for
  the microphysics.
- `RAD_SAMPLE_EVERY=N` writes `rad_sample_<TAG>_s<step>.npz` (the evolved bins,
  `dT_rad`, the hour): the states the emulator is trained from.
- Flags: `run_prod.py --rad-engine --rad-ref --rad-ref-every --rad-ref-cache
  --rad-sample-every --rad-emu-ckpt --rad-emu-anom`; every knob is in
  `docs/CONFIGURATION.md`.

## The reference cache

One `.npz` per absolute archive hour under
`inputs/cache/rad_ref/rad_ref_<key>/h<hour>.npz` (gitignored; `key.json`
beside them). The key hashes everything that determines the fields: the
archive file, the band levels, the optics flags and tables, the gases, the
reference-aerosol source and binning (`coupling.rad_ref_src()`), and the
jax-rrtmgp commit — so a change to any of them gets a fresh directory, never a
stale field. Fields per hour:

| field | what | dtype | written by |
|---|---|---|---|
| `hr0`, `d0_*` | the reference-aerosol solve: heating on the band levels, TOA/surface fluxes, AOD | float64 | the loop on a miss, or the builder |
| `hr_clear`, `hr_clear_sw`, `dc_*` | the zero-aerosol solve at `T_CESM` (total + SW) | float32 | builder, or the emulator on a miss |
| `hr_dT`, `hr_dT_sw`, `ddT_*` | response to a uniform +1 K on the band levels (diagnostic; see below) | float32 | same |
| `net_<ckpt>` | the aerosol network's output on the reference aerosol (`RAD_EMU_ANOM=net`) | float32 | the emulator |

Entries can be partial; a reader asks for the fields it needs and a miss on
those computes and stores them. Batch build:

```
CUDA_VISIBLE_DEVICES=N P_LO_HPA=1 P_HI_HPA=150 INIT_BIN=so4 \
  python3 scripts/radiation/build_ref_cache.py --hours 0:721:6      # ~12 s per hour
```

Run the builder under the same environment as the run (the band levels and
`INIT_BIN` are part of the key — the first attempt here built a 21-level
cache because `driver_fast.py`'s `P_HI_HPA=150` was not in the environment).
A 30-day run's 121 hours take ~25 min; a year ~5 h on one GPU; 24 MB per hour
with the clear-sky fields.

Gate (MEASURED, `scripts/radiation/test_rad_cache.py`, PASS): a served hour
equals the computed one bit for bit in `hr0` and every diagnostic, the
SW/LW-split solve equals the default solve bit for bit, and the serve costs
0.02–0.03 s against 2.8 s to compute. In the loop the anomaly with a cached
reference costs 3.5 s on a quiet GPU (`outputs/rad_emulator/profile_cache_path.py`),
i.e. the evolved solve plus the host work; the 8.5 → ~3 s of the scoping note.

## The emulator

### What it computes

The evolved call is decomposed as

```
hr1(optics, T_CESM + dT)  =  hr_clear(T_CESM)                      cached, exact
                           + [hr_clear(T_CESM + dT) - hr_clear(T_CESM)]   PlanckNet
                           + [hr1(optics, T_eff) - hr_clear(T_eff)]       AerNet
```

and the two TOA fluxes the same way (`sw_dn_toa` and the surface fluxes are
the cached clear-sky values; the model does not consume the surface fluxes).
In anomaly mode the reference term is either the exact cached physics
(`RAD_EMU_ANOM=phys`) or the aerosol network on the reference aerosol at
`dT = 0` (`RAD_EMU_ANOM=net`, the default); the latter is cached per hour and
checkpoint (`net_<digest>`), so a cached hour costs no reference work at all.
With `net` the anomaly is `PlanckNet(dT) + AerNet(evolved) - AerNet(reference)`:
the network's error on the background, where the two aerosols are nearly the
same, cancels instead of landing in `dT_rad` every step.

### Why the Planck term is a network

The scoping note planned to cache `∂hr/∂T` and linearise. That does not work,
and the reason is worth knowing because it is a property of the physics:
RRTMGP's Planck sources at the cell faces come from a WENO5 reconstruction of
the temperature profile (`optics_base.reconstruct_face_values`), whose
smoothness weights move with the local profile. The response to a UNIFORM
warming of the band is exactly linear (−0.0195 K/day per K at 52 hPa for
0.05–1 K probes, MEASURED), but the response to a single-level perturbation
is a smooth curve whose slope changes by 2× between 0.1 K and 1 K, and the
day-30 `dT_rad` is exactly such a profile (0.11 → 0.28 → 0.40 → 0.09 K on
adjacent levels at the injection height). MEASURED,
`scripts/radiation/check_planck_jacobian.py`
(`outputs/rad_emulator/check_planck_jacobian.log`): the full 24×24
finite-difference Jacobian applied to the day-30 `dT_rad` leaves a 36 % rms
error (max 0.026 K/day against a 0.068 K/day term); the row-sum shortcut 85 %;
any 4–12-solve probing of the Jacobian 38–61 %. Reverse-mode AD through the
solve asks for 255 GB for 48 latitude rows. So the term is learned: inputs the
gas column, surface and the dT profile — both raw and divided by its rms, the
SHAPE, without which the standardised raw dT is invisible at the 0.03 K rms
of a 30-day run — output multiplied by rms(dT), so `dT = 0` gives exactly 0.
Two consequences for the model itself: the TOA OLR responds by +0.43 W m⁻² per
K of uniform band warming (MEASURED), the same order as the −0.06 W m⁻² aerosol
forcing the run reports, so the reported forcing already contains the Planck
response of the warmed layers; and `dT_rad` profiles with sharp vertical
structure are integrated by a solver whose response to them is not linear.

### The aerosol network

Inputs (`src/radiation/rad_features.py`, one definition for labels and
inference, 2,437 per column): T, log q, log O₃, log CH₄ on the 55 native levels;
surface T, albedo, cos(zenith); dT on the band levels; log1p(τ/10⁻⁷), ω, g per
band (14 SW + 16 LW) × 24 band levels; the column τ per band. No position, no
time. Output head (`--head mixed`, the one that survived the sweep below): the
LW heating is a per-band, per-level *efficiency* predicted by the network and
contracted with the exact band optical depths (linear-in-τ, local emission and
absorption), the OLR delta likewise with the column τ; the SW heating and the
SW TOA delta are a gated MLP output (non-local: backscattered sunlight raises
ozone absorption in layers that hold no aerosol), the gate being the mean
log column optical depth. Both are exactly zero for an aerosol-free column;
SW rows are exactly zero at night, as in the solver.

### Training data and training

`scripts/radiation/label_columns.py`: takes saved 3-D states (the final
states of earlier runs plus `rad_sample_*` dumps of a 30-day training run at a
different injection — 8 Tg/yr, 15 °N, July start, `runs/raddata_15N_jul`),
pairs each with a random archive hour (hours ≥ 2000, so the first weeks of
1996 are never seen), scales the bins per column by a log-uniform factor in
[0.1, 10] (8 % of the columns set to zero aerosol, 22 % replaced by the MAM4
reference aerosol), scales a state's `dT_rad` per column by U[0, 2.5] capped at
3 K (12 % set to zero), runs the exact physics on the full globe (the evolved
call split SW/LW, the clear call at `T_eff`, the clear call at `T_CESM` from
the cache) and keeps a stratified sample of ~14,000 columns per globe (every
column with AOD ≥ 0.01, half of 0.001–0.01, 12 % of the rest). ~16 s per globe
alone, 38 s beside four other runs. Held-out validation
(`labels/val_phys30d`): the day-30 state of `bench_phys30d` — the scenario of
the in-loop A/B, never in training — at 16 January hours, every column.
`scripts/radiation/train_rad_mlp.py`: residual MLP, per-feature
standardisation, per-output rms scaling, MSE, AdamW + one-cycle cosine, the
whole set on the GPU (`--x-cpu` keeps the inputs in pageable host memory and
moves each batch through a one-batch pinned staging buffer, for a set larger
than the GPU); exports `.npz` weights the JAX adapter evaluates.
**Host memory (2026-10-16).** The loader streams the shards into one
preallocated array, filters rows on the way, standardises the aerosol set in
place and never bulk-pins: ~255 GB (cum head) / ~210 GB (mixed) for the v1–v3
set, and it refuses to start when the set would exceed 80 % of `MemAvailable`.
The previous version held the 160 GB set three to four times over and pinned
253 GB of it; two such trainers (`aer_v6mixed`, `aer_cum_v3_lr1e-4`) took the
1 TB, swap-less node down on 2026-10-15 21:55 UTC (and `aer_v5s1` had died with
SIGBUS at 181 GB on 10-14). `user.slice` is now capped at 880 GB so an overrun
is a clean OOM kill, not a hang. The mean/std are accumulated in float64 since
then, so checkpoints are not bit-identical to the pre-fix recipe.

**v4 data (2026-10-14).** Two more sampled physics runs give TRAJECTORY states
of the two regimes the emulator is used in, rather than day-30 end states:
`runs/raddata_30NS_jan` (the A/B scenario itself, `RAD_SAMPLE_EVERY=2`, 60
states over the month) and `runs/lr1yr_d177_sample` (the year-long `lr1yr`
run resumed from its day-177 checkpoint for 30 days with sampling: AOD
0.07, `dT_rad` to 4.6 K, forcing −2.3 W m⁻², 59 states). Labels
`train_v2jan` and `train_v2yr` (150 globes each, the year set with
`--dt-cap 5`) and `planck_noise_v2` (120 Planck-only globes from both),
3.9 M more columns. A second held-out set, `labels/val_yr_d207`: the day-207
state at 4 of its own hours, every column — the big-plume regime the first
set does not cover. Two things the year states taught the trainer: the ×10
augmentation of a 0.07-AOD state makes AOD > 1 columns whose τ-contracted
outputs blow the batch loss up to 10¹²  at the peak learning rate (never
seen in the v1–v3 trainings; the best-epoch checkpoint survives it but the
run is slowed and the per-output rms scale is dominated by those columns)
— `--max-aod 0.5` drops the 2 % of columns above five times the real year
maximum; and the Planck net at `--lr 1e-3` diverges outright on the larger
set (loss 1–10⁴ from epoch 50) while `--lr 3e-4` trains cleanly and better.

### Network sweep (MEASURED, held-out columns, err/signal rms; `outputs/rad_emulator/ckpt/*.log`, `eval_ckpt_v1_v2.log`)

| aerosol network | data | SW heating | LW heating | plume (AOD > 0.01) | TOA SW up | OLR |
|---|---|---|---|---|---|---|
| gate head (MLP × column-τ gate) | 0.76 M | 21 % | 13 % | 12 % | 11 % | 8.5 % |
| τ-contraction head everywhere | 0.76 M | 60 % | 9.4 % | 18 % | 16 % | 7.2 % |
| τ-contraction + gated non-local term | 0.76 M | 30 % | 8.5 % | 12 % | 16 % | 7.9 % |
| **mixed**: gated SW, τ-contracted LW | 0.76 M | 22 % | 5.4 % | 7.7 % | 13 % | 3.7 % |
| mixed, × cos(zenith) on SW | 0.76 M | 24 % | 5.3 % | 8.6 % | 20 % | 4.7 % |
| mixed, 1024 × 5 | 0.76 M | 23 % | 4.6 % | 7.8 % | 14 % | 3.8 % |
| mixed, 1024 × 5 (`v1`) | 2.66 M | 12 % | 3.8 % | 5.1 % | 7.2 % | 3.2 % |
| **mixed, 1024 × 5, τ-weighted ω/g (`v2`)** | 3.82 M | **7.3 %** | **3.1 %** | **3.5 %** | **5.0 %** | **2.1 %** |
| mixed2 (TOA deltas τ-contracted) + bias penalty | 3.82 M | 8.8 % | 6.5 % | — | 5.2 % | 2.7 % |

Planck network: 41 % (LW) without the shape input, 18 % with it; `v2` (768 × 4,
3.82 M columns): LW 10.9 %, OLR 3.5 %, SW 17 % of a 2e-5 K/day signal.

What the sweep says. Capacity is not the limit (1024 × 5 ≈ 768 × 4 at equal
data); DATA is — every doubling of the labelled set took SW from 22 to 12 to
7 %, as the microphysics emulator's history also found. The physical head
structure matters more than either: a pure τ-contraction breaks the SW part
because SW aerosol heating is non-local, and the τ-WEIGHTING OF THE ALBEDO
INPUTS was worth as much as the data (SW 12 → 7 %): it was found in the loop,
where a fresh polar nucleation burst (100× the reference number in the
ultrafine bins at 5–10 hPa, τ ~ 10⁻¹⁰) put the LW ω inputs 50–100 σ outside the
training distribution for one step — 2 × 10⁻² K/day errors over the whole
cap from layers the physics does not even see. The penalty on the batch-mean
error did not move the held-out TOA bias (it is a generalisation bias, ±1 % of
the signal, changing sign from one training to the next), which is why the
forcing diagnostic is taken from the physics (below).

## How it is judged

### Offline

`scripts/radiation/test_rad_emulator.py` (contract, structural zeros, both
anomaly modes, timing, and the error against the physics on the day-30 state
of `bench_phys30d` at hour 720): PASS with `v2`. The emulated call takes
0.14 s against 2.68 s for the physics on one H200 (19×), the anomaly with a
cached reference 0.18 s; the zero-aerosol, zero-dT call returns the cached
clear-sky field bit for bit; SW rows are zero at night.
`scripts/radiation/eval_ckpt.py` gives the per-AOD-bin error and bias of any
checkpoint on the held-out shards (`outputs/rad_emulator/eval_ckpt_v1_v2.log`).

### In the loop: the 30-day A/B

Four runs of the `bench_phys30d` scenario (10 Tg SO₂/yr, zonal rings at
±30°, 55 hPa, physics microphysics, January 1996 start — none of its states
or hours in the training set), all with the reference cache on:

| run | engine | anomaly reference | GPU |
|---|---|---|---|
| `bench_phys30d_rc` | RRTMGP | exact (cached) | control; reproduces `bench_phys30d` (10-10) to 3 × 10⁻⁵ in number and 3 × 10⁻⁸ in mass, i.e. GPU reproducibility |
| `bench_rademu30d_phys` / `_v2phys` | emulator v1 / v2 | exact (cached) | `RAD_REF=rrtmgp` every 4 steps |
| `bench_rademu30d_net` / `_v2net` | emulator v1 / v2 | networks | same |
| `bench_rademu30d_v3phys` | emulator v3 (production until 2026-10-14) | exact (cached) | `RAD_REF=rrtmgp` every 3 steps, `RAD_EMU_ARF=exact` |
| `bench_rademu30d_v4phys` | emulator v4 (`aer_v4` + `planck_v4`; `aer_v4` later discarded) | exact (cached) | same; the host shared with a training job |
| `bench_rademu30d_v5phys` | emulator v5 (`current`) | exact (cached) | same |

Gates (`docs/RADIATION_EMULATOR_SCOPE.md` §5.3), MEASURED with
`scripts/radiation/compare_dT.py`, `scripts/validation/compare_runs.py`
(`outputs/emulator_bench/compare_rad_v*.png`) and
`scripts/radiation/plot_rad_bench.py` (`outputs/rad_emulator/rad_bench_*.png`):

| gate (§5.3) | target | v1 phys-ref | v1 net-ref | v2 phys-ref | v2 net-ref | v3 phys-ref | v4 phys-ref (aer net later found unstable, see v4/v5) | **v5 phys-ref (production)** |
|---|---|---|---|---|---|---| --- | --- |
| `dT_rad` day 30, point-wise \|error\| rms over the band | — | 0.0025 K (8.2 % of dT rms) | 0.0025 K | 0.0022 K (7.0 %) | 0.0026 K | **0.0018 K (6.0 %)** | **0.0018 K (5.8 %)** | **0.00086 K (2.8 %)** |
| `dT_rad` day 30, point-wise p99 | — | 0.0095 K | 0.0090 K | 0.0074 K | 0.0082 K | **0.0064 K** | **0.0067 K** | **0.0031 K** |
| `dT_rad` day 30, point-wise max | 0.05 K | 0.081 K | 0.069 K | 0.64 K (one column, see below) | 0.78 K (same) | **0.079 K** (5 cells of that column) | **0.031 K** (PASS; the v3 column at 0.03) | **0.0117 K** |
| `dT_rad` day 30, zonal-mean max | 0.02 K | 0.013 K | 0.011 K | 0.011 K | 0.011 K | **0.0097 K** | 0.011 K | **0.0034 K** |
| `arf_toa_avg` day 10–30, networks' own | 2 % | −12 % | −33 % | +14 % | −22 % | +14 % (networks' samples in `rad_bench`) | −18 % (networks' samples) | +6.4 % (networks' samples) |
| `arf_toa_avg` day 30, networks' own | — | −1.6 % | +5.4 % | −6.1 % | +7.7 % | — | −0.5 % | +5.4 % |
| `arf_toa` with `RAD_EMU_ARF=exact` | 2 % | — | — | — | — | **−0.5 %** (instantaneous samples vs the control at the same hours, max 0.0002 W m⁻²; the reported 72-h window reads −0.0526 against the control's 72-h equivalent −0.0529) | as v3 (identical aerosol state) | as v3 (identical aerosol state) |
| number / mass / SO₂ / r_eff / AOD, every day | noise | 0.00–0.01 % | 0.00 % | 0.00–0.01 % | 0.00 % | 0.00–0.01 % | 0.00 % | 0.00 % |
| radiation stage, median s/step | ≤ 0.5 s | 0.73 | 3.8 (first pass) | 0.68 | 1.0 | **0.62** (+2.8 s every 3rd step for the exact forcing) | 0.92 (host shared with a training job; emulated call 0.78 s) | **0.67** |
| heating-anomaly error, run-mean rms / signal rms | 5 % (§5.1) | 62 % | 62 % | 18.6 % | 21.5 % | **15.2 %** | **15.3 %** | **7.9 %** (7.2 % from day 10) |
| heating-anomaly error, day 29 rms / signal rms | — | 20 % | 21 % | 17 % | 19 % | 16 % | 11.9 % | 6.4 % |

The run-mean ratio is dominated by the first days, when the physics anomaly
is 10⁻⁴ K/day and the emulator error is 3–5 × 10⁻⁴; the error itself is flat
in time (figure, top left). The point-wise maximum is ONE column at 16.5 °S,
157.5 °E and its two neighbours (16 of 1.3 M cells above 0.05 K in `v2phys`,
66 in `v2net`, all within three grid cells): from day 14 the emulated `dT_rad`
there climbs at 0.04 K/day while the physics stays at 0.05 K. Diagnosed
offline (`outputs/rad_emulator/probe_outlier.py`): with the same aerosol
state and `dT = 0` the aerosol network matches the physics in that column
to 10⁻³ K/day; with the run's own `dT` — a zig-zag of ±0.1 K at 1–6 hPa, 6 σ
outside the training `dT` distribution, which the emulator's own earlier
error produced — the PLANCK network is wrong by 0.05–0.15 K/day and the
loop feeds on itself. The training `dT` profiles were scaled copies of real
`dT_rad` fields, all smooth; a Planck-only labelling mode (one clear-sky
solve per globe) with per-level noise added to the profiles
(`label_columns.py --planck-only --dt-noise`, `labels/planck_noise_v1`, 240
globes in 50 min) is the fix: the retrained Planck network (`v3`, LW 8.9 %
against 10.9 % offline, max error halved) answers that column's zig-zag `dT`
to 10⁻² K/day, and in the loop (`bench_rademu30d_v3phys`) the column ends the
month at 0.05–0.08 K instead of 0.64 K — still the only place above 0.05 K,
and still the place to watch. Everything else in the table improved with it.

On the forcing row: the `RAD_EMU_ARF=exact` samples agree with the control's
instantaneous forcing at the same hours to 0.5 % (rms 0.0001 W m⁻²), i.e. the
emulated trajectory's forcing IS the physics' to that precision; the
`arf_toa_avg` the run prints is 12 % weaker at day 30 only because its 4 exact
samples span 72 h instead of 24 h while the forcing doubles in ten days.
Compare like with like (`compare_dT.py` prints both).

The microphysics is untouched by construction and the table shows it: number,
mass, SO₂, r_eff and AOD agree with the control to the printed precision at
every day. The heating-anomaly error in the loop is flat in time at
3–5 × 10⁻⁴ K/day rms while the physics anomaly grows to 3 × 10⁻³ K/day
(day-30 ratio 17 %; the run-mean ratio is dominated by the first
days, when the anomaly is below 10⁻³ K/day and the error is not). Where
`dT_rad` is wrong it is wrong in the plume: the zonal-mean error map peaks at
30 °S, 40 hPa, the injection ring one month on.

### v4/v5: trajectory states of both regimes (2026-10-14)

Where v3's in-loop error lives was measured first: the time-mean of the
signed zonal-mean error carries 94 % of its amplitude (a persistent bias,
not noise), above the plume at 1–30 hPa and over the south polar cap, and
offline on `val_phys30d` the three pieces are equal — aerosol-net SW
2.3 × 10⁻⁴, aerosol-net LW 1.8 × 10⁻⁴, Planck LW 2.1 × 10⁻⁴ K/day, which add
to the 3.6 × 10⁻⁴ seen in the loop. Resuming v3 from the day-207 state of
the year-long run gave a 21.5 % anomaly error with a 0.22 K/day maximum on
the first step: that regime was outside its training set. The two
trajectory label sets above were added and the architecture kept.

**A net whose training spiked is damaged even when its best epoch looks
fine.** `aer_v4` (the v3 recipe at `--lr 1e-3` on the larger set) had
batch-loss spikes of 10²–10¹² through its training and a respectable
held-out score (SW 9.8 %, LW 3.4 %); in the 30-day loop it matched v3 and
even removed the 0.079 K column (`bench_rademu30d_v4phys`: max 0.031 K).
Resumed from the day-207 state it ran at 10 % anomaly error for eleven
steps and then returned 674 K/day in one column at hour 5106
(`runs/rademu_v4_d207_dump`, `outputs/rad_emulator/probe_blowup2.log`,
`probe_column.log`): an ordinary column at 22 °S, 356 °E (AOD 0.04,
`dT` ≤ 1.1 K, physics heating 0.02 K/day) whose upper-troposphere humidity
sits at −7 to −16 σ of the training inputs — and on that latitude row 54 of
288 columns were above 5 K/day. `dT_rad` reached −168 K the next step and
NaN the one after. v3 and the lower-learning-rate net give 0.5 K/day on the
same input. Two consequences: `aer_v4` was discarded, and the adapter now
clamps the networks' outputs (`RAD_EMU_CLIP`, ±20 K/day on the heating,
25× that on the TOA deltas, NaN → 0, every touched cell counted and logged
— a no-op on a healthy run, since the physics never exceeds ~8 K/day even
in the ×10-augmented labels), so one column can no longer take a year-long
run down silently.

**v5 = `aer_v4b` (`--lr 3e-4 --max-aod 0.5`, 200 epochs, four early spikes
and none after epoch 80, best epoch 200) + `planck_v4` (`--lr 3e-4`),
`inputs/rad_emulator/v5`.** Offline, err/signal rms (`eval_ckpt.py`,
`outputs/rad_emulator/eval_*_val*.log`):

| held-out set | net | SW hr | LW hr | plume LW | TOA SW up | OLR |
|---|---|---|---|---|---|---|
| `val_phys30d` (day 30, AOD 0.008) | aer v3 → v5 | **7.3 → 3.3 %** | **3.1 → 1.4 %** | 2.7 → 1.2 % | **5.0 → 2.2 %** | 2.1 → 0.9 % |
| | planck v3 → v4 | 15 → 11 % | **8.9 → 7.0 %** | 6.1 → 4.6 % | 18 → 16 % | 4.7 → 5.0 % |
| `val_yr_d207` (day 207, AOD 0.07) | aer v3 → v5 | **20.2 → 1.9 %** | **5.1 → 1.1 %** | 5.0 → 1.1 % | **9.2 → 0.6 %** | 2.7 → 0.6 % |
| | planck v3 → v4 | 10.2 → 3.1 % | **8.5 → 3.4 %** | 8.3 → 3.3 % | 9.7 → 3.1 % | 2.2 → 1.4 % |

The aerosol net's global TOA SW bias on the day-30 state is +0.0013 W m⁻²
against v3's +0.0074 (the ±1 % bias that made the forcing a physics
diagnostic is now ±0.2 %).

**In the big-plume loop** — v3 and v5 each resumed from the day-207 state
of `lr1yr` for 30 days (`runs/rademu_v3_d207`, `runs/rademu_v5_d207`;
`RAD_REF=rrtmgp` every 3 steps; the control is `lr1yr` itself over the same
hours; AOD 0.07, `dT_rad` 0.56 K rms / 4.9 K max, forcing −2.3 to −2.6
W m⁻²), MEASURED (`outputs/rad_emulator/compare_resume.py`):

| day-207 resume, 30 days | v3 | **v5** |
|---|---|---|
| heating-anomaly error, run-mean rms / signal rms | 28.6 % (21 % → 29 %, growing) | **5.6 % (5.3 → 6.1 %, flat)** |
| max heating error at day 30 | 0.26 K/day | 0.073 K/day |
| `dT_rad` rms vs the physics, day 30 | +2.0 % | **+0.1 %** |
| `dT_rad` max, day 30 (physics 5.38 K) | 5.31 K | 5.42 K |
| number burden vs the physics, day 30 | −0.56 % | **−0.03 %** |
| mass / AOD / r_eff | 0.00 / −0.04 / +0.03 % | 0.00 / 0.00 / 0.00 % |
| networks' own instantaneous forcing vs the physics, run mean | −1.3 % | **+0.1 %** |
| reported `arf_toa_avg` (exact samples) vs the physics | within 1.2 % | within 1.2 % |
| clamp events | 0 | 0 |
| radiation stage | 0.7 s (9.0 s while the run also built the cache for these hours) | 0.73 s |

The 5 % target of §5.1 is met to the digit in the regime where v3 was out
of range, and the networks' own forcing is now good enough that the
physics samples are a check rather than the diagnostic.

**In the 30-day loop** (`bench_rademu30d_v5phys`, the v5 column of the
gate table): `dT_rad` error 0.00086 K rms (2.8 % of the field; v3 6.0 %),
p99 0.0031 K, maximum 0.0117 K, zonal-mean maximum 0.0034 K — every
`dT_rad` gate with a 4–7× margin; heating-anomaly error 7.9 % run-mean
(v3 15.2 %), 7.2 % from day 10, 6.4 % at day 30, still above the 5 % of §5.1
because the first days' signal is below the 3 × 10⁻⁴ K/day floor; the
networks' own forcing +6.4 % over days 10–30 (v3 +14.8 %), still outside the
2 % gate early in a run, so `RAD_REF=rrtmgp` every 3 steps with
`RAD_EMU_ARF=exact` stays the production setting; radiation stage 0.67 s;
no clamp events. `current -> v5` as of 2026-10-14.

### v6: toward 1 % (2026-10-14 → 10-17)

Target: `dT_rad` error below 1 % of the field at day 30 (v5: 2.8 %), which
needs the heating-anomaly error at day 30 near 2.5 % (v5: 6.4 %). The
run-mean anomaly metric cannot reach 1 % because the first days' signal is
10⁻⁴ K/day against a flat error floor; it is reported separately.

**Where v5's error sits** (MEASURED on the day-30 state of `val_phys30d`,
per-level rms error in K/day; scratch script `decomp_v5.py`, lost with the
node reboot, to be redone from `eval_ckpt.py`):

| piece | above the plume (1–20 hPa) | in the plume (50–75 hPa) | relative to its own signal |
|---|---|---|---|
| Planck net LW | 1.1 × 10⁻⁴ | 3 × 10⁻⁴ | 7 % overall; **13.6 % on the 92 % of columns with `dT` rms < 0.05 K**; 18 % on smooth profiles |
| aerosol net SW | 4–9 × 10⁻⁵ | 1.5 × 10⁻⁴ | 3.3 % |
| aerosol net LW | 0.5–2 × 10⁻⁵ | 2 × 10⁻⁴ | 1.4 % |
| total | 1.2 × 10⁻⁴ | 4 × 10⁻⁴ | the in-loop floor of 1.4 × 10⁻⁴ at every level |

In the loop the error is flat in time (1.6 × 10⁻⁴ at step 0, 2.1 × 10⁻⁴ at
day 30) while the signal grows from 1.1 × 10⁻⁴ to 2.9 × 10⁻³ K/day, which
is why the run-mean ratio is set by the early days. Per-level biases are
±1–4 × 10⁻⁵ (Planck, alternating sign) and ≤ 10⁻⁵ (aerosol): the error is
no longer a global bias.

**The Planck term is linear where the network fails.** Clear-sky solves at
hour 720 with `T + a·u`, `u` the day-30 `dT_rad` (rms 0.03 K, max 0.8 K),
`a` = 0.01, 0.1, 1 (scratch script `noise_floor.py`, same fate):

| perturbation | response per unit `a`, rms | r(0.01) − r(0.1) | r(0.1) − r(1) |
|---|---|---|---|
| the profile `u` | 2.29 × 10⁻³ K/day | 1.06 × 10⁻⁵ (0.5 %) | 9.3 × 10⁻⁵ (4 %) |
| uniform +1 K | 0.137 K/day | 1.7 × 10⁻³ (1.2 %) | 1.1 × 10⁻² (8 %) |

Solver noise at the in-loop scale is ~10⁻⁷ K/day, so at 30-day amplitudes
the Planck term is 96 % a state-dependent Jacobian times `dT`; the WENO5
face reconstruction (`weno5_node_to_face_for_rrtmgp` in
`models/jax-rrtmgp/rrtmgp/interpolation.py`) is the non-linear part and
matters only at K-scale `dT`. A plain MLP learns that bilinear product
badly (7–14 %). The earlier finite-difference Jacobian check
(`check_planck_jacobian.log`, 35 % error) used a 1 K step, already in the
non-linear regime; it was not evidence against a linear model at small `dT`.

**New heads** (`train_rad_mlp.py --head`, `rad_emulator.py`; results below):

- `jac` (Planck): the net emits a 50 × 24 Jacobian `J(state, dT)` and a
  residual `R`; `y = J · (dT / g_scale) + R · g² / (1 + g)` with
  `g = rms dT / g_scale`. Linear at small `dT`, free at large `dT`. Its
  gate vector `G` is `[dT / g_scale, g]` (25 floats).
- `cum` (aerosol): LW as `mixed` (τ-contracted); SW heating = per-band,
  per-level efficiencies contracted with five depths built on the fly from
  the SW band optics — local absorption `A = τ(1 − ω)`, cumulative `A` above
  and below the level, cumulative backscatter `B = τω(1 − g)/2` above and
  below — plus the gated free term; `swup = Σ_b e_b · B_col,b` + gated free
  term. `G` = `tau_feats` + raw SW `ssa` + raw SW `g` (1423 floats, pinned on
  the host under `--x-cpu`); the adapter captures `ssa_raw`/`g_raw` before
  the τ-weighting. (Native radiation column: 55 levels, krad 15–69 of the
  70 CESM levels; the 24 coupled band levels are rows 12–35 of it.)

**Plan, in order of expected payoff.** (1) Planck `jac` head: 7 % → 2–4 %
(the solver's own non-linearity at 30-day amplitudes is 4 %; the residual
term covers the year regime) — the largest single term above the plume.
(2) Aerosol `cum` head + doubled labels (`train_v3jan`, `train_v3yr`, 300
globes each): SW 3.3 % → 1.5–2 % (every doubling so far halved SW: 22 → 12
→ 7.3 → 3.3 %). (3) Seed ensembles of 2–3 nets: typically 20–30 % more,
~0.1 s per extra net in the stage. (4) If the small-`dT` Planck regime still
lags: weight the loss by 1/(rms `dT` + 0.05 K)² or add a low-noise
`--planck-only --dt-noise 0.01` label set. Not promising: capacity alone,
`--lr 1e-3` (spikes).

**Planck `jac` head, MEASURED (2026-10-15).** `ckpt/planck_jac.npz`: the v4
recipe (768 × 4, `--lr 3e-4`, 300 epochs, the same 11.1 M columns as
`planck_v4`) with `--head jac`; best epoch 297; loss clean throughout, 13 s
per epoch; ahead of `planck_v4`'s trajectory from the first epoch (LW 18.5 %
against 32.6 % at epoch 1, 11.4 % against 14.0 % at epoch 78). Held-out
err/signal rms (`eval_ckpt.py`, `outputs/rad_emulator/eval_planck_jac_val*.log`):

| held-out set | Planck net | SW hr | LW hr | TOA SW up | OLR |
|---|---|---|---|---|---|
| `val_phys30d` (day 30, AOD 0.008) | v5 → jac | 10.6 → **6.9 %** | 7.0 → **4.3 %** | 15.5 → 14.1 % | 5.0 → **1.3 %** |
| its AOD 10⁻³–10⁻² columns (90 % of the set) | v5 → jac | 13.6 → 8.2 % | 9.4 → **5.3 %** | 19.5 → 17.9 % | 6.7 → **1.6 %** |
| `val_yr_d207` (day 207, AOD 0.07) | v5 → jac | 3.1 → 3.5 % | 3.4 → 3.2 % | 3.1 → 3.6 % | 1.4 → 1.3 % |

The gain is where §3 predicted it: the small-`dT` regime of the 30-day run,
where the Planck term is linear (LW 7.0 → 4.3 %, OLR 5.0 → 1.3 %, biases at
the 0.1 % level; the v5 net's biases on those columns were 4–14 % of the
mean). The big-plume regime, K-scale `dT` where the solver itself is
non-linear, is unchanged to ±0.4 points: the residual term carries it as the
gated MLP did. Against the plan's 2–4 % the result is 4.3 % on the full
day-30 set. Not yet done for this net: `test_rad_emulator.py`, the 30-day A/B
and the day-207 resume with `aer` = v5 and `planck` = jac as the v6 candidate.

**Aerosol `cum` head at `--lr 3e-4`: stopped at epoch 53 of 200.** Epoch-mean
loss above 1 in 17 of 53 epochs, up to 1.7 × 10⁹ (`aer_v4b` at the same
learning rate: 4 such epochs, 1.8 × 10⁸ at most), and behind `aer_v4b` at the
same epoch (SW 14.8 % against 12.5 %, LW 4.7 % against 3.3 %). The head
contracts five depths × SW bands × levels per output level, an order of
magnitude more τ-weighted terms than `mixed`, so its loss tail is heavier at
the same step size; by the v4 rule (a net whose training spiked is damaged)
the run was discarded (`ckpt/aer_cum.log`). Relaunched at `--lr 1e-4` on the
v1–v3 labels (`aer_cum_v3_lr1e-4`), beside the pure data-doubling test
`aer_v6mixed` (the v5 recipe unchanged on the v1–v3 labels, 16.4 M columns
against v5's 8.0 M).

**State on 2026-10-15 evening.** Everything launched on 2026-10-14 was killed
by the `/data` outage (NFS `EPFNOSUPPORT`; the client-side evidence for the
storage admin is §7 of
a local status note of 2026-10-14, not in this repo) and the GPU host was
rebooted at 23:00 UTC, after which the host `/dev/nvidia*` nodes had to be
recreated by hand (local script, not in this repo). The label sets were
resumed with `--start` at the first missing globe and a new seed (so the
draws do not repeat the globes already on disk) and are complete:
`train_v3jan` and `train_v3yr`, 300 globes each. Do not start two label jobs
in the same second after a reboot: `jax-rrtmgp`'s optics loader
(`data_loader_base.py`) creates `/tmp/netcdf/data` without `exist_ok` and
the second job dies with `FileExistsError` (harmless once the directory
exists). Every v6 job is one command, `sudo -u ubuntu
scripts/radiation/launch_v6.sh <stage> <gpu>` (stages `labels-jan`,
`labels-yr`, `planck-jac`, `planck-s1`, `aer-s1`, `aer-cum`, `aer-cum-v3`,
`aer-mixed-v3`, `aer-cum-v3-lr1e-4`, `smoke-jac`, `smoke-cum`; every launcher
sets `HDF5_USE_FILE_LOCKING=FALSE`, without which the archive open hangs in D
state under the NFS lock manager). Running: `aer_v6mixed` (GPU 6) and
`aer_cum_v3_lr1e-4` (GPU 1), ~5 h each. Then: `eval_ckpt.py` on both held-out
sets, the better aerosol net + `planck_jac` → `inputs/rad_emulator/v6cand`,
`test_rad_emulator.py`, the 30-day A/B (`run_prod.py --rad-engine emulator
--rad-ref rrtmgp --rad-ref-every 3 --rad-emu-ckpt inputs/rad_emulator/v6cand`)
and the day-207 resume (`outputs/rad_emulator/compare_resume.py`); the seed
ensembles after that.

**v6 RESULTS (2026-10-16/17).** Both 16.4 M-column aerosol trainings
finished after the node hang (`g265` froze under two old-loader trainers,
see "Standing caveats" below and the trainer's host-memory
note in `launch_v6.sh`). Held-out err/signal rms (`eval_ckpt.py`,
`outputs/rad_emulator/eval_v6aer_*.log`, `eval_v6ens2_*.log`,
`eval_planck_v3_*.log`); day 30 = `val_phys30d`, day 207 = `val_yr_d207`:

| net | SW hr d30 / d207 | LW hr d30 / d207 | TOA SW up | OLR |
|---|---|---|---|---|
| v5 aerosol (`aer_v4b`, mixed head, 8.0 M cols) | 3.3 / 1.9 % | 1.4 / 1.1 % | 2.2 / 0.6 % | 0.9 / 0.6 % |
| `aer_cum_v3_lr1e-4` (cum head, 16.4 M) | **1.9 / 1.7 %** | **0.9 / 0.8 %** | 1.5 / 0.4 % | 0.9 / 0.4 % |
| `aer_v6mixed` (mixed head, 16.4 M: the pure data-doubling test) | 2.2 / 1.4 % | 1.0 / 0.8 % | 1.7 / 0.4 % | 0.6 / 0.4 % |
| cum + mixed, mean of the two (`v6ens2`) | **1.6 / 1.4 %** | **0.8 / 0.7 %** | 1.2 / 0.3 % | 0.6 / 0.4 % |
| Planck `planck_jac` (11.1 M cols) | 6.9 / 3.5 % | 4.3 / 3.2 % | 14.1 / 3.6 % | 1.3 / 1.3 % |
| Planck 2-seed mean (`planck_jac` + `planck_jac_s1`) | 6.5 / 3.2 % | 3.8 / 2.9 % | 13.7 / 3.2 % | 1.3 / 1.2 % |
| Planck `planck_jac_v3` (same net, 19.5 M cols) | **5.2 / 2.1 %** | **3.3 / 2.4 %** | 13.2 / 2.3 % | — |

Doubling the labels did for SW what every doubling before it did (3.3 →
2.2 % with the head unchanged); the `cum` head adds another 0.3 points on
the 30-day set and nothing on the big-plume set, where the τ-contracted
`mixed` head was already at 1.4 %. The two heads' errors are largely
independent, so their mean is 20 % better than either for one more
forward pass (+0.03 s). For the Planck net the lever is data, not seeds:
two seeds of the same recipe are strongly correlated (10 % from the
mean), 1.75× the columns gives 20 % on both sets (error ∝ N^-0.4 so far),
and averaging the data-doubled net with the older nets is WORSE than the
new net alone (LW 3.3 → 3.3 / 2.4 → 2.5 %: the weaker members drag the
mean). Where the Planck residual sits (`planck_by_dt.py`, scratch): the
relative LW error is 3–4 % at every `dT` amplitude from 0.05 K to 1 K
(the columns carrying 53 % / 88 % of the squared error on the two sets),
peaks at 7 % only for 0.005–0.02 K profiles that hold 1 % of the error,
and the zig-zag columns are not worse — a ~3 % error in the
state → Jacobian map itself, not the small-`dT` regime; the planned
1/(rms `dT`)² loss weighting would polish the 1 %-share columns and was
not tried.

**Checkpoint sets** (`inputs/rad_emulator/`): `v6cand` = cum + `planck_jac`
(= `current` since 2026-10-17, user decision); `v6mix` = mixed +
`planck_jac`; `v6ens2` = cum + mixed + `planck_jac`; `v6ens3` = `v6ens2` +
`planck_jac_s1`; `v6ens4` = cum + mixed + `planck_jac_v3`. A directory
holding `aer.npz` + `aer_*.npz` (and/or `planck.npz` + `planck_*.npz`) is
loaded as a mean-of-members ensemble (`EnsembleNet`/`load_net` in
`rad_emulator.py`; `eval_ckpt.py --aer-ens/--planck-ens`); the startup line
says "N-member ensemble". All pass `test_rad_emulator.py`.

**In the 30-day loop** (`compare_dT.py ffsl30d_K24brack <runs>`; the
reference is the FFSL-K24 physics run, every emulator run identical to it
except the radiation engine; `RAD_REF=rrtmgp` every 3rd step as the in-run
A/B; gates: point-wise 0.05 K, zonal-mean 0.02 K, forcing 2 %):

| run | `dT_rad` error rms, K / % of field | p99 / max, K | zonal-mean max, K | heating-anomaly err/signal, run mean / days 25–30 | radiation call, s |
|---|---|---|---|---|---|
| v5 (`e2e_rad30d`) | 0.00093 / 3.1 % | 0.0034 / 0.012 | 0.0036 | 7 % / 6.7 % | 0.73 |
| `v6cum` (= `current`) | 0.00031 / **1.0 %** | 0.0012 / 0.026 | 0.0012 | 4.0 / 3.8 % | 0.51 |
| `v6mix` | 0.00031 / 1.0 % | 0.0012 / 0.008 | 0.0011 | 4.2 / 3.9 % | 0.51 |
| `v6ens2` (2 aerosol nets) | 0.00027 / 0.9 % | 0.0010 / 0.013 | 0.0010 | 3.6 / 3.5 % | 0.54 |
| `v6ens3` (+ 2 Planck seeds) | 0.00025 / **0.8 %** | 0.0010 / 0.012 | 0.0009 | 3.4 / 3.2 % | 0.69 |
| `v6ens4` (2 aerosol + `planck_jac_v3`) | 0.00025 / **0.8 %** | 0.0009 / 0.013 | 0.0009 | **3.2 / 3.0 %** | 0.57 |
| `v6ens5` (3 aerosol: + seed-1 `cum` + `planck_jac_v3`) | 0.00023 / **0.7 %** | 0.0009 / 0.009 | 0.0009 | 3.0 / 2.9 % | 0.69 |

The 1 % target is met by every v6 set, and the ensembles buy another 10–20 %:
the second aerosol member costs 0.03 s for 0.1 points, the second Planck
member 0.15 s for 0.1 (the `jac` head emits a 50 × 24 Jacobian per column),
so `v6ens2` is the value point under the 0.5 s stage budget and `v6ens3`
the accuracy point — until `v6ens4`: the data-doubled Planck net buys what
the second Planck seed bought (anomaly 3.6 → 3.2 %, per-level 2.2–6.0 %
against 2.5–7.6 %) for 0.03 s instead of 0.15 s, and is the best set on
every gate; its trajectory is identical to `v6ens3` to 0.03 % in
`dT_rad` rms. A third aerosol member (the seed-1 `cum` net, `aer_cum_v3_s1`;
held-out SW 1.4 / 1.4 % for the three against 1.6 / 1.4 % for the pair —
for the aerosol nets seeds DO help, unlike the Planck net) takes the
max-error column down to 0.009 K and the anomaly to 3.0 %, but a second
`cum` member costs 0.12 s (its five-depth contraction is the expensive
part; the `mixed` member cost 0.03 s), so `v6ens4` stays the recommendation. The late-run absolute error now peaks IN the plume
(43–74 hPa, 2.4–2.8 × 10⁻⁴ K/day, 3 % of the signal there), the relative
error is largest at 1–3 hPa and 88–143 hPa (5–8 %), and the zonal-mean
error is flat in latitude (0.2–0.3 × 10⁻⁴ K), so relative to the signal the
polar caps dominate. The one outlier is the 16.5° S / 157.5° E column of
`v6cum` at 1–7 hPa (0.026 K, level-alternating; the column that ran away
under v2): on the physics state or with `dT = 0` both aerosol nets are
exact there, but on the run's own state the physics itself answers the
level-alternating `dT` with a level-alternating heating (±2–4 × 10⁻³ K/day
at 1–3 hPa) and `planck_jac` reproduces it to 10⁻³ — a self-sustaining mode
of the clear-sky Planck response (the WENO5 non-linearity), seeded by the
`cum` net's early error in that column and then maintained consistently;
the per-step error there is flat over the run, it is halved under the
ensemble (the `mixed` member has none), and `v6mix` has no such column
(`outputs/rad_emulator/probe_v6col.py`). `arf_toa_avg` in the first three
v6 runs reads −11 % against the physics for the window reason explained
under "The forcing diagnostic" below (they ran with `RAD_EMU_ARF=exact`);
with the networks' forcing (`emu`, the default since 2026-10-17) the same
runs read −0.3 to −0.4 % (gate 2 %), and `rad_bench`'s sampled ARF agrees
to the noise of its 0.017 W m⁻² mean (cum +0.6 %, ens2 −2.1 %).

**Pure-network mode** (`e2e_rad30d_v6net`: `v6cand`, no `RAD_REF` physics
call, `RAD_EMU_ANOM=phys` against the exact reference cache, forcing from
the networks every step): the trajectory is identical to the
`RAD_REF`-every-3 run to the last digit (that physics call is diagnostic
only), the networks' TOA forcing at matched hours is within 0.5 % of the
physics over days 20–30 (per-sample rms 1.1 %; days 1–10 are meaningless at
10⁻⁴ W m⁻²), `arf_toa_avg` −0.3 %; radiation stage MEAN 0.71 s against
1.45 s with the every-3rd-step physics call (3.04 s physics-only), median
0.70 s because the forcing costs a second network pass. `RAD_REF` is now an
A/B instrument, not a crutch.

**Day-207 resume** (`runs/rademu_v6_d207`: `v6cand` restarted from the
`lr1yr_d177_sample` checkpoint at hour 5034, 30 days against the `lr1yr`
physics continuation; `outputs/rad_emulator/compare_resume.py rademu_v5_d207
rademu_v6_d207`): heating-anomaly err/signal 4.9 % flat (v5 5.6 %, v3
28.6 %), max heating error at day 30 0.051 K/day (v5 0.073), `dT_rad` rms
+0.0 % (v5 +0.1 %), `dT_rad` max 5.375 against 5.384 K physics (v5 5.415),
N burden −0.03 % (same as v5), networks' ARF −2.453 against −2.453 W m⁻²
(+0.0 %; v5 +0.1 %), 0 clamp events, call 0.66 s. On the probe level
(51.7 hPa) against `lr1yr` at matched frame hours the field error is
0.28 / 0.38 / 0.37 % of the field at +7 / +15 / +29 days (saturating;
v5 0.34 / 0.53 / 0.49 %), max 0.048 K (v5 0.081), zonal-mean max
0.008–0.012 K (v5 0.016). The big-plume regime gains 25–30 %, less than
the 30-day regime (its held-out SW error was already 1.9 %), and sits well
under 1 % of the field either way. The same resume with `v6ens4`
(`rademu_v6ens4_d207`): anomaly err/signal 3.7 % flat (`v6cand` 4.9), max
heating error at day 30 0.034 K/day (0.051), networks' ARF −2.454 against
−2.453 W m⁻², `arf_avg` equal to the physics at every checkpoint (networks'
forcing default), probe-level field error 0.20 / 0.30 / 0.30 % of the field
at +7 / +15 / +29 days (area-weighted; `v6cand` 0.25 / 0.37 / 0.35, v5
0.32 / 0.53 / 0.48), max 0.050 K (0.054 / 0.092), call 0.75 s on the
big-plume state (0.66) — another 25 % in this regime too.

**One year with `v6ens4`** (`runs/rademu1yr_v6ens4`, 2026-10-17/18: the
scenario of `ffsl1yr_K24` with physics microphysics and the radiation
emulator, `RAD_REF` every 3rd step as the A/B, resumed once from its
checkpoint when `/data` filled; `compare_runs.py ffsl1yr_K24
rademu1yr_v6ens4 --days 30 90 180 270 365`): burdens, r_eff and AOD 0.0 %
all year; TOA forcing +0.2 / +0.1 / +0.1 / −0.1 / +0.2 % of −0.06 / −0.62 /
−1.83 / −2.70 / −2.92 W m⁻²; probe-level `dT_rad` field error 0.52 / 0.33 /
0.32 / 0.31 / 0.70 % of the field (max 0.007 → 0.093 K); day-365 state rms
0.0054 K = 0.8 % of a 4.4 K field, p99 0.020 K, max 0.137 K, zonal-mean max
0.040 K. The relative 1 % target holds for the whole year; the fixed 30-day
gates (0.05 K point-wise, 0.02 K zonal-mean) are exceeded at year scale by
the plume's absolute size and should be read as relative there. The in-run
heating-anomaly error is 3.2–3.7 % to day 270 and then creeps to 4.4 %
(days 270–330) and 5.4 % (330–365), with three single-sample outliers
(0.23 / 0.29 / 0.71 K/day at the 36–62 hPa plume levels, no persistence)
and the field error moving from the NH plume core to the SH subtropics
after day 345 — a coverage edge (the year-end state, AOD 0.095, r_eff
310 nm, lies beyond the latest training state, day 207 at AOD 0.07), not a
feedback: forcing stays within 0.3 % every quarter. Late-year states as
labels (`coupled_state_*_ckpt.npz` is a valid `label_columns.py --states`
input) are the next data lever for the aerosol nets.

**Planck data: a mixture optimum, not a volume law (2026-10-17/18).** Four
more Planck `jac` trainings, same net and recipe, differing only in the
label mix (held-out LW err/signal, day 30 / day 207):

| training set | columns | Planck-only share | LW |
|---|---|---|---|
| `planck_jac`: v1–v2 aerosol labels + `planck_noise_v1/v2` | 11.1 M | 28 % | 4.3 / 3.2 % |
| `planck_jac_v3`: v1–v3 aerosol labels + `planck_noise_v1/v2` | 19.5 M | 16 % | **3.3 / 2.4 %** |
| `planck_jac_v3nn`: v1–v3 aerosol labels only | 16.4 M | 0 % | 5.0 / 3.4 % |
| `planck_jac_v5`: v3 set + 168 noise-free Planck-only globes (`planck_smooth_v1`) | 24.1 M | 31 % | 4.2 / 3.0 % |
| `planck_jac_v4`: v3 set + 600 noisy Planck-only globes (`planck_noise_v3`, deleted) | 36.3 M | 55 % | ≈ 8 % (stopped at epoch 214, 2.3× v3 at every epoch) |

The net learns the state → Jacobian map from the aerosol-labelled
real-trajectory columns (whose Planck target is the residual of two full
solves); a small share of zero-aerosol Planck-only columns pins the
clear-sky and zig-zag behaviour, a large share — noisy or smooth — shifts
the per-feature and per-output normalisation toward columns whose optics
inputs are identically zero and degrades the net. `planck_jac_v3` stays the
production Planck net. The next data lever for all three nets is more
real-trajectory labels, in particular late-year states, with the
Planck-only sets held near 15 % of the columns.

**Open on 2026-10-18.** Late-year labels and a retrain of `cum`/`mixed`/`jac`
on them (the one-year run's coverage edge above); needs ~50 GB on `/data`,
which was full on 2026-10-17 (storage admin). Recommendation standing with
the user: `current → v6ens4`, strictly better than `v6cand` on every gate at
+0.06 s per call, and verified over a full year.

### The forcing diagnostic

**`arf_toa_avg` window artifact (2026-10-16).** The trajectory diagnostic is the
mean of the last `ARF_AVG_H/6` = 4 FINITE instantaneous ARF samples. A physics
run samples every step, so the window is 24 h; an emulator run with
`RAD_REF=rrtmgp RAD_REF_EVERY=3` has a finite sample every third step, so the
same window spans 72 h, and while the forcing is still growing (day 30 of the
30 N/S rings: -0.048 to -0.065 W/m² over two days) the emulator run's average
reads ~12 % weaker than the physics run's at the same hour. The exact samples
at the same steps agree to 0.5 % (`e2e_rad30d` vs `ffsl30d_K24brack`, v5 and
v6 alike). Compare forcing between runs with the per-sample `arf_toa` at
matching hours, or the `rad_bench` ARF, not `arf_toa_avg`.

**v6 in the loop (2026-10-16).** `runs/e2e_rad30d_v6cum` (`inputs/rad_emulator/v6cand`
= `aer_cum_v3_lr1e-4` + `planck_jac`) and `runs/e2e_rad30d_v6mix` (`v6mix` =
`aer_v6mixed` + `planck_jac`), 30 days vs `e2e_rad30d` (v5): heating-anomaly
err/signal 7 % → 4 % (both), hr1 3.9 → 1.9 / 2.0 %, sampled ARF +4 % → -1 / 0 %,
call 0.64 → 0.51 s; aerosol trajectory unchanged (0.0 %). Held-out
(`eval_ckpt.py`, `outputs/rad_emulator/eval_v6aer_*.log`), day 30 / day 207:
v5 (= `aer_v4b`) SW 3.3 / 1.9 %, LW 1.4 / 1.1 %; cum SW 1.9 / 1.7, LW 0.9 / 0.8,
swup 1.5 / 0.4, OLR 0.9; mixed SW 2.2 / 1.4, LW 1.0 / 0.8, OLR 0.6. The
trainer's own `_log.json` val metric is not comparable across runs; quote
`eval_ckpt.py`. `current` -> `v6cand` since 2026-10-17 (user decision); the day-207 resume and
the seed ensembles are still open.

The networks reproduce the aerosol's TOA effect (0.7 W m⁻² mean over the
daylit globe) to 5 % rms with a ±1 % global bias whose sign changes from one
training to the next (v1 −0.007, v2 +0.008, the bias-penalised variant
−0.009 W m⁻²). The forcing the run reports is the global mean of
*evolved minus reference*, 3–4 % of that effect, so the bias is 10–30 % of it
in the first weeks and a few per cent at day 30 (−0.060 W m⁻² in the control;
v1: −1.6 % with the exact reference, +5.4 % with the network reference; the
day 10–30 means are 12–33 % off). No training lever moved this — it is a
generalisation bias on a quantity whose relevant part is below the network's
resolution — so the diagnostic is taken from the physics instead:
`RAD_EMU_ARF=exact` (the default under `RAD_REF=rrtmgp`) reports `arf_toa`
from the A/B samples only, the other steps report NaN and the trailing window
skips them; with `RAD_REF_EVERY=3` the samples cycle through the four local
times, so the 24-h window is still a diurnal mean (3 days wide) for 0.9 s per
step. Verified on a 3-day run (`runs/rademu_arf_smoke`). The prognostic
`dT_rad` is the emulator's throughout; only the reported flux is exact.

### Cost

Per 6-h step, median over the 30-day runs on one H200 while six other jobs
shared the box (so the other stages are inflated; the radiation stage is
GPU-bound and clean):

| configuration | radiation stage | note |
|---|---|---|
| physics, no cache (`bench_phys30d`, 10-10) | 8.44 s | 2 solves + MAM4 re-binning |
| physics + reference cache (`bench_phys30d_rc`) | 11.1 s under load, **3.1 s** quiet | 1 solve; `outputs/rad_emulator/profile_cache_path.py` |
| emulator, exact reference (`_phys`) | **0.73 s** | 1 network call + optics + host prep |
| emulator, network reference (`_net`) | 3.8 s first pass, ~1 s thereafter | first pass re-bins MAM4 to store the per-hour network reference |
| emulator + `RAD_REF_EVERY=3` exact forcing | ~1.6 s | 0.9 s of physics amortised |

The scoping note's ladder had radiation at ~0.3 s; the 0.7 s is host work
(archive reads, transposes, the wt% field, 0.07 s), the optics (0.03 s), the
chunk loop and the cache read; the networks themselves are ~0.1 s. The
whole-step numbers from these runs (27–38 s) are not comparable to the
scoping note's 21 s because of the shared box; the stage numbers are.

## Standing caveats and what is next

- **`dT_rad` gate.** (v4 met every `dT_rad` gate in the 30-day loop, the
  point-wise maximum included — 0.031 K — and then blew up in the big-plume
  regime, see v4/v5; the rest of this bullet is the v3 state.)
  v2 meets the zonal-mean gate and the point-wise gate at
  the 99th percentile (0.007 K against 0.05) but not at the maximum, because
  of the one runaway column above. That is a feedback through the Planck
  network on a `dT` shape it never saw; the noisy-profile retraining addresses
  the coverage, and `RAD_REF=rrtmgp` every few steps is the instrument that
  finds the next such column (its per-step maximum error jumps a decade when
  one appears). Away from it the error is flat in time at 3–5 × 10⁻⁴ K/day —
  a persistent small bias on the plume columns (3.5 % of the plume heating
  offline), not accumulating noise; more labelled plume columns at day 20–30
  of a ±30° run would be the next lever, since the sweep says data, not
  capacity.
- **The forcing is a physics diagnostic**, not an emulated one (above).
- **Out-of-distribution columns.** The one found (a polar nucleation burst)
  is now harmless by construction (τ-weighted albedo inputs, ±10 σ input
  clip). The `RAD_REF=rrtmgp` A/B is the way to find the next one; keep it on
  (every 3–4 steps costs under 1 s per step) until the emulator has run a
  year.
- **Scope.** The networks were trained on this archive, this band, these Mie
  tables and gases; the cache key refuses a stale cache but nothing refuses a
  stale network. `RAD_EMU_CKPT` names the weights; retrain when any of those
  change (`label_columns.py` + `train_rad_mlp.py`, ~4 h for the data, ~1 h to
  train, on one GPU).
- **The physics itself**: the Planck response to a sharp `dT_rad` profile is
  non-linear at the 0.5 K scale through the WENO5 face temperatures, and the
  stratosphere's OLR response to its own warming (+0.43 W m⁻² K⁻¹) is in the
  reported forcing. Neither is an emulator issue; both were invisible before.

## Files

```
src/radiation/rad_cache.py          the per-hour cache (keys, partial entries, atomic writes)
src/radiation/rad_cache_build.py    clear-sky / reference entry builders shared by the scripts
src/radiation/rad_features.py       the one feature/target definition
src/radiation/rad_emulator.py       RadEmulator: JAX networks, heating()/heating_anomaly(), A/B, forcing diagnostic
src/radiation/radiation.py          split-mode solve, zero-aerosol call, cache-aware reference(), cache_key()
scripts/radiation/build_ref_cache.py           batch-build the cache for a range of hours
scripts/radiation/test_rad_cache.py            gate: bit-identical cache, split solve
scripts/radiation/check_planck_linearization.py / check_planck_jacobian.py   why the Planck term is a network
scripts/radiation/label_columns.py             physics-labelled columns from saved states
scripts/radiation/train_rad_mlp.py             the trainer (heads, gates, export)
scripts/radiation/eval_ckpt.py                 offline error/bias of a checkpoint by AOD bin
scripts/radiation/test_rad_emulator.py         gate: adapter contract, timing, day-30 error
scripts/radiation/compare_dT.py                gate 1: dT_rad after 30 days
scripts/radiation/plot_rad_bench.py            the A/B figure
inputs/rad_emulator/<version>/{aer,planck}.npz weights (gitignored; `current` -> the production set)
inputs/cache/rad_ref/<key>/h*.npz              the cache (gitignored)
outputs/rad_emulator/                           logs, label shards, checkpoints, figures
runs/raddata_15N_jul                            the 30-day training-data run (RAD_SAMPLE_EVERY=2)
runs/raddata_30NS_jan, runs/lr1yr_d177_sample   the v4 training-data runs (A/B scenario; year-long run resumed at day 177)
runs/rademu_v{3,5}_d207                          v3 / v5 resumed from the day-207 state, RAD_REF every 3 (control: lr1yr)
runs/rademu_v4_d207_dump                        the aer_v4 blow-up, 14 steps with a state dump at every call
inputs/rad_emulator/v5                          aer_v4b + planck_v4 (the v4 directory was deleted: its aerosol net is unstable)
inputs/rad_emulator/v6cand                      aer_cum_v3_lr1e-4 + planck_jac = `current` since 2026-10-17 (in-loop anomaly err 4 %, sampled ARF -1 %)
inputs/rad_emulator/v6mix                       aer_v6mixed + planck_jac (the alternative: better OLR, SW 2.2 %)
outputs/rad_emulator/labels/val_yr_d207         the big-plume held-out set
```
