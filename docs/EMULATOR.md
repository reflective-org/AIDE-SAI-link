# The TOMAS emulator at the microphysics seam

**Status (2026-10-10):** wired in, validated against the handoff's own numbers,
benchmarked against the fast physics in the production loop. The fast physics
(`MICRO_ENGINE=fast`) remains the default; nothing on that path changed.

This page records what the emulator is, how it is attached, the conventions the
adapter had to learn the hard way, and the benchmark — both the one-step
accuracy against `tomas_jax.fast` as ground truth and the speed. Numbers quoted
here are from `runs/emu_dumps/` (offline) and `runs/bench_phys30d/` /
`runs/bench_emu30d/` (the 30-day A/B), all on one H200, 2026-10-09/10.

## What it is

`/data/tomas-emulator/EMULATOR_HANDOFF.md` (2026-10-09) hands over bin-token
transformers trained on `tomas_jax.fast` trajectories — the **same engine
`driver_fast.py` runs** — at a 30-minute grid, as one-call-per-6-hours
operators. The recommended checkpoint, `bintf_big_rebal` (BinTF d=384, 6 layers,
7.2 M parameters), reports 6 h total-number / total-mass errors of 3.1 % / 2.6 %
(median, injection cells) and 2.9 µs per cell in fp16 with CUDA graphs.

Because the training target *is* the fast physics, an A/B between the two in
this model is a clean measurement of emulation error, not a comparison of two
physics.

## How it is attached

`driver_fast.py` has always filled the microphysics seam
(`coupling.run_microphysics_full`) with one batched engine. It now reads

| env | meaning |
|---|---|
| `MICRO_ENGINE=fast\|emulator` | which engine advances the run |
| `MICRO_REF=fast\|emulator` | optional: also run this one on the same input, compare, discard |
| `MICRO_REF_EVERY=N` | compare every N steps (default 1) |

The emulator adapter is `src/microphysics/tomas_emulator.py`, a drop-in with
the same signature, the same mixing-ratio ↔ per-box conversion and the same
two-moment consistency clip as `tomas_fast.py`; only the engine call differs.
`src/microphysics/micro_bench.py` wraps two adapters into one for `MICRO_REF`.
`run_prod.py` exposes all of it as `--micro-engine`, `--micro-ref`, `--emu-*`
(`docs/CONFIGURATION.md`).

The run process must carry **both** `jax[cuda12]` (advection, radiation,
settling, and the fast engine when it is the reference) and `torch`. No
existing environment did, so `/data/aide_emu_venv` was built (jax 0.6.2 +
torch 2.11 cu128, the AIDE-SAI-link pins otherwise); `run_prod.py` execs
whatever python launched it:

```bash
mkdir -p runs/emu30d && cd runs/emu30d
/data/aide_emu_venv/bin/python /data/AIDE-SAI-link/src/run_prod.py --gpu 4 \
    --micro-engine emulator --micro-ref fast \
    --inj-so2 10 --inj-lat 30 --inj-mirror 1 --n-hours 720 --out-tag emu30d
python3 /data/AIDE-SAI-link/scripts/validation/plot_micro_bench.py emu30d --vs phys30d
```

## The interface contract, in this tree's units

| emulator input | from coupling.py |
|---|---|
| state `[0:40]` log Nk, `[40:80]` log Mk (dry SO4), `[80]` log H2SO4, `[81]` log SO2, all per box | per-box = mixing ratio × ρ_air, box = `BOXVOL` = 1 m³, exactly the box the emulator was trained at. **Floors before the log: Nk 1e-20, Mk and the gases 1e-30** (read off the training cache; the "empty" token sits at log 1e-20 for number and log 1e-30 for mass — getting this wrong moves it by 23 log units and the adapter test's control case shows what that does) |
| forcing (13 × 6) at t0, t0+30 min, …, t0+6 h: log P, T, RH, log OH, log SO2 source, log k_dil | T, P, RH are constant over the coupling step (T = it0 snapshot + accumulated radiative ΔT), so repeated. OH comes from `coupling.oh_sza` at substep centres and is interpolated onto the 13 endpoints (`_oh_to_endpoints`, second order; 60 samples/step as for the physics). The SAI release is added to the gas field at the step *start*, before advection, so there is **no in-step source**: channel 4 is log 1e-30. No dilution in a grid cell: channel 5 is log 1e-30 |
| output: state at t0 + 6 h | exp, bins with Nk < 1e-5 per box (the occupancy threshold the model learned) zeroed with their mass, back to mixing ratio. The next step's two-moment clip catches any residual inconsistency and books it, as for the physics |

Precision: fp16 autocast + `torch.compile(mode="reduce-overhead")`, the
handoff's deployment path (measured lossless vs fp32 there). **bf16 is refused**
(it widens the 14-day tails). The compiled graph has a fixed batch (`EMU_BS`,
16384); the last batch is padded.

## Two conventions the first coupled step taught

**1. Night-time OH must not be zero.** `coupling.oh_sza` is a parabola in
cos(SZA) that is *exactly* zero past the terminator. The emulator's training
OH never was: the generator's diurnal curve bottoms out at 0.006 cm⁻³, and
log OH has mean 11.2, σ 4.6 — so log(1e-30) = −69 sits 17 σ below anything
the network saw. On the first coupled step every cell that received a zero
sample (|lat| > 60° in January, plus the terminator band) was answered with a
**spurious 1e8 cm⁻³ nucleation burst from a 13 cm⁻³ background**; 1000 such
cells carried 99.8 % of a +5000 % global number error while the per-cell
median error was 0.3 % (`runs/emu_dumps/ab_step0.txt`). Sweeping the floor on
that state (`ab_step0_ohfloor*.txt`, active-cell total-N p90): 0.01 cm⁻³ →
38.5 %, 1 → 20.5 %, 100 → 16.3 %, global error unchanged. `EMU_OH_FLOOR=100`
is the default: a value the generator's nights reached routinely, and
chemically nothing (k·OH·6 h ≈ 2e−6 of the SO2). RH is clamped to the
generator's [0.01, 1] for the same reason (0.04 σ, immaterial).

**2. SO2 must not be zero either.** The emulator-driven run (first pass,
`runs/bench_emu30d_v0`) went from a 0.3 % one-step mass error to a burden
**50× the initial one by day 4**: at step 7 it answered one plume cell with
137× the physics' mass, and the fabricated sulfur compounded. The trigger
(`runs/emu_dumps_emu7/ab_step7_noproj.txt`) was a cell whose SO2 had been
advected to exactly zero: the training set's prognostic SO2 never fell below
~1e-17 kg/box (log −39), so log(1e-30) is 7 σ outside it. The physics-driven
A/B never shows this because the physics does not produce such cells; the
emulator does (its SO2 answer in a depleted plume is tiny, and the next
advection floors it to zero). `EMU_SO2_FLOOR=1e-17` (0.1 molecule cm⁻³)
removes it: on the same state the global mass error goes +26 % → +0.2 %.

**3. Sulfur conservation cannot be enforced on this emulator.** A per-box
projection (gas sulfur can only fall, aerosol sulfur can only rise, total
conserved — `EMU_PROJECT=sulfur`) was tried as the fix for the above before
the SO2 floor was found. It makes the one-step error *worse*: active-cell p90
goes 16 % → 72 % for both N and M (`ab_step0_final.txt` vs `ab_step0_gpuprep.txt`).
The emulator's SO2 answer is good to ~1 % per step, but the physics only
converts 1–2 % of the SO2 per step, so the gas error is the size of the
signal, and crediting it to the aerosol — which in a clean cell holds far less
sulfur than the gas — turns a 1 % gas error into a 10–1e7× aerosol error.
Without it the emulator violates per-box sulfur conservation by 0.3 / 2.4 /
6.7 % (p50 / p90 / p99 per step; the physics: 0 / 0.01 / 0.02 %), with a
burden-weighted total of +0.3 %. Only the two-moment clip is applied
(`EMU_PROJECT=clip`, the default), as TOMAS's own mnfix would.

**4. The handoff's accuracy is reproduced by the adapter**
(`scripts/validation/test_emulator_adapter.py`, v9_sai test cells through the
production conversion path): total-N 3.0 / 20 %, total-M 2.7 / 11 % (p50 / p90)
against the handoff's 3.1 / 16 and 2.6 / 13. With the number and mass floors
deliberately swapped the median goes to 28 %, which is the guard.

## Benchmark against the fast physics (ground truth)

Design. Two 30-day runs of the production scenario (10 Tg SO2/yr, zonal
rings at ±30°, 51.7 hPa, full radiation), on the same GPU type, in the same
process shape:

* `bench_phys30d` — advanced by the **physics**, emulator run on every state
  as the reference: the emulator's one-step error along the physics
  trajectory ("teacher-forced").
* `bench_emu30d` — advanced by the **emulator**, physics run on every state:
  the emulator's own trajectory, and whether the states it drives itself into
  are still ones the physics agrees with (the drift test).

The two `coupled_timeseries_*.npz` give the trajectory-level comparison;
each `micro_bench_*.npz` gives the per-step one-step error and the timing of
both engines on the same cells.

Both runs completed 120 steps (2026-10-10, one H200 each, ~90 min). Figures
are in `outputs/emulator_bench/` (`<TAG>_bench_{onestep,bins,timing,zonal,traj}.png`).

### Speed

Steady-state wall time per 6 h coupling step, 1,327,104 cells, same GPU,
same process; step 0 carries JIT / `torch.compile` warm-up (38 s physics, 49 s
emulator) and is excluded.

| | fast physics | emulator | ratio |
|---|---|---|---|
| microphysics stage (`t_primary` / `t_reference`, includes the shared mixing-ratio ↔ per-box conversion and clip) | 14.3–14.6 s | 7.7–7.9 s | **1.8×** |
| of which the engine itself | `run_fast` 11.8–12.0 s | `emulate_cells` 5.6 s, of which the model forward 3.95 s = **2.98 µs/cell** (the handoff's 2.9) | 2.1× (engine), 3.0× (forward) |
| full production step (advect 6.4 + micro + settle + bc 1.3 + radiation 8.5; no reference engine) | ≈ 30.8 s | ≈ 24.1 s | **1.28×** |

What the handoff's 17× (73.5 s vs 4.4 s per 1.5 M cells) does not transfer:
TOMAS-JAX fast is adaptive and the global grid is mostly quiescent, so it
costs 9 µs/cell here, not 49. The emulator's cost is fixed. Of the emulator
stage's 7.8 s only 4 s is the network; 0.7 s host prep, 0.4 s device→host,
0.5 s clip and ~2 s of float64 mixing-ratio bookkeeping shared with the
physics adapter are the rest, and would be the next thing to trim (moving
the conversion onto the GPU would bring the stage to ~5 s, 2.9×).

### One-step accuracy along the physics trajectory (`bench_phys30d`)

Per-step percentiles over cells, then the median over the 120 steps
(`max` = the worst step, always step 0 — the fresh MAM4 state plus the first
injection pulse).

| quantity | cells | p50 | p90 | p99 |
|---|---|---|---|---|
| total number | active (1.1–1.2 M) | **0.17 %** | 1.95 % (max 18.4) | 25 % |
| total number | plume (top 1 %, 13 k) | 17.6 % | 51 % | 89 % |
| total SO4 mass | active | **0.26 %** | 0.93 % (max 19.6) | 2.8 % |
| total SO4 mass | plume | 0.58 % | 2.6 % | 5.9 % |
| SO2 | active / plume | 1.3 % / 0.65 % | 3.2 % / 1.6 % | |
| H2SO4(g) | active / plume | 17 % / 220 % | 89 % / 1e9 % | |

Burden-weighted global one-step error: **mass +0.16 % every step**; SO2
within ±0.3 %; **number +31.6, +30.7, +20.6, +3.6 % on the first four steps
(the fresh injection-ring burst over-predicted ~1.8×), then within ±9 %**
(median −0.9 %); H2SO4(g) −40 % (the emulator holds far less gas-phase acid
than the physics, with no consequence for the aerosol).

Per bin, the median error is below 1 % for every bin above 4 nm and 0.2–0.3 %
above 20 nm; the p90 is 42 % in the 2 nm bin falling to <1 % above 100 nm —
the nucleation-mode bins are where the emulator is loose. On "burst" bins
(fine-mode bins above 1e3 cm⁻³) the per-bin error is 19 / 70 % (p50 / p90).
Per level, the p90 of total number is 11–20 % at 43–74 hPa (the plume
levels) and under 2.5 % everywhere else.

### The emulator's own trajectory (`bench_emu30d`)

The one-step error on the states the emulator drives itself into stays the
same size as above (active p50 / p90: number 0.30 / 1.6 %, mass 0.29 / 1.2 %;
plume mass 0.41 / 1.25 %) — no blow-up in 30 days, where the first pass (no SO2
floor) reached 50× the initial burden by day 4. But the per-step errors are
systematic and compound. Day-30 state, emulator-driven vs physics-driven:

| diagnostic | physics | emulator | difference |
|---|---|---|---|
| SO4 burden / M0 | 1.378 | 1.368 | **−0.7 %** (+2.5 % at day 5–10, then crossing) |
| number burden / N0 | 103.9 | 67.9 | **−35 %** (−7 % day 5, −19 % day 10, −22 % day 20) |
| SO2 burden | 0.393 | 0.506 | +29 % |
| H2SO4(g) burden | 6.5e-4 | 1.0e-3 | +55 % |
| r_eff at 52 hPa | 106 nm | 122 nm | +14 % |
| mass-mean Dp | 597 nm | 669 nm | +12 % |
| AOD 550 nm | 0.00766 | 0.00840 | +10 % |
| TOA forcing (24 h mean) | −0.060 W m⁻² | −0.074 W m⁻² | +22 % |
| cumulative settled mass | 0.0140 | 0.0131 | −7 % |
| cumulative two-moment clip mass added / M0 | 1e-6 | 1.6e-3 | the emulator's answers are not self-consistent; the clip repairs them |

Reading it: the emulator **under-produces number in the plume** (its global
one-step number error settles at −15 % after day 5, and the self-rollout
loses a third of the number burden by day 30) and **under-oxidises SO2**
(+29 % SO2 left) while **matching the aerosol mass** — which it can only do
by not conserving sulfur (+0.16–0.3 % mass per step, burden-weighted, 120
times). Fewer, larger particles follow: r_eff +14 %, AOD +10 %, forcing +22 %.
For a 30-day SAI experiment the mass and AOD are usable to ~10 %; the number
and size are not, and nothing here says what day 90 or day 365 look like.

### What this says about using it (first pass; superseded below)

* As a **physics substitute in production** on the first pass: no. 1.8× on
  a stage that was ~45 % of the step bought 1.28× per step, and the
  trajectory drifted in number (−35 % at day 30) and size (+14 %).
* The three things proposed to change the verdict: (1) train on states this
  model actually produces, (2) the injection-ring burst, (3) sulfur closure
  in the network. The second pass below resolves all three — the first by
  refutation, the other two by routing and by an exact closure after the
  network — and the verdict is now yes, within ~1 % for a month at 3× on
  the stage.

## Second pass (2026-10-13): speed, the hybrid plume, and training on this model's own states

The first benchmark gave 1.8× on the microphysics stage and a trajectory that
lost a third of its number burden in a month. The second pass attacks both,
with the radiation left untouched. Runs: `runs/bench_phys30d_v2` /
`runs/bench_emu30d_v2` (the first benchmark re-run with the new adapter),
`runs/bench_hyb1_30d` / `runs/bench_hyb3_30d` (hybrid plume at 1 % and 3 %),
`runs/bench_ft2hyb1_30d` (fine-tuned network + hybrid 1 %), all 30 days of
the production scenario with the physics as in-loop reference.

### Speed: 14.5 s → 3.0 s on the microphysics stage

Two changes, both in `tomas_emulator.py`, neither touching the physics path:

1. **GPU-resident adapter.** The state arrives from coupling as jax device
   arrays; the adapter now takes them zero-copy through DLPack
   (`torch.from_dlpack`), does the mixing-ratio ↔ per-box conversion, the
   two-moment clip, the floors and the diagnostics in torch float64 on the
   device, and hands the answer back the same way. The 2 s of host float64
   bookkeeping, 0.7 s host prep and 0.4 s device→host copy of the first pass
   are gone: 7.8 s → 4.3 s at identical output.
2. **Triage** (`EMU_TRIAGE=1`, default). Measured on the dumped states, the
   physics changes number or mass by more than 1 % in only 45–49 % of cells
   per step, and those cells hold 4–5 % of the mass. A cell is QUIET when
   the sulfur that could reach its aerosol in 6 h (all its H2SO4 plus what
   k₁·OH can oxidise) is below 1e-3 of its aerosol mass, its number is below
   300 cm⁻³, the H2SO4 the chemistry can sustain against its condensation
   sink is below 1e5 cm⁻³ (no nucleation), and less than 10 % of its number
   is below 10 nm. Those cells — 30–33 % of the grid holding 80 % of the
   mass — get the exact pseudo-first-order SO2 + OH step with the acid
   condensed onto the existing bins in proportion to their surface area and
   no change in number, instead of the network. The physics itself moves
   them by 0.13 % (median) per step; the emulator's own error on them was
   0.10–0.15 %.

| in-loop wall time per step, 1.33 M cells, median over 120 steps | physics | emulator v1 | emulator v2 (GPU-resident + triage) |
|---|---|---|---|
| microphysics stage | 14.5 s | 7.9 s | **3.4 s** (4.3×); 3.0 s without the sampler |
| of which the network forward | — | 3.95 s | 2.6–2.8 s on the 67–71 % of cells it sees (2.95 µs/cell) |
| full production step (advect 6.4–7.0 + micro + bc 1.3 + radiation 8.5) | ≈ 31 s | ≈ 24 s | **≈ 19 s** (1.65×) |

The one-step accuracy is unchanged on the network's cells and better on the
quiet ones: the burden-weighted global mass error per step along the physics
trajectory went +0.15 % → +0.10 % (v1 → v2); every other statistic of
`bench_phys30d_v2` is within noise of `bench_phys30d`, and the v2 self-rollout
reproduces the v1 trajectory (number −34.7 % at day 30, r_eff +15.6 %, AOD
+9.7 %).

Where the step's time goes now: radiation 8.5 s, advection 7 s, microphysics
3 s, boundary/polar 1.5 s. Microphysics is no longer the critical path; the
next factor of two per step is in advection and radiation, not here.
Within the microphysics stage the remaining levers are small: widening the
triage (fine-mode threshold 0.1 → 0.3 is another ~6 % of cells, 0.15 s),
calling the operator every second step with a 12 h chain (the chain-trained
checkpoints support it; ~1.5 s per step on average at an unmeasured
splitting error), or a d=256 distillation of the network (~1.3 s). None was
pursued: 1.5 s of 19 is not worth an accuracy risk until the two larger
stages move.

### The hybrid plume (`EMU_PHYS_TOP`)

Attribution first. On the emulator-driven step-7 state the emulator's
signed global number error is −10.7 % per step; ranking cells by total
sulfur per box (SO4 + SO2 + H2SO4):

| top cells by sulfur | hold | signed global N error carried |
|---|---|---|
| 0.1 % (1.3 k) | 34 % of all particles | −5.0 % |
| 1 % (13 k) | 80 % | **−9.4 %** (of −10.7) |
| 3 % (40 k) | 82 % | −9.8 % |
| 10 % (133 k) | 86 % | −10.0 % |

The whole number error is the injection ring. `EMU_PHYS_TOP=f` hands the
top fraction `f` of cells to `tomas_jax.fast` with the production knobs
(through `tomas_fast`), the triage and the network take the rest. The
subset is padded with empty cells to a multiple of `EMU_PHYS_PAD` (4096) so
`run_fast`'s jit sees one shape: the first call compiles for ~20 s, every
later one costs **0.22 s at 1 % and 0.50 s at 3 %** per step (measured in
the loop; adaptive substepping is cheap on 13–40 k cells, and a single-call
offline timing reports the compile and misleads).

One-step, on the step-7 state (physics as truth):

| | emulator | hybrid 1 % | hybrid 3 % |
|---|---|---|---|
| global signed number error | −10.7 % | **−1.3 %** | **−0.9 %** |
| global mass / SO2 error | +0.05 / +0.12 % | +0.04 / +0.12 % | +0.04 / +0.11 % |
| plume total-number error p50 / p90 | 17.6 / 51 % | 0.00 / 0.06 % | 0.00 / 0.02 % |
| active cells p90 (number / mass) | 3.3 / 5.9 % | 3.3 / 5.9 % | 3.3 / 5.9 % |
| burst cells, per-bin number p50 / p90 | 0 / 13 % | 0 / 13 % | 0 / 13 % |
| microphysics stage in the loop | 3.0 s | 3.0 s | 3.3 s |

Self-rollout, 30 days, each run's burden against the physics run at the
same hour (`scripts/validation/compare_runs.py`, figure
`outputs/emulator_bench/compare_chem30d.png`):

| day 30 vs physics | emulator | hybrid 1 % | hybrid 3 % |
|---|---|---|---|
| number burden | −34.7 % | −15.9 % | **−8.2 %** |
| mass burden | −0.3 % | −1.3 % | −2.3 % |
| SO2 burden | +28 % | +38 % | +42 % |
| H2SO4(g) burden | +54 % | +60 % | +68 % |
| r_eff | +15.6 % | +0.9 % | −1.2 % |
| AOD 550 nm | +9.7 % | +2.7 % | 0.0 % |
| TOA forcing | −26 % | −1.9 % | +7.6 % |
| microphysics stage | 3.4 s | 3.7 s | 4.1 s |

The plume on the physics fixes the size (r_eff, AOD) and most of the
number; the number still drifts (−0.8 % at day 5, −7 % at day 20, −16 % at
day 30 for 1 %) and the gas side gets worse: the SO2 burden runs 38–42 %
high while the mass runs low. The one-step global error per step is
+1 % in number and 0.00 % in mass and SO2 — the drift is not visible at
one step, which is the point of the next section.

### Fine-tuning on the coupled model's own states

The handoff model was trained on box-model trajectories from CESM-sampled
conditions. The states this model hands it are different in two known ways
(the floors above are the symptoms): SO2 advected to exactly zero and
night-time OH of zero, and more generally the post-advection,
post-settling size distributions. The pipeline:

* `MICRO_SAMPLE_EVERY=4` (`src/microphysics/micro_sampler.py`) writes a
  stratified sample of the engine's input — plume (top 1 % by sulfur, up to
  half the budget), active, quiet (the triage set) and empty cells — with
  the cell's column position and hour so OH can be reconstructed.
* `scripts/emulator/label_chains.py` integrates each sampled cell for 48 h
  with `tomas_jax.fast` as a box model (the production knobs, OH from the
  SZA parabola at the cell's own position and hour, 30-minute checkpoints) and
  writes chains in the emulator's dataset format with the adapter's floors
  applied. 111 µs per cell per half hour: a 10 k-cell sample is 24 s.
* `scripts/emulator/finetune_bintf.py` warm-starts the handoff checkpoint and
  trains on K-chains of the AIDE samples (weight 3) mixed with the original
  populations (60 k chains, so nothing is unlearned), teacher-forced 6 h
  loss and the handoff's burst-bin weighting, two stages (24 h chains, then
  48 h); the score that picks `best_aide.pt` is the held-out plume + active
  p90 at 6 h.

Data: both v2 runs sampled every 4 steps (8 000 cells + plume), 600 000
chains of 97 points; held out: the samples from day 25 on (100 k chains).

Result: `best_aide.pt` (48 h stage, epoch 3) takes the held-out plume
one-step number error from 15.9/51 % to 1.9/8.7 % (p50/p90) and the 48 h
chained error from 18/44 % to 2.9/8.4 %, with the original test sets not
degraded; a second run with a higher AIDE weight and learning rate was
worse. Both checkpoints were then run for 30 days with the hybrid plume
and the closure (next section): both are worse than the handoff model in
every trajectory quantity. The score and the rollout disagree because
the score is a per-cell teacher-forced error on the sampled strata and
the rollout is driven by what the network does to the dense populations
it creates itself.

### Sulfur bookkeeping is the seam: `EMU_PROJECT=chem`

What the hybrid runs exposed. With the plume on the physics, every rollout
still drifted, and in a way the one-step tables did not predict: the
number and size came right, the SO2 burden went +38 % by day 30 and the
mass bent the other way; the fine-tuned network over-shot instead (number
+30 %, mass +5 % at day 25). On the step-7 state the handoff network
oxidises 0.38 % of the SO2 per step where the physics oxidises 0.76 %,
while adding 0.10 % of aerosol mass against the physics' 0.057 % — it
makes the right mass from the wrong amount of gas, per step, 120 times.
The fine-tuned one gets the SO2 right and still over-adds mass by 45 %.
Both are errors of 1e-3 per step relative to the burden; both compound into
tens of percent in a month. The two-moment clip cannot see them and the
first pass's sulfur projection made them worse because it took the
network's gas answer as the truth.

The gas side of a stratospheric step is not something a network needs to
learn. The SO2 chemistry is pseudo-first-order with the Troe rate
k₁(T, P, RH) from `tomas_jax.physics.so2_chemistry` and no other sink, so
what the physics consumes in a step is exact from the inputs; the H2SO4 it
produces relaxes to a production / condensation-sink balance, and the
physics' own sink (Fuchs–Sutugin, binary density, `ALPHA_COND`,
`tomas_jax.fast.condensation._condensation_sink_cell`) can be evaluated on
the network's output distribution. `EMU_PROJECT=chem` does exactly that
and gives the aerosol what the gas lost: the network's per-bin mass change
is kept (coagulation shifts, which it gets right) and the residual against
the budget is deposited by sink fraction, then re-clipped. The network is
left with the two things it is good at — number and the shape of the
distribution. Keeping the network's H2SO4 (capped) instead of the
analytic one was tried first and was worse (active-cell mass p90 5.9 → 9.9
%: its under-predicted gas acid was forced into the aerosol).

One-step on the step-7 state, hybrid 1 % in all three:

| active cells, p50 / p90 | clip, handoff net | chem, handoff net | chem, fine-tuned net |
|---|---|---|---|
| total number | 0.32 / 3.3 % | 0.32 / 3.3 % | **0.15 / 1.8 %** |
| total SO4 mass | 0.22 / 5.9 % | 0.00 / 3.8 % | **0.00 / 1.9 %** |
| SO2 | 0.94 / 16 % | exact | exact |
| H2SO4(g) | 7.9 / 42 % | 1.3 / 21 % | 1.3 / 21 % |
| global signed error N / M / SO2 | −1.3 / +0.04 / +0.12 % | −1.3 / 0.00 / 0.00 % | **−0.05 / 0.00 / 0.00 %** |
| per-cell sulfur closure p90 | 7.9 % | 0.12 % | 0.05 % |

Self-rollout, 30 days (`runs/bench_chem_hyb1_30d`, `runs/bench_ft2chem_hyb1_30d`,
`runs/bench_ftchem_hyb1_30d`), all hybrid 1 %:

| day 30 vs physics | hybrid 1 %, clip | chem, handoff net | chem, fine-tuned ep2 | chem, fine-tuned best |
|---|---|---|---|---|
| SO2 burden | +38 % | **0.0 %** (all month) | 0.0 % | 0.0 % |
| mass burden | −1.3 % | +1.1 % | +3.2 % | +1.8 % |
| H2SO4(g) burden | +60 % | +19 % | +21 % | — |
| number burden | −15.9 % | +7.8 % | +13.9 % | +16.5 % |
| r_eff | +0.9 % | −0.9 % | +6.6 % | +1.9 % |
| AOD 550 nm | +2.7 % | +6.5 % | +33 % | +21 % |
| TOA forcing | −1.9 % | −16 % | −124 % | −74 % |
| microphysics stage | 3.7 s | 4.1 s | 4.1 s | 4.0 s |

The gas budget is closed to the digit, and the mass follows (the residual
+1 % is settling acting on a slightly different size distribution). Two
things the table also says:

* **The fine-tuned checkpoints do not transfer.** Their held-out chain
  scores are 3–6× better than the handoff model's (plume number p50/p90
  15.9/51 → 1.9/8.7 % at 6 h, 18/44 → 2.9/8.4 % over 48 h) and every
  rollout is worse — number, AOD and forcing all over-shoot. A per-cell
  teacher-forced score, even on this model's own states, does not predict
  the closed-loop drift; the 30-day run is the only judge and both are
  retired.
* **The number now drifts the other way, and it is the closure that
  exposed it.** The per-step burden-weighted number error along the
  closure run's own trajectory grows from +0.6 % to +17.5 % over the month
  (it is +1 % without the closure), all positive, all in bins 0–6, at
  36–143 hPa (the injection levels), from 30° S to 60° N: the diffuse part
  of the plume. Fed the physics' H2SO4 instead of its own estimate — which
  runs 50 % low, and so suppressed its own nucleation — the network
  over-nucleates. These cells hold little sulfur, so the plume ranking does
  not catch them (the top 1 % by sulfur carries 88 % of the number error on
  day 2 and almost none on day 30), and the spurious particles mostly die
  again in the next step's coagulation, which is why the burden error
  (+8 %) is far smaller than the per-step one. The hybrid-only run's good
  forcing (−1.9 %) was two errors cancelling.

### The year-scale sulfur leak and its fix (`EMU_CHEM_NCLIP`, 2026-10-17/18)

The 30-day budget does not hold for a year. `runs/micro1yr_c5` (the production
emulator with `EMU_PHYS_COARSE=0.05`, physics radiation) against `ffsl1yr_K24`
drifted to mass +22.1 %, number −22.6 %, AOD +25.1 % and forcing −24.1 %
(stronger) at day 365 with SO2 exact; the full-emulator year (`allemu1yr_v6c5`,
radiation v6cand) is identical to 0.1 %, so the radiation emulator holds for a
year and the drift is the microphysics seam's. The run's budget terms show a
microphysics SOURCE excess with the SO2 sink unchanged, i.e. sulfur created in
the adapter. A per-class one-step sulfur budget on the day-30 state (network /
quiet-analytic / physics-routed cells) finds the network class gaining
+0.018 % of its sulfur per step against +0.002 % for the physics, the other two
classes exact; (1 + 1.7e-4)^1460 is the +28 %.

The mechanism was the last line of `project_chem`: after the aerosol is given
exactly the sulfur the gas lost, the two-moment bounds were restored by
re-clipping MASS, which creates mass wherever the network's number implies a
higher minimum (M < N·xk_lo) and silently lifts the bins that the additive
residual deposit had driven below zero. `EMU_CHEM_NCLIP=1` (default since
2026-10-18) replaces that: a negative residual (the network added more mass
than the budget allows) is removed multiplicatively, number is reduced where
mass is short (N := M/xk_lo), and where mass exceeds a bin's maximum it is
clipped and redistributed to the cell's bins with room (two passes; the
remainder is logged as `chem_lost`). Two earlier variants went NaN by step 3
(raising number into nearly empty bins; negative mass turned into negative
number) although their one-step tests were perfect — never ship an adapter
change on one-step tests alone. One-year validation (`micro1yr_nclip1b` vs the
legacy `micro1yr_c5`, day 365 vs `ffsl1yr_K24`): mass −0.1 % (legacy +22.1 %),
SO2 +0.3 % (same), number −16.3 % (−22.6 %), AOD +19.3 % (+25.1 %), forcing
−19.1 % (−24.1 %), r_eff +6.9 % (+5.9 %); at day 30 number is −2.2 % against
−0.7 % (the spurious number is gone), everything else equal or better.
`EMU_CHEM_NCLIP=0` restores the old behaviour.

What remains at a year is the size distribution at exact mass: the network's
number deficit above the plume (one-step −1 to −4 % per step of the number in
network cells at 25–30 hPa) leaves fewer, larger particles, hence AOD +19 % and
forcing 19 % too strong, and the coarse-bin even/odd sawtooth (bins 22–32,
±15 %, an accumulated 0.3–1.6 % per-step bias in network cells) costs 7 % of the
30-day forcing at equal AOD. Both are the network's own errors; adapter-side
filters (`EMU_BIN_CORR`, `EMU_NYQ_DAMP`, `EMU_SAW_DAMP`, all inert at their
defaults) and four chain fine-tunes (`frontier_bintf_aide_coarse1–4`) were tried
and refuted on 2026-10-17 (see the local session log of that day, not in this repo).
Routing more cells to the physics by coarse-bin mass (`EMU_PHYS_COARSE`) buys
forcing at a cost: 0 / 5 / 10 % → +7.5 / +5.4 / +4.5 % at 4.9 / 6.2 / 8.5 s per
step. Gate change: judge the emulator by a one-year run and a per-class sulfur
budget, not by the 30-day state.

### Routing the dense cells (`EMU_PHYS_NUM`)

Which cells carry it. The closure run's own day-30 state was dumped by
resuming its checkpoint for one step, and both engines run on it offline
(`scripts/validation/micro_ab_offline.py`), physics as truth. The top 4 %
of cells by absolute signed number error carry 96 % of the +18.4 % global
error, so it is concentrated; and they are not nucleating cells. They are
dense fine populations — a median of 1e4 particles per cm³ against a grid
median of 70, at 200 K and 60 hPa, the diffuse remains of the injection
ring — in which the physics halves the number in a step by coagulation
(median N_out/N_in 0.57) and the network removes a quarter (0.74–0.80);
only a third of them grow in the network at all. It is under-coagulation
of a dense nucleation mode, the burst-shape floor the emulator's own
benchmarks report, met here on a population the sulfur ranking does not
see because it holds little mass. Rankings that can be computed from the
inputs alone, fraction of the signed global number error carried by the
top cells:

| rank cells by | 1 % | 2 % | 4 % | 8 % | 16 % |
|---|---|---|---|---|---|
| total particle number per box | 37 % | 63 % | **89 %** | **98 %** | 100 % |
| steady-state H2SO4 (the triage's `Hss`; first ranking tried) | 45 % | 60 % | 73 % | 81 % | 86 % |
| the network's own number growth | 54 % | 57 % | 58 % | 58 % | 58 % |
| total sulfur per box (the plume ranking) | 2 % | 21 % | 47 % | 69 % | 94 % |

`EMU_PHYS_NUM=f` hands the top fraction `f` of the non-plume cells by
total number per box to the physics as a second routed set (the same
padded `run_fast` call as the plume). On the day-30 state, one step:

| routed to the physics | global signed number error / step | active cells number p90 | burst cells per-bin p50 / p90 | H2SO4 global | physics time in the loop |
|---|---|---|---|---|---|
| plume 1 % only | +18.4 % | 7.2 % | 20 / 85 % | +5.4 % | 0.22 s |
| plume 3 % only | +12.5 % | 5.5 % | 8 / 68 % | +2.2 % | 0.50 s |
| plume 1 % + `Hss` 4 % | +5.0 % | 4.5 % | 0 / 46 % | −7.2 % | 1.1 s |
| plume 1 % + `Hss` 10 % | — | — | — | — | 2.3 s |
| plume 1 % + number 2 % | +4.9 % | 5.1 % | 0 / 41 % | −2.4 % | |
| plume 1 % + number 4 % | **+1.4 %** | 3.5 % | 0 / 2 % | −1.4 % | **1.1 s** |
| plume 1 % + number 8 % | **+0.35 %** | 1.8 % | 0 / 1 % | −0.1 % | ≈ 2 s |

(mass error 0.01–0.03 % in every row; a first offline call on a new
subset size reports the jit compile, the times are from the 30-day runs).

Self-rollout, 30 days, exact closure and hybrid 1 % in every column
(`runs/bench_chem_hyb1nuc10_30d` is the `Hss` ranking at 10 %,
`runs/bench_chem_hyb1num{4,8}_30d` the number ranking; figure
`outputs/emulator_bench/compare_final30d.png`):

| day 30 vs physics | no second set | `Hss` 10 % | **number 4 %** | number 8 % |
|---|---|---|---|---|
| number burden | +7.8 % | −1.5 % | **−1.2 %** | −0.8 % |
| mass burden | +1.1 % | +0.4 % | **+0.1 %** | 0.0 % |
| SO2 burden | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| H2SO4(g) burden | +19 % | +4.6 % | +18 % | +18 % |
| r_eff | −0.9 % | −1.8 % | −1.1 % | −1.0 % |
| AOD 550 nm | +6.5 % | +1.1 % | **0.0 %** | −1.1 % |
| TOA forcing | −16 % | +3.0 % | +7.2 % | +10 % |
| number burden, worst day of the month | +12 % | −2.6 % | −1.4 % | −1.2 % |
| per-step global number error (median, first → last fifth) | +0.6 → +17 % | +0.5 → +2.3 % | +0.1 → +0.3 % | +0.1 → +0.1 % |
| microphysics stage, median of 120 steps | 4.1 s | 5.9 s | **4.9 s** | 5.6 s |

Number by 4 % is the production configuration (`EMU_PHYS_TOP=0.01
EMU_PHYS_NUM=0.04 EMU_PROJECT=chem`, now the adapter's defaults): the
number, mass, SO2 and AOD trajectories stay within 1.2 % of the physics for
the month at a third of its cost on this stage. Going to 8 % buys per-step
accuracy the trajectory does not need. Two things it leaves open: the
H2SO4(g) burden runs 18 % high (the network cells hold the analytic
steady state, which is an upper bound on a 6 h mean when the sink is
growing; it is 0.1 % of the sulfur and does not feed back), and the TOA
forcing is 7 % weaker than the physics' at the same AOD and a 1 % smaller
r_eff — steady from day 20 on, so not noise; the radiation was not touched
in this pass and the cause has not been located.

### What this pass says

* **As a physics substitute for a month: yes, at 3× on the stage.**
  `--micro-engine emulator` with the defaults tracks the physics run's
  burdens, size and AOD to about 1 % over 30 days of the production
  scenario at 4.9 s per step on the microphysics stage against 14.5 s.
  The full step is ≈ 20 s against 31 s; what remains is radiation (8.5 s)
  and advection (7 s), untouched.
* **The network does one thing in this configuration:** the shape and
  number evolution of the 95 % of cells that are neither the injection
  ring nor dense. Everything with a closed form — the gas chemistry, the
  H2SO4 balance, the sulfur budget — is computed, and everything the
  network is known to get wrong — the burst and the coagulation of a dense
  fine mode — goes to the physics, selected by two one-line rankings on
  the inputs. The three fractions are knobs, and the benchmark shows what
  each buys.
* **Fine-tuning on this model's states is refuted as a lever**, twice: a
  3–6× better held-out chain score and a worse month, both times. The
  closed-loop benchmark is the only judge; the chain pipeline stays for
  the record and for a DAgger round if someone wants to pursue it, but
  routing beat it at no training cost.
* **The first-pass verdict's three items** are resolved differently than
  proposed: (1) training on own states — refuted; (2) the injection burst
  — routed; (3) sulfur closure in the network — done after it, exactly.

## Standing caveats

* **Injection-ring bursts are over-predicted and dense fine modes are
  under-coagulated by the network** (the "burst floor" the emulator team's
  own audits flagged). With the defaults both populations go to the
  physics (`EMU_PHYS_TOP`, `EMU_PHYS_NUM`, second pass below); with them
  off, the fresh-pulse cells come out ~1.8× the physics' number and the
  diffuse plume gains +18 % in number per step by day 30.
* **Above ~10 hPa the emulator is unreliable** (per-level median error 5–40 %
  in total number). Those levels hold a negligible share of the burden and
  less than 1 % of the training set (log P below its p1), so the global
  numbers do not see it — but a diagnostic read at 1–7 hPa should not trust
  the emulator.
* **H2SO4(g) is not the network's.** With `EMU_PROJECT=chem` (default)
  the network cells carry the analytic production / condensation-sink
  steady state, and the burden runs ~18 % above the physics' over a month;
  it is 0.1 % of the sulfur and the mass does not see it. With `clip` the
  network's own H2SO4 is 50 % low in the plume and feeds the SO2 drift.
* **TOA forcing is ~7 % weaker than the physics' at the same AOD** in the
  production configuration (steady from day 20 on). Not located; the
  radiation was not touched in either pass.
* **Fine-tuning the network on this model's own states made the rollouts
  worse** both times it was tried, despite 3–6× better held-out chain
  scores. Do not judge a checkpoint by the chain score.
* The emulator was trained on trajectories with **no in-step source** of the
  kind `INJ_*` applies at step start, and with CESM-sampled stratospheric and
  upper-tropospheric conditions; a scenario outside that (very high
  injection rates, SO2 > 2.9e15 cm⁻³, OH > 2.5e7) is outside the training
  envelope and `micro_ab_offline.py` prints how many cells are.
* **Checkpoints across an engine switch** are allowed and warn (`PHYS_CFG`
  gained a fifth, appended field). A timeseries that spans the switch is two
  models.

## Files

| | |
|---|---|
| `src/microphysics/tomas_emulator.py` | the adapter; `describe()` is what the run header prints |
| `src/microphysics/micro_bench.py` | the in-loop A/B wrapper and `compare()` |
| `scripts/validation/test_emulator_adapter.py` | adapter vs the handoff's cached test cells (GPU, ~1 min) |
| `scripts/validation/micro_ab_offline.py` | A/B on a `DUMP_PREMICRO` state, with the worst cells and their inputs |
| `scripts/validation/plot_micro_bench.py` | the figures from `micro_bench_<TAG>.npz` (+ `--vs <TAG2>` for trajectories) |
| `scripts/validation/compare_runs.py` | the trajectory benchmark: burdens, r_eff, AOD, TOA forcing of any runs against a reference run, table + figure |
| `src/microphysics/micro_sampler.py` | `MICRO_SAMPLE_EVERY`: stratified samples of the engine's input for training data |
| `scripts/emulator/label_chains.py`, `finetune_bintf.py` | the sampled cells as 48 h physics chains; fine-tuning on them (refuted as a lever, kept for the record) |
| `/data/tomas-jax-output/ft_launch.sh`, `dagger_round.sh`, `label_loop.sh` | the fine-tune / DAgger launchers used (outside the repo) |
| `/data/aide_emu_venv` | jax + torch run environment (`/data/aide_emu_venv_build.sh` rebuilds it) |
| `/data/tomas-emulator/EMULATOR_HANDOFF.md` | the emulator team's handoff: checkpoints, contract, reference numbers |
