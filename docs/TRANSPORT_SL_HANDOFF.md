# Semi-Lagrangian transport for AIDE-SAI-link — status, findings, and the plan

Handoff written 2026-10-13 for whoever wires the JCM transport operator into
this model. Everything tagged MEASURED was produced by a command whose script
and log are named here; everything else is a design proposal.

---

## 0. The short version

- With the emulator microphysics in production the coupled 6-h step costs
  **21.3 s** on one H100 (physics microphysics: 31.0 s). Advection is **6.6 s**
  of it, radiation 8.5 s, microphysics 4.9 s. MEASURED, §1.
- The JCM "fast transport" is **dinosaur's `semi_lagrangian` module** (Google
  NeuralGCM dynamical core, JAX, Apache 2.0): fixed-iteration departure points
  in Cartesian coordinates, cubic Lagrange gather, quasi-monotone limiter. It
  is a JAX module, so it drops into this JAX model without a framework change.
- On this model's exact problem size (82 tracers × 24 levels × 192 × 288,
  3-D, cubic + limiter, 6 sub-steps per 6 h) it costs **49 ms per step**
  on an H100 — **135× less** than the current scheme. MEASURED, §3.
- If advection goes to ~0.1 s the step is ~15 s (physics 31 → 24.5 s), and the
  emulator+SL configuration is **2.1×** the physics step instead of 1.46×.
  Radiation then is more than half of what is left.
- Four things stand between the benchmark and a wired-in scheme (§5): the
  CESM grid has nodes AT the poles and dinosaur refuses those (a ~10-line
  patch, §5.1); the vertical coordinate is pressure with open top/bottom
  faces (§5.2, §5.3); and **the operator does not conserve mass**. §6 is the
  conservation design: a *closed extended column* plus a *Bermejo–Conde local
  proportional fixer*, which gives conservation to roundoff with an exactly
  known target and keeps the limiter's positivity. Try the trivial global
  fixer first for the A/B, then the local one.
- Validation gates and an ordered task list are in §7–§8. Environment and
  run recipes in §9.

---

## 1. Where the cost stands (MEASURED)

Medians over steps 2–119 of the two 30-day benchmark runs
(`runs/bench_phys30d`, `runs/bench_chem_hyb1num4_30d`; stage times from the
`[prof]` lines, engine times from the `[bench]` lines because those runs
carried both microphysics engines for the A/B):

| stage | fast physics | emulator (production) |
|---|---|---|
| CESM read | 0.2 s | 0.2 s |
| **advection** (`fct_lr`, 82 tracers) | **6.6 s** | **6.6 s** |
| microphysics | 14.6 s | 4.9 s |
| settling | 0.02 s | 0.02 s |
| BC + polar refresh | 1.2 s | 1.2 s |
| radiation | 8.4 s | 8.5 s |
| **whole step** | **31.0 s** | **21.3 s** |

Why advection is 6.6 s: `src/advection/fct_lr.py` is Lin–Rood flux-form PPM
with a Zalesak limiter. It is CFL-limited in *both* horizontal sweeps (flux
form has no integer-shift FFSL), and the sub-step is set by the smallest
resolved cell: at `LAT_FREEZE = 80°` the longitude spacing is
R·cos(80°)·2π/288 ≈ 24 km, so with `ADV_CFL = 0.5` and |u| ~ 100 m/s the
sub-step is ~120 s → **~180–270 sub-steps per 6 h** (the `nsub=` number in
the `[prof]` lines). Each sub-step is three PPM sweeps over 82 × 1.33 M cells.
That is a structural cost, not an implementation one; `fct_fast.py`'s header
already lists "FFSL in latitude" as the planned fix.

A semi-Lagrangian step has no CFL limit: it is one departure-point solve per
level plus one gather per tracer, whatever the wind.

## 2. What JCM did, and what it found (from `/data/TRACER_TRANSPORT_EMULATOR_RESULTS.md`)

- Operator: `dinosaur.semi_lagrangian` — `horizontal_departure_points`
  (Robert-style fixed point on the trajectory midpoint, 2 iterations,
  Cartesian unit-sphere formulation à la IFS 48r1, Diamantakis & Váňa 2022),
  `GridInterpolator(order='cubic', limiter='quasi_monotone')` (Bermejo &
  Staniforth 1992: cubic value clipped to the range of the 2×2 bracketing
  corners → no new extrema, positivity preserved), `transport_scalar_2d`,
  and a 3-D variant `departure_points_3d` / `interpolate_3d` / `transport_scalar`
  with linear-or-cubic vertical Lagrange on *fixed* vertical nodes.
- Cost in JCM (T63 × 63 levels, 24 tracers, H200): 1.22 ms for the remap,
  2.92 ms for the whole step, 68× the JCM dycore.
- Accuracy: the analytic operator with true winds explained R² 0.82–0.87 of
  6-h tracer tendencies on the archive; the learned correction added 0.047 on
  single steps and was *harmful* over rollouts. Lesson: **use the analytic
  operator, do not learn a correction.**
- Conservation: JCM enforced mass with a **global proportional mass fixer**;
  without it the operator drifted ~1.3 % over 38 steps. Positivity and the
  maximum principle held exactly over 8,763 steps.
- Its multi-year failure was the **missing vertical remap** (horizontal-only
  operator froze the pulse centroids). Here the band is only 24 levels and
  the vertical exchange through the open faces is ~10 %/day of the slab
  gross, so the 3-D operator is required from the start, not a refinement.
- Air moves 1.7 cells (median) and up to 32 cells per 6 h at T63; here with a
  finer grid it is more. Sub-stepping K = 1 → 8 in JCM bought +0.017 R² and
  plateaued by K = 4.

## 3. What I measured here (MEASURED)

`scripts/validation/bench_sl_transport.py`, log in
`outputs/emulator_bench/bench_sl_transport.log`. Synthetic smooth winds and
tracers on a Gaussian grid with the model's node counts (288 × 192), 24
levels, 82 tracers, float32, GPU 0, median of 10 jitted calls:

| configuration | per 6-h step |
|---|---|
| horizontal only, K = 1 | 4.8 ms |
| 3-D (cubic h, linear v, limiter), K = 1 | 13.0 ms |
| horizontal only, K = 6 | 19.2 ms |
| **3-D, K = 6 (the configuration to use)** | **48.8 ms** |
| departure points alone, 24 levels, K = 1 | 0.56 ms |

Compile 1–3 s. This is a *cost* measurement only: the winds are synthetic and
nothing is checked for accuracy. Memory was not an issue on an otherwise idle
80 GB card (one 82-tracer float32 stack is 1.1 GB; the cubic gather holds a
16-point stencil's worth of indices/weights, not 16 copies of the field).

## 4. The interface the new operator has to satisfy

`src/coupling.py` builds the tracer stack and calls one function per step
(lines ~2225–2250):

```python
qb = jnp.concatenate([num, mas, so2[None], h2so4[None]], axis=0)   # (82, nlev, nlat, nlon) mixing ratios
out = advect_step_batch(qb, u0, v0, w0, u1, v1, w1, qfrozb=qfroz,
                        lat=lat, dp=DP, lat_freeze=LAT_FREEZE, dt_total=STEP_SEC[, return_vflux=True])
qb, nsub[, vfl] = out
```

bound at import as
`functools.partial(fct_lr.advect_step_batch, cfl=ADV_CFL, dtype=float32 if ADV_F32)`.
Everything the SL path needs to know:

- **Layout** is `(tracer, lev, lat, lon)`; dinosaur's nodal layout is
  `[lon, lat]` with level leading. Two transposes per call, cheap.
- **Grid**: `lat = linspace(-90, 90, 192)` *with nodes at both poles*,
  `lon = arange(288) * 1.25` starting at 0. Levels index 0 = top
  (1.2 hPa) … 23 = bottom (143 hPa), pressure increasing with index.
  `DP = np.gradient(PLEV_PA)`; there are no explicit interface pressures,
  `fct_lr._lr_vert` builds `xe = [0, cumsum(dp)]`.
- **Winds**: `winds(t)` (coupling.py:1461) returns CESM `U`, `V` [m/s] and
  `OMEGA` [Pa/s] at the level *centers* for hourly index `t`. The step passes
  only the two snapshots bracketing the 6-h step (`it0`, `it1 = it0 + 6`)
  and `fct_lr` interpolates linearly between them at each sub-step. **The
  hourly winds in between are on disk and currently unused** — the SL path
  should read `winds(it0 + k)` for k = 0..6 and take six 1-h sub-steps.
- **Vertical velocity**: with `ADV_WCONT=1` (default) `fct_lr` discards the
  interior CESM omega and rederives face omega from discrete continuity so
  its own divergence terms cancel (`fct_fast._omega_continuity`). For the SL
  path that is unnecessary: use CESM omega directly, interpolated to faces
  (conservation is handled by the fixer, §6, not by wind consistency). The
  top-face anchor and the sign convention: omega > 0 = descent = increasing
  pressure = increasing level index.
- **Polar caps**: rows with |lat| > 80° are not advected; `fct_lr` stirs each
  cap into one cell and resets it to `qfroz` after the step; `coupling.py`
  refreshes `qfroz` on those rows from MAM4 every step (coupling.py:2455).
  dinosaur has no polar singularity (Cartesian departure points, cross-pole
  halo), so the cap hack is not *needed* for stability — but keep the
  post-step reset so the BC bookkeeping (`B_adv_pol`, `B_bc`) stays what it
  is. First pass: advect all rows, then overwrite the cap rows from `qfroz`.
- **Open vertical faces** (`ADV_WCONT=1`): `qfroz[:, 0]` and `qfroz[:, -1]`
  are the reservoir concentrations (`N_BC_TOP = N_BC_BOT = 1`) that inflowing
  air carries across the top/bottom faces; `fct_lr._lr_vert` uses them as
  `bc_top/bc_bot` for out-of-domain departure points and returns the signed
  face fluxes (`vfl`, used for `B_vf_in/out`). The other mode,
  `ADV_WCONT=0`, seals the faces and *Dirichlet-pins* the edge levels to the
  reservoir after the step ("clamp", coupling.py:1583–1600). §6 argues the
  SL path should use the Dirichlet mode.
- **Return**: `(qb, nsub)` plus `vfl` only if `return_vflux` is in the
  signature (`ADV_VFLUX` is probed with `inspect.signature`,
  coupling.py:1574). An SL driver without `return_vflux` just turns that
  diagnostic off; nothing else depends on it. Return `nsub = K`.
- **dtype**: `ADV_F32=1` runs the sweeps in float32 and casts back to
  float64; keep that (the cubic gather in float32 is fine; do the fixer's
  sums in float64).
- `TRACER_CHUNK` splits the stack to cap memory; honour it or ignore it (one
  float32 stack is 1.1 GB).
- The CI gate `scripts/validation/test_conservation.py` imports `fct_fast`
  and `fct_lr` directly and must keep passing untouched. Add a new test for
  the SL module rather than editing that one.

## 5. Blockers, each with its fix

### 5.1 Nodes at the poles (dinosaur refuses the CESM grid)

MEASURED: `sh.Grid(longitude_nodes=288, latitude_nodes=192, latitude_spacing='equiangular_with_poles')`
reproduces the CESM latitudes to 0.0000°. The refusal is only in the
cross-pole halo: `semi_lagrangian._latitude_halo_axis` mirrors the first
`_HALO_WIDTH = 2` rows about the pole (`south = -π - phi[1::-1]`), which for
`phi[0] = -π/2` duplicates the pole node and would give 0/0 Lagrange
weights, so it raises `'grids with nodes at the poles are not supported'`.

Fix (in a *copy* of the module under `src/advection/`, not in
`/data/dinosaur-sl`): when `phi[0] == -π/2`, mirror rows 1..H instead of
0..H-1 — `south = -π - phi[H:0:-1]`, `north = π - phi[-2:-H-2:-1]` — and the
same index shift in `_extend_with_pole_halo`
(`rolled[..., H:0:-1]` / `rolled[..., -2:-H-2:-1]`). The pole row itself
stays a normal node (all 288 longitudes hold the same value in CESM; the
antipodal roll is then a no-op there). Alternatively drop the two pole rows
and treat them as part of the 80° caps, which the coupling already resets.
Test: a cosine bell advected over the pole by solid-body rotation
(Williamson et al. 1992 case 1 with α = 90°) must come back where it
started with no new extrema.

### 5.2 Pressure levels as the vertical coordinate

`departure_points_3d` takes `vertical_nodes=VerticalNodes(centers, boundaries)`
— any fixed, increasing node coordinate — and `sigma_dot` = the rate of
change of that coordinate at the `layers + 1` boundaries. Use
`centers = PLEV_PA` (increasing with index, which is what it requires),
`boundaries = [0, cumsum(DP)]` shifted so the centers sit inside their
layers (or geometric means of adjacent centers — be explicit, the choice
changes nothing to first order), and `sigma_dot = omega` [Pa/s]
interpolated to the boundaries, zero at the two outermost faces only if the
faces are sealed (§6). Departure pressures are clipped to
`[centers[0], centers[-1]]` with constant extrapolation beyond — exactly the
"closed extended column" §6 relies on. It wants a `coords` object only for
`coords.horizontal`; pass a `CoordinateSystem` built from the Grid and a
dummy `SigmaCoordinates`, or copy the function and take `grid` directly.
Vertical interpolation: `vertical_order='linear'` to start (the band is
thin and log-p spacing is ~1.25× per level); the IFS-style cubic degrades to
linear at the ends anyway.

### 5.3 Open faces and the BC rows

See §6. The recommendation is the Dirichlet mode: advect the band plus the
two reservoir levels as a closed column, then reset the reservoir levels
(and the polar rows) from `qfroz` as `ADV_WCONT=0` already does. The
exchange through the faces is then booked in `B_bc`, not in `B_vf_in/out`.

### 5.4 Trajectory length and the winds

Air moves up to ~2000 km (≈20 cells) in 6 h here. The departure solve
converges when `dt · max‖∇V‖ < 1`; with 1-h sub-steps and stratospheric
shear that holds comfortably, with a single 6-h step it may not near the
jet. Use K = 6 with the hourly CESM winds (one `winds()` read per hour,
which the CESM read stage already does for T/RH/OH), winds at each
sub-step's midpoint by averaging the bracketing hours. The 49 ms number is
for K = 6.

## 6. Conservation: can we conserve mass? Yes — here is how

### 6.1 Why the operator does not, and why a fixer needs a known target

Point-wise semi-Lagrangian remap is an interpolation, not a flux balance;
the sum over cells of the gathered values is not the sum before. Over a
closed domain the error is small and random (JCM: 1.3 % over 38 steps with
*no* fixer) and the standard remedy in every operational SL model (IFS,
UM, GEM) is a *mass fixer*: scale or shift the field after the step so that
the total equals a known target. **The target has to be known.** On the
closed sphere it is simply the mass before the step. In an *open* slab with
flux boundaries the target is "mass before + net face inflow", and the SL
gather cannot report the inflow separately from its own interpolation error
— a departure point below the bottom face pulls reservoir concentration into
the band and that is both the physical inflow and the thing a fixer would
try to undo. So:

### 6.2 Design: closed extended column + Dirichlet reservoir levels

1. Advect the stack on the **extended column** = band + the `N_BC_TOP`
   top and `N_BC_BOT` bottom reservoir levels (they are already part of
   `qb`; the reservoir values live in `qfroz` and are rewritten each step).
   Clip departure pressures to the extended column's center range (dinosaur
   does this). The extended column is then **closed**: nothing enters or
   leaves, and the exact target for every tracer is its extended-column
   mass before the step, `M0 = Σ_cells A·dp·q`.
2. Apply the mass fixer (§6.3) per tracer to hit `M0` to roundoff.
3. Overwrite the reservoir levels and the polar rows from `qfroz`
   (the `ADV_WCONT=0` "clamp" path in `coupling.py` already does exactly
   this, and books the change in `B_bc`). The physical face exchange is now
   "whatever the reset put in or took out", explicit and diagnosable, which
   is the same accounting the model used before `ADV_WCONT`.

With `N_BC_* = 1` the reservoir is one level deep; the departure points of
the bottom band level reach ~0.1 level per hour (JCM: 0.097 layers per 6 h at
T63; here the omega-driven exchange is ~10 %/day of the slab), so one
reservoir level is enough for 1-h sub-steps. If a departure point would go
*past* the reservoir level it is clipped to it, which is the constant
extrapolation the current scheme also uses.

Cost of this choice: the transport across the slab faces is no longer a flux
boundary. `BC_BOT_AER` (scaling of the aerosol inflow) still works because it
scales the reservoir value. `B_vf_in/out` go away; `B_bc` carries the
exchange. The 30-day A/B against `fct_lr` (§7) is what says whether the
difference matters.

### 6.3 The fixer: Bermejo–Conde local proportional correction (recommended)

Priestley (1993), Bermejo & Conde (2002), as used in the IFS
(Diamantakis & Flemming 2014). Per tracer, per step:

```
q_H  = cubic, quasi-monotone gather            (the answer)
q_L  = linear gather at the same departure points   (one extra gather, 'linear' order; ~+30 % cost)
m_a  = A_a · dp_a                               (cell air mass weight, float64)
D    = Σ_a m_a q_H,a  −  M0                     (the mass error, float64)
w_a  = max(0,  sign(D) · (q_H,a − q_L,a))^β     β = 1 (or 2 to concentrate the correction)
λ    = D / Σ_a m_a w_a
q_a  = q_H,a − λ w_a
```

The correction is taken only from cells where the high-order answer
overshoots the low-order one in the direction of the error, i.e. where the
interpolation *created* the spurious mass, so it is local, it preserves
positivity (`q_H − λw ≥ q_L ≥ 0` because the limiter keeps `q_L` within the
corner range), and it leaves smooth regions untouched. If `Σ m_a w_a = 0`
(no cell overshoots in the needed direction — happens only when D ≈ 0) fall
back to the global fixer. Run it in float64 on the sums; the field itself can
stay float32.

**Bin consistency.** Number and mass of a bin are two tracers remapped by
the *same* departure points, so away from the limiter their ratio is
preserved exactly (the gather is linear in the field). The limiter and the
fixer act per tracer and can move the mean particle mass outside the bin
bounds; the coupler's two-moment clip (`coupling.py:944`, `clipM/M0` in the
step log) already repairs that and reports it. Watch `clipM/M0` in the A/B:
it is ~1e-7 with `fct_lr` and should stay small.

### 6.4 Alternatives, ranked

| option | conserves | locality | cost | verdict |
|---|---|---|---|---|
| **global proportional fixer** (JCM): `q ← q · M0/M1` per tracer | exact | none — rescales the whole field | free | **do this first** for the 30-day A/B; it is three lines and isolates the transport accuracy question from the conservation question |
| **Bermejo–Conde local fixer** (§6.3) | exact | corrects where the overshoot is | +1 linear gather | **the production answer** |
| air-mass-weighted SL: transport ρ and ρq along the same trajectories, `q' = (ρq)'/ρ'` (the `fct_lr` trick in SL form) | no | — | +1 tracer | preserves constants and consistency but does not conserve; not a fixer substitute |
| finish `fct_fast`'s plan: FFSL in latitude, large-Courant vertical, keep flux form | exact, by construction | exact local | big rewrite of the PPM sweeps; still a sweep per sub-step over the vertical/x-limited count | the conservative-by-construction route if SL + fixer fails the A/B; weeks, not days |
| cell-integrated SL (CSLAM / SLICE) | exact local | exact | polygon remap, large implementation | not for a first pass |

## 7. Validation plan (gates, in order)

1. **Unit tests, CPU, seconds** (new file `scripts/validation/test_sl_transport.py`,
   do not touch `test_conservation.py`):
   - a constant field is returned unchanged to roundoff (SL does this by
     construction; flux form does not — a good sanity check of the halo/pole
     patch);
   - Williamson case 1 (cosine bell, solid-body rotation, α = 0° and 90°)
     one full revolution: L2 error < a few %, max ≤ 1 (no new extrema), min ≥ 0;
   - after the fixer, `|M1 − M0|/M0 < 1e-12` for a random divergent wind on
     the closed extended column;
   - bin consistency: advect `(N, M)` with `M/N` inside the bounds and count
     violations before the clip.
2. **In-loop timing**: a 1-day run with `PROFILE=1`; expect `advect` ≲ 0.2 s
   including transposes and the reads (vs 6.6 s).
3. **30-day A/B against `fct_lr`** with the *physics* microphysics
   (`--micro-engine fast`), same scenario as the emulator benchmark
   (`--n-hours 720 --inj-so2 10 --inj-lat 30 --inj-mirror 1 --frame-every 24`),
   judged with `scripts/validation/compare_runs.py bench_phys30d <tag>` on
   the same table (number/mass/SO2 burden, r_eff, AOD, TOA forcing at days
   1/5/10/20/30) and `plot_emulator_summary.py`'s Hovmöller/map panels.
   There is no "truth" between two numerical schemes; the questions are
   whether the global quantities agree to the ~1 % the emulator already
   holds, whether the plume filaments (day-30 map) have the same width, and
   whether the slab's face exchange (`B_bc` vs `B_vf_in/out`) is the same
   size. Run K = 1, 2, 6 to show convergence in the sub-step count.
4. Then the production combination: `--micro-engine emulator` + SL
   transport vs `bench_phys30d`.

## 8. Ordered task list

1. Copy `dinosaur/semi_lagrangian.py` (and the three small modules it imports
   from: `spherical_harmonic.Grid` for the nodal mesh, `coordinate_systems`,
   `sigma_coordinates`, `tree_math`) into `src/advection/sl_core.py`, or
   vendor dinosaur as a submodule under `models/` like `tomas-jax` and patch
   the halo there. Patch §5.1. Unit test the pole crossing.
2. Write `src/advection/sl_lr.py::advect_step_batch(qb0, u0..w1, lat, dp, qfrozb, dt_total, *, K=6, dtype, hourly_winds=None)`
   with the signature of §4 (ignore `cfl`, return `(qb, K)`), doing:
   transpose → departure points per 1-h sub-step (3-D, pressure nodes, CESM
   omega at faces) → 82 cubic gathers (`jax.vmap` over tracers) → fixer
   (global first) → transpose back. Keep the reservoir/polar reset in
   `coupling.py` (the `ADV_WCONT=0` path).
3. Add an `ADV_SCHEME=lr|sl` env/flag in `coupling.py` (bind the partial
   from it, like `ADV_CFL`/`ADV_F32`), `--adv-scheme` in `run_prod.py`,
   row in `docs/CONFIGURATION.md`. Default stays `lr`.
4. Pass the hourly winds: `winds(it0 + k)` for k = 1..5 in the step loop when
   `ADV_SCHEME=sl` (the read is 0.2 s/step for the 6-h snapshots; hourly is
   ~1 s — fine, or read U/V/OMEGA for the whole step at once).
5. Gates 7.1–7.2.
6. 30-day A/B (7.3) with the global fixer. Decide from the table.
7. Bermejo–Conde fixer, re-run the A/B, promote to default if it holds.
8. Docs: a section in `docs/PROCESSES.md` (transport), `CHANGELOG.md`,
   `MANIFEST.md`; the standing-caveats list in `MANIFEST.md` if anything
   is left open.

Nothing in `src/radiation/` is to be touched (standing instruction), and
`--micro-engine fast` must stay byte-for-byte unchanged.

## 9. Environment and recipes

- Repo: `/data/AIDE-SAI-link`, branch `feat/frame-levels`. Files are owned by
  `ali:ubuntu`; `/data` is NFS with root-squash, so write as `ubuntu`:
  `cat file | sudo -n -u ubuntu tee /data/.../dest`. Compile-check with
  `sudo -n -u ubuntu /data/aide_emu_venv/bin/python -m py_compile <file>`.
- Python: `/data/aide_emu_venv` (jax 0.6.2 cuda12 + torch; `run_prod.py`
  execs whatever python launched it). The venv JCM used for dinosaur
  (`/data/jaxgcm_venv`) points at a uv-managed python that no longer exists
  on this node — do not use it. dinosaur's runtime dependencies were
  installed into `aide_emu_venv` on 2026-10-13 (`tree-math pint
  scikit-learn dask fsspec pandas xarray-tensorstore`); the source is
  `/data/dinosaur-sl` (the dinosaur checkout JCM used, with the
  `semi_lagrangian` module and its tests) — use
  `PYTHONPATH=/data/dinosaur-sl` until it is vendored.
- GPU as `ubuntu`:
  `sudo -n -u ubuntu env CUDA_VISIBLE_DEVICES=<n> LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=/data/dinosaur-sl /data/aide_emu_venv/bin/python <script>`.
  GPUs 1–3 are reserved; use 0 and 4–7. `nvidia-smi` is
  `sudo -n chroot /run/nvidia/driver nvidia-smi`.
- A production-style run, from `runs/<TAG>/`:
  `LD_LIBRARY_PATH=... /data/aide_emu_venv/bin/python ../../src/run_prod.py --gpu <n> --micro-engine fast --n-hours 720 --inj-so2 10 --inj-lat 30 --inj-mirror 1 --frame-every 24 --out-tag <TAG>`
  (run in tmux as ubuntu; ~45–65 min for 30 days). `--dry-run` prints the
  resolved environment.
- Reference runs to compare against: `runs/bench_phys30d` (fast physics,
  `fct_lr`), `runs/bench_chem_hyb1num4_30d` (production emulator,
  `fct_lr`). Both carried `MICRO_REF` so their wall times include two
  microphysics engines — use the per-stage `[prof]` medians, not the step
  wall clock.
- A full day-30 state for offline tests:
  `runs/chemdump/debug_nucleation/premicro_state_step120.npz`
  (`num, mas (40,24,192,288)`, `so2, h2so4, T3d, pres3d, rh3d`, `lat, lon, plev`).
  Winds for the same hours come from the CESM h1 archive via
  `coupling.winds()` (see `inputs/README.md` for `CESM_DIR`).

## 10. Files

| path | what |
|---|---|
| `src/advection/fct_lr.py`, `fct_fast.py`, `fct_core.py` | the current scheme; `fct_lr` is production, the others hold the shared pieces and the FFSL-in-x experiment |
| `src/coupling.py:2225–2250, 1461, 1550–1600, 2455–2490` | the advect call, `winds()`, the BC modes, the reservoir/polar refresh |
| `scripts/validation/test_conservation.py` | the CI conservation gate for the current scheme (leave it) |
| `scripts/validation/bench_sl_transport.py` + `outputs/emulator_bench/bench_sl_transport.log` | the 49 ms measurement |
| `scripts/validation/compare_runs.py`, `plot_emulator_summary.py` | the 30-day A/B table and the figure page |
| `/data/dinosaur-sl/dinosaur/semi_lagrangian.py` | the operator (1145 lines, well documented; read `_latitude_halo_axis`, `_extend_with_pole_halo`, `GridInterpolator`, `interpolate_3d`, `departure_points_3d`, `transport_scalar`) |
| `/data/jcm_tracer_emulator/b0_reference_operator.py` | how JCM drove it (K sub-steps, midpoint winds) |
| `/data/TRACER_TRANSPORT_EMULATOR_RESULTS.md`, `/data/JCM_TRACER_TRANSPORT_SPEC.md` | JCM's findings and spec |
| `docs/EMULATOR.md` | the microphysics side of the speed story, for context |
