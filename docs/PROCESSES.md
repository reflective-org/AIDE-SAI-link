# Process reference — transport and microphysics

Companion to `README.md`, `COUPLING_VARIABLES.md` and `BOUNDARY_CONDITIONS.md`.
Written against the configuration that produced `zonal90d` (2026-07-26): 90 days,
10 Tg SO2/yr equatorial zonal ring at 51.7 hPa, physical micro + radiation.

**Read this first — which driver ran.** `zonal90d` used
`driver_fast.py`, which monkeypatches `coupling.run_microphysics_full`
with `tomas_jax.fast.run_fast` and swaps the advection operator. Several
`coupling.py` code paths (`_chain_substep`, `NUC_NH3`/`NUC_ORG`/`NUC_FION`,
`NUC_FN_MAX`, `MICRO_SUBSTEPS`) are therefore **dead** in this run. The run log's
header names the active driver; check it before reasoning about any of them.

---

# 1. Transport

## 1.1 What is transported

**82 tracers**, advected together with shared winds each 6 h step:

| tracers | count |
|---|---|
| aerosol number, per bin | 40 |
| aerosol dry mass, per bin | 40 |
| SO2 (gas) | 1 |
| H2SO4 (gas) | 1 |

All 82 share one wind field, so the substep count is common to all — the batch is
exact, not an approximation. The count is `2·NBINS + 2`, so `N_BINS` changes it
(a transport-only run at `N_BINS=1` advects 4); under `MICRO=off` the aerosol rows
are independent passive tracers differing only in fall speed, which is what makes
one run a whole size-dependence of drainage.

## 1.2 Scheme: Lin-Rood flux form

`fast_advection/fct_lr.py`. PPM reconstruction, Zalesak limiter, dimensionally
split (x, y, z sweeps), CFL 0.5, f32.

The predecessor (`fct_fast.py`) used the **advective form**: flux divergence plus
a `+q*div(c)` correction. That form is *consistent* — a constant field is
preserved exactly — but **not conservative**. Global burden changes by
`sum(q*div)`, which vanishes only if the discrete 3-D wind is divergence-free.

Lin-Rood removes this **by construction rather than by fixing the winds**.
Following Lin & Rood (1996), carry air mass as a prognostic field and advect the
pair `(rho, rho*q)` in pure flux form:

```
rho'    = rho    - div(F_air)/area      F_air = C * m_face * rho_face
(rho q)'= (rho q)- div(F_trc)/area      F_trc = F_air * q_face
q'      = (rho q)' / rho'
```

Every flux appears twice with opposite signs, so `sum(area*dp*rho*q)`
**telescopes to the boundary fluxes alone** — exactly, to roundoff, for any wind
field, divergent or not, regardless of splitting order. The wind's inconsistency
no longer destroys tracer mass; it appears as `rho` drifting from 1, an explicit
diagnosable field.

> This property matters MORE for emulator winds, not less: a learned wind field
> will not satisfy discrete continuity, and flux form makes that harmless.

Cost: ~5x more substeps than the advective form (no integer-shift FFSL available
in flux form). Isolated conservation: **~1e-15/day**.

`rho_reset=True` remaps air mass back onto the fixed pressure grid each step, so
the caller's ordinary burden diagnostic stays exact without persisting `rho`. The
rescaling is ~1%/step; resetting every step keeps it small.

## 1.3 Four bugs fixed first

Residual improved ~50x. All four are geometry, not numerics:

| bug | consequence |
|---|---|
| wrong grid spacing | systematic transport error |
| no cos(phi) metric | area weighting wrong away from equator |
| raw CESM omega instead of continuity | divergence residual ~5e-6 1/s |
| polar int32 overflow | corruption in cap rows |

## 1.4 Vertical velocity: rederived, not read

Archived CESM omega does not satisfy discrete continuity — interpolated to fixed
pressure levels, subsampled to 6 h, re-interpolated in time, and the slab's
top/bottom faces forced to zero. So omega is rederived by integrating continuity
downward from the top face:

```
omega_{k+1/2} = omega_{k-1/2} - dp_k * (Sx + Sy)_k
```

using the SAME discrete divergence operators the x- and y-sweeps apply, which
cancels their divergence terms to roundoff. **Interior divergence residual drops
from ~5e-6 1/s to ~4e-22 1/s.** The top face is anchored to CESM omega there; a
constant offset in the anchor shifts through-slab flux but not conservation.

## 1.5 Polar caps

**Problem 1 — cell sizes.** The +-90 rows are half-cells, `ac ~ dphi/8`, an **8:1
volume ratio** to their neighbour. PPM assumes uniform cell volumes and is invalid
across such a jump: **1000x overshoots and negative mass** in the +-89/90 rows,
which was **99% of the residual mass drift**.

**Problem 2 — zonal Courant number.** `cx ~ u*dt/(R*cos(lat)*dlam)` diverges as
`lat -> 90`, reaching ~1e18 and **overflowing int32**.

**Fix:** treat each cap as one well-mixed bucket. Everything poleward of 80 deg
(22 rows) becomes, per level, the area-weighted mean of its cells.

Where it bites in `_lr_step_3d`:

```
-- stir caps --          BEFORE anything
x sweep      cx forced to 0 inside caps
y sweep      mf = 0 on faces internal to caps
z sweep      unmodified
-- stir caps --          AFTER everything
```

* `_mix_caps` runs **twice per substep**, on `rho` and `rhoq` separately (each
  conservative) so `q = rhoq/rho` comes out well mixed.
* **x sweep:** `cx` forced to zero inside caps. Not because zero is the right
  wind — the row was just made zonally uniform, and a uniform row is invariant
  under any east-west shift. Identical answer, dodges the overflow. This is why
  the stir MUST precede the x sweep.
* **y sweep:** internal cap faces get `mf = 0`. Mass crosses the cap EDGE only,
  so PPM never reconstructs across the bad volume jump.
* **z sweep:** untouched.

**Exactly conservative** — an area-weighted mean leaves `sum(ac*q)` unchanged.
Hemispheres handled separately. Cap-edge Courant number ~0.02 (cap is 1.5% of
global area).

**Cost:** no structure inside 80-90 deg. Irrelevant for equatorial SAI:
`B_adv_pol = +2.9e-4 M0` vs `B_adv_np = -1.09 M0`, i.e. 0.03%.

> **Do not confuse with the polar refresh.** `coupling.py:1690` writes CARMA/CESM
> values into `qfroz` at polar rows, all levels, every step. In `ADV_POLAR=zonal`
> (`pol_mode=1`, this run) that write is **dead**: the line applying it
> (`fct_lr.py:214`) is in the `else` branch. The caps are stirred, NOT clamped.
> The write only reaches the top/bottom rows of `qfroz`, which feed the vertical
> faces. The legacy `pol_mode=0` clamp was the dominant term (~55%) in the old
> -0.3%/day mass leak.

## 1.6 Initial and boundary conditions

> [!IMPORTANT]
> **This section records one specific CARMA-IC experiment, not the production
> configuration.** It is kept because the reasoning below (why `BC_BOT_AER`
> matters, what the bottom face does) still applies. What has changed since:
>
> | | this section | production today |
> |---|---|---|
> | IC/BC source | static CARMA frame (`AER_SRC=carma`) | per-step MAM4 (`AER_SRC=mam4`) |
> | band | 1–100 hPa, BCs at 13.3 / 87.8 hPa | 1–150 hPa, BCs at 1.2 / 143 hPa |
> | bottom aerosol inflow | `BC_BOT_AER=0` (aerosol-free) | `BC_BOT_AER=1.0` (full reservoir) |
> | gases | Dirichlet-clamped (`BC_GAS=clamp`) | open faces (`BC_GAS=flux`, default since 2026-07-30) |
>
> The current settings are echoed in every run header; `docs/CONFIGURATION.md`
> lists the defaults and `MANIFEST.md` explains why each is what it is.
>
> The reasoning below applies **directly** to the advection-only experiment, which
> runs `BC_BOT_AER=0` *and* `BC_TOP_AER=0` on a prescribed uniform PSD
> (`AER_SRC=fixed`) over 0.03–150 hPa: both faces are aerosol-free, so the band
> drains and nothing refills it. What differs from this section is only the IC —
> one uniform mixing ratio instead of a CARMA frame.

**IC:** CARMA, `cesm2.2_CARMA16node_freerun_1wk_19910601_1deg`, frame 0.
PRSUL + MXAER projected onto 40 TOMAS bins as dry sulfate-equivalent radius at
rho = 1923 kg/m3, via **sub-bin remap**. Direct binning produced a day-0 "comb"
(every other bin exactly zero); fixing it raised baseline AOD 16%.
Initial burdens: N = 9.643e15, M = 7.667e-4 (**0.669 Tg**).

> 99.2% of the CARMA IC's NUMBER sits in the bottom two levels (73% at 87.8 hPa
> alone) as ~17600 /cm3 of 2-8 nm particles. Full-slab N/N0 measures that
> tropopause pile, not stratospheric aerosol. Use `DIAG_CORE_HPA=20,55`.

**BC:**

| | setting |
|---|---|
| BC levels | 1 at each end (`N_BC_TOP = N_BC_BOT = 1`) — 13.3 and 87.8 hPa |
| face type | FLUX (continuity omega, open faces) |
| outflow | free, both faces |
| aerosol inflow, top face | 1x static CARMA reservoir (one frozen frame) |
| aerosol inflow, bottom face | **0x — aerosol-free upwelling** (`BC_BOT_AER=0`) |
| aerosol edge levels | prognostic, not overwritten (`BC_EDGE=open`) |
| gas inflow + edge levels | **CESM** SO2/H2SO4, Dirichlet-clamped, re-read every 6 h step (`BC_GAS=clamp`) |
| polar rows (\|lat\|>80, 22 rows) | stirred, mass-conserving (`ADV_POLAR=zonal`) — **not clamped** |
| mass fixer | none — open system by construction |

`BC_BOT_AER` and `BC_EDGE` came from that run's environment, NOT the code
defaults. (At the time those defaults were 1.0 and `clamp`; `BC_EDGE` now
resolves to `open` whenever `ADV_WCONT=1`, which is itself the default.) Record
the env, not the defaults.

**Why `BC_BOT_AER=0`** — two reasons pointing the same way:
1. *Physical.* Air crossing 88 hPa in the tropics is Brewer-Dobson ascent of
   tropospheric air with essentially no stratospheric sulfate. Gases are
   deliberately NOT scaled — tropospheric air genuinely is the SO2 source.
2. *Numerical.* `BC_BOT_AER=1.0` would continuously pump the CARMA IC's 87.8 hPa
   ultrafine pile (~17600 /cm3 of 2-8 nm) into the model, ~10%/day of slab air.

`coupling.py:162` flags this as a **first-order control on the steady-state
burden**, not a minor switch.

**Cost:** exported aerosol never returns. Since 88 hPa is above the tropical
tropopause, some exported mass is still stratospheric in reality. Residence time
is a **lower bound**, steady-state burden an **underestimate**. The justification
— BDC asymmetry, clean tropical inflow vs one-way extratropical descent — is
**unverified**. Cheap check: is bottom-face inflow tropics-concentrated?

## 1.7 Diagnostics and verification

Transport is budgeted in two telescoping stages, `B_adv_np` (non-polar) and
`B_adv_pol` (caps), plus a **vertical face diagnostic** (`B_vf_in`, `B_vf_out`)
that integrates actual flux through the open faces. Subtracting:

```
residual = (B_adv_np + B_adv_pol) - (vf_in + vf_out)
         = (-1.0934 + 0.0003)     - (-1.0927)
         = -6.47e-4 M0
```

Everything advection did that cannot be explained by measured face flux is
numerical error.

| metric | value | what it claims |
|---|---|---|
| staged budget closure | 4e-16 /step | **audit, not accuracy** — the six terms are burden differences at checkpoints, so they telescope by construction. Proves no stage is unaccounted for; would read 4e-16 even if the physics were wrong. |
| flux-form conservation (isolated) | ~1e-15/day | scheme property in a CLOSED domain |
| **advection numerical residual** | **6.5e-4 over 90 d** | **the real accuracy number** — in situ, with open faces, polar stirring, per-step rho remap and sweep splitting all active |
| face exchange at day 90 | top -0.031, bottom -1.06, net -1.09 M0 | both faces are net EXPORTERS |

Residual in perspective: 0.06% of the advective signal, 0.016% of final burden,
0.06% of the physical sink rate, ~0.43 Gg against a 2.79 Tg burden.

Note the face labels mislead: `in`/`out` mean top/bottom face, not inflow and
outflow. Both are negative. Top-face export (3%) is real — the plume spreads
upward into 13-20 hPa. These are NET per-face globally summed values and cannot
separate gross inflow from gross outflow.

---

# 2. Microphysics

## 2.1 What is evolved

| array | contents |
|---|---|
| `Nk` | number per bin, 40 values [#/m3] |
| `Mk` | mass per bin x species — `SRTSO4=0` prognostic dry sulfate, `SRTH2O=1` water (diagnostic, re-equilibrated each step) |
| `Gc` | `GH2SO4=0`, `GSO2=1` |

`ICOMP=2`, `IDIAG=1` — only sulfate is prognostic dry mass. The full TOMAS-JAX
core carries `ICOMP=44`.

## 2.2 Bin grid

| | |
|---|---|
| bins | 40 |
| smallest boundary | `xk0 = 4.553e-24 kg` -> **1.70 nm** diameter |
| spacing | mass-doubling -> diameter ratio 2^(1/3) = **1.2599** |
| largest boundary | **17.5 um** |
| density convention | 1770 kg/m3; bin edges are MASS boundaries, do not vary with RH |

The 1.7 nm start was chosen so the Dunne nucleation cluster lands near the bin-0
geometric mean.

## 2.3 Chain and substepping

One 6 h coupling step = **60 inner steps x 360 s** (`FAST_DT=360`). Each inner
step (`tomas_jax/fast/step.py:80`):

```
0. gas production (host tendencies)
1. SO2 + OH chemistry           -> H2SO4
2. nucleation                   -> new particles in bin 0
   MNFIX
3. water equilibrium            (so coag sees current wet sizes)
4. coagulation                  adaptive Euler substeps + MNFIX
5. condensation                 PPM, CFL substeps
6. water equilibrium
   MNFIX
```

Three MNFIX calls and two water re-equilibrations per 360 s step.

### On `FAST_DT = 360`

**Not a TOMAS default.** TOMAS-JAX's full model runs at **60 s**
(tomas-jax `docs/architecture.md:394`; coagulation there uses fixed 3-10 substeps per 60 s
step). 360 s is the GPU-fast reduced model's own design point — a 6x coarsening
adopted for performance (`fast/run.py:5`: "6 XLA program invocations with no host
round-trips"). It propagates as a default through `make_fast_step`, `run_fast`
and `driver_fast.py:50`.

It is empirically supported, but note the direction of the argument: the accuracy
machinery was built to make 360 s viable, not the reverse. Calibration measured
AT 360 s (tomas-jax `docs/gpu_fast.md`):

| state | fixed 3x120 s | adaptive dt*lambda/0.05 |
|---|---|---|
| jagged high-N (adversarial) | **-12% mass**, clamp +73% | closure 6e-11, sig-bin error <=2% |
| realistic lognormals (1e3-1e5/cm3) | closure ~1e-4..1e-1 | n_sub 1-15, closure <=3.5e-8, dist error <=0.8% |
| nucleation burst (1e6/cm3 bin 0) | — | closure 3e-8, sig-bin error <=~1% |

One residual dt-dependence is documented (`fast/condensation.py:189`): strong
growth can advect a donor bin's whole population upward within a step, stranding
deposited mass that MNFIX then destroys — "worst case ~2%/step of a cell's mass
at dt=360 s; the full model has the same leak, smaller at dt=60 s." The fast
model fixes it by redistributing to surviving bins.

**GAP: no dt-convergence sweep has been run in the COUPLED model.** All of the
above is box-level. In the plume (tropical, 51.7 hPa) tau_cond ~ 695 s, so 360 s
is ~0.5 condensation e-folds of operator-splitting exposure (60 s would be
~0.09). But see 2.4a — the plume value is NOT representative of the domain.

### 2.4a H2SO4 lifetime — why the gas must be advected

The H2SO4 condensation lifetime `tau_cond = 1/CS`, `CS ~ sum(N*r^2)`, collapses
wherever there is little aerosol surface area. Day-90 distribution over all cells:

| pct | 5th | 25th | 50th | 75th | 95th |
|---|---|---|---|---|---|
| tau_cond | 0.38 h | 2.87 h | **6.61 h** | 11.62 h | 22.90 h |

**The median cell's H2SO4 lifetime EXCEEDS the 6 h coupling step.** Area-weighted,
**49% of the domain** has `tau_cond > 6 h`; **99.3%** exceeds the 360 s substep.

| level | median tau_cond | cells with tau > 6 h |
|---|---|---|
| 13.3 hPa | 21.8 h | **100%** |
| 20.1 hPa | 11.6 h | 73% |
| 29.7 hPa | 7.3 h | 54% |
| **51.7 hPa (plume)** | **2.9 h** | **1.8%** |
| 87.8 hPa | 7.9 h | 64% |

The injection ring is the ONE place H2SO4 is short-lived. So advecting H2SO4 is
not bookkeeping — over half the domain it is a genuinely transported long-lived
vapour, and pinning it to its grid cell would be a real transport error.

**Where this bites is the size distribution, not mass.** Total H2SO4 burden is
~0.26 Gg against 2.79 Tg aerosol, so placement changes mass by <=1e-4. But
placement decides whether vapour **nucleates or condenses**: high-CS cells
condense it onto existing particles, low-CS cells accumulate it until it
nucleates. That is an r_eff lever.

Independent confirmation in-code: the `BC_GAS` comment records that clamping
H2SO4 at the 13.3 hPa top level acted as "an infinite gas SOURCE feeding
nucleation with no particle SINK" — that level went from 0.3% to ~50% of the
model's total number in 24 h. Same physics: at 13.3 hPa, 100% of cells have
`tau_cond > 6 h`.

## 2.4 Processes

### Chemistry
SO2 + OH (+M) -> H2SO4, **Sun et al. (2022) Troe** rate with H2O enhancement,
pseudo-first-order analytic decay, product by stoichiometry (98/64). OH
prescribed from CESM. Note this differs from canonical TOMAS in CESM/GEOS-Chem,
where the host chemistry does the oxidation and TOMAS receives an H2SO4
production rate. Worth a methods sentence.

### Nucleation — Dunne (2016) BINARY-NEUTRAL ONLY
`Jbn` only. No ions (`Jbi`/`Jti`), no NH3 (`Jtn`), no organics (Riccobono).
`nucleation_step(Nk, Mk, Gc, temp, boxvol, dt, fn_scale)` takes only a scale
factor (`FAST_FN_SCALE=1.0`).

> `NUC_ORG`, `NUC_NH3`, `NUC_FION` and `NUC_FN_MAX` in `coupling.py` belong to the
> full-model path and are **dead in this run**. Binary-only nucleation gives ~3x
> fewer particles and ~12% higher AOD than the physical ternary scheme.

Three deliberate deviations from the full model:
1. **100% of cluster mass deposited as SO4.** Full model deposits 90% SO4 / 10%
   organic, with the organic fraction created from nothing.
2. **Sulfur-conserving gas depletion:** `dGc(H2SO4) = dM(SO4) * 98/96`. The full
   model's unclamped path removes H2SO4 kg equal to SO4 kg added, **creating
   sulfur at the 2% level**. Total-S budget now closes exactly.
3. **Exact analytic integration, not substepping.** With T fixed over the step,
   `dG/dt = -A*G^p`, p = 3.95, closed form
   `G(t) = (G0^(1-p) + (p-1)*A*t)^(1/(1-p))`. Validated against a 1e4-substep
   reference. `G(t) > 0` always, so there is no clamp branch and **no rate cap is
   needed**.

### Coagulation — frozen Brownian kernel, adaptive Euler
Kernel frozen per outer step; forward Euler + MNFIX per substep. Substep count
adaptive, **shared across the whole batch**:

```
lambda_k = kij_kk*N_k + sum_{j>k} kij_kj*N_j + 2*K1M_k/xk_k
n_sub    = clip(ceil(dt * max_over_cells_bins(lambda) / c_max), 1, cap)
```

Terms are per-particle loss frequencies: self-coagulation, collision with larger
bins, promotion by mass flux from below.

`c_max = 0.05` (keeps content-significant bins within ~1% of converged),
cap = 256. Typical 1-15 per 360 s; nucleation-burst cells demand 100+. Fixed
coarse substeps lose >10% of mass to the positivity clamp in bursts. Cost at
scale is managed by stiffness-sorted chunking (`FAST_SORT=1`), not by lowering
the cap.

Two rejected shortcuts, for the record: masking near-empty bins out of the lambda
criterion loses 3% of mass on adversarial states (a bin with negligible content
can still carry the whole promotion flux); net-rate-based criteria miss
gain-dominated stiffness entirely.

### Condensation — PPM, shared capped CFL loop
Per-cell CFL substep counts (up to ~200 in bursts) replaced by ONE shared loop of
`n_glob = min(max_over_cells(n_sub), 40)`. Cells needing fewer substeps run at a
smaller Courant number — **strictly more accurate, never less**; only the capped
case degrades, reported via `cap_hit`. The `lax.cond` dump/PPM/no-op branches
become `jnp.where` masks (under batching every branch runs anyway).

### Water — Tabazadeh (1997)
Equilibrium wt% where solution water vapour pressure matches ambient. Replaces
the ISORROPIA ammonium-bisulfate fit. Fitted T = 185-260 K, 10-80 wt%; outside
that range `ln P = a + b/T + c/T^2` is **smoothly extrapolated, not clamped** —
clamping at 260 K would bias an entire global run's troposphere. RH clipped to
[1e-3, 99]%.

### MNFIX — mass/number consistency repair
Three phases: empty-bin reset, extreme out-of-range trim, partial transfer to a
computed target bin. Vectorized version replaces three sequential 40-iteration
loops (~120 dependent kernel launches) with a one-hot scatter, 2 fixed sweeps.
Number and mass **conserved by construction**; only operation order differs from
`mnfix.f`.

## 2.5 Conservation architecture

**(a) Two-moment clip at the coupling boundary.** Advected `Nk` and `Mt` arrive
independently, so `Nk*xk_lo <= Mt <= Nk*xk_hi` can be violated.
`driver_fast.py:146` clips and reports burden-weighted add/remove:
`clipM/M0 +2.2e-07 / -1.3e-07` at day 90 — seven orders below the signal.

**(b) MNFIX**, three times per 360 s step, conservative by construction.

**(c) Staged budget.** `B_micro` is the burden difference across the whole micro
stage — **+4.274 M0**, a genuine mass source because gas->particle conversion
happens inside it.

## 2.6 Verification

| check | value | |
|---|---|---|
| sulfur closure, end-to-end | 1.99 Tg SO2 oxidized -> 2.86 Tg aerosol (1.437 vs 1.531) | 6% |
| clip magnitude | +2.2e-07 / -1.3e-07 M0 | negligible |
| nucleation analytic vs 1e4 substeps | converged | ok |
| coagulation vs converged reference | ~1% at c_max=0.05 | ok |
| operator-split error (30 d stress test) | r_eff <1%, mass exactly 0% | ok |
| accumulation-mode number | 75 /cm3 | plausible |
| **effective radius** | **0.18-0.20 um** | vs 0.3-0.5 expected |
| total number | 46,000 /cm3 | **not physical** |

### Operator-split A/B (2026-07-27)

Real day-90 state, identical harness both arms, only the position of condensation
differs. Canonical = nucleation first; cond-first = condensation first. The two
**bracket** the true simultaneous solution, since they give nucleation its maximum
and minimum possible claim on the shared vapour.

| horizon | total number | accum-mode number | r_eff | dry mass |
|---|---|---|---|---|
| one 6 h step | ratio 0.21 | 1.0000 | +1.7% | 1.0000 |
| 30 d (SO2 replenished, no dilution — stress test) | -98.4% | -1.79% | **+0.53%** | **-0.00%** |

Bracket **narrows** monotonically (3.01% at day 1 -> 0.53% at day 30): the
splitting error is self-correcting, because r_eff is mass-driven and mass is
identical between arms. **Do not change the ordering.** Defensible statement:
*microphysical operator splitting contributes <1% to effective radius and 0% to
aerosol mass over 30 days, under conditions chosen to maximize the difference.*

Caveat to carry: total number in the nucleation mode is ordering-dependent at the
60x level and is **not a reportable quantity**.

## 2.7 Limitations

**Structural:**
1. **Ordering** — nucleation precedes condensation, inherited from tomas-jax's
   `_CANONICAL_PROCESS_ORDER`, which `make_step` enforces with a warning on
   deviation. Bounded as above.
2. **Binary-only nucleation** — ~3x fewer particles, ~12% higher AOD than
   physical ternary.
3. **r_eff too small** — 0.20 vs 0.3-0.5 um. The open item; governs diminishing
   returns, so it is the one bias that could move a headline result.

**Parameterization:**
4. **OH prescribed with no plume depletion** (`coupling.py:1091`), SO2 at 84x
   background -> lifetime 17.3 d vs ~35 d observed. A host-coupling gap, not a
   TOMAS setting.
5. **Reduced species set** — sulfate + water only.
6. **Frozen coagulation kernel** per 360 s outer step.

**Unreported diagnostics.** `coag_overflow` (mass past the 17.5 um top bin),
`cond_cap_hit` and `coag_cap_hit` are computed every step but printed only under
`DEBUG`/`PROFILE`. A silent cap degrades accuracy invisibly — these belong in the
standard log for production runs.

---

# 3. Open items

| item | status |
|---|---|
| dt-convergence sweep in the coupled model | **never run**; box-level only |
| r_eff plateau above/below 0.3 um | unresolved; needs the long run |
| bottom-face inflow tropics-concentrated? | unverified; justifies `BC_BOT_AER=0` |
| gross vs net face exchange | `vface` gives net only |
| polar `qfroz` write at levels 1-9 | inert in `pol_mode=1`; should be gated to stop implying a clamp |
| cap-hit counts in standard log | not surfaced |
