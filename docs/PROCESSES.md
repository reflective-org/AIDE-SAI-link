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

# 0. One coupling step

Everything runs on the `STEP_HOURS` cadence (6 h). A run of `N_HOURS` is
`N_HOURS / STEP_HOURS` **coupling steps**, and each step walks the stage list
below once, in this order. The ordering is not incidental — several stages are
placed where they are for a stated reason, noted in the code at each one.

| # | Stage | Off when | Timed as |
|---|---|---|---|
| 0 | SAI source — continuous SO2 (and optional H2SO4) into the injection cells | `INJ_SO2_TG_YR=0` | — |
| 1 | **Transport** — all 82 tracers advected on shared winds | — | `advect` |
| 2 | **Microphysics** — SO2 chemistry, nucleation, coagulation, condensation (`MICRO_ENGINE=fast`, the `tomas_jax.fast` physics, or `emulator`, the learned 6 h operator — `docs/EMULATOR.md`) | `MICRO=off` | `micro` |
| 2b | **Settling** — implicit upwind column sweep, the one true aerosol sink | `SETTLE=0` | `settle` |
| 3 | Vertical BC — refill the edge slabs from the MAM4 reservoir | — | `bc+polar` |
| 4 | Polar-cap refresh — freeze the target for the *next* step's advection | — | `bc+polar` |
| 4a | Reservoir refresh for the open vertical faces | — | `bc+polar` |
| 5 | **Radiation** — evolved bins → heating → prognostic `dT` | `RAD=0` | `radiation` |

Reading the ordering:

* **Stage 0 before transport**, so a fresh pulse is advected and oxidized in the
  same step it is emitted. It touches only the gas, so the aerosol staged budget
  is independent of this choice.
* **Stage 2b after stage 2**, so particles that just grew settle at their new
  size within the same step.
* **Stages 3/4/4a after the physics**, so the pinned levels are exactly the CESM
  reservoir at each step boundary. Stage 4 writes `qfroz`, which is consumed by
  the *next* step's stage 1 — the polar caps and the open faces are always one
  step behind by construction.
* **Stage 5 last**, on the end-of-step state. Its `dT` feeds the *next* step's
  microphysics, so the aerosol → radiation → temperature loop closes across the
  step boundary, not within it.

**Three nested levels of "step"** — the word is overloaded in the logs, and they
are three different things:

```
run  =  N_HOURS/STEP_HOURS coupling steps      ("step 8/8 (h 48)")
  step  =  the 8 stages above, once
    stage 1  =  ~168 CFL-limited advection substeps   ("nsub=168")
    stage 2  =  N_FAST_STEPS inner steps of FAST_DT  (60 x 360 s, driver_fast)
```

(Stage 2's inner stepping is driver-dependent: under `driver_fast.py` it is
`N_FAST_STEPS` x `FAST_DT`, and `MICRO_SUBSTEPS` is dead — see the note at the
top of this file. Standalone `coupling.py` uses `MICRO_SUBSTEPS` instead.)

`nsub` is chosen per step from the wind field, so it varies run to run; it is the
substep count shared by all 82 tracers (see 1.1).

With `PROFILE=1` each step prints its own breakdown, which is how the stage
costs above are measured:

```
[prof] s=0 read=0.28s advect=10.72s micro=0.23s settle=0.00s (nsub=164)
[prof] s=0 bc+polar=2.33s
```

`read` is the CESM h1 I/O for the step and is not a stage; `advect`, `micro` and
`settle` are stages 1, 2 and 2b, and `bc+polar` covers 3, 4 and 4a together.
Radiation prints its own `radiation=` line on the steps it runs (`RAD_EVERY`).
Step 0 carries JIT compilation, so read steady-state cost from step 1 onward.

The `micro` timer opens *before* the `MICRO` branch, so it always includes the
`RELHUM` read — wet settling and the wet-size diagnostics need RH too, which is
why it is read outside the branch. Under `MICRO=off` that read is the entire
`micro` number, as in the line above.

**`MICRO=off` is not a physics configuration.** It disables stage 2 outright so
that a timing run exercises the production step loop with only transport live —
the aerosol moves but never evolves. Combined with `RAD=0 SETTLE=0` and no
injection it leaves stages 1, 3, 4 and 4a, which is the transport-only benchmark
in `MANIFEST.md`'s Running section. A typo in `MICRO` is rejected at import
rather than falling through to the legacy `coag` path.

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
exact, not an approximation.

## 1.2 Scheme: Lin-Rood flux form

`src/advection/fct_lr.py` (`ADV_SCHEME=lr`; since 2026-10-15 production runs
its large-Courant form `ffsl_lr.py`, §1.9, which imports these sweeps). PPM
reconstruction, Zalesak limiter, dimensionally split (x, y, z sweeps), CFL 0.5,
f32.

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

## 1.8 Semi-Lagrangian alternative (`ADV_SCHEME=sl`, 2026-10-13)

**The operator as it stands (`ADV_SL_VERT=flux ADV_SL_SPLIT=strang`, the
defaults since 2026-10-14): lr with its two CFL-limited horizontal sweeps
replaced by one semi-Lagrangian remap, Strang-composed with lr's vertical
sweep.** Per sub-step (K = `ADV_SL_K` = 6, one per hourly CESM wind):
(1) `fct_lr._lr_vert` on every column for HALF the sub-step — the production
flux-form PPM vertical remap with its open faces, reservoir inflow, positivity
limiter and reported face fluxes — in advective form `(ρq)'/ρ'` with the air
mass ρ carried; (2) 2-D departure points per level, cubic gather with the
quasi-monotone limiter, and a per-(tracer, level) mass fixer whose target is
Σ ac·ρ·q with ρ advanced by the Jacobian of the departure map (the
Lagrangian air-mass ratio, second order like the departure points;
`ADV_SL_JAC=1`) — each level is a closed sphere, so the target is exact;
(3) the second half vertical sweep from that ρ, after which the O(dt²)
bookkeeping remainder `ρ − 1` is sent through the column's own faces at the
local upwind mixing ratio (`_remainder_sweep`), so the air mass is exactly 1
again, a uniform field stays uniform and `dM = f_top − f_bot` holds without a
fixer (the global column fixer remains as a roundoff closure); (4) the polar caps stirred with `fct_fast._mix_caps`
(`ADV_POLAR=zonal`). The 2026-10-13 composition (horizontal then vertical, air
mass frozen at 1) is `ADV_SL_SPLIT=lie`, first order at the open face; see
"the open-face defect" below for why it was replaced. Face omega is
`lr`'s continuity derivation (`ADV_SL_OMEGA=cont`). Consequences: `ADV_WCONT=1`,
`BC_EDGE=open`, `BC_BOT_AER`, `B_vf_in/out` and the "advection numerical
residual" all mean exactly what they mean under `lr` (the residual is at
roundoff by construction, 1e-16 in the 1-day run). In-loop: `advect` 0.38 s
per step vs 6.6 s; the hourly wind reads add ~1 s to `read`. Day 1 against
`bench_phys30d`: mass, SO2, H2SO4 and AOD within 0.1 %, number +3.4 %.

**30-day A/B of the split operator (2026-10-13, `runs/slsplit30d_K6`,
`slsplit30d_K2`, `slsplit30d_K6loc` vs `runs/bench_phys30d`, physics
microphysics, same injection scenario;
`outputs/emulator_bench/compare_slsplit_ab30d.png`), day 30 relative to `lr`:**

| run | number | mass | SO2 | r_eff | AOD 550 | TOA forcing | H2SO4(g) | bottom-face export /M0 |
|---|---|---|---|---|---|---|---|---|
| lr (reference) | — | — | — | — | — | — | — | −0.085 |
| split K=6, global fixer | +0.9 % | −2.1 % | +0.9 % | −3.8 % | −1.6 % | +7.2 % | +8.5 % | −0.103 |
| split K=2, global | +0.7 % | −3.2 % | +0.7 % | −4.2 % | −2.0 % | +10.2 % | +8.1 % | −0.118 |
| split K=6, local fixer | +3.5 % | −2.7 % | +0.5 % | −4.9 % | −1.9 % | +6.1 % | +8.7 % | −0.117 |

Days 1–10 are within ~1 % on every burden. The whole day-30 difference is
again in the slab-face term, but now a modest one — and, as the one-year pair
showed a day later, NOT a physical one (see "the open-face defect" below): the
split exports 20–40 % more aerosol through the bottom face than `lr` (both
budget to roundoff; `micro` +0.48 vs +0.49, `settle` −0.028 vs −0.027). The
two horizontal operators place the plume's lower edge differently (SL cubic +
quasi-monotone limiter vs PPM + Zalesak with ~200 sub-steps), and the
bottom-face flux is what that edge sees. K = 6 with the global fixer is the
best configuration; the local fixer buys nothing here. The mass gap grows
linearly after day 10 (−0.4 % at day 10, −1.3 % at 20, −2.1 % at 30), i.e. it
has NOT saturated in 30 days; the TOA-forcing gap sits at +7 % from day 10 on
and does not grow (it tracks r_eff, not burden). Under continuous injection
the burden ratio must saturate at the ratio of the effective removal rates;
the one-year pair `runs/lr1yr`, `runs/slsplit1yr_K6` (and `_K24`) measures
where (launched 2026-10-14).

**Which scheme is nearer the truth (2026-10-14).** Four 30-day diagnostics
(`runs/lrdiag30d_cfl025`, `sldiag30d_K6brack`, `sldiag30d_K12`,
`sldiag30d_K24`), day 30 relative to the production `lr` run:

| run | mass | cumul. bottom export | number | AOD | TOA forcing |
|---|---|---|---|---|---|
| lr at CFL 0.25 | +0.06 % | −0.0 % | −0.0 % | +0.07 % | −0.3 % |
| split K=6, hourly winds | −2.10 % | −21.5 % | +0.9 % | −1.6 % | +7.2 % |
| split K=6, bracketing winds (lr's input) | −1.92 % | −19.0 % | −1.9 % | −1.0 % | +4.1 % |
| split K=12 | −1.99 % | −18.6 % | +0.8 % | −1.6 % | +7.2 % |
| split K=24 | −1.94 % | −17.2 % | +0.8 % | −1.6 % | +7.2 % |

`lr` is converged in its own sub-stepping (halving the CFL moves nothing).
The split moves toward `lr` as K grows (export excess 21.5 → 18.6 → 17.2 %;
with the Lie composition, partly because its face error is first order in the
sub-step, below) and the hourly winds account for ~2.5 points. But "toward lr"
is NOT "toward the truth": `scripts/validation/compare_hz_operators.py` runs
both production entry points (w = 0, caps at 80°, f32, 6 h steps) on analytic
cases, 12 days, L2 error and surviving peak (1 = exact):

| case | lr (CFL 0.5) | split K=6 | split K=24 |
|---|---|---|---|
| Williamson 1, solid body along the equator | 0.019 / 0.970 | **0.0015 / 0.9985** | 0.062 / 0.909 |
| Williamson 1, axis tilted 45° | 0.013 / 0.966 | **0.0125 / 0.976** | 0.036 / 0.947 |
| Nair–Lauritzen 4, deformational, reversing | 0.132 / 0.777 | **0.058 / 0.915** | 0.232 / 0.664 |

SL at K = 6 is 2–12× more accurate than `lr` and keeps the peak; `lr`'s 70–250
PPM sub-steps per 6 h are the diffusive scheme. More SL sub-steps make it
WORSE (every gather smooths), which is why K = 24 "converges toward lr": it
converges toward lr's diffusion. So the 20 % export spread is dominated by
`lr`'s numerical diffusion of the plume's lower edge, and the split is the
better estimate of the two. K must match the wind cadence (hourly → K = 6),
never exceed it. `lr` stays the default until the one-year pair and the user's
call; the 17× cheaper transport is also the more accurate one.

**The open-face defect and the Strang fix (2026-10-14).** The one-year pair
(`runs/lr1yr`, `runs/slsplit1yr_K6`, Lie composition, 10 Tg/yr continuous
injection) showed the gap does not saturate while the burden is still
building — it accelerates — and only levels off once `lr`'s own burden does
(day ~330, M/M0 ≈ 11; both complete):

| day | mass | cumul. bottom export | settling | AOD | TOA forcing | lr burden M/M0 |
|---|---|---|---|---|---|---|
| 30 | −2.1 % | +17 % | −3.5 % | −1.6 % | +7 % | 1.4 |
| 90 | −5.4 % | +39 % | −7.6 % | −4.8 % | +7 % | 3.4 |
| 180 | −14.1 % | +62 % | −10.0 % | −13.2 % | +13 % | 6.7 |
| 270 | −28.6 % | +76 % | −14.3 % | −27.3 % | +28 % | 9.6 |
| 365 | −37.1 % | +75 % | −17.3 % | −34.6 % | +36 % | 11.1 |

(export excess here against `lr1yr` itself, which differs from
`bench_phys30d` by the lr run's own sampling of the forcing.) A scheme that
ends a year 37 % low in burden and 36 % high in forcing is unusable, whatever
its interior accuracy.

The day-30 per-level mass against `lr` was the clue: every level from 1 to
50 hPa −22 to −37 %, 88–122 hPa +12 to +22 %, the same in every latitude band
— a downward redistribution of the whole burden, not a difference at the
plume's edge. The analytic cases above are non-divergent and could not see it.
A closed-column 3-D test with a DIVERGENT, time-reversing flow
(`scripts/validation/compare_3d_operators.py`: Nair–Lauritzen case 3 in the
horizontal × a zero-mean vertical profile so the column stays closed, uniform
24 levels 20–143 hPa, 12 days, exact answer = the initial field, `KDEF` =
deformation strength) puts the split AHEAD of `lr` in the interior:

| composition (K) | L2, KDEF=2 | L2, KDEF=5 |
|---|---|---|
| lr (CFL 0.5) | 0.0089 | 0.0250 |
| Lie, air mass frozen (6) | 0.0067 | 0.0244 |
| Lie, air mass carried (6) | 0.0068 | — |
| **Strang (6)** | **0.0064** | **0.0234** |
| Strang (2) | 0.0070 | — |
| Strang (24) | 0.0158 | 0.0305 |

So the operators are fine and the whole coupled-run gap is the open bottom
face. Mechanism: Lie splitting applies a whole sub-step's horizontal
convergence to the column before the vertical sweep sees any of it, and at
143 hPa that is `dt·div ≈ 0.3` of the cell's air in one hour. With the air
mass frozen at 1, the per-level fixer absorbed that as a uniform O(dt·div)
rescale of the level every sub-step — the spurious transfer that drained the
upper levels. Carrying the air mass `ρ_h = 1 − dt·div(v)` through the
horizontal step (`ADV_SL_RHO=1`) puts the fixer at O(dt²) and repairs the
vertical distribution (mid-levels −3 to −5 % of `lr` instead of −25 to −49 %),
but the bottom-face flux then rides on ρ ≈ 1.3 for the whole hour: first order
in the sub-step, and more sub-steps are not available because every SL gather
smooths. Strang composition (`ADV_SL_SPLIT=strang`: half vertical sweep,
horizontal remap with ρ carried, half vertical sweep) centres the convergence
on the vertical motion, which makes the face term second order with still one
gather per sub-step; vertical sweeps are cheap. 30-day A/B, day 30 relative to
the production `lr` run (`runs/sldiag30d_K6brack`, `slrho30d_K6`,
`slrho30d_K24`, `slstrang30d_K6`):

| composition | mass | cumul. bottom export | reservoir inflow | settling | number | AOD | TOA forcing |
|---|---|---|---|---|---|---|---|
| Lie, ρ frozen, K=6 (hourly winds) | −2.10 % | +21.5 % | +38 % | +5 % | +0.9 % | −1.6 % | +7.2 % |
| Lie, ρ frozen, K=6 (bracketing winds) | −1.92 % | +15.1 % | +37 % | −3.5 % | −1.9 % | −1.0 % | +4.1 % |
| Lie, ρ carried, K=6 | −2.37 % | +37.1 % | +3.8 % | +0.1 % | +3.7 % | −3.1 % | +14.3 % |
| Lie, ρ carried, K=24 | −0.72 % | +9.9 % | −4.7 % | +0.7 % | +2.6 % | −1.6 % | +7.5 % |
| Strang, Eulerian air mass, K=6 | −1.15 % | +15.8 % | −11.3 % | +0.0 % | +2.7 % | −1.8 % | +8.2 % |
| **Strang, Jacobian + remainder sweep, K=6 (default)** | **−0.48 %** | **+7.8 %** | **−16.2 %** | **+0.6 %** | **+2.9 %** | **−1.4 %** | **+6.0 %** |

Under Strang the settling and microphysics terms match `lr` to 0.1 %, the
reservoir inflow (38 % high under the frozen-ρ split) is now 11–16 % low, and
the mass gap grows more slowly each decade (−0.41, −0.85, −1.15 % with the
Eulerian air mass; −0.22, −0.40, −0.48 % with the default, whose export excess
is flat at 8 % from day 10) instead of faster. The remaining export spread (~16 %, flat from day 10)
is what the analytic tests attribute to `lr`'s diffusion of the plume's lower
edge plus the hourly winds (~2.5 points); `runs/slstrang10d_K24` (10 days)
checks that the face term has converged in K at K=6, and the one-year Strang
run `runs/slstrang1yr_K6` (launched 2026-10-14, physics microphysics, same
scenario as `lr1yr`) answers whether the gap saturates. `ADV_SL_SPLIT=strang`
is the default under `sl`; `ADV_SL_RHO` only affects `lie`. `sl` stays
opt-in: `ffsl` K = 24 (§1.9) became the production default on 2026-10-15.

**Convergence in K, and the two remaining first-order terms (2026-10-14).**
Ten-day Strang runs at K = 2, 6, 12, 24 (`runs/slstrang10d_K2`, `_K12`,
`_K24`; K = 2 blows up — `dt·div ≈ 0.9` at the face puts the air mass on its
floor — so K ≥ 6 is required, not just preferred): the export excess over `lr`
at day 10 was 14.7 / 7.9 / 4.5 % at K = 6 / 12 / 24, halving with each
doubling of K — FIRST order, with Richardson limits of +1 % in export and
−0.1 % in mass. So in the coupled model the split converges to `lr` at the
face and `lr` is the converged answer there; the question is only how far
K = 6 sits from it, because more gathers are not an option. Two sources were
found, each a local O(dt²) error that a global fixer was spreading uniformly
every sub-step (which makes it first order over the run): (a) the Eulerian
air mass `1 − dt·div` is the Lagrangian one to first order only, so the
per-level fixer still moved ~0.3·(dt·div)² of a level (≈2 % per hour at the
143 hPa face); the Jacobian of the departure map, `|∂(λ_d, μ_d)/∂(λ, μ)|` by
central differences in the equal-area coordinates, cuts that load 30× at
dt = 1 h (`_jacobian`); (b) after the two half sweeps the air mass is 1 only to
O(dt²), and dividing it out plus the global column fixer again smeared a
local remainder over the globe; `_remainder_sweep` returns it through the
column's own faces at the local upwind mixing ratio instead, after which no
fixer is needed. Both are `ADV_SL_JAC=1` (default). Day 10 relative to `lr`:

| Strang variant | K | mass | cumul. bottom export | reservoir inflow | settling |
|---|---|---|---|---|---|
| Eulerian air mass, column fixer | 6 | −0.41 % | +14.7 % | −8.1 % | −0.2 % |
| Eulerian air mass, column fixer | 12 | −0.26 % | +7.9 % | −8.2 % | +0.0 % |
| Eulerian air mass, column fixer | 24 | −0.18 % | +4.5 % | −7.9 % | +0.2 % |
| Jacobian air mass, column fixer | 6 | −0.25 % | +10.0 % | −4.3 % | +0.2 % |
| Jacobian air mass, column fixer | 24 | −0.15 % | +3.3 % | −6.9 % | +0.3 % |
| **Jacobian + remainder sweep (default)** | **6** | **−0.22 %** | **+8.4 %** | **−16.5 %** | **+0.1 %** |
| Jacobian + remainder sweep | 24 | −0.13 % | +2.6 % | −8.1 % | +0.2 % |

(`runs/sljac10d_K6`, `_K24`, `runs/slrem10d_K6`, `_K24`; the Lie split of
2026-10-13 was +9.6 % at day 10 but +21.5 % by day 30 and accelerating.) The
K = 6 export excess went 14.7 → 10.0 → 8.4 %; what is left still halves from
K = 6 to K = 24, so a first-order term of a few percent remains (candidates:
the PPM half sweeps with the hourly wind held fixed over the sub-step, the
cap stirring, the time interpolation of the winds that K > 6 implies).
`advect` is 0.8 s/step with the extra gather and sweep (was 0.5). The
one-year run was restarted with this default as `runs/slrem1yr_K6`
(2026-10-14; the Eulerian-Strang one-year `slstrang1yr_K6` was stopped at
day 98 and is superseded).

**The handoff's original design (`ADV_SL_VERT=sl`), kept for the record.**
`src/advection/sl_lr.py` on the kernels in `src/advection/sl_core.py`, which
are dinosaur's `semi_lagrangian` module (Google NeuralGCM, Apache 2.0) vendored
without its dependencies and with one patch: the CESM grid has rows AT the
poles, which dinosaur refuses because its cross-pole halo would duplicate the
pole node; `sl_core` mirrors rows 1..2 instead of 0..1 about a pole row, and the
pole row stays an ordinary node. The design and its measurements are in
[TRANSPORT_SL_HANDOFF.md](./TRANSPORT_SL_HANDOFF.md).

**Why.** `fct_lr` is CFL-limited in both horizontal sweeps and the sub-step is
set by the 24 km longitude spacing at the 80° cap edge: ~180–270 sub-steps and
6.6 s of a 21 s emulator step. A semi-Lagrangian step has no CFL limit — one
departure-point solve per level plus one gather per tracer per sub-step,
whatever the wind — and costs ~49 ms at this problem size (offline, H100,
`scripts/validation/bench_sl_transport.py`).

**What a step does.** K = `ADV_SL_K` sub-steps (default 6, one per hourly CESM
wind; `coupling.py` reads the five hourly snapshots between the two step ends
and `sl_lr` resamples them onto the sub-step nodes). Per sub-step: 3-D
departure points by two fixed-point iterations on the trajectory midpoint in
geocentric Cartesian coordinates (no polar singularity; the IFS 48r1
formulation), with CESM omega at the level centers as the vertical velocity in
pressure, departure pressures clipped to the center range; then one cubic
(4×4) × linear (2) Lagrange gather per tracer with the Bermejo–Staniforth
quasi-monotone limiter (clip to the 2×2×2 bracketing corners: no new extrema,
positivity preserved); then the mass fixer. After the K sub-steps the polar
rows (|lat| > 80°) are overwritten from `qfroz`, as `ADV_POLAR=freeze` does.

**Conservation.** Point-wise SL remap is an interpolation, not a flux balance,
so the column sum is not preserved (the raw operator drifts ~0.5–1 %/6 h on a
random divergent wind in the unit test; JCM saw 1.3 % over 38 steps). The
standard remedy (IFS, UM, GEM) is a mass fixer with a *known* target, and the
target is known because the column is **closed**: nothing enters through the
end faces (constant extrapolation beyond the end levels), so the target of every
tracer is its extended-column mass before the sub-step, Σ dp·ac·q, with the
same dp·ac weight the step log's `M/M0` uses. `ADV_SL_FIXER=global` rescales
the tracer to it; `local` (Bermejo & Conde 2002, as in the IFS) removes the
error only from cells where the cubic gather overshoots a linear gather at the
same departure points in the direction of the error, i.e. where the
interpolation created the spurious mass, and falls back to the global scaling
when no cell does (a smooth field). Both hit the target to float64 roundoff:
the gathers run in float32 but the stack is carried between sub-steps in
float64 and the sums are float64.

**Boundaries.** Because the column is closed there is no face flux to book, so
`coupling.py` sets `ADV_WCONT=0` under `sl`: `BC_EDGE=clamp`, the reservoir
levels and the polar rows are Dirichlet-reset after the step and the exchange
with the reservoir is what the reset puts in or takes out, booked in `B_bc`
(`B_vf_in/out` are off — `sl_lr.advect_step_batch` has no `return_vflux`, so
the `ADV_VFLUX` probe answers False). `BC_BOT_AER` is **not** applied in the
clamp path (it scales the open-face inflow value), so it is a no-op under `sl`
at present. Bin consistency: number and mass of a bin are gathered along the
same trajectories, so their ratio is preserved away from the limiter and the
fixer; the coupler's two-moment clip repairs the rest and `clipM/M0` reports it.

**Validation.** `scripts/validation/test_sl_transport.py` (CPU, ~30 s, on the
model's own pole-row grid): constant field preserved to roundoff across the
poles; Williamson case 1 (cosine bell, solid-body rotation, α = 0° and 90°,
one revolution in 48 × 6 h) L2 = 2.5 %, max ≤ 1, min ≥ 0; fixer closure
|M1−M0|/M0 ≤ 1e-15 (global and local) with positivity on a random divergent
3-D wind; 0 % bin-bound violations on a (N, M) pair; the driver contract.

**30-day A/B of the closed-column design (`ADV_SL_VERT=sl`; 2026-10-13,
`runs/sl30d_K6`, `sl30d_K1`, `sl30d_K2`, `sl30d_K6loc` vs `runs/bench_phys30d`,
physics microphysics, same injection scenario;
`outputs/emulator_bench/compare_sl_ab30d.png`): the gate is NOT met.**
In-loop, `advect` is 0.35 s/step (median) against 6.6 s, and the hourly wind
reads add ~1.0 s to the `read` stage. Microphysics (+0.49 M0) and settling
(−0.025 M0) agree between all five runs to the third digit, but at day 30 the
SL runs are −9 to −14 % in mass burden, −12 to −15 % in SO2, −7 to −14 % in AOD
and +25 to +40 % in number burden, growing with K (K=1 closest to `lr`). The
whole difference is the slab-face term: `lr` exchanges −0.085 M0 through the
bottom face over 30 days (`B_vf`), the SL runs −0.23 (K=1) to −0.29 (K=6) M0
through the Dirichlet edge-level reset (`B_bc`). Resetting the whole bottom
LEVEL every step is a ~3× stronger sink than the flux through its lower FACE
(and a stronger ultrafine-number source from the aerosol-rich 143 hPa
reservoir), which is the risk the handoff's §6.2 named. The operator is fine;
the boundary design is what had to change — which is the split operator
above (`ADV_SL_VERT=flux`), built the same day.

## 1.9 `lr` at large Courant number (`ADV_SCHEME=ffsl`, 2026-10-14)

**Why.** The one-year pair settled which scheme the budget quantities
converge to: the split semi-Lagrangian scheme converges TO `lr` at the open
face as its sub-step count grows (§1.8), so `lr` is the reference there, and
its only defect is the 70–250 CFL sub-steps per step that the 24 km zonal
spacing at the cap edge forces (6.6–7 s of the step). Lin & Rood (1996, §3)
lift that limit with the flux-form semi-Lagrangian (FFSL) extension that
FV3 uses near its poles, and `src/advection/ffsl_lr.py` is `fct_lr` with it:
the flux through a face whose Courant number is N + f is the content of the
N whole upwind cells plus the PPM sliver of fraction f of the next one, still
exact flux form (every flux appears twice), so conservation, the open faces,
the reservoir inflow, the Zalesak limiter, the cap stirring, the continuity
omega and the face-flux diagnostic are `fct_lr`'s, imported. For |c| < 1 the
sweep is `fct_lr._lr_sweep` bit for bit on periodic rows. Two deliberate
differences: the sliver is measured in the donor cell's own area rather than
the face length (the same divergence to O(dφ²); it is what keeps a whole-cell
shift exact on the sphere and the pole half-cell positive without closing the
cap-internal faces), and the reservoir inflow is scaled by the edge air mass
(`fct_lr` passes the mixing ratio alone, a ~1 % approximation at its 100 s
sub-steps that would break a uniform field at hourly ones). The vertical
remap has no Courant limit, so the sub-step count `ADV_FFSL_K` is free; the
default 6 is one per hourly CESM wind, and raising it is a plain convergence
knob (each sub-step costs ~1/200 of a production `lr` step and fewer remaps
mean less diffusion than production `lr`, not more — unlike the SL scheme,
where every extra gather smooths). `ADV_FFSL_SPLIT=strang` (default) orders
the sub-step z/2, horizontal, z/2 and alternates x,y / y,x, for the same
reason the SL split needed it at the face.

**Gates.** `scripts/validation/test_ffsl_transport.py` (CPU, ~100 s): the
sweep equals `fct_lr`'s at |c| < 1, conserves Σ a ρ q to roundoff at |c| up to
8 and keeps a uniform mixing ratio uniform, a bell shifted 3.7 cells per step
lands where the exact shift puts it with the accuracy of a 0.7-cell step, the
3-D driver closes `dM = f_top − f_bot` and keeps a uniform field with the
random divergent winds at K = 6 and 12, and Williamson 1 at hourly sub-steps
(Courant up to ~6 at the cap edge) is as accurate as `lr` at CFL 0.5 with a
tenth of the sub-steps. Against the analytic truth (`compare_hz_operators.py`,
`compare_3d_operators.py`):

| case | lr (CFL 0.5) | ffsl K=6 | ffsl K=24 | SL split K=6 |
|---|---|---|---|---|
| Williamson 1, axis tilted 45° (L2 / peak) | 0.013 / 0.966 | 0.014 / 0.996 | 0.019 / 0.954 | 0.0125 / 0.976 |
| Nair–Lauritzen 4, strong deformation (L2 / peak) | 0.132 / 0.777 | 0.225 / 0.961 | 0.113 / 0.801 | 0.058 / 0.915 |
| closed-column 3-D divergent, mild deformation (L2) | 0.0089 | **0.0065** | 0.0076 | 0.0067 |

The strong-deformation case is the one weakness: at hourly sub-steps with
Courant numbers of several, the sequential x-then-y sweeps leave an air-mass
splitting error of O(dt² ∂u/∂x ∂v/∂y) per sub-step that `lr`'s 100 s
sub-steps never see and that Lin & Rood's inner/outer operator cross terms
remove (never needed in `fct_lr`, not implemented here). More sub-steps
remove it too, and at K = 24 the scheme is still 4× cheaper than `lr`.

**Coupled A/B (10 days, physics microphysics, vs `bench_phys30d`; `runs/ffsl10d_K6`, `_K12`, `_K24`):**

| scheme | advect s/step | mass | cumul. bottom export | settling | AOD | number |
|---|---|---|---|---|---|---|
| lr at CFL 0.25 (lr's own convergence) | ~14 | +0.02 % | +0.4 % | −0.0 % | +0.0 % | +0.0 % |
| SL split, Strang + Jacobian + remainder, K=6 (§1.8 default) | 0.8 | −0.22 % | +8.4 % | +0.1 % | −0.2 % | +2.8 % |
| SL split, same, K=24 | 2.6 | −0.13 % | +2.6 % | +0.2 % | −0.2 % | +2.9 % |
| **ffsl K=6** | **0.6** | −0.44 % | +22 % | +1.0 % | −0.45 % | +1.9 % |
| ffsl K=12 | 0.9 | −0.18 % | +9.0 % | +0.4 % | −0.25 % | +2.3 % |
| **ffsl K=24** | **1.6** | **−0.05 %** | **+2.5 %** | +0.1 % | −0.15 % | +2.5 % |

(`lr` itself: 7.0 s.) At K = 24 the large-Courant `lr` is the closest thing to
`lr` of everything built, at 4.4× lower transport cost; at K = 6 the
splitting error above shows up at the face exactly as the deformation case
predicted, and the K series converges between first and second order (the
alternating x,y / y,x order is second order only over pairs of sub-steps).
K = 24 is the operating point; the two ways to make K = 6–12 viable, if
transport ever matters again once the microphysics and radiation are
emulated, are Lin & Rood's cross terms (two extra advective-form inner
sweeps per sub-step) and a cheaper sweep kernel (the FFSL sweep costs ~2× an
`lr` sweep per sub-step: f64 cumsum, five gathered PPM arrays).

**30-day A/B at K = 24 (`runs/ffsl30d_K24`, day 30 relative to `lr`):**

| scheme | mass | cumul. bottom export | settling | SO2 | AOD | number | r_eff | TOA forcing |
|---|---|---|---|---|---|---|---|---|
| lr at CFL 0.25 | +0.06 % | +0.1 % | −0.05 % | +0.03 % | +0.07 % | −0.01 % | +0.03 % | −0.3 % |
| old Lie split K=6 | −1.92 % | +15 % | −3.5 % | +1.0 % | −1.0 % | −1.9 % | −2.8 % | +4.1 % |
| SL split, §1.8 default, K=6 | −0.48 % | +7.8 % | +0.6 % | −0.2 % | −1.4 % | +2.9 % | −0.9 % | +6.0 % |
| **ffsl K=24** | **−0.15 %** | **+1.2 %** | +0.14 % | +0.00 % | −0.9 % | +2.4 % | −0.95 % | +5.1 % |

The budget terms (mass, export, settling, SO2) are at `lr`'s own convergence
level. The number, r_eff, AOD and forcing differences are shared with the SL
scheme although their budgets differ by 10×, so they are not the transport
operator's: both new schemes advected on the hourly CESM winds in these runs
(`ADV_SL_HOURLY=1`), `lr` on the two step snapshots interpolated in time; the
next paragraph puts `ffsl` on `lr`'s winds, which is now the default.

**Wind cadence (`ADV_SL_HOURLY=0`, 2026-10-14).** Production will receive
winds as six-hourly snapshots (from CESM today, from an atmospheric emulator
later), so `lr`'s input — the two step-end snapshots, linear in time at every
sub-step — is the production cadence, and the hourly CESM mode is a
diagnostic luxury. `runs/ffsl10d_K24brack` is ffsl K = 24 on those
bracketing winds (`runs/ffsl30d_K24brack` is the same to 30 days), relative to
`lr`:

| ffsl K=24 on | day | mass | cumul. export | settling | SO2 | AOD | number | r_eff | TOA forcing | H2SO4 |
|---|---|---|---|---|---|---|---|---|---|---|
| hourly winds | 10 | −0.05 % | +2.5 % | +0.12 % | +0.01 % | −0.15 % | +2.5 % | +0.9 % | (noise) | −0.5 % |
| bracketing winds (production) | 10 | −0.04 % | +1.8 % | +0.05 % | +0.00 % | −0.10 % | −0.3 % | +0.0 % | (noise) | +0.0 % |
| hourly winds | 30 | −0.15 % | +1.2 % | +0.14 % | +0.00 % | −0.9 % | +2.4 % | −0.95 % | +5.1 % | +3.4 % |
| **bracketing winds (production)** | **30** | **−0.16 %** | **+0.8 %** | −0.09 % | +0.09 % | **−0.30 %** | **−0.39 %** | −0.17 % | **+2.0 %** | +0.06 % |

Number, r_eff, H2SO4, AOD and forcing collapse toward `lr`: those residuals
were the wind cadence, not the operator. On production-cadence winds ffsl
K = 24 is within 2 % of `lr` on every quantity at day 30, with the budget
terms within 0.2 % (export +0.8 %, which is `lr`'s own sub-stepping
convergence level). The day-10 forcing is −0.002 W m⁻² and its percentage
is noise at that size. The bracketing mode also reads one snapshot per step

instead of six: `read` 0.2 s against 1.5 s, so the production step is
advect 1.5 s + read 0.2 s where `lr` is 7.0 + 0.2 s. The hourly-versus-
bracketing difference (≈2.7 % in number, ≈3 % in forcing, ≈3 % in H2SO4 at
day 30) is the value of sub-six-hour wind variability; if it ever
matters, the cheap way to recover it is interval-mean mass fluxes from the
wind source at the same cadence, not finer output.

**Production default (2026-10-15).** On that evidence `ffsl` K = 24 on the
bracketing winds is the production default: `ADV_SCHEME=ffsl`,
`ADV_FFSL_K=24`, `ADV_SL_HOURLY=0` are the defaults in `coupling.py`,
`driver_fast.py` and `ffsl_lr.py`, so a bare `run_prod.py` advects at 1.5 s +
0.2 s read per step where it spent 7.0 + 0.2 s. `ADV_SCHEME=lr` reproduces the
previous default byte for byte (`fct_lr.py` is untouched and `ffsl_lr.py`
imports from it), and `lr` remains the reference every transport change is
measured against. The standing acceptance test is the one-year `ffsl` run
against `runs/lr1yr`: the gap must stay within a few percent and stop growing.

> **2026-11-05: that acceptance test FAILED, the cause was found and fixed, and
> the first test against a truth outside the `lr` family reverses the ranking.
> See §1.10 before using any number in this section.**

---

## 1.10 The one-year verdict, the FFSL sliver bug, and the CH4 truth test (2026-11-05)

**The one-year acceptance test of §1.9 was run (2026-10-18) and never written
up; it failed.** `runs/ffsl1yr_K24`, `_K48` against `runs/lr1yr` (coupled,
physics microphysics, RRTMGP, 10 Tg/yr at 30N/S), day 365 relative to `lr`:

| | lr CFL 0.25 | ffsl K=24 | ffsl K=48 |
|---|---|---|---|
| mass burden | +0.4 % | −4.6 % | −3.8 % |
| AOD 550 | +0.4 % | −4.2 % | −3.6 % |
| TOA forcing | −0.3 % | +5.2 % | +4.4 % |
| number, day 270 | +0.1 % | +10.8 % | +13.2 % |

The gap grows with time and doubling K barely moves it. Transport-only
(`runs/transport1yr_{lr,ffsl,sl}`, MICRO=off RAD=0): ffsl −5.2 % mass,
sl −2.9 % at a year. The "within 0.2 % at 30 d" of §1.9 was a crossing
point, not agreement.

**Cause: the FFSL sliver measure.** A 10-day transport-only K ladder
(`runs/lrT10d`, `ffslT10d_K{24,48,96,192}`) showed ffsl converging in K to a
limit that is NOT `lr` (bottom export +1.1 / −1.6 / −2.9 / −3.6 % at
K = 24/48/96/192) and delivering **70 % more mass into the polar caps at every
K**. The whole-cell part of the FFSL flux is exact, but the fractional sliver
was measured in the donor cell's area, `f·a_j`, while `fct_lr` measures it at
the face, `c·m_face`. The two differ by `tan(φ)·dφ/2`, 4.5 % at the 80° cap
edge, with opposite sign for poleward and equatorward flux, so the small NET
cap exchange was biased by 70 % and the bias did not shrink with K (it is
exact only as f → 1). `ffsl_lr.py` now has `ADV_FFSL_FRAC = cell | face |
exact`: `face` is `fct_lr`'s convention (bit-identical at |c| < 1,
`test_ffsl_transport.py` test 1), `exact` is the true spherical sliver area
`(sin φ_face − sin(φ_face ∓ f dφ))/dφ`, exact at both limits, and is the
default. Same ladder with `exact` (`runs/ffslX10d_K{24,48,96,192}`), day 10
relative to `lr`:

| K | polar-cap delivery | bottom export | mass |
|---|---|---|---|
| 24 | +0.6 % | +4.3 % | −0.10 % |
| 96 | +0.2 % | +0.5 % | −0.01 % |
| 192 | +0.2 % | −0.1 % | +0.01 % |

ffsl now converges to `lr`. What is left at K = 24 is the first-order
open-face splitting error of §1.8, which the old measure had been cancelling
against; it needs K ≈ 96 (6 s/step, the cost of `lr`) to go away.
Over a year, transport-only (`runs/transport1yr_ffslX_K24`, `_K48` against
`transport1yr_lr`): mass −2.2 % at K=24 (was −5.2 % with the old measure) and
−1.0 % at K=48 (day 346), bottom export +2.3 % / +0.7 %, polar-cap delivery
−17 % / −14 % (was +72 %). The fix halves the one-year drift and each doubling
of K halves it again, as a first-order term should.

**A second lr-family fact.** `lr`'s "advection numerical residual" is not
roundoff over a year: +0.032 M0 at CFL 0.5 and +0.093 M0 at CFL 0.25
(`runs/lr1yr_cfl025`) — it GROWS with the sub-step count, is 0.3–0.8 % of the
final burden, and is the whole of the `lr` CFL 0.25-vs-0.5 difference. `lr`
is therefore not an unambiguous target; the budgets below are consistent
with `lr` creating mass at the few-tenths-of-a-percent-per-year level.

**The truth test.** `scripts/validation/ch4_truth_test.py` advects CESM's own
hourly CH4 field (inert on a one-year timescale below ~30 hPa; CESM moved it
with its FV dycore, Lin–Rood FFSL PPM on this grid with floating Lagrangian
levels) from 1996-01-01 with one of this model's schemes on the same winds,
exactly as `coupling.py` calls it (6 h steps, bracketing winds, open faces
and caps fed by the truth, f32), and scores it against CESM's CH4 every step:
RMSE, bias, pattern correlation over |lat| ≤ 80, p ≥ 30 hPa, dp·cos
weighted. Persistence (IC held) is the do-nothing baseline. Output:
`outputs/ch4_truth/<tag>.npz` and `.log`. One tracer, 15–70 min per scheme-year.

RMSE relative to `lr` CFL 0.5 (whose year-mean RMSE is 2.99e-8 mol/mol, 29 %
of persistence and 18 % of the truth's standard deviation):

| scheme | d30 | d90 | d180 | d270 | d365 | year mean | corr d365 | advect s/step (1 tracer) |
|---|---|---|---|---|---|---|---|---|
| **sl K=12** | 0.74 | 0.87 | 0.63 | 0.68 | 0.81 | **0.75** | 0.985 | 0.29 |
| sl K=24 | 0.74 | 0.86 | 0.68 | 0.76 | 0.88 | 0.80 | 0.986 | 0.57 |
| **sl K=6** | 0.79 | 1.00 | 0.84 | 0.85 | 0.80 | **0.87** | 0.982 | 0.22 |
| lr CFL 0.5 (Noah's `fct_lr`) | 1 | 1 | 1 | 1 | 1 | 1.00 | 0.974 | 0.14 (7 s at 82 tracers) |
| ffsl exact K=96 | 1.02 | 1.06 | 1.17 | 1.53 | 1.03 | 1.21 | 0.967 | 2.2 |
| lr CFL 0.25 | 1.01 | 1.02 | 1.12 | 1.13 | 1.13 | 1.09 | 0.973 | 1.0 |
| sl K=6, hourly winds (`ADV_SL_HOURLY=1`) | 0.81 | 1.05 | 0.92 | 1.05 | 0.81 | 0.95 | 0.982 | 0.08 |
| sl K=3 | 0.93 | 1.33 | 1.51 | 1.62 | 1.09 | 1.37 | 0.972 | 0.11 |
| ffsl exact K=24 (fixed) | 1.47 | 1.73 | 2.91 | 5.26 | 2.00 | 2.98 | 0.898 | 0.53 |
| ffsl cell K=24 (production 10-15 → 11-05) | 2.09 | 2.90 | 5.17 | 6.10 | 4.18 | 4.63 | 0.385 | 0.59 |
| persistence | 2.80 | 3.23 | 4.33 | 3.36 | 2.59 | 3.46 | 0.780 | — |

Structure at day 365 (zonal-mean RMS, 30–143 hPa): `sl` K=6 and K=12 beat
`lr` in every latitude band and level group; K=12 is the optimum of the `sl`
ladder (K=3 1.37, K=6 0.87, K=12 0.75, K=24 0.80 — at K=24 the polar-band
error is back at `lr`'s and the bias grows, the smoothing-per-gather effect
of §1.8 setting in); `ffsl` K=24's excess error is at
60–80S and at 60–120 hPa (the large-Courant x/y split without Lin & Rood's
cross terms, plus the open-face term); `lr` is biased HIGH by ~3 % of the mean
at 40–75 hPa where `sl` is unbiased — so the "10–16 % more bottom export than
lr" that §1.8 counted against `sl` is `lr` under-exporting, not `sl`
over-exporting. `lr` at CFL 0.25 is WORSE than at 0.5 (its residual grows with
sub-steps). Feeding `sl` the hourly wind snapshots instead of the two
bracketing ones does not help (0.95 vs 0.87), so `ADV_SL_HOURLY=0` stays.
Above ~10 hPa every scheme is worse than persistence: CH4 is
chemically active there (and the top face is open), which is why the score
starts at 30 hPa.

**Verdict.** For transport against the only truth available, the split
semi-Lagrangian scheme (`ADV_SCHEME=sl`, dinosaur horizontal gather +
`fct_lr`'s flux-form vertical, Jacobian air mass + remainder sweep, Strang,
K = 12 recommended, K = 6 nearly as good) is the most accurate scheme in the
tree AND the cheapest (0.5 s/step at K=6, ~0.9 s at K=12, for 82 tracers, vs
1.5 s ffsl and 7 s lr). To run it: `ADV_SCHEME=sl ADV_SL_K=12` (everything
else at its defaults: `ADV_SL_VERT=flux ADV_SL_SPLIT=strang ADV_SL_JAC=1
ADV_SL_HOURLY=0`). The default `ADV_SCHEME` in `coupling.py`, `driver_fast.py`
and `run_prod.py` is still `ffsl` and should be switched. `ffsl` at any affordable
K is worse than `lr`; with the sliver fix it is a correct large-Courant `lr`
and is kept for that (and for emulated winds that violate discrete
continuity, where flux form needs no fixer), but it is no longer the
recommended production transport. `lr` remains the regression reference for
`ffsl` only.

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
