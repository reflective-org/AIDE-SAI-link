# aide_sai_core — the advection-only experiment (`advection-mip`)

[![CI](https://github.com/reflective-org/aide_sai_core/actions/workflows/ci.yml/badge.svg)](https://github.com/reflective-org/aide_sai_core/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](./LICENSE)

**This branch runs one experiment: transport of a prescribed aerosol size
distribution through prescribed winds, with settling either on or off.** No
microphysics, no radiation, no injection. The point is that re-running this
identical code against another model's winds isolates inter-model spread in
**transport** (and sedimentation) from everything else the coupled model does.

For the full coupled SAI model — TOMAS microphysics, RRTMGP radiation, an SO2
source — check out `main` and read its README. Nothing below needs any of that,
and the flags below switch all of it off.

```
CESM h1 hourly ──► U, V, OMEGA ──────► transport (Lin-Rood flux-form)
               └─► T, RELHUM ────────► wet size, hence fall speed (SETTLE=1 only)

  IC      one uniform, time-invariant PSD in every cell of the band (AER_SRC=fixed)
  SOURCE  none. Both faces are aerosol-free (BC_TOP_AER=0 BC_BOT_AER=0)
  SINK    whatever leaves the band -- settling and the advective flux, counted separately
  OFF     MICRO=off, RAD=0, INJ_SO2_TG_YR=0
```

- **State**: 40 mass-doubling bins × (number, dry SO4 mass) = 80 advected 3-D
  fields on the native CESM f09 grid (192 × 288), 6 h coupling step. The two gas
  tracers ride along in the same 82-row batch, inert with no microphysics.
- **Under `MICRO=off` the bins are independent passive tracers**, differing only
  in fall speed (∝ D², ∝ D in the slip limit). One run is therefore the whole
  size-dependence of drainage at once: the small bins are effectively an
  age-of-air tracer, the large bins are settling-dominated.
- **Cost**: ~12 s/step on one H100 at 33 levels, so 90 days ≈ 1.2 h and 2 years
  ≈ 10 h. (The coupled model's 90 days is ~33 h — microphysics is ~94% of it.)

> [!NOTE]
> CI covers what a GPU-less runner without the CESM archive can cover — that the
> tree imports, that advection still conserves mass, and that the closed-form
> settling physics is unchanged. Those are the two things this experiment
> measures. See [docs/VALIDATION.md](./docs/VALIDATION.md).

## Why a uniform mixing ratio

A constant mixing-ratio field is an **exact steady state of the advection
operator alone**: flux form moves ρ·q, so a constant q stays constant to
roundoff for any wind field, divergent or not. That makes every departure from
the initial field attributable — it is settling, the open faces, or scheme
error, never "the winds stirred a gradient I put in myself". With `SETTLE=0` it
makes the run a pure advection **accuracy** test with no analytic solution
needed: whatever structure appears is the scheme's own error.

Uniform *concentration* would instead impose a ~100× vertical mixing-ratio
gradient across the band and the two effects would be inseparable.

## Installation

The coupled model needs two sibling repos; **this experiment needs one.**

| repo | needed here? |
|---|---|
| [`reflective-org/tomas-jax`](https://github.com/reflective-org/tomas-jax) branch **`gpu-fast`** | **yes** — `coupling.py` imports it at module scope for the bin grid, even at `MICRO=off` |
| [`climate-analytics-lab/jax-rrtmgp`](https://github.com/climate-analytics-lab/jax-rrtmgp) | **no** — `RAD=0` drops radiation and this dependency, patch included |

```bash
git clone https://github.com/reflective-org/aide_sai_core.git
git -C aide_sai_core checkout advection-mip

# microphysics -- the gpu-fast branch, not main. Clone it BESIDE this repo
# (or set TOMAS_JAX_PATH) and it is found automatically.
git clone -b gpu-fast https://github.com/reflective-org/tomas-jax

pip install -r aide_sai_core/requirements.txt
pip install --upgrade "jax[cuda12]>=0.6.2"   # the CPU wheel cannot do these runs
```

Do **not** activate the tomas-jax `.venv` — it is CPU-only jaxlib with no
xarray. The working combination is system `python3` with the GPU jax wheel.

Verify before touching CESM data — this resolves tomas-jax the same way the model
does, so it fails the same way a run would:

```bash
REPO=$PWD/aide_sai_core
export LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
python3 -c "
import sys; sys.path.insert(0, '$REPO')
from tomas_jax.core.config import xk_boundaries   # right repo, found via sibling layout
from tomas_jax.fast import run_fast               # right branch: gpu-fast only
import jax; print('deps ok:', jax.devices())      # GPU wheel loads
"
```

A pass is one line, `deps ok: [CudaDevice(id=0)]`. The `export` is only needed in
a bare shell: this box keeps `libcuda.so.1` outside the loader path and
`run_prod.sh` prepends the same directory itself. **JAX falls back to CPU
silently**, so if a step takes minutes instead of seconds, check `jax.devices()`
first.

Figures need `matplotlib` + `cartopy`; the GIFs additionally need `ffmpeg` on
`PATH`.

## CESM forcing

The model reads hourly `h1` files for **14 variables** — `U V OMEGA T
num_a{1,2,3} so4_a{1,2,3} SO2 H2SO4 OH RELHUM`. A transport-only run actually
uses few of them (`U V OMEGA` for transport, `T`/`RELHUM` for the wet size), but
**all 14 are opened at startup, so a missing file stops the run.** The 7
radiation variables are not read at all at `RAD=0`.

Layout: `$CESM_DIR/hour_1/$CESM_PREFIX.h1.<VAR>$CESM_SUF`. Only levels inside the
band are read (~5.3 MB per variable-hour), so a subset is enough — but at full
vertical resolution the archive is TBs, so subset in level and time before
copying one. Full variable list and units:
[docs/COUPLING_VARIABLES.md](./docs/COUPLING_VARIABLES.md).

**On the shared H100 box these runs were made on, set none of this** — the
defaults in `coupling.py` resolve to the FWHIST archive under `/data`.
`CESM_DIR`/`CESM_PREFIX`/`CESM_SUF` matter only off that box.

## Running

**Run from a directory that is not the repo.** Every output path is relative to
the working directory, so that directory *is* the run, and these runs write GBs
of `.npz`. `run_prod.sh` refuses to start with the repo as `$PWD` rather than let
that happen silently.

Two flags are load-bearing and easy to get wrong:

* **`DRIVER=coupling.py`** — the default `driver_fast.py` exists only to swap in
  the batched microphysics engine and **exits at `MICRO=off`**. `coupling.py`
  uses the same advection and the same settling, so the two agree.
* **`P_LO_HPA=0.03 P_HI_HPA=150`** — the domain. The lid has to be raised off its
  `1` hPa default; the floor is stated explicitly even though `150` *is* the
  default, because where the faces sit governs the whole result and a command
  that names both is self-describing. See [the lid and the
  floor](#the-lid-and-the-floor).

```bash
REPO=$HOME/noah/coupling_prod          # this clone
RUNS=$HOME/noah/advection_mip_runs     # anywhere BUT the repo
mkdir -p "$RUNS" && cd "$RUNS"

# smoke test: one 6 h coupling step, a couple of minutes with startup and JIT.
# Do this first on any new archive -- it proves the CESM paths and the GPU.
DRIVER=coupling.py MICRO=off RAD=0 AER_SRC=fixed \
  BC_TOP_AER=0 BC_BOT_AER=0 P_LO_HPA=0.03 P_HI_HPA=150 \
  N_HOURS=6 OUT_TAG=mip_smoke $REPO/run_prod.sh
```

### With settling — the drainage experiment

`SETTLE=1` and `WET_SETTLING=1` are the defaults, so **settling is on unless you
turn it off**. Fall speed uses the *wet* H2SO4/H2O droplet at the model's own T
and RH, which is what makes the settling channel nearly model-independent and the
advective channel the interesting one.

```bash
DRIVER=coupling.py MICRO=off RAD=0 AER_SRC=fixed \
  BC_TOP_AER=0 BC_BOT_AER=0 P_LO_HPA=0.03 P_HI_HPA=150 \
  N_HOURS=2160 FRAME_EVERY=24 OUT_TAG=mip_settle $REPO/run_prod.sh
```

90 days (`N_HOURS=2160` = 360 steps), ~1.2 h. The band drains and nothing
refills it, so the burden only falls; `{TAG}_drain.png` is the figure this run
exists for.

### Without settling — pure advection, and age of air

```bash
DRIVER=coupling.py MICRO=off RAD=0 AER_SRC=fixed \
  BC_TOP_AER=0 BC_BOT_AER=0 P_LO_HPA=0.03 P_HI_HPA=150 \
  SETTLE=0 \
  N_HOURS=2160 FRAME_EVERY=24 OUT_TAG=mip_nosettle $REPO/run_prod.sh
```

Same run, one flag. Now the tracer is passive, the initial field is an exact
steady state, and everything that appears is scheme error plus what the open
faces do. The remaining fraction of *original* air is then an age-of-air
diagnostic — note that **90 days shows almost nothing** in the interior
(0.99–1.00 of original air remains through 8.6–43 hPa); the Brewer-Dobson
drainage this is aimed at needs **years**, and mass loss on a 90-day run is
almost entirely the 100–143 hPa layer exchanging with the troposphere, because
that is where the air mass is.

For a multi-year version, raise `FRAME_EVERY` and consider `N_BINS=1`:

```bash
DRIVER=coupling.py MICRO=off RAD=0 AER_SRC=fixed \
  BC_TOP_AER=0 BC_BOT_AER=0 P_LO_HPA=0.03 P_HI_HPA=150 \
  SETTLE=0 N_BINS=1 \
  N_HOURS=17520 FRAME_EVERY=120 OUT_TAG=mip_2yr $REPO/run_prod.sh
```

> [!WARNING]
> `N_BINS=1` is only valid **with `SETTLE=0`**. With no fall speed and no
> microphysics the 40 bins are identical passive tracers, so 40 of them is 40
> copies of one answer. With `SETTLE=1` it collapses the fall-speed spectrum the
> drainage run is measuring. Side effect: one bin spanning the whole mass grid
> leaves no bin-bound tying number to mass, so `Dp(M/N)` in the log is
> meaningless — analyse `frames_zm_mas` and ignore the number moment.

`FRAME_EVERY` must be raised for anything much longer than 90 days: **the whole
frames history is rewritten at every frame**, so I/O cost grows as (frames)².
A 2-year run at `FRAME_EVERY=24` would write tens of GB and spend hours
re-serialising; at 120 (5-day frames) it is 146 frames.

### Resuming, and pairing tags

```bash
RESUME=1 DRIVER=coupling.py MICRO=off RAD=0 AER_SRC=fixed \
  BC_TOP_AER=0 BC_BOT_AER=0 P_LO_HPA=0.03 P_HI_HPA=150 \
  N_HOURS=2160 FRAME_EVERY=24 OUT_TAG=mip_settle $REPO/run_prod.sh
```

State is checkpointed every `FRAME_EVERY` hours, so a killed run continues and a
long run can be stopped early and still plots. **Repeat every flag, including
`SETTLE`.** The injection scenario is stamped into the checkpoint and a mismatched
`RESUME` is refused; physics-mode flags (`SETTLE`, `WET_*`, `ADV_VPOS`) only
*warn*, so `SETTLE=0` onto a `SETTLE=1` checkpoint will run. Always pair a
configuration with its own `OUT_TAG` — outputs and checkpoints are keyed by it
and reusing a tag overwrites the other run's results.

### The tagged-pulse variant

`run_pulse_bdc.sh` is the same transport-only machinery seeded differently: only
the tropical lower stratosphere (`FIXED_LAT_MAX_DEG=15`, `FIXED_P_LO_HPA=40`,
`FIXED_P_HI_HPA=90`) instead of uniformly, with `SETTLE=0 N_BINS=1`. A uniform IC
can only show the circulation as the *absence* of tracer arriving, which needs
4–6 years; a pulse shows the ascent directly from the first month, as the thing
that moves — the model analogue of the water-vapour tape recorder.

```bash
cd "$RUNS"
$REPO/run_pulse_bdc.sh 15                      # 15 model years, ~12 h, ~3.8 GB
$REPO/run_pulse_bdc.sh 1                       # ~50 min -- do this one first

python3 $REPO/pulse_progress_abs.py pulse_15yr # one lat-pressure panel per year
python3 $REPO/pulse_deep_branch.py  pulse_15yr # ascent/descent Hovmoller (+ --log)
python3 $REPO/gif_run.py pulse_15yr --log --massdens --decades 5 --stride 4
```

Every setting in that launcher is a decision, documented in its own header;
`MANIFEST.md` records the seeded window, the measured transport rates, and the
caveats that go with them. Both pulse figure scripts import only numpy and
matplotlib and read `coupled_frames_<TAG>.npz` from the working directory, so
anyone handed that one file can regenerate every panel with no GPU, no CESM and
no sibling repos.

## Outputs and figures

Each run writes, into the launch directory:

| file | what |
|---|---|
| `coupled_final_<TAG>.npz` | full 3-D prognostic state at the end of the run |
| `coupled_frames_<TAG>.npz` | snapshots every `FRAME_EVERY` hours — column integrals, zonal-mean cross-sections, probe level |
| `coupled_timeseries_<TAG>.npz` | per-step scalars, including the resolved `D_*` drain counters |
| `coupled_state_<TAG>_ckpt.npz` | restart checkpoint (+ `_frames_`/`_timeseries_` twins) |

Frames are the big file: ~50 MB per frame with the full 40-bin grid (≈5 GB for
90 days at `FRAME_EVERY=24`), ~3.5 MB at `N_BINS=1`.

```bash
python3 $REPO/plot_run.py mip_settle
python3 $REPO/gif_run.py  mip_settle --log --massdens
```

Both read `coupled_*_<TAG>.npz` from the **current working directory** and write
`<TAG>_*.png` beside them:

| figure | what |
|---|---|
| `<TAG>_drain.png` | **the figure of this experiment** — decay and e-folding time, the settling/advective split, and that split by latitude and by size |
| `<TAG>_crosssection.png` | zonal-mean lat-height filmstrip: the vertical transport a map cannot show |
| `<TAG>_filmstrip.png` | column-integral maps through time |
| `<TAG>_sizedist.png` | dN/dlogDp evolution, global mean and 15S–15N |
| `<TAG>_dashboard.png` | burdens, effective size, budget closure |

The two channels in `drain.png` are never merged into one curve: settling depends
on the model's temperature and nothing else and should be nearly
model-independent, while the advective flux through the same face *is* the
residual circulation, and is where two dycores are expected to disagree. The
radiative panels of the dashboard and the `dTrad` GIFs are skipped by themselves
at `RAD=0`.

**The budget printout closes to roundoff** (`sum` vs `M/M0-1`) — that line is the
model auditing itself, and on a transport-only run it is the whole result. If a
change breaks that closure, the change is wrong.

## Knobs that matter here

Every knob is an environment variable, read **once at module import** and echoed
in the run header, so any log is self-describing. The header prints
`==> TRANSPORT-ONLY run` only when all of it is really in force, and names
whatever is still active otherwise — `AER_SRC=fixed` with radiation or
microphysics still on is a legitimate run, but it is not this experiment, and the
difference is invisible in the output files.

| variable | default | here |
|---|---|---|
| `DRIVER` | `driver_fast.py` | **`coupling.py`** — required at `MICRO=off` |
| `MICRO` | `full` | **`off`** — bins become independent passive tracers |
| `RAD` | `1` | **`0`** — drops radiation and the jax-rrtmgp dependency |
| `AER_SRC` | `mam4` | **`fixed`** — the prescribed uniform PSD |
| `BC_TOP_AER` / `BC_BOT_AER` | `1` / `1` | **`0` / `0`** — both faces aerosol-free, so nothing refills the band |
| `SETTLE` | `1` | `1` for drainage, `0` for pure advection |
| `WET_SETTLING` | `1` | leave on — fall speed on the wet droplet at the model's T and RH |
| `P_LO_HPA` / `P_HI_HPA` | `1` / `150` | **`0.03`** / 150 (see below) |
| `N_HOURS` | `2160` (90 d) | run length **in forcing hours**; `17520` = 2 yr |
| `FRAME_EVERY` | `24` | raise to `120` beyond ~90 days |
| `N_BINS` | `0` (= 40) | `1` only with `SETTLE=0` |
| `FIXED_PSD` | `lognormal` | `FIXED_DG_NM=200`, `FIXED_SIGMA=1.6`, `FIXED_N=1e8` #/kg |
| `OUT_TAG` | — | always set it, one per configuration |
| `ADV_F32` / `ADV_CFL` | `1` / `0.5` | production precision — **validate advection changes here**, not at the f64/cfl=0.2 module defaults |

`FIXED_N`'s absolute scale is irrelevant: with no microphysics the system is
linear in the aerosol, so every reported drained fraction is scale-free. 1e8 #/kg
is ~8 #/cm³ at 50 hPa / 210 K, so the printed concentrations are recognisable
rather than arbitrary. `FIXED_DG_NM` is the one knob that really changes the
answer — settling goes as D².

A second experiment worth one line: `FIXED_P_HI_HPA=30` starts the run with a
real vertical gradient for the winds to act on **immediately**, instead of one
settling has to create first. Under a uniform IC the winds only enter at second
order in time.

Full reference: [docs/CONFIGURATION.md](./docs/CONFIGURATION.md). Every default
and the reasoning behind it: **[MANIFEST.md](./MANIFEST.md), the canonical
reference for this tree** — where `docs/` disagrees with it, MANIFEST is current.

> [!IMPORTANT]
> `run_prod.sh` **hard-sets** `INIT_BIN`, `STATE_CKPT`, `FAST_CELL_CAP`,
> `ADV_VPOS`, `DEBUG` and `PROFILE`. Passing those on the command line is
> silently ignored — see [docs/CONFIGURATION.md](./docs/CONFIGURATION.md).

## The lid and the floor

Both faces of the band are open, and on a transport-only run the box's own
boundaries are the main thing that can masquerade as physics.

**The lid was the contaminant at 1 hPa.** At that top face, 6.9% of the domain's
air descends through it per 90 days carrying q = 0 (`BC_TOP_AER=0`) — which
*dilutes without appearing in the top-face mass term at all*, since that term
only counts aerosol leaving. `P_LO_HPA=0.03` (33 levels) cuts it to ~0.8%, so the
domain drains through its base, and `BC_TOP_AER=1` — which would stop it draining
at all — is not needed. The forcing is 70-level to 4.5e-6 hPa, so the lid is a
free choice; tropical zonal-mean ascent turns poleward at ~2.1 hPa in this
forcing, so the old 1 hPa lid was already inside the circulation.

**The floor is at 143 hPa** (`P_HI_HPA=150` snaps to the nearest model level),
1–3 km above the real extratropical tropopause, and most of the drainage happens
poleward of 35°. Tracer is therefore removed somewhat early: a measured decay
rate is a **lower bound** on the true stratospheric lifetime, and it is honest to
say "left the 143 hPa domain", not "removed".

Two more standing cautions for reading these runs:

* **Do not read tracer descent off Eulerian zonal-mean omega.** In the winter
  polar stratosphere it has the *opposite sign* to the residual circulation that
  actually moves tracer.
* Diagnostics and plots report concentrations at **STP**; the kernels keep
  ambient density. Don't mix the two.

## Where the seams are

Read this as a **dycore–aerosol coupler** with CESM and TOMAS plugged in — the
whole reason for an advection-only mode is to swap the meteorology and compare.
`AER_SRC` selects the aerosol source, and `driver_fast.py` swaps the microphysics
engine into `coupling.py` at import rather than branching inside it. The
meteorology reader is **not** generic yet — it expects the CESM `h1` layout — so
treat a new forcing source as work, not configuration.

## Citations

The parameterizations this experiment composes are other people's work:

- Lin & Rood (1996), *Mon. Weather Rev.* 124, 2046 — flux-form advection
- Adams & Seinfeld (2002), *J. Geophys. Res.* 107, 4370 — TOMAS sectional bin grid
- Tabazadeh et al. (1997), *Geophys. Res. Lett.* 24, 1931 — H2SO4/H2O equilibrium composition (the wet size)
- Tang (1997), *J. Geophys. Res.* 102, 1883 — solution density

## License

This project is released under the Apache 2.0 License - see the [LICENSE](./LICENSE) file for details.
