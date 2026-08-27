# runs/

Raw model output, exactly as a run writes it. One directory per `OUT_TAG`.

Everything here is **disposable and gitignored** (only this README is tracked).
A single 90-day production run writes ~5 GB per checkpoint cycle, so nothing in
this tree should ever be committed.

## Launching a run

`src/run_prod.sh` writes every output to the **current working directory** and
refuses to run from inside the source tree, so launch from a tag directory here:

```bash
mkdir -p runs/myrun && cd runs/myrun
N_HOURS=6 OUT_TAG=myrun INJ_SO2_TG_YR=10 ../../src/run_prod.sh
```

Always pair a scenario with its own `OUT_TAG`: outputs and checkpoints are keyed
by it, so reusing a tag overwrites the other scenario's results.

## What a run writes

| file | contents |
|---|---|
| `coupled_final_<TAG>.npz` | full 3-D end state |
| `coupled_timeseries_<TAG>.npz` | per-step scalar diagnostics and budget terms |
| `coupled_frames_<TAG>.npz` | probe-level frames for the filmstrip |
| `coupled_state_<TAG>_ckpt.npz` | restart checkpoint (`RESUME=1` picks it up) |
| `coupled_frames_<TAG>_ckpt.npz`, `coupled_timeseries_<TAG>_ckpt.npz` | checkpoint-cycle copies |

## Existing directories

| tag | what it is |
|---|---|
| `smoke/` | 6 h single-step smoke test (was `../../../sai_runs/`) |
| `prod1d/` | 1-day run (was loose files in `/home/susanne/linking/`) |

Figures derived from these runs belong in `outputs/`, not here.
