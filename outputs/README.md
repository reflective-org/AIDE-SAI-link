# outputs/

Derived products: figures, tables and anything else produced *from* a run rather
than *by* one.

Contents are **gitignored** (only this README is tracked) — regenerate them from
the run in `runs/<TAG>/` rather than committing them.

The distinction from `runs/`:

| | `runs/<TAG>/` | `outputs/` |
|---|---|---|
| written by | the model (`src/run_prod.py`) | analysis and plotting scripts |
| contents | `.npz` state, checkpoints, frames | figures, tables, summaries |
| reproducible by | re-running the model (hours) | re-running a plot script (seconds) |

`scripts/utils/plot_run.py <TAG>` currently writes its dashboard/filmstrip/sizedist PNGs next
to the run data. Repointing the plotting and analysis scripts at this directory
is part of the separate validation/plotting reorganisation — see
`docs/DEFERRED.md`.
