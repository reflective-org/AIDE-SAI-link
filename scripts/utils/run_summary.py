#!/usr/bin/env python3
"""Per-run cost summary: parse a run log's PROFILE lines into <TAG>_summary.md.

  python3 run_summary.py [TAG] [--log FILE]      # TAG defaults to the only *.log here

Reads the log from the CURRENT WORKING DIRECTORY and writes beside it, the same
convention as plot_run.py / gif_run.py -- so the working directory is the run.

WHY A LOG PARSER, AND WHY THAT IS TEMPORARY. The per-step stage timings exist
only as `[prof]` stdout lines; nothing in coupled_timeseries_<TAG>.npz records
wall clock (checked 2026-08-28: it stores physics and budget terms only). So the
cost of a run is recoverable only if somebody tee'd the log, and the timing
figures quoted in run_prod.py's header cannot be confirmed against any run output
that exists. The durable fix is for coupling.py to put stage timings in the
timeseries npz; this script should then read that instead of scraping stdout.

REPORTS A TREND, NOT JUST A MEAN -- and the trend is how we learned the mean is
safe. The expectation was that microphysics cost would climb as the injected
burden builds (stiffer cells -> more adaptive substeps), making any run-mean
meaningless. The 1-year run (runs/prod1yr/, 2026-08-28) measured the opposite:
quarter means 35.8/34.1/35.5/34.2 s per step, no stage drifting >15%, over a 12.8x
mass increase. So cost is FLAT and the mean is representative.

Keep the trend table anyway. It is the check that establishes that, it is cheap,
and it is what would catch the assumption breaking under a config nobody has run
yet -- a different engine, resolution, or injection rate. Advection is the one
stage that does move, and it tracks nsub from the winds, not the run's age.
"""
import os
import re
import sys
import glob
import argparse

p = argparse.ArgumentParser()
p.add_argument("tag", nargs="?", default=None)
p.add_argument("--log", default=None, help="log file (default: the only *.log here)")
args = p.parse_args()

if args.log:
    log = args.log
elif args.tag and os.path.exists(f"{args.tag}.log"):
    log = f"{args.tag}.log"                  # <TAG>.log is what the run chain writes
else:
    # post.log is the chain's own post-processing log, never a run log -- it is
    # present in every completed run directory, so leaving it in the candidate
    # list made bare auto-detect fail exactly where it is most wanted.
    cands = [c for c in sorted(glob.glob("*.log")) if c != "post.log"]
    if len(cands) != 1:
        sys.exit(f"run_summary: cannot pick a run log here, found {cands or 'none'}; "
                 f"pass --log or a TAG matching <TAG>.log")
    log = cands[0]
text = open(log, errors="replace").read()

TAG = args.tag
if TAG is None:
    m = re.search(r'coupled_timeseries_(.+?)\.npz', text) or \
        re.search(r'saved coupled_final/timeseries/frames_(.+?)\.npz', text)
    if not m:
        sys.exit("run_summary: could not infer TAG from the log; pass it explicitly")
    TAG = m.group(1)

# ---- parse the two PROFILE line shapes -------------------------------------
S = {}
for line in text.splitlines():
    m = re.search(r'\[prof\] s=(\d+) read=([\d.]+)s advect=([\d.]+)s '
                  r'micro=([\d.]+)s settle=([\d.]+)s \(nsub=(\d+)\)', line)
    if m:
        S.setdefault(int(m.group(1)), {}).update(
            read=float(m.group(2)), advect=float(m.group(3)), micro=float(m.group(4)),
            settle=float(m.group(5)), nsub=int(m.group(6)))
    for pat, key in ((r'\[prof\] s=(\d+) bc\+polar=([\d.]+)s', 'bc'),
                     (r'\[prof\] s=(\d+) radiation=([\d.]+)s', 'rad')):
        m2 = re.search(pat, line)
        if m2:
            S.setdefault(int(m2.group(1)), {})[key] = float(m2.group(2))
if not S:
    sys.exit(f"run_summary: no [prof] lines in {log} -- was the run launched with PROFILE=1?")

LBL = [('micro', 'Microphysics'), ('rad', 'Radiation'), ('advect', 'Advection'),
       ('bc', 'BC + polar caps'), ('read', 'CESM read'), ('settle', 'Settling')]
# radiation only appears on RAD_EVERY steps, so require a key on MOST steps, not all
keys = [k for k, _ in LBL if sum(k in v for v in S.values()) > 0.5 * len(S)]
full = [s for s in sorted(S) if all(k in S[s] for k in keys)]
if not full:
    sys.exit("run_summary: no step has a complete set of stage timings")
ss = [s for s in full if s > 0] or full          # exclude step 0 (JIT) when possible


def med(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


mean = {k: sum(S[s][k] for s in ss) / len(ss) for k in keys}
total = sum(mean.values())
grid = re.search(r'  (grid \d+x\d+, \d+ levels \([^)]*\))', text)
wall = re.search(r'Complete in (\d+)s', text)
inj = re.search(r'  (SAI injection: .*)', text)
micro_mode = re.search(r'  (micro mode: .*)', text)
nsteps = re.search(r'step +\d+/(\d+)', text)

out = [f"# {TAG} — run cost summary\n"]
meta = [f"`OUT_TAG={TAG}`"]
if grid:
    meta.append(grid.group(1))
if wall:
    w = int(wall.group(1))
    meta.append(f"wall clock **{w} s** ({w/3600:.1f} h)" if w >= 3600 else f"wall clock **{w} s**")
else:
    meta.append("**run did not finish** (no 'Complete in' line)")
out.append(", ".join(meta) + f", {len(S)} steps profiled"
           + (f" of {nsteps.group(1)}" if nsteps else "") + ".  ")
if inj:
    out.append(inj.group(1) + "  ")
if micro_mode:
    out.append(micro_mode.group(1) + "\n")

out += ["", f"Steady state = steps {min(ss)}–{max(ss)}"
        + (" (step 0 excluded: it carries JIT compilation)." if 0 in S and min(ss) > 0 else "."),
        "", "| Stage | Mean | Median | min–max | % of step |", "|---|---|---|---|---|"]
for k, l in sorted(((k, l) for k, l in LBL if k in keys), key=lambda kl: -mean[kl[0]]):
    v = [S[s][k] for s in ss]
    out.append(f"| {l} | **{mean[k]:.2f} s** | {med(v):.2f} s | {min(v):.2f}–{max(v):.2f} | "
               f"{100*mean[k]/total:.1f}% |")
out.append(f"| **Total** | **{total:.2f} s** | | | |")
if 0 in S:
    out.append(f"\nStep 0 (with JIT): **{sum(S[0].get(k, 0) for k in keys):.2f} s**"
               + "".join(f", {l.lower()} {S[0][k]:.2f} s" for k, l in LBL
                         if k in keys and k in S[0]) + ".")

ns = [S[s]['nsub'] for s in ss]
out.append(f"\nAdvection substeps `nsub` {min(ns)}–{max(ns)} (mean {sum(ns)/len(ns):.0f}) "
           f"→ **{1000*mean['advect']/(sum(ns)/len(ns)):.1f} ms/substep**.")

# ---- trend: does any stage drift over the run? -----------------------------
if len(ss) >= 8:
    q = max(1, len(ss) // 4)
    out += ["\n## Trend (quarters of the run)", "",
            "| Steps | " + " | ".join(l for k, l in LBL if k in keys) + " | Total |",
            "|---" * (len(keys) + 2) + "|"]
    for i in range(0, len(ss) - q + 1, q):
        blk = ss[i:i + q]
        if not blk:
            continue
        m = {k: sum(S[s][k] for s in blk) / len(blk) for k in keys}
        out.append(f"| {blk[0]}–{blk[-1]} | "
                   + " | ".join(f"{m[k]:.2f} s" for k, _ in LBL if k in keys)
                   + f" | {sum(m.values()):.2f} s |")
    first = {k: sum(S[s][k] for s in ss[:q]) / q for k in keys}
    last = {k: sum(S[s][k] for s in ss[-q:]) / q for k in keys}
    drift = [(k, l, last[k] / first[k]) for k, l in LBL
             if k in keys and first[k] > 0.05]
    grew = [f"{l} {r:.2f}x" for k, l, r in drift if r > 1.15 or r < 0.87]
    out.append("\n" + ("Last quarter vs first: " + ", ".join(grew) + "."
                       if grew else "No stage drifted more than 15% across the run."))

# Hand-written commentary lives in a SEPARATE file so regenerating the summary
# cannot destroy it -- the generated table is disposable, the reader's notes on
# what the run means are not.
notes = f"{TAG}_notes.md"
if os.path.exists(notes):
    out.append("\n" + open(notes).read().rstrip())

open(f"{TAG}_summary.md", "w").write("\n".join(out) + "\n")
print(f"wrote {TAG}_summary.md  ({len(S)} steps, {total:.2f} s/step steady)")
