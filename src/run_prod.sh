#!/usr/bin/env bash
# ============================================================================
# COMPATIBILITY SHIM -- the launcher is now src/run_prod.py
# ============================================================================
# This file used to BE the launcher. Its logic (the source-tree guard, the
# libcuda.so.1 search, the GPU pin, the production preset) moved to run_prod.py
# on 2026-09-01, where every knob is a documented --flag and `--help` is the
# inventory. Verified equivalent: run_prod.py hands driver_fast.py a
# byte-identical environment to the one this script used to build.
#
# It survives only as a redirect because run directories carry their own
# run_chain.sh that calls this path by name -- including runs that are mid-flight
# and would otherwise lose the ability to RESUME. The environment passes through
# untouched, which is exactly the interface this script always had, so
#     N_HOURS=8760 OUT_TAG=prod1yr run_prod.sh
# behaves as before, and --flags are forwarded too.
#
# DELETE THIS FILE once no runs/*/run_chain.sh references it.
# ============================================================================
exec python3 "$(dirname "$(readlink -f "$0")")/run_prod.py" "$@"
