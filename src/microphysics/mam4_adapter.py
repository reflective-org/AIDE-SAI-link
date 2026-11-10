"""The MAM4-JAX adapter: coupling.py's grid state <-> MAM4 modal box model.

"Run 3" of the SAI campaign: a MAM4 modal-microphysics variant of the TOMAS
sectional model. Same seam as tomas_fast.py / tomas_emulator.py (the
`run_microphysics_full` slot that driver_fast.py monkeypatches), same return
contract, but the prognostic aerosol state is MODAL:

    num : (4, nlev, nlat, nlon)  number mixing ratio per mode [#/kg]
    so4 : (4, nlev, nlat, nlon)  sulfate mass mixing ratio per mode [kg SO4/kg]

Mode order is MAM4-JAX's MODE_NAMES: accum, aitken, coarse, primary_carbon.
Mode 3 (primary carbon) is carried for layout parity with the core but is
always zero -- this run is sulfate only, and MAM4 gives that mode no so4 slot.

WHAT IS MAM4's AND WHAT IS SWAPPED (the spec for run 3):
  * condensation (gasaerexch, closed-form H2SO4 uptake), coagulation,
    calcsize/rename: MAM4-JAX's own, untouched -- that is what the comparison
    with TOMAS is about.
  * nucleation: Dunne et al. 2016 binary-neutral rate, EXACTLY
    tomas_jax.fast.nucleation.dunne_binary_neutral_rate, replacing
    Vehkamaki 2002 + the Wang & Penner PBL overlay. Only the rate leaf is
    swapped (`dunne_nuc_mosaic_1box` below has the signature of
    mer07_veh02_nuc_mosaic_1box); MAM4's sizing of the new particles
    (KK2002 growth from the cluster to the Aitken lower edge, the Aitken
    size constraints, the H2SO4 availability cap) is kept. See the docstring
    of dunne_nuc_mosaic_1box for exactly what happens to the cluster size.
  * water uptake: Tabazadeh 1997 H2SO4/H2O equilibrium, EXACTLY
    tomas_jax.fast.water.h2so4_equilibrium_wt, replacing Koehler +
    deliquescence/crystallisation hysteresis (`wateruptake_tabazadeh`).
  * SO2 + OH -> H2SO4: NOT in the core. coupling's so2_oxidation_step
    (tomas_jax.physics.so2_chemistry) runs before every inner core step on
    the per-box gas, exactly as tomas_jax.fast interleaves its own copy of the
    same analytic solution; the core's hard-coded 1e-16 mol/mol/s H2SO4 stub
    is disabled with configure_gas_netprod(h2so4=0, soa=0).

UNITS AND CONVENTIONS (why each one):
  * MAM4 carries so4_a with molecular weight 115.107 (NH4HSO4-equivalent,
    core.data.ADV_MASS); TOMAS carries SO4 at 96. coupling.bin_mam4 already
    reads the archive so4_a* AS kg SO4(96)/kg, so the modal state keeps the
    TOMAS meaning (total sulfate is comparable across the two engines, the
    IC/BC/settling/radiation code needs no change) and THIS file converts at
    the core boundary with SO4_TO_Q = 115.107/96 so the core's mole
    bookkeeping (1 mol H2SO4 -> 1 mol so4_a) stays exact. Sulfur budget:
    S = SO2*32/64 + H2SO4*32/98 + SO4*32/96.
  * The core gets mixing ratios (kg/kg, #/kg) -- its own units -- while the
    chemistry gets per-box kg at BOXVOL = 1 m3 (mixing ratio x rho_air),
    exactly as tomas_fast.py builds its FastState.
  * RH: coupling passes CESM's RELHUM (over ice on these levels) as a 0-1
    fraction. tomas_fast hands that fraction to Tabazadeh unchanged, so this
    adapter does the same (state['relhum']); stock MAM4 would instead derive
    RH over liquid water from q[0]/qsat_water and clamp it, and would collapse
    to DRY particles below rhcrystal = 0.35 -- i.e. no water at all on
    stratospheric sulfate. q[0] (water vapour) is set to RH*qsat_ice(T,p) so
    the stock path, when selected for the A/B, sees a physically consistent
    specific humidity.
  * Inner step MAM4_DT = 1800 s (the CAM physics step MAM4 runs at in the
    host model), 12 per 6 h coupling step; OH is block-averaged from
    coupling's 60 samples/step onto those 12 (coupling._oh_to_substeps).
  * float64 throughout (mam4_jax enables x64 at import; coupling runs x64).

IMPORT ORDER MATTERS, as for tomas_fast.py: this module imports coupling,
which reads the environment at import, so the driver's os.environ.setdefault
block must run first. Do not hoist this import.
"""
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths                      # noqa: E402 -- puts src/ subdirs on sys.path

import coupling as C               # noqa: E402 -- see IMPORT ORDER above
import settling                    # noqa: E402

import mam4_jax                                           # noqa: E402
from mam4_jax.coupling import amicphys as _amic           # noqa: E402
from mam4_jax.physics import newnuc as _nn                # noqa: E402
from mam4_jax.physics.calcsize import calcsize as _calcsize            # noqa: E402
from mam4_jax.physics.wateruptake import wateruptake as _wateruptake_stock  # noqa: E402
from mam4_jax.core import data as D                       # noqa: E402
from mam4_jax.core.constants import RHOH2O                # noqa: E402
from tomas_jax.fast.nucleation import dunne_binary_neutral_rate   # noqa: E402
from tomas_jax.fast.water import h2so4_equilibrium_wt     # noqa: E402
from tomas_jax.physics.nucleation import _MNUC            # noqa: E402  Dunne/Kulmala cluster mass
from tomas_jax.physics.so2_chemistry import so2_oxidation_step    # noqa: E402
from tomas_jax.core.config import (N_GAS_SPECIES, SRTSO4 as _G_H2SO4,  # noqa: E402
                                   SRTSO2 as _G_SO2, MW_SO4 as _MW_SO4_TOMAS)

# ---- knobs -------------------------------------------------------------------
MAM4_DT       = float(os.environ.get('MAM4_DT', '1800'))      # inner core step [s]
MAM4_CELL_CAP = int(os.environ.get('MAM4_CELL_CAP', '250000'))  # cells per chunk
# 'swap' = Dunne nucleation + Tabazadeh water (the run-3 physics);
# 'stock' = MAM4-JAX as shipped (Vehkamaki + Koehler), for the A/B only.
MAM4_PHYS     = os.environ.get('MAM4_PHYS', 'swap').lower()
if MAM4_PHYS not in ('swap', 'stock'):
    sys.exit(f"mam4_adapter: MAM4_PHYS must be swap|stock, got {MAM4_PHYS!r}")
N_INNER = int(round(C.DT_MICRO / MAM4_DT))
if abs(N_INNER * MAM4_DT - C.DT_MICRO) > 1e-6:
    sys.exit(f"mam4_adapter: MAM4_DT={MAM4_DT} does not divide the "
             f"{C.DT_MICRO:.0f} s coupling step")
# OH sampling: keep coupling's 60 samples/step (the same field the fast
# physics sees) and block-mean onto the N_INNER core steps.
if 'OH_SUBSTEPS' not in os.environ:
    C.OH_SUBSTEPS = 60

# the core's process-global configuration, set once at import
_amic.configure_condensation(backend='substep', n_substeps=4)   # exact H2SO4 closed form
mam4_jax.configure_gas_netprod(h2so4=0.0, soa=0.0)             # chemistry is ours

# ---- layout ------------------------------------------------------------------
NMODE      = D.NTOT_AMODE                                   # 4
PCNST      = D.PCNST                                        # 35
NUM_SLOTS  = np.asarray(D.LMAP_NUM, dtype=np.int32)         # [17, 22, 30, 34]
_SO4_ROW   = np.asarray(D.LMAP_AER[:, D.AMICPHYS_IAER_SO4]) # [10, 18, 25, -1]
SO4_VALID  = _SO4_ROW >= 0                                  # pcarbon has no so4 slot
SO4_SLOTS  = np.where(SO4_VALID, _SO4_ROW, 0).astype(np.int32)
H2SO4_SLOT = int(D.LMAP_GAS[1])                             # 6
H2OMMR_SLOT = 0
SIGMA_G    = np.asarray(D.SIGMAG_AMODE)                     # (1.8, 1.6, 1.8, 1.6)
ALNSG      = np.log(SIGMA_G)
DGNUM0     = np.asarray(D.DGNUM_AMODE)                      # calcsize's seed dgnum
RHO_SO4    = float(C.RHO_AER)                               # 1770, both engines
MW_SO4A_Q  = float(D.ADV_MASS[SO4_SLOTS[0] - 5])            # 115.107 (vmr factor of the so4 slots)
MW_H2SO4   = 98.0784
SO4_TO_Q   = MW_SO4A_Q / _MW_SO4_TOMAS                      # kg SO4(96) -> MAM4 so4_a mass
Q_TO_SO4   = 1.0 / SO4_TO_Q
ZMID_STRAT = 20000.0   # [m] any value above pblh=0: the PBL nucleation branch is off
                       # (and gone entirely under MAM4_PHYS=swap)


# =============================================================================
# the two physics swaps
# =============================================================================
# Dunne cluster expressed as a dry sulfate sphere at MAM4's sulfate density, so
# the KK2002 size correction starts from the same cluster TOMAS nucleates at
# (_MNUC = 3.47e-24 kg, Kulmala 2013 cluster; TOMAS's own cluster density is
# 1350 kg/m3 -> r 0.85 nm; at 1770 kg/m3 the same mass is d = 1.55 nm).
DCLUS_M = (6.0 * _MNUC / (np.pi * RHO_SO4)) ** (1.0 / 3.0)


def dunne_nuc_mosaic_1box(dtnuc, temp, rh, press, zm, pblh,
                          qh2so4_cur, qh2so4_avg, h2so4_uptkrate,
                          dplom_sect, dphim_sect, newnuc_method_flagaa=11):
    """Drop-in for mam4_jax.physics.newnuc.mer07_veh02_nuc_mosaic_1box with the
    Dunne 2016 binary-neutral rate in place of Vehkamaki 2002 (+ PBL overlay).

    Same signature and the same 8-tuple return, so the amicphys glue
    (_mam_newnuc_1subarea: Aitken-mode deposition, the size constraints and
    the H2SO4 cap) runs unchanged. Stage by stage, against the stock leaf:

      rate      J_bn = dunne_binary_neutral_rate(T, [H2SO4]) on the
                gasaerexch time-mean H2SO4 (qh2so4_avg, MAM4's
                newnuc_h2so4_conc_optaa=2 convention). NO clamps on T, RH or
                [H2SO4] (stock clamps T >= 230.15 K -- i.e. stock is pinned to
                its lower T bound everywhere in the stratosphere -- and
                1e4 <= [H2SO4] <= 1e11 cm-3). MAM4's exit gate J > 1e-6 cm-3
                s-1 (rateloge > -13.82) is kept: it is part of MAM4's handling,
                not of the rate.
      PBL       none. Dunne binary-neutral only (as in tomas_jax.fast).
      cluster   fixed: the Dunne/Kulmala cluster mass _MNUC (TOMAS nucleates
                every particle at this mass) as a d = 1.55 nm sulfate sphere.
                Stock derives the cluster from Vehkamaki's molecule count.
      KK2002    kept, with dnuc = that 1.55 nm cluster (stock: 2*r_cluster,
                floored at 1 nm). The new particles are grown to the Aitken
                lower edge dplom = exp(0.67 ln dgnumlo + 0.33 ln dgnum) =
                12.4 nm and the surviving rate is J * exp(nu/dfin - nu/dnuc),
                with the uptake rate h2so4_uptkrate the glue passes (MAM4-JAX
                passes a 1e-3 1/s placeholder there, upstream shortcut -- same
                for stock and swap, flagged in STATUS.md).
      mass      each new particle carries the mass of a dplom sphere; the gas
                removed is that mass (capped at 0.9999 of the available
                H2SO4). This is how MAM4 books nucleation: the growth from the
                cluster to 12 nm is charged to the gas here, not to
                condensation. TOMAS instead deposits the cluster mass itself
                (1 particle per 3.47e-24 kg) in bin 0 and lets condensation
                grow it -- so for the SAME J the two engines create very
                different NUMBERS of particles per unit sulfur (factor
                mass(dplom)/_MNUC ~ 500). That is the modal scheme's
                nucleation handling, and the comparison measures it.
    """
    del zm, pblh, newnuc_method_flagaa          # PBL overlay: not in Dunne
    rgas_local = _nn._RGAS_J_PER_K_PER_KMOL / 1.0e3   # J/K/mol
    avogad = 6.02214e26 / 1.0e3                       # 1/mol
    pi = jnp.pi
    onethird = 1.0 / 3.0

    cair = press / (temp * rgas_local)                # mol/m3
    so4vol_in = qh2so4_avg * cair * avogad * 1.0e-6   # molec/cm3

    # ---- the Dunne rate, unclamped --------------------------------------
    ratenuclt = dunne_binary_neutral_rate(temp, so4vol_in)        # cm-3 s-1
    rateloge = jnp.log(jnp.maximum(ratenuclt, 1.0e-38))
    rateloge = rateloge + jnp.log(jnp.maximum(1.0e-38, _nn._ADJUST_FACTOR_BIN_TERN_RATENUCL))
    rate_ok = rateloge > -13.82
    ratenuclt = jnp.exp(rateloge)
    ratenuclt_bb = ratenuclt * 1.0e6                              # #/m3/s
    dnclusterdt = jnp.where(rate_ok, ratenuclt_bb, 0.0)

    # ---- from here on: stock MAM4 stages 5-10, cluster size fixed --------
    rh_clamp = jnp.maximum(0.10, jnp.minimum(0.95, rh))
    wetvol_dryvol = 1.0 - 0.56 / jnp.log(rh_clamp)

    dpdry_clus = jnp.full_like(ratenuclt, DCLUS_M)
    isize_nuc = jnp.ones_like(jnp.asarray(so4vol_in), dtype=jnp.int32)
    igrow = (dpdry_clus <= dplom_sect).astype(jnp.float64)
    dpdry_part = jnp.where(
        dpdry_clus <= dplom_sect, dplom_sect,
        jnp.where(dpdry_clus >= dphim_sect, dphim_sect, dpdry_clus))
    voldry_part = (pi / 6.0) * (dpdry_part ** 3)

    dens_part = _nn._DENS_SULFACID
    dens_nh4so4a = jnp.full_like(rateloge, dens_part)
    mass_part = voldry_part * dens_part
    kgaero_per_moleso4a = 1.0e-3 * _nn._MW_SULFACID * (D.MW_SO4A_HOST / _nn._MW_SO4A)
    wet_volfrac_so4a = 1.0 / wetvol_dryvol

    # Kerminen-Kulmala 2002, as stock
    tmp_spd = 14.7 * jnp.sqrt(temp)
    gr_kk = 3.0e-9 * tmp_spd * _nn._MW_SULFACID * so4vol_in / (dens_part * wet_volfrac_so4a)
    dfin_kk = 1.0e9 * dpdry_part * (wetvol_dryvol ** onethird)
    dnuc_kk = jnp.maximum(1.0e9 * dpdry_clus, 1.0)
    gamma_kk = (0.23 * dnuc_kk ** 0.2 * (dfin_kk / 3.0) ** 0.075
                * (dens_part * 1.0e-3) ** (-0.33) * (temp / 293.0) ** (-0.75))
    tmpa_kk = jnp.maximum(h2so4_uptkrate * 3600.0, 0.0)
    tmpb_kk = 6.7037e-6 * (temp ** 0.75) / cair * 3600.0
    cs_prime_kk = tmpa_kk / (4.0 * pi * tmpb_kk * _nn._ACCOM_COEF_H2SO4)
    nu_kk = gamma_kk * cs_prime_kk / gr_kk
    factor_kk_grow = jnp.exp(nu_kk / dfin_kk - nu_kk / dnuc_kk)
    factor_kk = jnp.where(igrow > 0, factor_kk_grow, 1.0)
    ratenuclt_kk = ratenuclt_bb * factor_kk

    qmolso4a_del_max = jnp.maximum(0.0, ratenuclt_kk * dtnuc * mass_part) / (
        kgaero_per_moleso4a * cair)
    freduce = jnp.where(qmolso4a_del_max > qh2so4_cur,
                        qh2so4_cur / jnp.maximum(qmolso4a_del_max, 1.0e-30), 1.0)
    final_ok = rate_ok & (freduce * ratenuclt_kk > 1.0e-12)
    qh2so4_del_pos = jnp.minimum(0.9999 * qh2so4_cur, freduce * qmolso4a_del_max)
    qh2so4_del = jnp.where(final_ok, -qh2so4_del_pos, 0.0)
    qnh3_del = jnp.zeros_like(qh2so4_del)
    qso4a_del = -qh2so4_del
    qnh4a_del = jnp.zeros_like(qh2so4_del)
    qnuma_del_raw = 1.0e-3 * (qso4a_del * _nn._MW_SO4A) / jnp.maximum(mass_part, 1.0e-30)
    qnuma_del = jnp.where(final_ok, qnuma_del_raw, 0.0)
    return (isize_nuc, qnuma_del, qso4a_del, qnh4a_del,
            qh2so4_del, qnh3_del, dens_nh4so4a, dnclusterdt)


def wateruptake_tabazadeh(state):
    """Drop-in for mam4_jax.physics.wateruptake.wateruptake, sulfate only.

    Water on each mode from the Tabazadeh 1997 H2SO4/H2O equilibrium -- the
    exact tomas_jax.fast.water relations: wt% = h2so4_equilibrium_wt(T, RH%)
    and water = acid * (100/wt - 1) with acid = moles(so4) * 98 (wt% is defined
    on H2SO4 mass). RH comes from state['relhum'] (the coupling's 0-1 field,
    as for TOMAS), not from q[0]/qsat_water as stock does, and there is no
    crystallisation/deliquescence hysteresis: a stratospheric sulfate particle
    is a solution droplet at any RH.

    Outputs follow MAM4's conventions so the core's coagulation and
    condensation see what they expect: wet volume = dry volume + water/RHOH2O
    (volume-additive, as stock), dgncur_awet = dgncur_a * (wet/dry)^(1/3),
    wetdens = (dry mass + water)/wet volume with the dry mass in MAM4's
    115-mass convention (that is what q carries inside the core), qaerwat in
    kg water / kg air. Modes with no sulfate (primary carbon) get no water.
    """
    q = jnp.asarray(state['q'], dtype=jnp.float64)
    dgncur_a = jnp.asarray(state['dgncur_a'], dtype=jnp.float64)
    t = jnp.asarray(state['t'], dtype=jnp.float64)
    rh = jnp.asarray(state['relhum'], dtype=jnp.float64)

    so4_q = q[..., SO4_SLOTS] * jnp.asarray(SO4_VALID, dtype=jnp.float64)  # (..., 4) MAM4 mass
    moles = so4_q / MW_SO4A_Q                                 # kmol SO4 / kg air
    acid = moles * MW_H2SO4                                   # kg H2SO4 / kg air
    dryvolmr = so4_q / RHO_SO4                                # m3 / kg air

    wt = h2so4_equilibrium_wt(t, rh * 100.0)                  # (...,) in [10, 80]
    water = acid * (100.0 / wt[..., None] - 1.0)              # kg water / kg air
    wtrvolmr = water / RHOH2O

    v2ncur_a = 1.0 / ((jnp.pi / 6.0) * dgncur_a ** 3 * jnp.exp(4.5 * ALNSG ** 2))
    has_dry = dryvolmr > 1.0e-30
    ratio = jnp.where(has_dry, 1.0 + wtrvolmr / jnp.where(has_dry, dryvolmr, 1.0), 1.0)
    dgncur_awet = dgncur_a * ratio ** (1.0 / 3.0)
    wetdens = jnp.where(has_dry,
                        (so4_q + water) / jnp.where(has_dry, dryvolmr + wtrvolmr, 1.0),
                        RHO_SO4)
    del v2ncur_a   # not needed: qaerwat == RHOH2O * naer * wtrvol_per_particle == water
    return {**state, 'dgncur_awet': dgncur_awet, 'qaerwat': water, 'wetdens': wetdens}


# =============================================================================
# one core step for one cell
# =============================================================================
_PARAMS = mam4_jax.AmicphysParams()


def core_step(state, phys=MAM4_PHYS):
    """calcsize -> water -> amicphys for ONE cell (mam4_jax.driver.run_step's
    sequence, re-composed here because run_step binds the stock water
    uptake at import and is jitted). Under phys='swap' the Dunne leaf is
    installed into the newnuc module for the duration of the TRACE: amicphys
    looks the name up when traced, so the swap is baked into the compiled
    graph and the module is restored before anything else can see it.
    """
    state = _calcsize(state)
    if phys == 'swap':
        state = wateruptake_tabazadeh(state)
        saved = _nn.mer07_veh02_nuc_mosaic_1box
        _nn.mer07_veh02_nuc_mosaic_1box = dunne_nuc_mosaic_1box
        try:
            state = _amic.amicphys(state, _PARAMS)
        finally:
            _nn.mer07_veh02_nuc_mosaic_1box = saved
    else:
        state = _wateruptake_stock(state)
        state = _amic.amicphys(state, _PARAMS)
    return state


def _make_run_cells(phys, n_inner, dt_inner):
    """Build the jitted (cells x inner steps) integrator for one physics choice.

    Arguments of the returned function (all leading axis = cell, float64):
      q0 (ncell, 35) core tracer array; so2_kg (ncell,) per-box SO2;
      T, P, RH, rho (ncell,); OH (n_inner, ncell) [molec/cm3].
    Returns (q, so2_kg) after n_inner steps of: so2_oxidation_step on the
    per-box gases -> q[H2SO4] <- the new H2SO4 -> core_step.
    """
    dt = float(dt_inner)
    boxvol = float(C.BOXVOL)

    def chem_cell(h2so4_kg, so2_kg, T, P, RH, oh):
        Gc = jnp.zeros((N_GAS_SPECIES,), dtype=jnp.float64)
        Gc = Gc.at[_G_H2SO4].set(h2so4_kg).at[_G_SO2].set(so2_kg)
        Gc = so2_oxidation_step(Gc, T, P, boxvol, dt, oh, RH)
        return Gc[_G_H2SO4], Gc[_G_SO2]

    def cell_state(q, T, P, RH):
        return {
            'q': q, 'qqcw': jnp.zeros_like(q),
            'dgncur_a': jnp.asarray(DGNUM0),         # calcsize recomputes it from q
            't': T, 'pmid': P, 'cldn': jnp.zeros_like(T),
            'zmid': jnp.full_like(T, ZMID_STRAT), 'pblh': jnp.zeros_like(T),
            'relhum': RH, 'deltat': jnp.asarray(dt),
        }

    def step_cell(q, T, P, RH):
        return core_step(cell_state(q, T, P, RH), phys)['q']

    v_chem = jax.vmap(chem_cell)
    v_step = jax.vmap(step_cell, in_axes=(0, 0, 0, 0))

    @jax.jit
    def run(q0, so2_kg0, T, P, RH, rho, OH):
        def body(carry, oh):
            q, so2_kg = carry
            h2so4_kg, so2_kg = v_chem(q[:, H2SO4_SLOT] * rho, so2_kg, T, P, RH, oh)
            q = q.at[:, H2SO4_SLOT].set(h2so4_kg / rho)
            q = v_step(q, T, P, RH)
            return (q, so2_kg), None
        (q, so2_kg), _ = jax.lax.scan(body, (q0, so2_kg0), OH, length=n_inner)
        return q, so2_kg
    return run


_RUN_CACHE = {}


def get_run_cells(phys=MAM4_PHYS, n_inner=N_INNER, dt_inner=MAM4_DT):
    key = (phys, int(n_inner), float(dt_inner))
    if key not in _RUN_CACHE:
        _RUN_CACHE[key] = _make_run_cells(*key)
    return _RUN_CACHE[key]


# =============================================================================
# modal state <-> core q array
# =============================================================================
def pack_q(num_m, so4_m, h2so4, T, P, RH):
    """(ncell,4) num [#/kg], (ncell,4) so4 [kg SO4/kg], (ncell,) h2so4 [kg/kg]
    -> (ncell, 35) core tracer array. so4 converted to MAM4's 115-mass; water
    vapour slot = RH * qsat_ice (coupling's own formula, so the stock water
    path sees a consistent humidity)."""
    ncell = num_m.shape[0]
    q = np.zeros((ncell, PCNST), dtype=np.float64)
    q[:, NUM_SLOTS] = np.maximum(num_m, 0.0)
    q[:, SO4_SLOTS[SO4_VALID]] = np.maximum(so4_m[:, SO4_VALID], 0.0) * SO4_TO_Q
    q[:, H2SO4_SLOT] = np.maximum(h2so4, 0.0)
    q[:, H2OMMR_SLOT] = np.asarray(RH) * C.qsat_ice(np.asarray(T, np.float64), np.asarray(P, np.float64))
    return q


def unpack_q(q):
    """(ncell,35) -> (num (ncell,4), so4 (ncell,4) [kg SO4(96)/kg], h2so4 (ncell,))"""
    q = np.asarray(q)
    num_m = q[:, NUM_SLOTS]
    so4_m = np.zeros((q.shape[0], NMODE), dtype=np.float64)
    so4_m[:, SO4_VALID] = q[:, SO4_SLOTS[SO4_VALID]] * Q_TO_SO4
    return num_m, so4_m, q[:, H2SO4_SLOT]


# =============================================================================
# the seam: drop-in for coupling.run_microphysics_full on the MODAL state
# =============================================================================
def run_microphysics_full_mam4(num, so4, so2, h2so4, temp3d, pres3d, rh3d, oh3d,
                               wgt3d):
    """Same signature/return contract as tomas_fast.run_microphysics_full_fast,
    with the modal state (4 modes) in place of the 40 bins:

      num, so4 : (4, nlev, nlat, nlon) -> num2, so4_2 of the same shape
      so2, h2so4 : (nlev, nlat, nlon) mixing ratios -> same
      clip_add, clip_rem : 0.0, 0.0. MAM4 has no two-moment clip: calcsize
        keeps mass and adjusts NUMBER when a mode's diameter leaves its
        [dgnumlo, dgnumhi] range. Under DEBUG/PROFILE the number that calcsize
        moved is printed instead (it is the modal analogue of the clip).
    """
    nmode, nlev, nlat, nlon = num.shape
    assert nmode == NMODE, f"expected {NMODE} modes, got {nmode}"
    ncell = nlev * nlat * nlon

    T_f = np.asarray(temp3d, np.float64).reshape(ncell)
    P_f = np.asarray(pres3d, np.float64).reshape(ncell)
    RH_f = np.asarray(rh3d, np.float64).reshape(ncell)
    rho_f = P_f / (C.RD * T_f)
    oh_arr = np.asarray(oh3d)
    OH_f = (C._oh_to_substeps(oh_arr, ncell, nsub_out=N_INNER).T
            if oh_arr.ndim == 4 else np.repeat(oh_arr.reshape(1, ncell), N_INNER, 0))

    num_c = np.asarray(num, np.float64).reshape(NMODE, ncell).T
    so4_c = np.asarray(so4, np.float64).reshape(NMODE, ncell).T
    q0 = pack_q(num_c, so4_c, np.asarray(h2so4, np.float64).reshape(ncell), T_f, P_f, RH_f)
    so2_kg = np.maximum(np.asarray(so2, np.float64).reshape(ncell), 0.0) * rho_f

    run = get_run_cells()
    t0 = time.time()
    q_out = np.empty_like(q0); so2_out = np.empty_like(so2_kg)
    for a in range(0, ncell, MAM4_CELL_CAP):
        b = min(a + MAM4_CELL_CAP, ncell)
        qo, so = run(jnp.asarray(q0[a:b]), jnp.asarray(so2_kg[a:b]),
                     jnp.asarray(T_f[a:b]), jnp.asarray(P_f[a:b]), jnp.asarray(RH_f[a:b]),
                     jnp.asarray(rho_f[a:b]), jnp.asarray(OH_f[:, a:b]))
        q_out[a:b] = np.asarray(qo); so2_out[a:b] = np.asarray(so)
    dt_run = time.time() - t0

    num2_c, so4_2c, h2so4_2 = unpack_q(q_out)
    if os.environ.get('DEBUG') or os.environ.get('PROFILE'):
        print(f"  [mam4] {ncell} cells x {N_INNER} steps (dt={MAM4_DT:.0f}s, phys={MAM4_PHYS}) "
              f"in {dt_run:.1f}s | dN(calcsize+nuc+coag)/N0 = "
              f"{(num2_c.sum() - num_c.sum()) / max(num_c.sum(), 1e-300):+.3e}", flush=True)

    num2 = num2_c.T.reshape(NMODE, nlev, nlat, nlon)
    so4_2 = so4_2c.T.reshape(NMODE, nlev, nlat, nlon)
    so2_2 = (so2_out / rho_f).reshape(nlev, nlat, nlon)
    h2so4_2 = h2so4_2.reshape(nlev, nlat, nlon)
    return num2, so4_2, so2_2, h2so4_2, 0.0, 0.0


# =============================================================================
# modes <-> TOMAS bins (for the bin-based consumers: Mie optics, settling)
# =============================================================================
# Pure numpy, copied from coupling.bin_mode and extended to bin the MASS with
# the mass-weighted lognormal (same sigma, median mass mg*exp(s^2)) instead of
# num_bin*MMID: both moments then round-trip EXACTLY per mode, which the
# settle -> refit cycle needs (num*MMID is only ~2 % mass-consistent, and a
# 2 %/step leak would be a sink). Sizes use the TOMAS convention (SO4(96) at
# 1770 kg/m3), identical to bin_mam4's binning of the archive, so the binned
# state is what the TOMAS run would see for the same archive values.
_LNXK = np.log(C.XK_NP)                      # (41,)


def _phi(z):
    # standard-normal CDF, as coupling._phi (the /sqrt(2) was missing in the
    # first version of this file: number was conserved but placed in bins drawn
    # with width s/sqrt(2), so num*MMID recovered only 47 % of the mode mass)
    return 0.5 * (1.0 + np.asarray(jax.scipy.special.erf(jnp.asarray(z) / np.sqrt(2.0))))


def mode_bin_fractions(num_m, so4_m, sigma_g):
    """Lognormal fractions of one mode's number and mass per TOMAS bin.
    num_m, so4_m: (G,). Returns (fn, fm), each (NBINS, G), zero where invalid."""
    G = num_m.shape[0]
    s = 3.0 * np.log(sigma_g)
    valid = (num_m > 0) & (so4_m > 0)
    amm = np.where(valid, so4_m / np.maximum(num_m, 1e-300), 1.0)
    lnmg = np.log(amm) - 0.5 * s * s             # number-median mass
    lnmg_mass = lnmg + s * s                     # mass-median mass
    fn = np.diff(_phi((_LNXK[:, None] - lnmg[None, :]) / s), axis=0)
    fm = np.diff(_phi((_LNXK[:, None] - lnmg_mass[None, :]) / s), axis=0)
    fn[:, ~valid] = 0.0; fm[:, ~valid] = 0.0
    return fn, fm


def modes_to_bins(num_m, so4_m):
    """(4, ...) modal -> (NBINS, ...) binned num/mas plus the per-(bin, mode)
    fractions needed to refit after settling. Returns
    num_bin, mas_bin, fn (NBINS, 4, G), fm (NBINS, 4, G) with G = prod(...)."""
    shape = num_m.shape[1:]
    G = int(np.prod(shape))
    nm = np.asarray(num_m, np.float64).reshape(NMODE, G)
    sm = np.asarray(so4_m, np.float64).reshape(NMODE, G)
    fn = np.zeros((C.NBINS, NMODE, G)); fm = np.zeros((C.NBINS, NMODE, G))
    for m in range(NMODE):
        fn[:, m], fm[:, m] = mode_bin_fractions(nm[m], sm[m], SIGMA_G[m])
    num_bin = (fn * nm[None]).sum(1).reshape(C.NBINS, *shape)
    mas_bin = (fm * sm[None]).sum(1).reshape(C.NBINS, *shape)
    return num_bin, mas_bin, fn, fm


def bins_to_modes(num_bin, mas_bin, fn, fm, num_m, so4_m):
    """Refit the modes from (settled) bins: each bin's number/mass is handed
    back to the modes in the proportion they contributed to that bin before
    settling (fn, fm from modes_to_bins), conserving total number and mass per
    cell exactly. On a static state this is the identity up to the lognormal
    tails that fall outside the 40-bin grid (measured in STATUS.md)."""
    shape = num_bin.shape[1:]
    G = int(np.prod(shape))
    nb = np.asarray(num_bin, np.float64).reshape(C.NBINS, G)
    mb = np.asarray(mas_bin, np.float64).reshape(C.NBINS, G)
    nm = np.asarray(num_m, np.float64).reshape(NMODE, G)
    sm = np.asarray(so4_m, np.float64).reshape(NMODE, G)
    wn = fn * nm[None]; wm = fm * sm[None]                  # (NBINS, 4, G)
    sn = wn.sum(1, keepdims=True); smm = wm.sum(1, keepdims=True)
    # share of bin k owned by mode m (bins nobody filled keep their content
    # with the mode of the nearest filled bin's dominant owner is overkill:
    # such bins are empty before settling, so give them to the mode whose
    # size range contains the bin -- see _BIN_HOME_MODE)
    share_n = np.where(sn > 0, wn / np.where(sn > 0, sn, 1.0), _BIN_HOME[:, :, None])
    share_m = np.where(smm > 0, wm / np.where(smm > 0, smm, 1.0), _BIN_HOME[:, :, None])
    num_new = (share_n * nb[:, None, :]).sum(0).reshape(NMODE, *shape)
    so4_new = (share_m * mb[:, None, :]).sum(0).reshape(NMODE, *shape)
    return num_new, so4_new


def _bin_home_mode():
    """Fallback owner of a bin that was empty before settling: the sulfate
    mode whose [dgnumlo, dgnumhi] range is nearest the bin's dry diameter (in
    log space). Primary carbon never owns a bin."""
    dp = C.DP_BIN * 1e-9
    home = np.zeros((C.NBINS, NMODE))
    lo = np.asarray(D.DGNUMLO_AMODE); hi = np.asarray(D.DGNUMHI_AMODE)
    for k in range(C.NBINS):
        best, bestd = 0, np.inf
        for m in range(NMODE - 1):            # modes 0..2 only
            d = 0.0 if lo[m] <= dp[k] <= hi[m] else min(abs(np.log(dp[k] / lo[m])), abs(np.log(dp[k] / hi[m])))
            if d < bestd:
                best, bestd = m, d
        home[k, best] = 1.0
    return home


_BIN_HOME = _bin_home_mode()


# =============================================================================
# per-mode size diagnostics (TOMAS size convention: SO4(96) at 1770 kg/m3)
# =============================================================================
def mode_dgnum(num_m, so4_m):
    """Dry number-median diameter per mode [m]; 0 where the mode is empty."""
    valid = (num_m > 0) & (so4_m > 0)
    amm = np.where(valid, so4_m / np.where(valid, num_m, 1.0), 0.0)
    shp = (NMODE,) + (1,) * (np.ndim(num_m) - 1)
    return np.where(valid, np.cbrt(6.0 * amm / (np.pi * RHO_SO4)
                                   / np.exp(4.5 * ALNSG.reshape(shp) ** 2)), 0.0)


def mode_reff(num_m, so4_m):
    """Effective radius over the modes [m]: 0.5 * sum N dg^3 e^{4.5 s^2} /
    sum N dg^2 e^{2 s^2} (the lognormal moments), summed over modes."""
    dg = mode_dgnum(num_m, so4_m)
    shp = (NMODE,) + (1,) * (np.ndim(num_m) - 1)
    s2 = ALNSG.reshape(shp) ** 2
    m3 = (num_m * dg ** 3 * np.exp(4.5 * s2)).sum(0)
    m2 = (num_m * dg ** 2 * np.exp(2.0 * s2)).sum(0)
    return 0.5 * m3 / np.maximum(m2, 1e-300)


def describe():
    return (f"mam4_jax {mam4_jax.__version__} ({N_INNER} x {MAM4_DT:.0f}s inner steps, "
            f"phys={MAM4_PHYS}: "
            + ("Dunne-2016 nucleation + Tabazadeh-1997 water" if MAM4_PHYS == 'swap'
               else "stock Vehkamaki + Koehler")
            + f", cond=substep, gas_netprod=0, cap {MAM4_CELL_CAP} cells/chunk)")


# =============================================================================
# the three hooks driver_mam4.py binds into coupling.py (see STATUS.md patch)
# =============================================================================
def modal_fill(ds_by_var, t, levs, lat_idx=None):
    """aer_fill replacement under AER_SRC=mam4: the archive's modes at h1 hour
    index t (relative to H0), NOT binned. Same signature/role as
    coupling.bin_mam4 (IC, open-BC slabs, polar refresh), returning
    num (4, nlev_s, nlat_s, nlon) [#/kg] and so4 (4, ...) [kg SO4/kg] with the
    archive so4_a* read as kg SO4/kg exactly as bin_mam4 does. Mode 3 = 0."""
    def read(var):
        da = ds_by_var[var][var].isel(time=C.H0 + t, lev=levs)
        if lat_idx is not None:
            da = da.isel(lat=lat_idx)
        return da.values.astype(np.float64)
    n1 = read('num_a1')
    z = np.zeros_like(n1)
    num = np.stack([n1, read('num_a2'), read('num_a3'), z])
    so4 = np.stack([read('so4_a1'), read('so4_a2'), read('so4_a3'), z])
    return num, so4


# ---- jax.numpy versions of the modes<->bins conversions (2026-11-05 pm) ----
# The numpy modes_to_bins/bins_to_modes above cost 11.5 s + ~5 s per full-grid
# call (the smoke: settle 18.5 s and +11 s inside radiation, vs 0.04 s and
# 13.9 s for TOMAS). These are the same formulas on device, jitted, with the
# per-mode loop kept explicit so the (41, G) intermediates are 0.44 GB each
# instead of 1.7 GB for a (41, 4, G) block. The numpy versions stay as the
# reference for runs/mam4_dev/jax_port_test.py.
_LNXK_J = jnp.asarray(_LNXK)
_S3_J = jnp.asarray(3.0 * ALNSG)                 # (4,) ln-mass width per mode
_BIN_HOME_J = jnp.asarray(_BIN_HOME)             # (NBINS, 4)


def _phi_j(z):
    return 0.5 * (1.0 + jax.scipy.special.erf(z / jnp.sqrt(2.0)))


def _mode_fractions_j(nm, sm, m):
    """fn, fm (NBINS, G) of mode m from nm, sm (G,); zero where invalid."""
    s = _S3_J[m]
    valid = (nm > 0) & (sm > 0)
    amm = jnp.where(valid, sm / jnp.where(valid, nm, 1.0), 1.0)
    lnmg = jnp.log(amm) - 0.5 * s * s
    fn = jnp.diff(_phi_j((_LNXK_J[:, None] - lnmg[None, :]) / s), axis=0)
    fm = jnp.diff(_phi_j((_LNXK_J[:, None] - (lnmg + s * s)[None, :]) / s), axis=0)
    return jnp.where(valid[None], fn, 0.0), jnp.where(valid[None], fm, 0.0)


@jax.jit
def modes_to_bins_j(num_m, so4_m):
    """jnp version of modes_to_bins: (4, ...) -> num_bin, mas_bin (NBINS, ...)."""
    shape = num_m.shape[1:]
    nm = num_m.reshape(NMODE, -1); sm = so4_m.reshape(NMODE, -1)
    nb = jnp.zeros((C.NBINS, nm.shape[1])); mb = jnp.zeros_like(nb)
    for m in range(NMODE):
        fn, fm = _mode_fractions_j(nm[m], sm[m], m)
        nb = nb + fn * nm[m][None]; mb = mb + fm * sm[m][None]
    return nb.reshape(C.NBINS, *shape), mb.reshape(C.NBINS, *shape)


@jax.jit
def bins_to_modes_j(num_bin, mas_bin, num_m, so4_m):
    """jnp version of bins_to_modes; recomputes the ownership fractions from the
    pre-settling modes instead of taking them as arguments."""
    shape = num_bin.shape[1:]
    nb = num_bin.reshape(C.NBINS, -1); mb = mas_bin.reshape(C.NBINS, -1)
    nm = num_m.reshape(NMODE, -1); sm = so4_m.reshape(NMODE, -1)
    wn = []; wm = []
    for m in range(NMODE):
        fn, fm = _mode_fractions_j(nm[m], sm[m], m)
        wn.append(fn * nm[m][None]); wm.append(fm * sm[m][None])
    wn = jnp.stack(wn, 1); wm = jnp.stack(wm, 1)                  # (NBINS, 4, G)
    sn = wn.sum(1, keepdims=True); smm = wm.sum(1, keepdims=True)
    share_n = jnp.where(sn > 0, wn / jnp.where(sn > 0, sn, 1.0), _BIN_HOME_J[:, :, None])
    share_m = jnp.where(smm > 0, wm / jnp.where(smm > 0, smm, 1.0), _BIN_HOME_J[:, :, None])
    num_new = (share_n * nb[:, None, :]).sum(0)
    so4_new = (share_m * mb[:, None, :]).sum(0)
    return num_new.reshape(NMODE, *shape), so4_new.reshape(NMODE, *shape)


# aer_to_bins is called up to three times per step on the SAME (num, mas)
# (radiation, size diagnostics, frames); keep the last result keyed by the
# input arrays' identity. The cache holds references to the inputs, so their
# ids cannot be recycled while the entry is alive.
_BINS_CACHE = {'key': None, 'inputs': None, 'out': None}


def aer_to_bins(num_m, so4_m):
    """coupling.aer_to_bins replacement: the modal state on the 40 TOMAS bins
    (number and mass, both moments conserved) for the Mie optics and the size
    diagnostics. jitted; memoised on the input arrays."""
    key = (id(num_m), id(so4_m))
    if _BINS_CACHE['key'] == key:
        return _BINS_CACHE['out']
    out = modes_to_bins_j(jnp.asarray(num_m), jnp.asarray(so4_m))
    _BINS_CACHE.update(key=key, inputs=(num_m, so4_m), out=out)
    return out


@jax.jit
def _settle_modal_j(num_m, so4_m, T3d, pres3d, dp, dt, dpw, rhow):
    nb, mb = modes_to_bins_j(num_m, so4_m)
    nb2, mb2, out_n, out_m = settling.settle_step(nb, mb, T3d, pres3d, dp, dt, dpw, rhow)
    n2, s2 = bins_to_modes_j(nb2, mb2, num_m, so4_m)
    return n2, s2, out_n, out_m


def settle_modal(num_m, so4_m, T3d, pres3d, dp, dt, dpw, rhow):
    """coupling.aer_settle replacement: bin the modes, run settling.settle_step
    on the bins exactly as coupling does for TOMAS (per-bin wet Stokes), refit
    the modes from the settled bins by the pre-settling per-(bin, cell)
    ownership fractions (bins_to_modes). Returns (num_m2, so4_m2, out_n, out_m)
    with out_* per (bin, lat, lon) in settle_step's q*dp units, so coupling's
    settle budget line is unchanged. One jit on device (was numpy, 18 s)."""
    return _settle_modal_j(jnp.asarray(num_m), jnp.asarray(so4_m), jnp.asarray(T3d),
                           jnp.asarray(pres3d), jnp.asarray(dp), float(dt),
                           jnp.asarray(dpw), jnp.asarray(rhow))


def settle_modal_np(num_m, so4_m, T3d, pres3d, dp, dt, dpw, rhow):
    """The numpy-path version (reference for jax_port_test.py)."""
    nm = np.asarray(num_m); sm = np.asarray(so4_m)
    nb, mb, fn, fm = modes_to_bins(nm, sm)
    nb2, mb2, out_n, out_m = settling.settle_step(jnp.asarray(nb), jnp.asarray(mb), T3d, pres3d,
                                                  dp, dt, dpw, rhow)
    n2, s2 = bins_to_modes(np.asarray(nb2), np.asarray(mb2), fn, fm, nm, sm)
    return jnp.asarray(n2), jnp.asarray(s2), out_n, out_m
