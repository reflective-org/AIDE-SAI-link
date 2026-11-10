"""Semi-Lagrangian transport kernels on a lon-lat grid (vendored from dinosaur).

ORIGIN
------
This is `dinosaur.semi_lagrangian` (Google NeuralGCM dynamical core, Apache
2.0, Copyright 2026 Google LLC; checkout at /data/dinosaur-sl) reduced to the
pieces the coupling needs and freed of dinosaur's own dependencies
(`spherical_harmonic.Grid`, `coordinate_systems`, `sigma_coordinates`,
`tree_math`). The algorithms are unchanged:

  * fixed-iteration departure points on the trajectory midpoint, solved in
    geocentric Cartesian coordinates on the unit sphere (no polar singularity;
    the IFS 48r1 formulation, Diamantakis & Vana 2022);
  * tensor-product Lagrange gather, linear (2^d) or cubic (4^d), uniform in
    longitude with periodic wrap, on the actual node latitudes with a
    cross-pole halo, linear or cubic in the vertical on fixed nodes with
    constant extrapolation beyond the end nodes;
  * the quasi-monotone limiter of Bermejo & Staniforth (1992): the
    interpolated value is clipped to the range of the 2x2(x2) bracketing cell
    corners, so no new extrema are created and positivity is preserved.

WHAT IS DIFFERENT
-----------------
1. `SLGrid` replaces `spherical_harmonic.Grid`: a frozen, hashable record of
   the longitude count, the node latitudes and the radius, which is all the
   kernels ever read. It is a `jax.jit` static argument.
2. NODES AT THE POLES ARE SUPPORTED. The CESM finite-volume grid has rows at
   exactly +-90 deg, which dinosaur refuses: its halo mirrors rows 0..H-1
   about the pole, and for a pole row that duplicates the node and gives 0/0
   Lagrange weights. Here, when an end row sits on the pole, the halo mirrors
   rows 1..H (south) / -2..-H-1 (north) instead -- `_halo_source_slices` --
   so the extended latitude axis stays strictly increasing and the pole row
   itself is an ordinary node (the antipodal roll is a no-op there because
   every longitude on a pole row holds the same value). Grids without pole
   rows get dinosaur's original halo, unchanged.
3. The 3-D departure solve takes the vertical velocity AT THE LAYER CENTERS
   (the same nodes as the winds), not at the layer boundaries: the coupling
   has CESM omega at the centers and no interface pressures. The four
   components (three Cartesian wind components + omega) are then one batched
   trilinear gather at the trajectory midpoint. Departure pressures are
   clipped to the center range, i.e. constant extrapolation beyond the end
   levels -- the "closed extended column" that the mass fixer in sl_lr.py
   relies on.
4. The stencil (indices + weights) is a first-class object: computed once per
   sub-step from the departure points and applied to every tracer, and the
   mass fixer's extra linear gather reuses the departure points through a
   second, 2-point stencil.

Conventions (dinosaur's):
  * nodal fields are [..., lon, lat] (level leading for 3-D); the coupling's
    (tracer, lev, lat, lon) layout is transposed in sl_lr.py;
  * winds are true (east, north) velocities in m/s; omega is in Pa/s with
    omega > 0 = descent = increasing pressure = increasing level index;
  * Cartesian vectors carry (x, y, z) along a leading axis of size 3, z at
    the north pole.

References:
  Bermejo & Staniforth 1992, Mon. Wea. Rev. 120 (quasi-monotone limiter).
  Diamantakis 2014, ECMWF Seminar on Numerical Methods (SL review).
  Diamantakis & Vana 2022, QJRMS 148, 670-684 (Cartesian departure points).
  Staniforth & Cote 1991, Mon. Wea. Rev. 119 (SL review).
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# All `einsum`s should be done at highest available precision.
einsum = functools.partial(jnp.einsum, precision=jax.lax.Precision.HIGHEST)


# =============================================================================
# The grid.
# =============================================================================


@dataclasses.dataclass(frozen=True)
class SLGrid:
    """A nodal lon-lat grid: uniform longitudes from 0, arbitrary increasing
    latitudes (which may include the poles).

    Frozen and hashable so it can be a `jax.jit` static argument; the
    latitudes are stored as a tuple for that reason.

    Attributes:
      longitude_nodes: number of longitudes (must be even for the cross-pole
        halo).
      latitudes_rad: node latitudes in radians, strictly increasing.
      radius: sphere radius [m].
    """

    longitude_nodes: int
    latitudes_rad: tuple
    radius: float = 6.371e6

    @classmethod
    def from_latitudes(cls, lat_deg, longitude_nodes: int, radius: float = 6.371e6):
        lat = np.asarray(lat_deg, dtype=np.float64)
        if not np.all(np.diff(lat) > 0):
            raise ValueError('expected latitudes in increasing order')
        return cls(int(longitude_nodes), tuple(np.deg2rad(lat).tolist()), float(radius))

    @property
    def latitude_nodes(self) -> int:
        return len(self.latitudes_rad)

    @property
    def latitudes(self) -> np.ndarray:
        return np.asarray(self.latitudes_rad, dtype=np.float64)

    @property
    def longitudes(self) -> np.ndarray:
        return np.arange(self.longitude_nodes) * (2 * np.pi / self.longitude_nodes)

    @property
    def nodal_shape(self) -> tuple[int, int]:
        return (self.longitude_nodes, self.latitude_nodes)

    @property
    def nodal_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(lon[:, None], sin_lat[None, :]), broadcastable to `nodal_shape`."""
        return self.longitudes[:, None], np.sin(self.latitudes)[None, :]

    @property
    def south_pole_node(self) -> bool:
        return bool(np.isclose(self.latitudes_rad[0], -np.pi / 2, atol=1e-12))

    @property
    def north_pole_node(self) -> bool:
        return bool(np.isclose(self.latitudes_rad[-1], np.pi / 2, atol=1e-12))


# =============================================================================
# Spherical geometry helpers.
# =============================================================================


def lon_lat_to_cartesian(lon, sin_lat):
    """Unit vectors [3, *shape] pointing at (lon, lat)."""
    cos_lat = jnp.sqrt(1 - jnp.square(sin_lat))
    return jnp.stack([cos_lat * jnp.cos(lon), cos_lat * jnp.sin(lon), sin_lat], axis=0)


def cartesian_to_lon_sin_lat(r):
    """(longitude in [0, 2pi), sine of latitude) from unit vectors [3, ...]."""
    x, y, z = r
    lon = jnp.arctan2(y, x) % (2 * np.pi)
    sin_lat = jnp.clip(z, -1, 1)
    return lon, sin_lat


def cartesian_wind(u, v, lon, sin_lat):
    """(u, v) -> Cartesian components [3, ...] of the tangent vector field.

    The Cartesian components are three smooth scalar fields on the sphere
    (unlike (u, v), which are only piecewise smooth at the poles), so they can
    be interpolated componentwise.
    """
    cos_lat = jnp.sqrt(1 - jnp.square(sin_lat))
    sin_lon = jnp.sin(lon)
    cos_lon = jnp.cos(lon)
    return jnp.stack([
        -u * sin_lon - v * sin_lat * cos_lon,
        u * cos_lon - v * sin_lat * sin_lon,
        v * cos_lat,
    ])


def _normalize(r):
    return r / jnp.sqrt(jnp.square(r).sum(axis=0, keepdims=True))


# =============================================================================
# Gather-based interpolation.
# =============================================================================


def _lagrange_weights(x, nodes):
    """Lagrange weights [..., stencil] for points x [...] on `nodes`
    [..., stencil]; exact for polynomials of degree stencil-1."""
    stencil_size = nodes.shape[-1]
    diff = x[..., jnp.newaxis] - nodes
    denom = nodes[..., :, jnp.newaxis] - nodes[..., jnp.newaxis, :]
    eye = np.eye(stencil_size, dtype=bool)
    num = jnp.where(eye, 1.0, diff[..., jnp.newaxis, :])
    den = jnp.where(eye, 1.0, denom)
    return jnp.prod(num / den, axis=-1)


def _contract_stencil(values, *weights):
    """Contracts gathered stencil values [..., s1, s2(, s3)] with per-axis
    weights. einsum on CPU (fastest there); a fused broadcast-multiply-reduce
    on accelerators, where it fuses with the producing gather and the full
    stencil tensor is never materialised (dinosaur's measurement)."""
    if jax.default_backend() == 'cpu':
        n = len(weights)
        letters = 'ijkl'[:n]
        operands = ','.join(f'...{c}' for c in letters)
        return einsum(f'{operands},...{letters}->...', *weights, values)
    n = len(weights)
    result = values
    for axis, w in enumerate(weights):
        shape = w.shape[:-1] + (1,) * axis + (w.shape[-1],) + (1,) * (n - 1 - axis)
        result = result * w.reshape(shape)
    return result.sum(axis=tuple(range(-n, 0)))


_HALO_WIDTH = 2  # supports stencils up to 4 points wide in latitude.


def _halo_source_slices(grid: SLGrid) -> tuple[slice, slice]:
    """Which rows are mirrored into the south / north halo (in halo order,
    i.e. farthest-from-the-domain first).

    dinosaur mirrors rows 0..H-1 (south) and -1..-H (north). When the end row
    IS the pole (the CESM 0.9x1.25 grid) that would duplicate the pole node;
    the mirror image of row 0 about the pole is row 0 itself. So for a pole
    row the mirrored rows are 1..H / -2..-H-1 instead, and the pole row stays
    a normal node. Both cases keep the extended axis strictly increasing.
    """
    H = _HALO_WIDTH
    south = slice(H, 0, -1) if grid.south_pole_node else slice(H - 1, None, -1)
    north = slice(-2, -H - 2, -1) if grid.north_pole_node else slice(None, -H - 1, -1)
    return south, north


@functools.lru_cache(maxsize=None)
def _latitude_halo_axis(grid: SLGrid) -> np.ndarray:
    """Node latitudes extended with `_HALO_WIDTH` mirrored rows per pole.

    A point beyond the pole at "latitude" phi > pi/2 is the physical location
    (pi - phi, lambda + pi), so halo rows take coordinates mirrored about the
    pole. The extended axis is strictly increasing, which makes latitude (not
    its sine, which folds at the poles) the interpolation coordinate.
    """
    phi = grid.latitudes
    s, n = _halo_source_slices(grid)
    south = -np.pi - phi[s]
    north = np.pi - phi[n]
    extended = np.concatenate([south, phi, north])
    if not np.all(np.diff(extended) > 0):
        raise ValueError('halo-extended latitudes are not strictly increasing')
    return extended


def _extend_with_pole_halo(field, grid: SLGrid):
    """Appends `_HALO_WIDTH` cross-pole halo rows on both latitude ends of a
    [..., lon, lat] SCALAR field (halo values come from the antipodal
    longitude: a roll by half the longitude count)."""
    num_lon = grid.longitude_nodes
    if num_lon % 2:
        raise ValueError('cross-pole halos require an even number of longitude nodes')
    rolled = jnp.roll(field, num_lon // 2, axis=-2)
    s, n = _halo_source_slices(grid)
    return jnp.concatenate([rolled[..., s], field, rolled[..., n]], axis=-1)


def _stencil_offsets(order: str) -> np.ndarray:
    if order == 'linear':
        return np.arange(2)
    if order == 'cubic':
        return np.arange(-1, 3)
    raise ValueError(f'unknown interpolation {order=}')


def _linear_cell_slice(order: str) -> slice:
    """The 2-point bracketing cell within a stencil of the given order."""
    start = -int(_stencil_offsets(order)[0])
    return slice(start, start + 2)


class HorizontalStencil(NamedTuple):
    """Gather indices/weights for horizontal interpolation at a set of points."""
    lon_index: jnp.ndarray     # [..., stencil]  (wrapped)
    lat_index: jnp.ndarray     # [..., stencil]  (into the halo-extended axis)
    lon_weights: jnp.ndarray   # [..., stencil]
    lat_weights: jnp.ndarray   # [..., stencil]
    extended_latitude_size: int
    order: str


def horizontal_stencil(grid: SLGrid, lon, sin_lat, order: str) -> HorizontalStencil:
    offsets = _stencil_offsets(order)
    lon = jnp.asarray(lon)
    # arctan2 rather than arcsin: finite gradient at |sin_lat| = 1.
    sin_lat = jnp.clip(jnp.asarray(sin_lat), -1, 1)
    cos_lat = jnp.sqrt(jnp.maximum(1 - jnp.square(sin_lat), 1e-12))
    phi = jnp.arctan2(sin_lat, cos_lat)

    lon_spacing = 2 * np.pi / grid.longitude_nodes
    lon_cell = jnp.floor(lon / lon_spacing).astype(jnp.int32)
    lon_stencil = lon_cell[..., jnp.newaxis] + offsets
    lon_nodes = lon_stencil * lon_spacing
    lon_weights = _lagrange_weights(lon, lon_nodes.astype(lon.dtype))
    lon_index = lon_stencil % grid.longitude_nodes

    phi_extended = _latitude_halo_axis(grid)
    lat_cell = (jnp.searchsorted(jnp.asarray(phi_extended, dtype=phi.dtype), phi,
                                 side='right') - 1).astype(jnp.int32)
    lat_cell = jnp.clip(lat_cell, -offsets[0], phi_extended.size - 1 - offsets[-1])
    lat_stencil = lat_cell[..., jnp.newaxis] + offsets
    lat_nodes = jnp.asarray(phi_extended, dtype=phi.dtype)[lat_stencil]
    lat_weights = _lagrange_weights(phi, lat_nodes)
    return HorizontalStencil(lon_index, lat_stencil, lon_weights, lat_weights,
                             phi_extended.size, order)


def _vertical_stencil(p, p_nodes, vertical_order: str):
    """(level_index, level_weights, bracket_offset) on fixed increasing nodes.

    p is clipped to the node range first (constant extrapolation). 'linear'
    uses the two bracketing nodes; 'cubic' a 4-point Lagrange stencil degraded
    to linear in the first and last cells (the IFS rule).
    """
    p_nodes = jnp.asarray(p_nodes)
    num_levels = p_nodes.shape[0]
    p = jnp.clip(jnp.asarray(p), p_nodes[0], p_nodes[-1])
    level_cell = (jnp.searchsorted(p_nodes, p, side='right') - 1).astype(jnp.int32)
    level_cell = jnp.clip(level_cell, 0, num_levels - 2)
    below = p_nodes[level_cell]
    above = p_nodes[level_cell + 1]
    fraction = ((p - below) / (above - below)).astype(p.dtype)
    if vertical_order == 'linear':
        level_weights = jnp.stack([1 - fraction, fraction], axis=-1)
        level_index = level_cell[..., jnp.newaxis] + np.arange(2)
        return level_index, level_weights, jnp.zeros_like(level_cell)
    if vertical_order != 'cubic':
        raise ValueError(f'unknown {vertical_order=}')
    start = jnp.clip(level_cell - 1, 0, num_levels - 4)
    level_index = start[..., jnp.newaxis] + np.arange(4)
    cubic_weights = _lagrange_weights(p, p_nodes[level_index].astype(p.dtype))
    bracket_offset = level_cell - start
    linear_weights = ((1 - fraction)[..., jnp.newaxis]
                      * jax.nn.one_hot(bracket_offset, 4, dtype=cubic_weights.dtype)
                      + fraction[..., jnp.newaxis]
                      * jax.nn.one_hot(bracket_offset + 1, 4, dtype=cubic_weights.dtype))
    boundary_cell = (level_cell == 0) | (level_cell == num_levels - 2)
    level_weights = jnp.where(boundary_cell[..., jnp.newaxis], linear_weights, cubic_weights)
    return level_index, level_weights, bracket_offset


class Stencil3D(NamedTuple):
    """Gather indices/weights for 3-D interpolation of a [lev, lon, lat] field
    at a set of points, computed once and applied to any number of fields."""
    flat_index: jnp.ndarray      # [..., sv, sh, sh] into the halo-extended field
    level_weights: jnp.ndarray   # [..., sv]
    lon_weights: jnp.ndarray     # [..., sh]
    lat_weights: jnp.ndarray     # [..., sh]
    bracket_offset: jnp.ndarray  # [...]  position of the lower bracketing level
    order: str
    extended_latitude_size: int


def stencil_3d(grid: SLGrid, lon, sin_lat, p, p_nodes, order: str = 'cubic',
               vertical_order: str = 'linear') -> Stencil3D:
    """Builds the gather stencil for points (lon, sin_lat, p) in a field whose
    levels sit at the increasing coordinates `p_nodes`."""
    h = horizontal_stencil(grid, lon, sin_lat, order)
    level_index, level_weights, bracket_offset = _vertical_stencil(p, p_nodes, vertical_order)
    ext = h.extended_latitude_size
    flat_index = ((level_index[..., :, jnp.newaxis, jnp.newaxis] * grid.longitude_nodes
                   + h.lon_index[..., jnp.newaxis, :, jnp.newaxis]) * ext
                  + h.lat_index[..., jnp.newaxis, jnp.newaxis, :])
    return Stencil3D(flat_index, level_weights, h.lon_weights, h.lat_weights,
                     bracket_offset, order, ext)


def gather_3d(field, grid: SLGrid, st: Stencil3D, limiter: str | None = 'quasi_monotone'):
    """Interpolates one [lev, lon, lat] field with a precomputed stencil.

    One 1-D gather on the flattened halo-extended field (the ravel is a
    bitcast; `mode='clip'` gives defined out-of-range semantics), then the
    tensor-product contraction. With the quasi-monotone limiter the result is
    clipped to the range of the 2x2x2 corners bracketing the point.
    """
    extended = _extend_with_pole_halo(jnp.asarray(field), grid)
    values = jnp.take(extended.ravel(), st.flat_index, mode='clip')
    result = _contract_stencil(values, st.level_weights, st.lon_weights, st.lat_weights)
    if limiter is None:
        return result
    if limiter != 'quasi_monotone':
        raise ValueError(f'unknown {limiter=}')
    cell = _linear_cell_slice(st.order)
    corners = values[..., :, cell, cell]
    # restrict to the two levels bracketing the departure p (identity for a
    # linear vertical stencil)
    index = (st.bracket_offset[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]
             + np.arange(2)[:, np.newaxis, np.newaxis])
    corners = jnp.take_along_axis(corners, index, axis=-3)
    return jnp.clip(result, corners.min(axis=(-3, -2, -1)), corners.max(axis=(-3, -2, -1)))


def gather_2d(field, grid: SLGrid, st: HorizontalStencil, limiter: str | None = None):
    """Interpolates one [lon, lat] field with a precomputed horizontal stencil."""
    extended = _extend_with_pole_halo(jnp.asarray(field), grid)
    flat_index = (st.lon_index[..., :, jnp.newaxis] * st.extended_latitude_size
                  + st.lat_index[..., jnp.newaxis, :])
    values = jnp.take(extended.ravel(), flat_index, mode='clip')
    result = _contract_stencil(values, st.lon_weights, st.lat_weights)
    if limiter is None:
        return result
    if limiter != 'quasi_monotone':
        raise ValueError(f'unknown {limiter=}')
    cell = _linear_cell_slice(st.order)
    corners = values[..., cell, cell]
    return jnp.clip(result, corners.min(axis=(-2, -1)), corners.max(axis=(-2, -1)))


class Stencil2DLevels(NamedTuple):
    """Horizontal gather stencil for per-level points, with flat indices into
    the halo-extended [lev, lon, lat] field so ONE gather serves every level
    (each level has its own departure points). Computed once per sub-step and
    applied to every tracer."""
    flat_index: jnp.ndarray     # [lev, lon, lat, sh, sh]
    lon_weights: jnp.ndarray    # [lev, lon, lat, sh]
    lat_weights: jnp.ndarray    # [lev, lon, lat, sh]
    order: str


def stencil_2d_levels(grid: SLGrid, lon, sin_lat, order: str = 'cubic') -> Stencil2DLevels:
    """lon, sin_lat: [lev, lon, lat] departure points of the nodes of each level."""
    h = horizontal_stencil(grid, lon, sin_lat, order)
    nlev = lon.shape[0]
    lev = jnp.arange(nlev, dtype=jnp.int32)[:, None, None, None, None]
    flat_index = ((lev * grid.longitude_nodes + h.lon_index[..., :, jnp.newaxis])
                  * h.extended_latitude_size + h.lat_index[..., jnp.newaxis, :])
    return Stencil2DLevels(flat_index, h.lon_weights, h.lat_weights, order)


def gather_2d_levels(field, grid: SLGrid, st: Stencil2DLevels, limiter: str | None = 'quasi_monotone'):
    """Interpolates one [lev, lon, lat] field level by level with a
    precomputed per-level stencil (see `gather_3d` for the limiter)."""
    extended = _extend_with_pole_halo(jnp.asarray(field), grid)
    values = jnp.take(extended.ravel(), st.flat_index, mode='clip')
    result = _contract_stencil(values, st.lon_weights, st.lat_weights)
    if limiter is None:
        return result
    if limiter != 'quasi_monotone':
        raise ValueError(f'unknown {limiter=}')
    cell = _linear_cell_slice(st.order)
    corners = values[..., cell, cell]
    return jnp.clip(result, corners.min(axis=(-2, -1)), corners.max(axis=(-2, -1)))


def interpolate_2d(field, grid: SLGrid, lon, sin_lat, order='cubic', limiter=None):
    """[lon, lat] or [lev, lon, lat] field at points shaped like `lon` (per
    level when both carry a leading level axis)."""
    field = jnp.asarray(field)
    if field.ndim == 2:
        return gather_2d(field, grid, horizontal_stencil(grid, lon, sin_lat, order), limiter)
    return jax.vmap(lambda f, lo, sl: gather_2d(f, grid, horizontal_stencil(grid, lo, sl, order),
                                                limiter))(field, lon, sin_lat)


def interpolate_3d(field, grid: SLGrid, lon, sin_lat, p, p_nodes, order='cubic',
                   limiter=None, vertical_order='linear'):
    """[lev, lon, lat] field at points (lon, sin_lat, p); see `gather_3d`."""
    return gather_3d(field, grid, stencil_3d(grid, lon, sin_lat, p, p_nodes, order, vertical_order),
                     limiter)


# =============================================================================
# Departure points.
# =============================================================================


class DeparturePoints(NamedTuple):
    """Departure points of trajectories arriving at grid points.

    cartesian: unit vectors [3, *points]; p: vertical coordinate of the
    departure points (None for horizontal-only trajectories).
    """
    cartesian: jnp.ndarray
    p: jnp.ndarray | None = None

    @property
    def lon(self):
        return cartesian_to_lon_sin_lat(self.cartesian)[0]

    @property
    def sin_lat(self):
        return cartesian_to_lon_sin_lat(self.cartesian)[1]


def horizontal_departure_points(u, v, grid: SLGrid, dt: float, *, iterations: int = 2,
                                initial_guess: DeparturePoints | None = None) -> DeparturePoints:
    """Departure points of horizontal trajectories arriving at the nodes.

    Fixed-point iteration on the trajectory midpoint (Robert-style; Diamantakis
    2014, Eq. 5) in Cartesian coordinates on the unit sphere, with the wind at
    the midpoint interpolated bilinearly. Converges when dt * max|grad V| < 1;
    two iterations give second-order departure points. `u`, `v` are
    [lon, lat] or [lev, lon, lat] (each level advected by its own wind).
    """
    u = jnp.asarray(u)
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    lon_mesh = jnp.asarray(lon_mesh, dtype=u.dtype)
    sin_lat_mesh = jnp.asarray(sin_lat_mesh, dtype=u.dtype)
    arrival = lon_lat_to_cartesian(jnp.broadcast_to(lon_mesh, u.shape),
                                   jnp.broadcast_to(sin_lat_mesh, u.shape))
    wind = cartesian_wind(u, v, lon_mesh, sin_lat_mesh)
    angular_dt = dt / grid.radius
    departure = arrival if initial_guess is None else initial_guess.cartesian
    for _ in range(iterations):
        midpoint = _normalize((arrival + departure) / 2)
        lon_mid, sin_lat_mid = cartesian_to_lon_sin_lat(midpoint)
        wind_mid = jax.vmap(lambda c: interpolate_2d(c, grid, lon_mid, sin_lat_mid, 'linear'))(wind)
        departure = _normalize(arrival - angular_dt * wind_mid)
    return DeparturePoints(cartesian=departure, p=None)


def departure_points_3d(u, v, omega, grid: SLGrid, dt: float, p_centers, *,
                        iterations: int = 2,
                        initial_guess: DeparturePoints | None = None) -> DeparturePoints:
    """3-D departure points of trajectories arriving at the layer centers.

    Horizontal update as `horizontal_departure_points`; the vertical position
    is updated jointly in the same fixed-point iteration, with omega (and the
    Cartesian wind) interpolated trilinearly at the trajectory midpoint. All
    four velocity components live at the layer centers `p_centers`
    (increasing). Departure pressures are clipped to [p_centers[0],
    p_centers[-1]]: constant extrapolation beyond the end levels, i.e. the
    column is CLOSED (nothing enters from outside the node range).

    Args:
      u, v: [lev, lon, lat] true winds [m/s] at the centers.
      omega: [lev, lon, lat] dp/dt [Pa/s] at the centers (> 0 descending).
      p_centers: [lev] increasing vertical coordinate of the levels [Pa].
      dt: time step [s]; trajectories are integrated backwards over dt.
    """
    u = jnp.asarray(u)
    p_centers = jnp.asarray(p_centers, dtype=u.dtype)
    lon_mesh, sin_lat_mesh = grid.nodal_mesh
    lon_mesh = jnp.asarray(lon_mesh, dtype=u.dtype)
    sin_lat_mesh = jnp.asarray(sin_lat_mesh, dtype=u.dtype)
    arrival = lon_lat_to_cartesian(jnp.broadcast_to(lon_mesh, u.shape),
                                   jnp.broadcast_to(sin_lat_mesh, u.shape))
    p_arrival = jnp.broadcast_to(p_centers[:, jnp.newaxis, jnp.newaxis], u.shape)
    vel = jnp.concatenate([cartesian_wind(u, v, lon_mesh, sin_lat_mesh),
                           jnp.asarray(omega, dtype=u.dtype)[jnp.newaxis]], axis=0)
    angular_dt = dt / grid.radius
    p_lo, p_hi = p_centers[0], p_centers[-1]

    if initial_guess is None:
        departure, p_departure = arrival, p_arrival
    else:
        departure, p_departure = initial_guess.cartesian, initial_guess.p
    for _ in range(iterations):
        midpoint = _normalize((arrival + departure) / 2)
        p_mid = (p_arrival + p_departure) / 2
        lon_mid, sin_lat_mid = cartesian_to_lon_sin_lat(midpoint)
        st = stencil_3d(grid, lon_mid, sin_lat_mid, p_mid, p_centers, 'linear', 'linear')
        vel_mid = jax.vmap(lambda c: gather_3d(c, grid, st, None))(vel)
        departure = _normalize(arrival - angular_dt * vel_mid[:3])
        p_departure = jnp.clip(p_arrival - dt * vel_mid[3], p_lo, p_hi)
    return DeparturePoints(cartesian=departure, p=p_departure)


def transport_scalar_2d(field, departure: DeparturePoints, grid: SLGrid,
                        order='cubic', limiter='quasi_monotone'):
    """Remaps a [lon, lat] or [lev, lon, lat] field along 2-D trajectories."""
    return interpolate_2d(field, grid, departure.lon, departure.sin_lat, order, limiter)


def transport_scalar(field, departure: DeparturePoints, grid: SLGrid, p_nodes,
                     order='cubic', limiter='quasi_monotone', vertical_order='linear'):
    """Remaps a [lev, lon, lat] field along 3-D trajectories."""
    if departure.p is None:
        raise ValueError('transport_scalar requires 3-D departure points')
    return interpolate_3d(field, grid, departure.lon, departure.sin_lat, departure.p,
                          p_nodes, order, limiter, vertical_order)
