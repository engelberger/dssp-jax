# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2020 NKI/AVL, Netherlands Cancer Institute
# Optimised JAX implementation of Solvent‑Accessible Surface Area (SASA)
# compatible with **JAX\xa00.6.0** (no deprecated ``jax.core`` symbols).
"""accessibility_optimized.py

Back‑ends implemented
=====================
"vmap2"  – fully‑vectorised *vmap‑of‑vmap* (≤\xa06\xa0k atoms).
"nlist" – lightweight neighbour list (no jax‑md/Haiku dependency, works to ≈150\xa0k atoms).
"loop"  – fall‑back that calls the original Python implementation.

The neighbour‑list code here is a *self‑contained* JAX implementation, so we
avoid the current incompatibility between **jax‑md/Haiku** and JAX\xa0≥0.6.  When
a compatible jax‑md wheel is released you can switch back by re‑enabling the
commented import section below.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Package‑local imports ------------------------------------------------------
from .constants import kRadiusC, kRadiusCA, kRadiusN, kRadiusO, kRadiusWater
from .types import ChainPytree  # type: ignore – project‑specific

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("dsspjax")

# ---------------------------------------------------------------------------
# Utility: Fibonacci lattice for surface dots
# ---------------------------------------------------------------------------

def generate_surface_dots(n_points: int) -> Tuple[jnp.ndarray, float]:
    """Return evenly distributed unit vectors + per‑dot area weight."""
    idx = jnp.arange(0, n_points, dtype=jnp.float32) + 0.5
    phi = jnp.arccos(1.0 - 2.0 * idx / n_points)
    theta = jnp.pi * (1.0 + math.sqrt(5.0)) * idx

    x = jnp.cos(theta) * jnp.sin(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(phi)
    dots = jnp.stack((x, y, z), axis=-1)
    return dots, float(4.0 * math.pi / n_points)


# ---------------------------------------------------------------------------
# Shared core – dot‑occlusion test, JIT‑compiled, differentiable
# ---------------------------------------------------------------------------

@jax.jit
def _atomic_sasa(
    coord: jnp.ndarray,
    radius: float,
    neighbour_coords: jnp.ndarray,
    neighbour_radii: jnp.ndarray,
    surface_dots: jnp.ndarray,
    dot_weight: float,
):
    probe_r = radius + kRadiusWater
    probe_r_sq = probe_r * probe_r

    surf_xyz = coord + surface_dots * probe_r  # (D,3)
    nbr_probe_r_sq = (neighbour_radii + kRadiusWater) ** 2  # (M,)

    @jax.vmap  # over dots
    def _is_occ(dot):
        d2 = jnp.sum((neighbour_coords - dot) ** 2, axis=1)
        return jnp.any(d2 < nbr_probe_r_sq)

    accessible = ~_is_occ(surf_xyz)
    n_acc = jnp.sum(accessible)
    return n_acc * dot_weight * probe_r_sq


# ---------------------------------------------------------------------------
# Backend 1 – fully vectorised vmap² (O(N²) memory)
# ---------------------------------------------------------------------------

def _sasa_vmap2(coords: jnp.ndarray, radii: jnp.ndarray, surface_dots: jnp.ndarray, dot_weight: float):
    n = coords.shape[0]
    if n > 6_000:
        raise MemoryError("vmap2 backend limited to 6\\xa0000 atoms – choose 'nlist'.")

    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1)

    probe_r = radii + kRadiusWater
    probe_r_sq = probe_r ** 2

    mask = dist_sq < (probe_r_sq[:, None] + probe_r_sq[None, :])
    mask = mask.at[jnp.diag_indices(n)].set(False)

    max_nbrs = int(mask.sum(axis=1).max())
    idx = jnp.where(mask, jnp.arange(n), -1)
    # Ensure padding matches max_nbrs dimension
    if max_nbrs >= n:
        actual_max_nbrs = mask.sum(axis=1).max().item()
        # Determine required columns. Ensure it's at least max_nbrs.
        required_cols = max(max_nbrs, idx.shape[1])
        if required_cols > idx.shape[1]:
            pad_width = required_cols - idx.shape[1]
            idx = jnp.pad(idx, ((0, 0), (0, pad_width)), constant_values=-1)
        # Trim if idx has more columns than needed (should not happen with where+arange)
        elif idx.shape[1] > required_cols:
             idx = idx[:, :required_cols]

    @jax.jit
    def _kernel():
        @jax.vmap
        def _per_atom(coord, rad, idx_row):
            valid = idx_row != -1
            nbr_indices = jnp.where(valid, idx_row, 0)  # safe gather indices
            # Keep full shape for coords
            nbr_xyz = coords[nbr_indices]
            # Mask radii: Set radius of invalid neighbors so probe radius becomes 0
            nbr_rad = jnp.where(valid, radii[nbr_indices], -kRadiusWater)
            # Pass full arrays to _atomic_sasa
            return _atomic_sasa(coord, rad, nbr_xyz, nbr_rad, surface_dots, dot_weight)

        return _per_atom(coords, radii, idx)

    return _kernel()


# ---------------------------------------------------------------------------
# Backend 2 – lightweight neighbour list (pure JAX, no jax‑md)
# ---------------------------------------------------------------------------

@dataclass
class _NeighborList:
    idx: jnp.ndarray  # (N, capacity) padded with -1


def _build_neighbor_list(coords: jnp.ndarray, cutoff: float) -> _NeighborList:
    """Brute‑force O(N²) neighbour list – sufficient up to ~150 k atoms."""
    n = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1)
    mask = (dist_sq < cutoff**2) & (~jnp.eye(n, dtype=bool))

    # Determine max number of neighbors
    max_neighbors = jnp.max(jnp.sum(mask, axis=1)).item()
    max_nbrs_int = int(max_neighbors)

    # Use sort trick for robust index extraction and padding
    col_indices = jnp.arange(n)
    # Set indices of non-neighbors to a large value (n) for sorting
    masked_col_indices = jnp.where(mask, col_indices, n)
    # Sort indices per row, non-neighbors (n) will be at the end
    sorted_indices = jnp.sort(masked_col_indices, axis=1)

    # Slice to keep only the first `max_nbrs_int` indices (actual neighbors)
    # This handles rows with fewer than max_neighbors automatically
    neighbor_idx = sorted_indices[:, :max_nbrs_int]

    # Replace the large value (n) used for sorting with -1 for padding indication
    final_idx = jnp.where(neighbor_idx == n, -1, neighbor_idx)

    return _NeighborList(idx=final_idx)


def _sasa_nlist(coords: jnp.ndarray, radii: jnp.ndarray, surface_dots: jnp.ndarray, dot_weight: float):
    cutoff = float(radii.max()) * 2.0 + 2.0 * kRadiusWater + 2.0
    nbrs = _build_neighbor_list(coords, cutoff)

    @jax.jit
    def _kernel():
        @jax.vmap
        def _per_atom(i):
            idx_row = nbrs.idx[i]
            valid = idx_row != -1
            nbr_indices = jnp.where(valid, idx_row, 0)
            # Keep full shape for coords
            nbr_xyz = coords[nbr_indices]
            # Mask radii
            nbr_rad = jnp.where(valid, radii[nbr_indices], -kRadiusWater)
            # Pass full arrays to _atomic_sasa
            return _atomic_sasa(coords[i], radii[i], nbr_xyz, nbr_rad, surface_dots, dot_weight)

        return _per_atom(jnp.arange(coords.shape[0]))

    return _kernel()


# ---------------------------------------------------------------------------
# Backend chooser
# ---------------------------------------------------------------------------

_BackendName = Literal["auto", "vmap2", "nlist", "loop"]


def _select_backend(n_atoms: int, request: _BackendName) -> str:
    if request != "auto":
        return request
    if n_atoms <= 6_000:
        return "vmap2"
    if n_atoms <= 150_000:
        return "nlist"
    return "loop"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_accessibility(
    chain: ChainPytree,
    *,
    n_dots: int = 960,
    backend: _BackendName = "auto",
):
    """Return a *new* ChainPytree with per‑residue ``accessibility`` field."""

    if not chain:
        return chain

    # 1. Extract atom arrays ------------------------------------------------
    bb_radii = {"N": kRadiusN, "CA": kRadiusCA, "C": kRadiusC, "O": kRadiusO}
    bb_order = ("N", "CA", "C", "O")

    coords_buf, radii_buf, atom2res = [], [], []
    for res_idx, res in enumerate(chain):
        bb = res["bb_coords"]
        for name in bb_order:
            xyz = getattr(bb, name)
            if not jnp.any(jnp.isnan(xyz)):
                coords_buf.append(xyz)
                radii_buf.append(bb_radii[name])
                atom2res.append(res_idx)
        sc_xyz = res["sidechain_coords"]
        sc_rad = res["sidechain_radii"]
        if sc_xyz.size:
            # Use extend with list comprehension for coordinates
            coords_buf.extend([row for row in sc_xyz])
            # Use extend with list() for radii array
            radii_buf.extend(list(sc_rad))
            atom2res.extend([res_idx] * sc_xyz.shape[0])

    if not coords_buf: # Check if buffer is empty after processing all residues
        log.warning("No valid atoms found to calculate accessibility.")
        # Return chain with NaN accessibility or handle as appropriate
        return [{**res, "accessibility": float('nan')} for res in chain]

    all_coords = jnp.stack(coords_buf)
    all_radii = jnp.asarray(radii_buf, dtype=jnp.float32)
    atom2res = jnp.asarray(atom2res, dtype=jnp.int32)

    n_atoms = all_coords.shape[0]
    selected_backend = _select_backend(n_atoms, backend)
    log.info("[cyan]SASA:[/] Using backend '%s' for %d atoms", selected_backend, n_atoms)

    # 2. Surface dots -------------------------------------------------------
    surface_dots, dot_w = generate_surface_dots(n_dots)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as bar:
        task = bar.add_task("Running SASA kernel", total=None)
        if selected_backend == "vmap2":
            atom_sasa = _sasa_vmap2(all_coords, all_radii, surface_dots, dot_w)
        elif selected_backend == "nlist":
            atom_sasa = _sasa_nlist(all_coords, all_radii, surface_dots, dot_w)
        else:  # 'loop'
            # Import the original (non-optimized) accessibility calculation dynamically
            # This assumes the original is in a module named 'accessibility_original.py'
            # or similar, or handled differently if it's the same file.
            # For now, let's assume a placeholder import mechanism.
            try:
                # Attempt to import the original implementation if needed
                from .accessibility_original import calculate_accessibility as _slow
                log.warning("Falling back to original 'loop' implementation.")
                return _slow(chain, n_dots=n_dots)
            except ImportError:
                log.error("Original 'loop' backend implementation not found. Cannot proceed.")
                # Return chain with NaNs or raise error
                return [{**res, "accessibility": float('nan')} for res in chain]

        bar.update(task, completed=1)

    # 3. Reduce atom → residue ---------------------------------------------
    residue_sasa = jax.ops.segment_sum(atom_sasa, atom2res, len(chain))
    residue_sasa_np = np.asarray(residue_sasa) # Convert to numpy for easier assignment

    # Calculate and log total SASA
    total_sasa = np.sum(residue_sasa_np)
    log.info(f"--> Accessibility Calculation Complete. Total SASA: [b]{total_sasa:.2f}[/] Å²", extra={"markup": True})

    # 4. Attach back to chain ----------------------------------------------
    return [
        {**res, "accessibility": float(residue_sasa_np[i])} for i, res in enumerate(chain)
    ]


__all__ = ["calculate_accessibility"]
