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
from typing import Literal, Tuple, List, Dict, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Package‑local imports ------------------------------------------------------
from .constants import kRadiusC, kRadiusCA, kRadiusN, kRadiusO, kRadiusWater, element_radii, kRadiusDefault
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
        log.info(f"Backend '{request}' explicitly requested.")
        return request

    # Default to 'nlist' if 'auto' is requested
    # The vmap2 backend uses O(N^2) memory and is often too demanding.
    log.debug("Backend set to 'auto', defaulting to 'nlist'.")
    # Keep the atom count checks mainly for logging/info
    if n_atoms <= 6_000:
        log.debug("    (Atom count %d <= 6000, 'vmap2' could be used if requested explicitly)", n_atoms)
    elif n_atoms > 150_000:
        log.warning("    (Atom count %d > 150000, 'nlist' backend might become slow)", n_atoms)

    # Currently, the 'loop' backend is not fully supported with ligands, so we default to nlist.
    # if n_atoms > 150_000:
    #     log.warning("Atom count > 150000, falling back to 'loop'.")
    #     return "loop"

    return "nlist" # Default to nlist when request is 'auto'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_accessibility(
    chain: ChainPytree,
    *,
    ligand_atoms: Optional[List[Dict[str, Any]]] = None,
    n_dots: int = 960,
    backend: _BackendName = "auto",
):
    """Calculates accessibility for protein residues, considering the environment.

    Includes protein atoms and optionally specified ligand atoms in the
    calculation environment that can occlude protein atoms.

    Args:
        chain: The input ChainPytree representing the protein.
        ligand_atoms: Optional flat list of dictionaries for target ligand atoms
                      (from load_cif_data), each containing 'coords', 'symbol', 'ligand_key'.
        n_dots: Number of points for surface dot generation.
        backend: SASA calculation backend to use ("auto", "vmap2", "nlist", "loop").

    Returns:
        A *new* ChainPytree with per‑residue ``accessibility`` field calculated
        considering occlusion by protein and specified ligand atoms.
    """

    # Handle empty input
    if not chain and not ligand_atoms:
        log.warning("Cannot calculate accessibility: No protein chain or ligand atoms provided.")
        return chain # Return empty chain if protein chain was empty
    if not chain:
        log.warning("Protein chain is empty, calculating SASA only for ligands is not yet supported by the return type.")
        # Cannot currently return ligand SASA, return empty chain
        return []

    # 1. Extract atom arrays ------------------------------------------------
    bb_radii = {"N": kRadiusN, "CA": kRadiusCA, "C": kRadiusC, "O": kRadiusO}
    bb_order = ("N", "CA", "C", "O")

    coords_buf, radii_buf, atom_to_entity_idx = [], [], []
    protein_atom_count = 0

    # Process protein atoms
    for res_idx, res in enumerate(chain):
        bb = res["bb_coords"]
        start_protein_atom_idx = len(coords_buf) # Index before adding this residue's atoms
        num_atoms_this_res = 0
        for name in bb_order:
            xyz = getattr(bb, name)
            if not jnp.any(jnp.isnan(xyz)):
                coords_buf.append(xyz)
                radii_buf.append(bb_radii[name])
                num_atoms_this_res += 1
        sc_xyz = res["sidechain_coords"]
        sc_rad = res["sidechain_radii"]
        if sc_xyz.size:
            coords_buf.extend([row for row in sc_xyz])
            radii_buf.extend(list(sc_rad))
            num_atoms_this_res += sc_xyz.shape[0]

        # Assign residue index to all atoms added for this residue
        atom_to_entity_idx.extend([res_idx] * num_atoms_this_res)
        protein_atom_count += num_atoms_this_res

    # Process ligand atoms if provided
    ligand_atom_count = 0
    ligand_key_map = {}
    ligand_start_idx = len(chain) # Assign indices starting after protein residues

    if ligand_atoms:
        for atom_dict in ligand_atoms:
            ligand_key = atom_dict['ligand_key']
            symbol = atom_dict['symbol'].upper()
            coords = atom_dict['coords'] # Should be numpy array from io.py

            # Assign a unique index to this ligand molecule if not seen before
            if ligand_key not in ligand_key_map:
                ligand_key_map[ligand_key] = ligand_start_idx
                ligand_start_idx += 1
            entity_idx = ligand_key_map[ligand_key]

            # Get radius (handle missing elements with default)
            radius = element_radii.get(symbol, kRadiusDefault)
            if symbol not in element_radii:
                log.warning(f"Using default radius {kRadiusDefault}Å for unknown ligand element: '{symbol}' in {ligand_key}")

            coords_buf.append(jnp.asarray(coords)) # Ensure JAX array
            radii_buf.append(radius)
            atom_to_entity_idx.append(entity_idx)
            ligand_atom_count += 1

    if protein_atom_count + ligand_atom_count == 0:
        log.warning("No valid atoms found after processing protein and ligands.")
        return [{**res, "accessibility": float('nan')} for res in chain]

    # Stack all coordinates and radii (protein + ligand)
    all_coords = jnp.stack(coords_buf)
    all_radii = jnp.asarray(radii_buf, dtype=jnp.float32)
    atom_to_entity_idx_arr = jnp.asarray(atom_to_entity_idx, dtype=jnp.int32)
    num_total_atoms = all_coords.shape[0]
    num_entities = ligand_start_idx # Total number of protein residues + unique ligands

    log.info(f"--> Prepared {protein_atom_count} protein atoms and {ligand_atom_count} ligand atoms for SASA environment.")

    # Select backend based on TOTAL number of atoms
    selected_backend = _select_backend(num_total_atoms, backend)
    log.info("[cyan]SASA:[/] Using backend '%s' for %d total atoms (%d entities)", selected_backend, num_total_atoms, num_entities)

    # 2. Surface dots -------------------------------------------------------
    surface_dots, dot_w = generate_surface_dots(n_dots)

    # 3. Run SASA calculation on ALL atoms ----------------------------------
    atom_sasa = jnp.zeros(num_total_atoms, dtype=jnp.float32) # Initialize
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TimeElapsedColumn(), transient=True) as bar:
        task = bar.add_task("Running SASA kernel on all atoms", total=None)
        if selected_backend == "vmap2":
            atom_sasa = _sasa_vmap2(all_coords, all_radii, surface_dots, dot_w)
        elif selected_backend == "nlist":
            atom_sasa = _sasa_nlist(all_coords, all_radii, surface_dots, dot_w)
        else:  # 'loop'
            # Loop backend would need significant modification to handle mixed protein/ligand
            # For simplicity, we'll raise an error or return NaN if loop is selected with ligands.
            if ligand_atoms:
                log.error("'loop' backend is not supported when ligands are included. Use 'vmap2' or 'nlist'.")
                return [{**res, "accessibility": float('nan')} for res in chain]
            else:
                try:
                    from .accessibility_original import calculate_accessibility as _slow
                    log.warning("Falling back to original 'loop' implementation for protein only.")
                    return _slow(chain, n_dots=n_dots)
                except ImportError:
                    log.error("Original 'loop' backend implementation not found. Cannot proceed.")
                    return [{**res, "accessibility": float('nan')} for res in chain]

        bar.update(task, completed=1)

    # 4. Aggregate SASA per entity (protein residue or ligand) ------------
    # Use segment_sum with the combined atom_to_entity_idx mapping
    entity_sasa = jax.ops.segment_sum(atom_sasa, atom_to_entity_idx_arr, num_entities)
    entity_sasa_np = np.asarray(entity_sasa) # Convert to numpy

    # Calculate and log total SASA (protein + ligand)
    total_sasa = np.sum(entity_sasa_np)
    # Extract total protein SASA (summing only the residue parts)
    total_protein_sasa = np.sum(entity_sasa_np[:len(chain)]) if len(chain) > 0 else 0.0
    log.info(f"--> Accessibility Calculation Complete. Total Protein SASA: [b]{total_protein_sasa:.2f}[/] Å² ([b]{total_sasa:.2f}[/] Å² including ligands)", extra={"markup": True})

    # Log individual ligand SASA values if ligands were processed
    if ligand_atoms:
        for ligand_key, ligand_idx in ligand_key_map.items():
            lig_sasa = entity_sasa_np[ligand_idx]
            log.info(f"    SASA for Ligand {ligand_key}: [b]{lig_sasa:.2f}[/] Å²", extra={"markup": True})

    # 5. Attach protein residue SASA back to protein chain -----------------
    # We only return the protein ChainPytree, updated with its SASA values.
    return [
        {**res, "accessibility": float(entity_sasa_np[i])} for i, res in enumerate(chain)
    ]


__all__ = ["calculate_accessibility"]
