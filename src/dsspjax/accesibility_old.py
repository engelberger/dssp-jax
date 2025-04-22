# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2020 NKI/AVL, Netherlands Cancer Institute
# Substantial portions of this code are derived from the DSSP C++ implementation
# by Maarten L. Hekkelman and contributors, licensed under the BSD-2-Clause license.
# Full C++ source and license available at: https://github.com/PDB-REDO/dssp
"""Solvent Accessible Surface Area (SASA) calculation logic."""

import jax
import jax.numpy as jnp
import numpy as np # Use numpy for mutable accumulator array
from typing import Tuple
import logging # Import logging
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn # Import rich progress

# Relative imports from within the package
from .types import ChainPytree
from .constants import (
    kRadiusWater, kRadiusN, kRadiusCA, kRadiusC, kRadiusO
)

# Get the logger instance
log = logging.getLogger("dsspjax")

# --- Helper Functions for SASA --- #

def generate_surface_dots(n_points: int) -> Tuple[jnp.ndarray, float]:
    """Generates approximately evenly distributed points on a unit sphere.

    Uses the Fibonacci spiral (or golden angle) method.

    Args:
        n_points: The approximate number of points to generate.

    Returns:
        A tuple containing:
          - dots: A JAX array of shape (n_points, 3) with unit vectors.
          - weight_per_dot: The surface area weight associated with each dot
            on a unit sphere (4 * pi / n_points).
    """
    # Using the golden angle/Fibonacci lattice method
    # Ref: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere/
    indices = jnp.arange(0, n_points, dtype=jnp.float32) + 0.5
    phi = jnp.arccos(1 - 2 * indices / n_points) # Latitude (polar angle)
    theta = jnp.pi * (1 + jnp.sqrt(5.0)) * indices # Longitude (azimuthal angle)

    x = jnp.cos(theta) * jnp.sin(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(phi)

    dots = jnp.stack([x, y, z], axis=-1)
    # Each dot represents an equal area patch. Total area = 4*pi for unit sphere.
    weight_per_dot = 4.0 * jnp.pi / n_points

    return dots, float(weight_per_dot)

@jax.jit
def _distance_sq_to_neighbors(point: jnp.ndarray, neighbor_coords: jnp.ndarray) -> jnp.ndarray:
    """Calculates squared Euclidean distance from a point to each neighbor coord."""
    # point: shape (3,)
    # neighbor_coords: shape (M, 3)
    # Output: shape (M,)
    diff = neighbor_coords - point # Broadcasting point
    return jnp.sum(diff * diff, axis=1)

@jax.jit
def _check_occlusion_for_point(surface_point: jnp.ndarray,
                                neighbor_coords: jnp.ndarray, # Shape (M, 3)
                                neighbor_probe_radii_sq: jnp.ndarray # Shape (M,)
                               ) -> bool:
    """Checks if a single surface point is occluded by any neighbor's probe sphere."""
    # Check if there are any neighbors to consider
    num_neighbors = neighbor_coords.shape[0]

    # Calculate distances only if neighbors exist
    dist_sq_to_point = _distance_sq_to_neighbors(surface_point, neighbor_coords)

    # Check if *any* neighbor distance is less than its probe radius squared
    is_occluded = jnp.any(dist_sq_to_point < neighbor_probe_radii_sq)

    # Return False if no neighbors, otherwise return occlusion status
    return jnp.where(num_neighbors == 0, False, is_occluded)

# Vmap the occlusion check over all surface points for a single target atom
_vmap_check_occlusion_over_dots = jax.vmap(
    _check_occlusion_for_point, in_axes=(0, None, None), out_axes=0
)

# Note: calculate_atom_sasa cannot be fully JITted if the debug print based on index is kept.
# For production, remove the target_atom_idx_debug argument and the conditional print,
# then uncomment the @jax.jit decorator.
@jax.jit
def calculate_atom_sasa(target_atom_idx_debug: int, # For optional debug print
                        target_atom_coord: jnp.ndarray,
                        target_atom_radius: float,
                        neighbor_coords: jnp.ndarray, # Shape (M, 3)
                        neighbor_radii: jnp.ndarray,  # Shape (M,)
                        surface_dots: jnp.ndarray,    # Shape (n_dots, 3)
                        dot_weight: float) -> float:
    """Calculates the Solvent Accessible Surface Area (SASA) for a single atom.

    Args:
        target_atom_idx_debug: Index of the atom (for debug printing only).
        target_atom_coord: Coordinates of the target atom.
        target_atom_radius: Van der Waals radius of the target atom.
        neighbor_coords: Coordinates of neighboring atoms.
        neighbor_radii: Van der Waals radii of neighboring atoms.
        surface_dots: Unit vectors for surface points (from generate_surface_dots).
        dot_weight: Surface area weight per dot (from generate_surface_dots).

    Returns:
        The solvent-accessible surface area for the target atom (in Angstrom^2).
    """
    # Calculate radii expanded by the water probe radius
    target_probe_radius = target_atom_radius + kRadiusWater
    neighbor_probe_radii = neighbor_radii + kRadiusWater
    neighbor_probe_radii_sq = neighbor_probe_radii * neighbor_probe_radii

    # Scale surface dots to the probe radius and translate to atom center
    # surface_points shape: (n_dots, 3)
    surface_points = target_atom_coord + surface_dots * target_probe_radius

    # Check occlusion for all surface points against all neighbors
    # is_occluded shape: (n_dots,)
    is_occluded = _vmap_check_occlusion_over_dots(
        surface_points, neighbor_coords, neighbor_probe_radii_sq
    )

    # Count accessible dots (where is_occluded is False)
    n_accessible_dots = jnp.sum(~is_occluded)

    # --- Optional Debug Print --- #
    # This print statement prevents full JIT compilation.
    # Remove for production code.
    # Use lax.cond or host_callback for JIT-compatible conditional printing if needed.
    # if target_atom_idx_debug < 5: # Print for first 5 atoms only
    #     # NOTE: Standard print doesn't work reliably inside JIT. Use jax.debug.print or remove.
    #     # print(f"    DEBUG SASA: Atom {target_atom_idx_debug}, n_accessible = {n_accessible_dots}")
    #     # Use jax.debug.print for JIT-compatible printing (prints during trace)
    #     jax.debug.print("    DEBUG SASA: Atom {idx}, n_accessible = {n_acc}",
    #                     idx=target_atom_idx_debug, n_acc=n_accessible_dots)
    # --- End Optional Debug Print --- #

    # Calculate SASA: Area = N_accessible * Area_per_dot_on_probe_sphere
    # Area_per_dot_on_probe_sphere = dot_weight_unit_sphere * probe_radius^2
    atom_sasa = n_accessible_dots * dot_weight * (target_probe_radius ** 2)

    return atom_sasa


# --- Main Accessibility Calculation --- #

def calculate_accessibility(chain: ChainPytree, n_dots: int = 960, neighbor_cutoff_margin: float = 1.0) -> ChainPytree:
    """Calculates SASA for each residue by summing atom SASA values.

    Includes both backbone and sidechain atoms in the calculation.
    Uses a distance-based cutoff to find potential neighboring atoms for each
    target atom to reduce the number of occlusion checks.

    Args:
        chain: The input ChainPytree.
        n_dots: Number of points for surface dot generation (higher is more accurate
                but slower). 960 is a common value.
        neighbor_cutoff_margin: Additional margin added to the theoretical max
                                neighbor distance (R_target + R_max_neighbor + 2*R_water)
                                for safety.

    Returns:
        The updated ChainPytree with the 'accessibility' field populated for each residue.
    """
    log.info(f"\n[bold cyan]Step 7: Accessibility Calculation[/] ({n_dots} dots/atom, cutoff margin {neighbor_cutoff_margin:.1f} Å)", extra={"markup": True})
    if not chain:
        return chain

    # --- 1. Precomputation --- #
    surface_dots, dot_weight = generate_surface_dots(n_dots)
    log.info(f"--> Generated {surface_dots.shape[0]} surface points per atom.")

    # --- 2. Extract All Atom Data --- #
    all_coords_list = []
    all_radii_list = []
    atom_to_residue_index = [] # Map global atom index back to residue index

    # Define backbone radii mapping
    bb_radii_map = {'N': kRadiusN, 'CA': kRadiusCA, 'C': kRadiusC, 'O': kRadiusO}
    bb_atom_order = ['N', 'CA', 'C', 'O'] # Consistent order

    for res_idx, res in enumerate(chain):
        coords = res['bb_coords']
        # Add backbone atoms (if coords are valid)
        for atom_name in bb_atom_order:
            atom_coord = getattr(coords, atom_name)
            if not jnp.any(jnp.isnan(atom_coord)):
                all_coords_list.append(atom_coord)
                all_radii_list.append(bb_radii_map[atom_name])
                atom_to_residue_index.append(res_idx)

        # Add sidechain atoms
        sc_coords = res['sidechain_coords']
        sc_radii = res['sidechain_radii']
        num_sc_atoms = sc_coords.shape[0]
        if num_sc_atoms > 0:
            all_coords_list.extend(list(sc_coords)) # Use list() for extend
            all_radii_list.extend(list(sc_radii))
            atom_to_residue_index.extend([res_idx] * num_sc_atoms)

    if not all_coords_list:
        print("Warning: No valid atom coordinates found in the chain for SASA calculation.")
        # Return chain with NaN accessibility
        updated_chain = []
        for res in chain:
            updated_chain.append({**res, 'accessibility': jnp.nan})
        return updated_chain

    # Convert lists to JAX arrays
    all_coords = jnp.stack(all_coords_list) # Shape (N_atoms_total, 3)
    all_radii = jnp.array(all_radii_list)   # Shape (N_atoms_total,)
    atom_to_residue_index_np = np.array(atom_to_residue_index, dtype=np.int32)
    n_total_atoms = all_coords.shape[0]
    log.info(f"--> Extracted {n_total_atoms} total atoms for SASA calculation.")

    # --- 3. Calculate SASA per Atom --- #
    # Accumulate SASA per residue using a NumPy array (mutable)
    residue_sasa_np = np.zeros(len(chain), dtype=np.float64)

    # Precompute max VdW radius for efficient cutoff calculation
    max_vdw_radius = jnp.max(all_radii) if n_total_atoms > 0 else 0.0

    # log.info(f"  Calculating SASA for {n_total_atoms} atoms...")
    # Main loop: Iterate through each atom as the target
    # Use rich.progress for a nice progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("--> Calculating SASA...", total=n_total_atoms)
        for target_idx in range(n_total_atoms):
            target_coord = all_coords[target_idx]
            target_radius = all_radii[target_idx]
            target_res_idx = atom_to_residue_index_np[target_idx]

            # --- 4. Find Neighbors (Optimized) --- #
            # Identify indices of all *other* atoms
            other_indices = jnp.arange(n_total_atoms) != target_idx
            other_coords = all_coords[other_indices]
            other_radii = all_radii[other_indices]

            if other_coords.shape[0] == 0: # Handle case of single-atom structure
                neighbor_coords_filtered = jnp.empty((0, 3), dtype=all_coords.dtype)
                neighbor_radii_filtered = jnp.empty((0,), dtype=all_radii.dtype)
            else:
                # Calculate distance-based cutoff squared
                # Max distance = R_target + R_neighbor_max + 2 * R_water + margin
                cutoff_dist = target_radius + max_vdw_radius + 2 * kRadiusWater + neighbor_cutoff_margin
                cutoff_dist_sq = cutoff_dist * cutoff_dist

                # Calculate squared distances to all *other* atoms
                dist_sq_to_others = _distance_sq_to_neighbors(target_coord, other_coords)

                # Filter neighbors based on the cutoff distance
                neighbor_mask = dist_sq_to_others < cutoff_dist_sq
                neighbor_coords_filtered = other_coords[neighbor_mask]
                neighbor_radii_filtered = other_radii[neighbor_mask]

            # --- 5. Calculate SASA for the Target Atom --- #
            # Pass target_idx for potential debug print inside
            atom_sasa = calculate_atom_sasa(
                target_idx,
                target_coord,
                target_radius,
                neighbor_coords_filtered, # Pass only the nearby neighbors
                neighbor_radii_filtered,
                surface_dots,
                dot_weight
            )

            # --- 6. Accumulate SASA for the Residue --- #
            # Add atom's SASA to its corresponding residue's total
            # Use NumPy array for in-place addition
            residue_sasa_np[target_res_idx] += float(atom_sasa) # Ensure float for numpy

            # Optional progress indicator
            # if (target_idx + 1) % 200 == 0 or target_idx == n_total_atoms - 1:
            #      print(f"    Processed atom {target_idx + 1}/{n_total_atoms}")
            progress.update(task, advance=1)

    # --- 7. Update Pytree with Final Residue SASA --- #
    updated_chain = []
    for i, res in enumerate(chain):
        # Store accessibility as a standard float
        updated_res = {**res, 'accessibility': float(residue_sasa_np[i])}
        updated_chain.append(updated_res)

    total_sasa = np.sum(residue_sasa_np)
    log.info(f"--> Accessibility Calculation Complete. Total SASA: [b]{total_sasa:.2f}[/] Å²", extra={"markup": True})
    return updated_chain 