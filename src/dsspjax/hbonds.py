# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2020 NKI/AVL, Netherlands Cancer Institute
# Substantial portions of this code are derived from the DSSP C++ implementation
# by Maarten L. Hekkelman and contributors, licensed under the BSD-2-Clause license.
# Full C++ source and license available at: https://github.com/PDB-REDO/dssp
"""Hydrogen bond calculations, partner finding, and bridge testing."""

import jax
import jax.numpy as jnp
from typing import Dict

# Relative imports from within the package
from .types import ChainPytree, BridgeType
from .constants import (
    kCouplingConstant, kMinimalDistance, kMinHBondEnergy,
    kInvalidHBondEnergy, kMinimalCADistanceSq
)
from .geometry import distance_sq # Need distance_sq for H-bond energy

# --- Hydrogen Position Calculation ---

def calculate_hydrogen_positions(chain: ChainPytree) -> ChainPytree:
    """Calculates the approximate position of the backbone amide hydrogen (H).

    Uses the standard DSSP approximation: H is 1.0 Angstrom from N along the
    vector from the previous residue's O to C atom.
    Sets H coordinates to NaN for Proline and the first residue.

    Args:
        chain: Input ChainPytree.

    Returns:
        Updated ChainPytree with 'H' coordinates calculated in bb_coords.
    """
    if not chain:
        return chain

    # Stack coordinates for vectorized operations
    coords_n = jnp.stack([r['bb_coords'].N for r in chain])
    coords_c = jnp.stack([r['bb_coords'].C for r in chain])
    coords_o = jnp.stack([r['bb_coords'].O for r in chain])
    is_proline = jnp.array([r['is_proline'] for r in chain], dtype=bool)
    n_res = len(chain)

    # Get C and O coordinates of the *previous* residue, handle boundaries
    prev_c = jnp.roll(coords_c, shift=1, axis=0)
    prev_o = jnp.roll(coords_o, shift=1, axis=0)
    # Set prev coords for the first residue to NaN to prevent calculation
    prev_c = prev_c.at[0].set(jnp.full(3, jnp.nan))
    prev_o = prev_o.at[0].set(jnp.full(3, jnp.nan))

    # Vector from previous O to previous C
    prev_co = prev_c - prev_o
    norm = jnp.linalg.norm(prev_co, axis=1, keepdims=True)

    # Handle potential zero-length vectors (e.g., at start or missing prev O/C)
    # or NaN vectors
    safe_norm = jnp.where((norm < 1e-6) | jnp.isnan(norm), 1.0, norm)
    norm_co = prev_co / safe_norm
    # Ensure normalization didn't create NaNs if prev_co was NaN
    norm_co = jnp.where(jnp.isnan(prev_co), jnp.nan, norm_co)

    # Place H approx 1.0 Angstrom from N along the O->C vector
    h_pos_calculated = coords_n + norm_co * 1.0

    # Create a mask: H exists if not Proline AND not the first residue
    mask = (~is_proline) & (jnp.arange(n_res) > 0)

    # Update the H coordinate in the Pytree
    updated_chain = []
    for i, r in enumerate(chain):
        # Use jnp.where to conditionally select the calculated H or NaN
        new_h = jnp.where(mask[i], h_pos_calculated[i], jnp.full(3, jnp.nan))
        # Further ensure NaN if calculation resulted in NaN (e.g., due to missing prev O/C)
        new_h = jnp.where(jnp.any(jnp.isnan(h_pos_calculated[i])), jnp.full(3, jnp.nan), new_h)

        updated_bb = r['bb_coords']._replace(H=new_h)
        updated_chain.append({**r, 'bb_coords': updated_bb})

    return updated_chain

# --- H-Bond Energy Calculation ---

@jax.jit
def _calculate_hbond_energy_jax_impl(n_d: jnp.ndarray, h_d: jnp.ndarray, c_a: jnp.ndarray, o_a: jnp.ndarray) -> float:
    """Calculates the H-bond energy between a donor (N-H) and acceptor (C=O).

    Uses the Kabsch-Sander electrostatic model.

    Args:
        n_d: Coordinates of donor nitrogen.
        h_d: Coordinates of donor hydrogen.
        c_a: Coordinates of acceptor carbon.
        o_a: Coordinates of acceptor oxygen.

    Returns:
        Calculated H-bond energy (in kcal/mol), or kInvalidHBondEnergy if
        the donor H position is invalid (NaN) or distances are too close.
    """
    n_d, h_d, c_a, o_a = map(jnp.asarray, [n_d, h_d, c_a, o_a])

    # Check if donor H position is valid (not NaN)
    h_ok = ~jnp.any(jnp.isnan(h_d))

    # Calculate squared distances between relevant atoms
    # Use the imported distance_sq from geometry module
    dHO2 = distance_sq(h_d, o_a) # H(donor) <-> O(acceptor)
    dHC2 = distance_sq(h_d, c_a) # H(donor) <-> C(acceptor)
    dNC2 = distance_sq(n_d, c_a) # N(donor) <-> C(acceptor)
    dNO2 = distance_sq(n_d, o_a) # N(donor) <-> O(acceptor)

    # Check if any distance is below the minimum allowed
    min_d_sq = kMinimalDistance * kMinimalDistance
    # Check includes NaN propagation: if any dist is NaN, d_ok is False
    d_ok = (dHO2 > min_d_sq) & (dHC2 > min_d_sq) & (dNC2 > min_d_sq) & (dNO2 > min_d_sq)

    # Calculate actual distances needed for the formula
    # Use safe sqrt: sqrt(max(0, x)) to avoid NaN from slightly negative values due to precision
    dHO = jnp.sqrt(jnp.maximum(0., dHO2))
    dHC = jnp.sqrt(jnp.maximum(0., dHC2))
    dNC = jnp.sqrt(jnp.maximum(0., dNC2))
    dNO = jnp.sqrt(jnp.maximum(0., dNO2))

    # Calculate energy using Kabsch-Sander formula
    # E = C * (1/r_HO - 1/r_HC + 1/r_NC - 1/r_NO)
    # Need to handle potential division by zero if distances are exactly kMinimalDistance
    # Add small epsilon to denominator to avoid division by zero, or rely on d_ok check
    energy = kCouplingConstant * (1.0 / (dHO + 1e-9) - 1.0 / (dHC + 1e-9) +
                                  1.0 / (dNC + 1e-9) - 1.0 / (dNO + 1e-9))

    # Clamp energy: DSSP uses a floor value, but no ceiling mentioned in original paper for this formula
    # However, the validity check uses kMaxHBondEnergy later.
    # energy = jnp.maximum(energy, kMinHBondEnergy)

    # Return energy only if H position is valid and distances are ok, otherwise return invalid energy
    # Note: Energy can be > 0; validity check (kMinHBondEnergy < E < kMaxHBondEnergy) happens later.
    return jnp.where(h_ok & d_ok, energy, kInvalidHBondEnergy)

calculate_hbond_energy_jax = _calculate_hbond_energy_jax_impl

# Vectorize the H-bond calculation for efficiency
# Map over donors (rows of N, H) for a fixed acceptor (C, O)
_vmap_hbond_donor = jax.vmap(_calculate_hbond_energy_jax_impl, in_axes=(0, 0, None, None), out_axes=0)
# Map the donor-vmapped function over acceptors (rows of C, O)
_vmap_hbond_acceptor = jax.vmap(_vmap_hbond_donor, in_axes=(None, None, 0, 0), out_axes=1) # out_axes=1 gives (donor, acceptor) matrix

@jax.jit
def _calc_all_hb_energies_impl(n_coords, h_coords, c_coords, o_coords) -> jnp.ndarray:
    """Calculates the full N x N H-bond energy matrix via vmap."""
    return _vmap_hbond_acceptor(n_coords, h_coords, c_coords, o_coords)

calculate_all_hbond_energies_jax = _calc_all_hb_energies_impl

# --- H-Bond Partner Finding ---

def find_potential_hbond_pairs(chain: ChainPytree) -> jnp.ndarray:
    """Identifies potential H-bond pairs based on C-alpha distance cutoff.

    This serves as a pre-filter to avoid calculating energies for distant pairs.

    Args:
        chain: Input ChainPytree.

    Returns:
        A boolean matrix (N x N) where True indicates CA distance is below cutoff.
    """
    if not chain:
        return jnp.zeros((0, 0), dtype=bool)

    coords_ca = jnp.stack([r['bb_coords'].CA for r in chain])
    n_res = len(chain)

    # Define a function to calculate squared distances from one CA to all others
    def _row_dist_sq(ca_i, all_ca):
        return jax.vmap(distance_sq, in_axes=(None, 0))(ca_i, all_ca)

    # Apply this function to all CAs to get the full distance matrix
    ca_dist_sq_matrix = jax.vmap(_row_dist_sq, in_axes=(0, None))(coords_ca, coords_ca)

    # Create a mask where distance is less than the cutoff
    mask = ca_dist_sq_matrix < kMinimalCADistanceSq
    # Exclude self-interactions (diagonal)
    mask = mask & (~jnp.eye(n_res, dtype=bool))

    # TODO: Consider adding chain break check here if necessary.
    return mask


def find_best_hbond_partners(energy_matrix: jnp.ndarray, valid_hbonds: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Finds the indices and energies of the two best H-bond partners for each residue.

    Determines the best two donors for each acceptor, and the best two acceptors
    for each donor, based on the calculated H-bond energies.

    Args:
        energy_matrix: The N x N matrix of H-bond energies.
        valid_hbonds: The N x N boolean matrix indicating valid H-bonds
                      (energy within [kMinHBondEnergy, kMaxHBondEnergy]).

    Returns:
        A dictionary containing:
          'acceptor_idx': (N, 2) array, indices of the 2 best donors for each acceptor.
          'acceptor_nrg': (N, 2) array, energies of the 2 best donors for each acceptor.
          'donor_idx': (N, 2) array, indices of the 2 best acceptors for each donor.
          'donor_nrg': (N, 2) array, energies of the 2 best acceptors for each donor.
          Indices are -1 and energies are kInvalidHBondEnergy if fewer than 2 valid
          partners exist.
    """
    n_res = energy_matrix.shape[0]
    if n_res == 0:
        empty_idx = jnp.full((0, 2), -1, dtype=jnp.int32)
        empty_nrg = jnp.full((0, 2), kInvalidHBondEnergy, dtype=jnp.float64)
        return {'acceptor_idx': empty_idx, 'acceptor_nrg': empty_nrg,
                'donor_idx': empty_idx, 'donor_nrg': empty_nrg}

    # Mask energies that are not valid H-bonds (set to infinity)
    masked_energies = jnp.where(valid_hbonds, energy_matrix, kInvalidHBondEnergy)

    # --- Find Best Acceptors (for each Donor i) ---
    # Goal: For each row `i`, find the indices `j` of the two smallest valid energies.
    # top_k finds maximum, so we minimize negative energy.
    neg_energies = -masked_energies
    # donor_neg_nrg[i, k] = k-th largest neg energy for donor i
    # donor_idx[i, k] = index j (acceptor) of the k-th largest neg energy for donor i
    donor_neg_nrg, donor_idx = jax.lax.top_k(neg_energies, k=2)
    donor_nrg = -donor_neg_nrg # Energies of best acceptors for each donor

    # --- Find Best Donors (for each Acceptor j) ---
    # Goal: For each column `j`, find the indices `i` of the two smallest valid energies.
    # Transpose the matrix and apply the same logic.
    neg_energies_T = -masked_energies.T # Now rows correspond to acceptors
    # acceptor_neg_nrg[j, k] = k-th largest neg energy for acceptor j
    # acceptor_idx[j, k] = index i (donor) of the k-th largest neg energy for acceptor j
    acceptor_neg_nrg, acceptor_idx = jax.lax.top_k(neg_energies_T, k=2)
    acceptor_nrg = -acceptor_neg_nrg # Energies of best donors for each acceptor

    # --- Handle cases where fewer than 2 valid partners exist ---
    # Check if the energy is still effectively infinity (or >= invalid threshold)
    is_invalid_acceptor_partner = acceptor_nrg >= (kInvalidHBondEnergy - 1.0)
    acceptor_idx = jnp.where(is_invalid_acceptor_partner, -1, acceptor_idx)
    acceptor_nrg = jnp.where(is_invalid_acceptor_partner, kInvalidHBondEnergy, acceptor_nrg)

    is_invalid_donor_partner = donor_nrg >= (kInvalidHBondEnergy - 1.0)
    donor_idx = jnp.where(is_invalid_donor_partner, -1, donor_idx)
    donor_nrg = jnp.where(is_invalid_donor_partner, kInvalidHBondEnergy, donor_nrg)

    return {
        'acceptor_idx': acceptor_idx, 'acceptor_nrg': acceptor_nrg, # Shape (n_res, 2)
        'donor_idx': donor_idx, 'donor_nrg': donor_nrg             # Shape (n_res, 2)
    }

def update_pytree_hbond_partners(chain: ChainPytree, hbond_partners: Dict) -> ChainPytree:
    """Updates the ChainPytree with the best H-bond partner information.

    Args:
        chain: The input ChainPytree.
        hbond_partners: The dictionary returned by `find_best_hbond_partners`.

    Returns:
        The updated ChainPytree with hbond_* fields populated.
    """
    updated_chain = []
    acc_idx = hbond_partners['acceptor_idx'] # Shape (n_res, 2)
    acc_nrg = hbond_partners['acceptor_nrg'] # Shape (n_res, 2)
    don_idx = hbond_partners['donor_idx']   # Shape (n_res, 2)
    don_nrg = hbond_partners['donor_nrg']   # Shape (n_res, 2)

    for i, res in enumerate(chain):
        updated_res = {**res} # Copy existing data
        # For residue i acting as DONOR, store best ACCEPTOR indices/energies
        updated_res['hbond_acceptor_1_idx'] = int(don_idx[i, 0]) # Ensure int
        updated_res['hbond_acceptor_1_nrg'] = float(don_nrg[i, 0]) # Ensure float
        updated_res['hbond_acceptor_2_idx'] = int(don_idx[i, 1])
        updated_res['hbond_acceptor_2_nrg'] = float(don_nrg[i, 1])
        # For residue i acting as ACCEPTOR, store best DONOR indices/energies
        updated_res['hbond_donor_1_idx'] = int(acc_idx[i, 0])
        updated_res['hbond_donor_1_nrg'] = float(acc_nrg[i, 0])
        updated_res['hbond_donor_2_idx'] = int(acc_idx[i, 1])
        updated_res['hbond_donor_2_nrg'] = float(acc_nrg[i, 1])
        updated_chain.append(updated_res)
    return updated_chain

# --- Beta Bridge Testing ---

@jax.jit
def _test_bridge_impl(i: int, j: int, n: int, hb_mat: jnp.ndarray) -> int:
    """Tests if residues i and j form a parallel or antiparallel beta bridge.

    Uses the criteria from the original DSSP paper (Kabsch & Sander, 1983).
    Requires the boolean H-bond matrix `hb_mat`.

    Args:
        i: Index of the first residue.
        j: Index of the second residue.
        n: Total number of residues (for boundary checks).
        hb_mat: N x N boolean matrix where hb_mat[donor, acceptor] is True if
                a valid H-bond exists.

    Returns:
        Integer value of the BridgeType enum (PARALLEL, ANTIPARALLEL, or NONE).
    """
    # Ensure indices are treated as needed for calculations
    i = jnp.asarray(i, dtype=jnp.int32)
    j = jnp.asarray(j, dtype=jnp.int32)

    # Indices relative to i and j for checking patterns
    ip1 = i + 1; im1 = i - 1
    jp1 = j + 1; jm1 = j - 1

    # Boundary checks (ensure indices are within [0, n-1])
    i_ok = (i >= 0) & (i < n)
    j_ok = (j >= 0) & (j < n)
    ip1_ok = (ip1 >= 0) & (ip1 < n)
    im1_ok = (im1 >= 0) & (im1 < n)
    jp1_ok = (jp1 >= 0) & (jp1 < n)
    jm1_ok = (jm1 >= 0) & (jm1 < n)

    # --- Parallel Check --- #
    # Type I: (i+1) -> j and j -> (i-1) H-bonds
    # Need ip1_ok, j_ok, im1_ok
    p1_possible = ip1_ok & j_ok & im1_ok
    # Access hb_mat safely using where or index clipping if needed, although
    # boundary checks should prevent out-of-bounds if hb_mat is N x N.
    # Assuming hb_mat access is safe if indices are ok.
    p1 = p1_possible & hb_mat[ip1, j] & hb_mat[j, im1]

    # Type II: (j+1) -> i and i -> (j-1) H-bonds
    # Need jp1_ok, i_ok, jm1_ok
    p2_possible = jp1_ok & i_ok & jm1_ok
    p2 = p2_possible & hb_mat[jp1, i] & hb_mat[i, jm1]

    is_parallel = p1 | p2

    # --- Antiparallel Check --- #
    # Type III: i -> j and j -> i H-bonds
    # Need i_ok, j_ok
    a1_possible = i_ok & j_ok
    a1 = a1_possible & hb_mat[i, j] & hb_mat[j, i]

    # Type IV: (i+1) -> (j-1) and (j+1) -> (i-1) H-bonds
    # Need ip1_ok, jm1_ok, jp1_ok, im1_ok
    a2_possible = ip1_ok & jm1_ok & jp1_ok & im1_ok
    a2 = a2_possible & hb_mat[ip1, jm1] & hb_mat[jp1, im1]

    is_antiparallel = a1 | a2

    # Determine result: Parallel takes precedence over Antiparallel if both match?
    # Original DSSP logic seems to imply separate checks, let's prioritize Parallel.
    # Return integer value of the Enum for JIT compatibility
    result_type = jnp.where(is_parallel, BridgeType.PARALLEL.value,
                            jnp.where(is_antiparallel, BridgeType.ANTIPARALLEL.value,
                                      BridgeType.NONE.value))
    # Return the integer value directly for JIT compatibility
    return result_type


# JIT compile with n as static argument for efficiency
_test_bridge_jax_static = jax.jit(_test_bridge_impl, static_argnums=(2,))

# Vectorize bridge testing over all pairs (i, j)
# vmap over i for a fixed j
_vmap_test_bridge_i = jax.vmap(_test_bridge_jax_static, in_axes=(0, None, None, None), out_axes=0)
# vmap over j for the i-vmapped function
_vmap_test_bridge_ij = jax.vmap(_vmap_test_bridge_i, in_axes=(None, 0, None, None), out_axes=1)

# JIT compile the fully vmapped function (n still static)
_calculate_bridge_matrix_core = jax.jit(_vmap_test_bridge_ij, static_argnums=(2,))

def calculate_bridge_types(n: int, hb_mat: jnp.ndarray) -> jnp.ndarray:
    """Calculates the N x N matrix of beta bridge types (Parallel/Antiparallel/None).

    Uses JIT and vmap for efficient calculation across all residue pairs.

    Args:
        n: Total number of residues.
        hb_mat: N x N boolean H-bond matrix (True if valid H-bond).

    Returns:
        An N x N integer matrix containing BridgeType enum values.
    """
    if not isinstance(n, int) or n < 0:
        raise TypeError(f"n must be a non-negative integer, got {n} ({type(n)})")
    if n == 0:
        return jnp.array([], dtype=jnp.int32).reshape(0, 0)

    # Ensure hb_mat has the correct shape
    if hb_mat.shape != (n, n):
        raise ValueError(f"hb_mat shape {hb_mat.shape} does not match n={n}")

    # Generate indices [0, 1, ..., n-1]
    idx = jnp.arange(n)

    # Call the JITted, vmapped core function
    bridge_matrix = _calculate_bridge_matrix_core(idx, idx, n, hb_mat)

    return bridge_matrix 