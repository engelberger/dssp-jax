# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2020 NKI/AVL, Netherlands Cancer Institute
# Substantial portions of this code are derived from the DSSP C++ implementation
# by Maarten L. Hekkelman and contributors, licensed under the BSD-2-Clause license.
# Full C++ source and license available at: https://github.com/PDB-REDO/dssp
"""Assignment of secondary structure types (Helix, Turn, Bend)."""

import jax
import jax.numpy as jnp
from typing import Tuple
import logging # Import logging

# Relative imports from within the package
from .types import ChainPytree, SecondaryStructureType, HelixFlagType
from .constants import (
    kMinPPHelixLength, kPPHelixPhiMin, kPPHelixPhiMax,
    kPPHelixPsiMin, kPPHelixPsiMax, kBendKappaThreshold
)
from .geometry import angle, dihedral_angle # Need geometry functions

# Get the logger instance
log = logging.getLogger("dsspjax")

# --- Angle Calculations (Phi, Psi, Kappa) --- #

def calculate_phi_psi_kappa(chain: ChainPytree) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculates backbone Phi, Psi, and C-alpha Kappa angles.

    Args:
        chain: Input ChainPytree.

    Returns:
        A tuple containing JAX arrays for phi, psi, and kappa angles (in degrees).
        Angles are NaN where undefined (e.g., boundaries, missing atoms).
    """
    n_res = len(chain)
    # Need at least 3 residues for Psi, 4 for Kappa, 2 for Phi
    if n_res < 2:
        nan_array = jnp.full(n_res, jnp.nan)
        return nan_array, nan_array, nan_array # Phi, Psi, Kappa
    if n_res < 3:
        nan_array = jnp.full(n_res, jnp.nan)
        phi, _, _ = calculate_phi_psi_kappa([chain[0], chain[1]]) # Calculate only Phi for first residue
        return phi, nan_array, nan_array
    if n_res < 4:
         phi, psi, _ = calculate_phi_psi_kappa(chain[:3]) # Calc Phi/Psi for first 2
         nan_array = jnp.full(n_res, jnp.nan)
         return phi, psi, nan_array


    # Stack coordinates
    coords_n = jnp.stack([r['bb_coords'].N for r in chain])
    coords_ca = jnp.stack([r['bb_coords'].CA for r in chain])
    coords_c = jnp.stack([r['bb_coords'].C for r in chain])

    # Get shifted coordinates for calculations, handling boundaries with NaN padding
    # This avoids explicit boundary checks in vmap, NaNs will propagate
    def _get_shifted_coords(arr, shift):
        rolled = jnp.roll(arr, shift, axis=0)
        if shift > 0: # Getting previous (e.g., shift=1 means index i-1)
            rolled = rolled.at[:shift].set(jnp.nan)
        elif shift < 0: # Getting next (e.g., shift=-1 means index i+1)
            rolled = rolled.at[shift:].set(jnp.nan)
        return rolled

    prev_c = _get_shifted_coords(coords_c, 1)   # C(i-1)
    next_n = _get_shifted_coords(coords_n, -1)  # N(i+1)
    prev2_ca = _get_shifted_coords(coords_ca, 2) # CA(i-2)
    next2_ca = _get_shifted_coords(coords_ca, -2)# CA(i+2)

    # --- Vectorized angle calculations using vmap --- #

    # Phi(i): Dihedral C(i-1) - N(i) - CA(i) - C(i)
    # vmap requires all inputs to have the same leading dimension (n_res)
    phi = jax.vmap(dihedral_angle)(prev_c, coords_n, coords_ca, coords_c)

    # Psi(i): Dihedral N(i) - CA(i) - C(i) - N(i+1)
    psi = jax.vmap(dihedral_angle)(coords_n, coords_ca, coords_c, next_n)

    # Kappa(i): Angle CA(i-2) - CA(i) - CA(i+2)
    kappa = jax.vmap(angle)(prev2_ca, coords_ca, next2_ca)

    # Boundary conditions are implicitly handled by NaN propagation from shifted coords

    return phi, psi, kappa

# --- Helix/Bend Pattern Identification --- #

def identify_helix_patterns(n_residues: int, valid_hbonds: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Identifies residues that ACCEPT H-bonds typical of standard helices.

    Checks for i <- i+3 (3_10), i <- i+4 (Alpha), and i <- i+5 (Pi) H-bond patterns.

    Args:
        n_residues: Total number of residues.
        valid_hbonds: N x N boolean matrix where valid_hbonds[donor, acceptor] is True.

    Returns:
        Tuple of boolean arrays (is_3_10, is_alpha, is_pi) where True at index `i`
        means residue `i` accepts the corresponding helical H-bond.
    """
    # is_X_hbond[i] is True if residue i is an ACCEPTOR for an X-type H-bond
    # from a donor at i+k (where k=3, 4, or 5)
    is_3_10_hbond = jnp.zeros(n_residues, dtype=bool)
    is_alpha_hbond = jnp.zeros(n_residues, dtype=bool)
    is_pi_hbond = jnp.zeros(n_residues, dtype=bool)

    # Check for i <- i+3 bonds (3_10)
    if n_residues >= 4:
        donor_indices_3 = jnp.arange(3, n_residues)
        acceptor_indices_3 = donor_indices_3 - 3
        # Check valid_hbonds[donor, acceptor]
        bonds_3_10 = valid_hbonds[donor_indices_3, acceptor_indices_3]
        is_3_10_hbond = is_3_10_hbond.at[acceptor_indices_3].set(bonds_3_10)

    # Check for i <- i+4 bonds (alpha)
    if n_residues >= 5:
        donor_indices_4 = jnp.arange(4, n_residues)
        acceptor_indices_4 = donor_indices_4 - 4
        bonds_alpha = valid_hbonds[donor_indices_4, acceptor_indices_4]
        is_alpha_hbond = is_alpha_hbond.at[acceptor_indices_4].set(bonds_alpha)

    # Check for i <- i+5 bonds (pi)
    if n_residues >= 6:
        donor_indices_5 = jnp.arange(5, n_residues)
        acceptor_indices_5 = donor_indices_5 - 5
        bonds_pi = valid_hbonds[donor_indices_5, acceptor_indices_5]
        is_pi_hbond = is_pi_hbond.at[acceptor_indices_5].set(bonds_pi)

    return is_3_10_hbond, is_alpha_hbond, is_pi_hbond

def identify_pp_bend_candidates(phi: jnp.ndarray, psi: jnp.ndarray, kappa: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Identifies residues potentially in Polyproline II helix or bends based on angles.

    Args:
        phi, psi, kappa: Arrays of backbone angles.

    Returns:
        Tuple of boolean arrays (is_pp_candidate, is_bend_candidate).
    """
    # Check if Phi/Psi are within the Polyproline II helix range
    # Handle potential NaNs in phi/psi: nan comparisons are False
    is_pp_candidate = (phi >= kPPHelixPhiMin) & (phi <= kPPHelixPhiMax) & \
                      (psi >= kPPHelixPsiMin) & (psi <= kPPHelixPsiMax)

    # Check if Kappa angle indicates a bend
    # Handle potential NaNs in kappa: nan > threshold is False
    is_bend_candidate = kappa > kBendKappaThreshold

    return is_pp_candidate, is_bend_candidate

# --- Secondary Structure Assignment Scan --- #

def assign_secondary_structure_scan_body(state, inputs):
    """Body function for jax.lax.scan to assign SS type iteratively.

    Processes one residue at a time, carrying forward state about running
    helix lengths and whether the previous residue ended a helix.

    Args:
        state: Tuple containing state from the previous residue (i-1):
               (prev_ss_type_val, helix_3_len, helix_a_len, helix_p_len, pp_len,
                prev_h3_end, prev_ha_end, prev_hp_end)
        inputs: Tuple containing data for the current residue (i):
                (current_beta_ss_type_val, hbond_3, hbond_4, hbond_5,
                 pp_candidate, bend_candidate)

    Returns:
        Tuple (next_state, output) where:
        next_state: Updated state for the next iteration (i+1).
        output: The assigned SecondaryStructureType value for residue i.
    """
    # Unpack state from previous iteration (i-1)
    (prev_ss_type_val, # SS type assigned to residue i-1 (as int value)
     helix_3_len, helix_a_len, helix_p_len, pp_len, # Running lengths
     prev_h3_end, prev_ha_end, prev_hp_end # Flags if helix ended at i-1
     ) = state

    # Unpack inputs for current residue i
    (current_beta_ss_type_val, # SS type from beta assignment (B/E/LOOP) (int)
     hbond_3, hbond_4, hbond_5, # H-bond patterns ACCEPTED by i (i <- i+k)
     pp_candidate, bend_candidate # Angle-based flags for i
     ) = inputs

    # --- Determine SS type for residue i --- #
    # Start with the Beta assignment (highest priority: E, B)
    current_ss_type_val = current_beta_ss_type_val

    # --- Helix Assignment (priority: Pi > Alpha > 3_10) --- #
    # Update running lengths based on H-bonds ACCEPTED by residue i
    new_helix_3_len = jnp.where(hbond_3, helix_3_len + 1, 0)
    new_helix_a_len = jnp.where(hbond_4, helix_a_len + 1, 0)
    new_helix_p_len = jnp.where(hbond_5, helix_p_len + 1, 0)

    # Check minimum length criteria for helices *ending at i*
    # DSSP definitions: 3_10 needs length >= 1, Alpha/Pi need length >= 2
    is_pi_helix_at_i = new_helix_p_len >= 2
    is_alpha_helix_at_i = new_helix_a_len >= 2
    is_3_10_helix_at_i = new_helix_3_len >= 1

    # Determine the highest priority helix type satisfied at residue i
    ss_if_helix = SecondaryStructureType.LOOP.value # Default if no helix
    ss_if_helix = jnp.where(is_3_10_helix_at_i, SecondaryStructureType.HELIX_3_10.value, ss_if_helix)
    ss_if_helix = jnp.where(is_alpha_helix_at_i, SecondaryStructureType.HELIX_ALPHA.value, ss_if_helix)
    ss_if_helix = jnp.where(is_pi_helix_at_i, SecondaryStructureType.HELIX_PI.value, ss_if_helix)

    # Assign helix type only if current assignment is LOOP/TURN/BEND/UNKNOWN
    # Do NOT override existing Beta Strand/Bridge assignments.
    can_assign_helix = (current_ss_type_val == SecondaryStructureType.LOOP.value) | \
                       (current_ss_type_val == SecondaryStructureType.TURN.value) | \
                       (current_ss_type_val == SecondaryStructureType.BEND.value) | \
                       (current_ss_type_val == SecondaryStructureType.UNKNOWN.value)

    current_ss_type_val = jnp.where(can_assign_helix & (ss_if_helix != SecondaryStructureType.LOOP.value),
                                    ss_if_helix, current_ss_type_val)

    # --- Polyproline Helix Assignment --- #
    # Update running length for PP helix based on angle criteria for residue i
    new_pp_len = jnp.where(pp_candidate, pp_len + 1, 0)
    # Check if minimum length is met ending at i
    is_pp_helix_at_i = new_pp_len >= kMinPPHelixLength

    # Assign PP only if current assignment is LOOP/TURN/BEND/UNKNOWN and length criterion met
    can_assign_pp = (current_ss_type_val == SecondaryStructureType.LOOP.value) | \
                    (current_ss_type_val == SecondaryStructureType.TURN.value) | \
                    (current_ss_type_val == SecondaryStructureType.BEND.value) | \
                    (current_ss_type_val == SecondaryStructureType.UNKNOWN.value)

    current_ss_type_val = jnp.where(can_assign_pp & is_pp_helix_at_i,
                                    SecondaryStructureType.HELIX_PP.value, current_ss_type_val)

    # --- Turn & Bend Assignment (lower priority) --- #

    # Turn (T): Assign if current assignment is LOOP/UNKNOWN and previous residue ended a helix
    # (i.e., previous had length > 0, current has length 0 for that helix type)
    # Note: DSSP defines Turn based on H-bonds, but often occurs after helices.
    # This simplified logic assigns Turn if *any* standard helix ended at i-1.
    is_turn_candidate = prev_h3_end | prev_ha_end | prev_hp_end
    can_assign_turn = (current_ss_type_val == SecondaryStructureType.LOOP.value) | \
                      (current_ss_type_val == SecondaryStructureType.UNKNOWN.value)

    current_ss_type_val = jnp.where(can_assign_turn & is_turn_candidate,
                                    SecondaryStructureType.TURN.value, current_ss_type_val)

    # Bend (S): Assign if current assignment is LOOP/UNKNOWN and kappa angle indicates bend
    can_assign_bend = (current_ss_type_val == SecondaryStructureType.LOOP.value) | \
                      (current_ss_type_val == SecondaryStructureType.UNKNOWN.value)
    is_bend = can_assign_bend & bend_candidate
    current_ss_type_val = jnp.where(is_bend, SecondaryStructureType.BEND.value, current_ss_type_val)

    # --- Prepare State for Next Iteration (i+1) --- #
    # Determine if any helices *end* at the current residue i
    # This happens if length was > 0 at i-1, but the H-bond pattern breaks at i
    h3_ends_here = (helix_3_len > 0) & (new_helix_3_len == 0)
    ha_ends_here = (helix_a_len > 0) & (new_helix_a_len == 0)
    hp_ends_here = (helix_p_len > 0) & (new_helix_p_len == 0)

    # Pack state for the next residue (i+1)
    next_state = (current_ss_type_val, # Pass current type value for next iter
                  new_helix_3_len, new_helix_a_len, new_helix_p_len, new_pp_len,
                  h3_ends_here, ha_ends_here, hp_ends_here # Pass end flags
                  )

    # Output for this residue i is its assigned SS type value
    output = current_ss_type_val

    return next_state, output

def assign_secondary_structure(
    chain: ChainPytree,          # Input chain *after* beta assignment
    valid_hbonds: jnp.ndarray   # Boolean H-bond matrix
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assigns secondary structure (Helices, Turns, Bends) using jax.lax.scan.

    Takes the chain with pre-assigned Beta Strand/Bridge (E/B) structures
    and applies rules for G, H, I, P, T, S assignments based on H-bonds and angles.

    Args:
        chain: ChainPytree *after* beta assignment (from assemble_beta_...).
        valid_hbonds: Boolean matrix indicating valid H-bonds.

    Returns:
        Tuple containing JAX arrays for:
          - ss_types: Final secondary structure assignments (integer enum values).
          - phi, psi, kappa: Calculated backbone angles.
    """
    n_res = len(chain)
    if n_res == 0:
        empty_i = jnp.array([], dtype=jnp.int32)
        empty_f = jnp.array([], dtype=jnp.float64)
        return empty_i, empty_f, empty_f, empty_f # ss, phi, psi, kappa

    # 1. Calculate geometric features (Phi, Psi, Kappa)
    phi, psi, kappa = calculate_phi_psi_kappa(chain)

    # 2. Identify structural patterns based on H-bonds and angles
    hbond_3, hbond_4, hbond_5 = identify_helix_patterns(n_res, valid_hbonds)
    pp_candidate, bend_candidate = identify_pp_bend_candidates(phi, psi, kappa)

    # 3. Get initial SS types (Beta assignments B/E or initial LOOP)
    initial_ss_types = jnp.array([res['secondary_structure'].value for res in chain])

    # 4. Define initial state for the scan
    # (prev_ss, h3_len, ha_len, hp_len, pp_len, prev_h3_end, prev_ha_end, prev_hp_end)
    initial_state = (
        SecondaryStructureType.LOOP.value, # Assume previous SS before first residue is Loop
        0, 0, 0, 0,                       # Initial helix lengths
        False, False, False              # Initial helix end flags
    )

    # 5. Prepare inputs for scan (pack per-residue info into tuple of arrays)
    scan_inputs = (
        initial_ss_types, hbond_3, hbond_4, hbond_5,
        pp_candidate, bend_candidate
    )

    # 6. Run the scan
    # The scan iterates through residues, applying assign_secondary_structure_scan_body
    _, ss_type_outputs = jax.lax.scan(
        assign_secondary_structure_scan_body, initial_state, scan_inputs
    )

    # 7. Return the final SS assignments and calculated angles
    return ss_type_outputs, phi, psi, kappa


# --- Post-processing and Pytree Updates --- #

def update_pytree_final_ss_angles(
    chain: ChainPytree,
    ss_types: jnp.ndarray,
    phi: jnp.ndarray,
    psi: jnp.ndarray,
    kappa: jnp.ndarray
    ) -> ChainPytree:
    """Updates the ChainPytree with final SS assignments and calculated angles.

    Args:
        chain: Input ChainPytree (typically after beta assignment).
        ss_types: Array of final SS type enum values from the scan.
        phi, psi, kappa: Arrays of calculated angles.

    Returns:
        Updated ChainPytree with 'secondary_structure', 'phi', 'psi', 'kappa' fields updated.
    """
    updated_chain = []
    for i, res in enumerate(chain):
        updated_res = {**res} # Start with existing data
        # Update SS type from the main scan result
        updated_res['secondary_structure'] = SecondaryStructureType(int(ss_types[i]))
        # Update calculated angles
        updated_res['phi'] = float(phi[i])
        updated_res['psi'] = float(psi[i])
        updated_res['kappa'] = float(kappa[i])
        updated_chain.append(updated_res)
    return updated_chain

def refine_helix_flags(chain: ChainPytree) -> ChainPytree:
    """Refines helix flags (START, MIDDLE, END, START_END) based on final SS assignments.

    Iterates through the chain with final SS assignments and determines the role
    of each helical residue within its helix segment.

    Args:
        chain: ChainPytree with final 'secondary_structure' assignments.

    Returns:
        Updated ChainPytree with helix_X_flag fields populated.
    """
    # log.info("--- Refining Helix Flags (Start/End/Middle) ---") # Covered in main.py step log
    n_res = len(chain)
    if n_res == 0:
        return chain

    # Extract final SS types
    ss_types = jnp.array([res['secondary_structure'].value for res in chain])

    # Define helper function to check if a type is any helix type (G, H, I, P)
    @jax.vmap
    def is_helix_type(ss_type_val):
        return ((ss_type_val == SecondaryStructureType.HELIX_3_10.value) |
                (ss_type_val == SecondaryStructureType.HELIX_ALPHA.value) |
                (ss_type_val == SecondaryStructureType.HELIX_PI.value) |
                (ss_type_val == SecondaryStructureType.HELIX_PP.value))

    is_helix_curr = is_helix_type(ss_types)

    # Get SS types of previous/next residues, handling boundaries
    ss_prev = jnp.roll(ss_types, shift=1)
    ss_next = jnp.roll(ss_types, shift=-1)
    # Assume non-helix neighbors at boundaries
    ss_prev = ss_prev.at[0].set(SecondaryStructureType.LOOP.value)
    ss_next = ss_next.at[n_res - 1].set(SecondaryStructureType.LOOP.value)

    is_helix_prev = is_helix_type(ss_prev)
    is_helix_next = is_helix_type(ss_next)

    # Determine flags based on transitions into/out of helical states
    is_start = is_helix_curr & (~is_helix_prev)
    is_end = is_helix_curr & (~is_helix_next)
    is_middle = is_helix_curr & is_helix_prev & is_helix_next
    is_start_end = is_start & is_end # Single residue helix

    # Assign flags based on conditions (priority: START_END > START/END > MIDDLE > NONE)
    flags = jnp.full(n_res, HelixFlagType.NONE.value, dtype=jnp.int32)
    # Assign MIDDLE first, then overwrite with START/END, then START_END
    flags = jnp.where(is_middle, HelixFlagType.MIDDLE.value, flags)
    flags = jnp.where(is_start & ~is_start_end, HelixFlagType.START.value, flags) # Start but not end
    flags = jnp.where(is_end & ~is_start_end, HelixFlagType.END.value, flags)     # End but not start
    flags = jnp.where(is_start_end, HelixFlagType.START_END.value, flags)       # Start and end

    # Update the specific helix flag fields in the Pytree
    updated_chain = []
    for i, res in enumerate(chain):
        updated_res = {**res}
        current_flag_val = flags[i]
        current_ss_type = SecondaryStructureType(ss_types[i])

        # Reset all flags first
        updated_res['helix_3_10_flag'] = HelixFlagType.NONE
        updated_res['helix_alpha_flag'] = HelixFlagType.NONE
        updated_res['helix_pi_flag'] = HelixFlagType.NONE
        updated_res['helix_pp_flag'] = HelixFlagType.NONE

        # Assign the calculated flag to the *correct* helix type field
        if current_ss_type == SecondaryStructureType.HELIX_3_10:
            updated_res['helix_3_10_flag'] = HelixFlagType(current_flag_val)
        elif current_ss_type == SecondaryStructureType.HELIX_ALPHA:
            updated_res['helix_alpha_flag'] = HelixFlagType(current_flag_val)
        elif current_ss_type == SecondaryStructureType.HELIX_PI:
            updated_res['helix_pi_flag'] = HelixFlagType(current_flag_val)
        elif current_ss_type == SecondaryStructureType.HELIX_PP:
            updated_res['helix_pp_flag'] = HelixFlagType(current_flag_val)

        updated_chain.append(updated_res)

    return updated_chain 