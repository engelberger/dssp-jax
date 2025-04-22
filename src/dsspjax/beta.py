# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2020 NKI/AVL, Netherlands Cancer Institute
# Substantial portions of this code are derived from the DSSP C++ implementation
# by Maarten L. Hekkelman and contributors, licensed under the BSD-2-Clause license.
# Full C++ source and license available at: https://github.com/PDB-REDO/dssp
"""Beta sheet ladder and sheet assembly logic."""

import jax.numpy as jnp
import collections
from typing import List, Dict, Set
import logging # Import logging

# Relative imports from within the package
from .types import ChainPytree, BridgeType, SecondaryStructureType

# Get the logger instance
log = logging.getLogger("dsspjax")

# --- Helper Classes and Functions ---

class Ladder:
    """Helper class to store information about a single beta ladder.

    A ladder represents a series of consecutive residues forming bridges
    of the same type (parallel or antiparallel) with another strand segment.
    """
    def __init__(self, ladder_id: int, bridge_type: BridgeType, i: int, j: int):
        self.ladder_id: int = ladder_id
        self.sheet_id: int = -1 # Assigned during sheet assembly
        self.type: BridgeType = bridge_type
        # Store residue indices involved in the ladder
        self.i_indices: List[int] = [i] # Indices on one side of the ladder
        self.j_indices: List[int] = [j] # Indices on the other side
        self.length: int = 1

    def extend_parallel(self, i: int, j: int):
        """Extends a parallel ladder by adding residues i and j."""
        self.i_indices.append(i)
        self.j_indices.append(j)
        self.length += 1

    def extend_antiparallel(self, i: int, j: int):
        """Extends an antiparallel ladder by adding residues i and j.

        Note: For antiparallel, j indices are effectively added in reverse order
        relative to i indices when considering sequence progression.
        We prepend `j` to maintain the pairing relative to `i` indices.
        """
        self.i_indices.append(i)
        # Prepend j to keep i[k] paired with j[k] correctly for the bridge
        self.j_indices.insert(0, j)
        self.length += 1

    def __repr__(self):
        return (
            f"Ladder(id={self.ladder_id}, sheet={self.sheet_id}, "
            f"type={self.type.name}, len={self.length}, "
            f"i={self.i_indices}, j={self.j_indices})"
        )

def _check_ladders_linked(ladder1: Ladder, ladder2: Ladder) -> bool:
    """Checks if two ladders are linked (share any residue index)."""
    # Use sets for efficient intersection checking
    set1_residues = set(ladder1.i_indices) | set(ladder1.j_indices)
    set2_residues = set(ladder2.i_indices) | set(ladder2.j_indices)

    return not set1_residues.isdisjoint(set2_residues)

# --- Main Beta Structure Assembly ---

def assemble_ladders_sheets_and_assign_beta(
    bridge_matrix: jnp.ndarray, # Shape (N, N), BridgeType integer values
    chain: ChainPytree          # Input chain Pytree
    ) -> ChainPytree:
    """Identifies beta ladders and sheets, assigns structure types B/E.

    This function implements the logic to:
    1. Identify consecutive beta bridges to form ladders.
    2. Group linked ladders into sheets using a breadth-first search.
    3. Assign ladder, partner, and sheet IDs to residues.
    4. Assign SecondaryStructureType BETA_BRIDGE (B) or BETA_STRAND (E)
       based on ladder participation and length.

    Args:
        bridge_matrix: N x N matrix from `calculate_bridge_types`.
        chain: The input ChainPytree (usually after H-bond partner assignment).

    Returns:
        The updated ChainPytree with beta structure information populated:
        - `beta_partner_1`, `beta_partner_2`: Indices of bridge partners.
        - `ladder_1`, `ladder_2`: IDs of ladders the residue participates in.
        - `is_parallel_1`, `is_parallel_2`: Type of the corresponding ladder.
        - `sheet_id`: ID of the sheet the residue belongs to.
        - `secondary_structure`: Updated to BETA_BRIDGE or BETA_STRAND if applicable.
    """
    # log.info("--- Assembling Beta Ladders and Sheets ---") # Covered in main.py step log
    n_res = len(chain)
    if n_res == 0:
        return chain

    ladders: List[Ladder] = []
    ladder_id_counter = 0

    # --- 1. Identify and Extend Ladders --- #
    # Iterate through the bridge matrix to find potential ladder starts/extensions.
    log.info("--> Identifying and extending ladders...")
    # Keep track of bridges already assigned to a ladder to avoid redundant checks
    assigned_bridges = set() # Store tuples (min(i,j), max(i,j), type)

    for i in range(n_res):
        for j in range(i + 1, n_res): # Only need to check upper triangle (i < j)
            bridge_type_ij = BridgeType(int(bridge_matrix[i, j]))
            bridge_type_ji = BridgeType(int(bridge_matrix[j, i]))

            # Determine the primary bridge type (prefer Parallel if both exist?)
            bridge_type = BridgeType.NONE
            if bridge_type_ij != BridgeType.NONE:
                bridge_type = bridge_type_ij
            elif bridge_type_ji != BridgeType.NONE:
                bridge_type = bridge_type_ji # Should be same type if bridge exists
                # Swap i, j to canonical order for assignment check? Let's base on i,j pair.

            if bridge_type != BridgeType.NONE:
                bridge_key = (i, j, bridge_type)
                if bridge_key in assigned_bridges:
                    continue

                found_extension = False
                # Attempt to extend existing ladders (iterate backwards for efficiency?)
                for k in range(len(ladders) - 1, -1, -1):
                    existing_ladder = ladders[k]
                    if existing_ladder.type != bridge_type:
                        continue

                    # Check if (i, j) can extend this ladder
                    last_i = existing_ladder.i_indices[-1]
                    first_j = existing_ladder.j_indices[0]
                    last_j = existing_ladder.j_indices[-1]

                    if i == last_i + 1: # Check if i continues the i-strand
                        # Parallel Check: j should continue the j-strand
                        if bridge_type == BridgeType.PARALLEL and j == last_j + 1:
                            existing_ladder.extend_parallel(i, j)
                            assigned_bridges.add(bridge_key)
                            found_extension = True
                            break
                        # Antiparallel Check: j should precede the first j
                        elif bridge_type == BridgeType.ANTIPARALLEL and j == first_j - 1:
                            existing_ladder.extend_antiparallel(i, j)
                            assigned_bridges.add(bridge_key)
                            found_extension = True
                            break

                # Create New Ladder if no extension was found
                if not found_extension:
                    new_ladder = Ladder(ladder_id_counter, bridge_type, i, j)
                    ladders.append(new_ladder)
                    assigned_bridges.add(bridge_key)
                    ladder_id_counter += 1

    # Filter out single-bridge ladders (length 1) as they only contribute 'B'
    # Keep them for sheet assignment, but track lengths for 'E' assignment.
    ladder_lengths = {lad.ladder_id: lad.length for lad in ladders}
    # log.info(f"  Found {len(ladders)} potential ladders (including length 1)." )

    # --- 2. Group Ladders into Sheets --- #
    # Use BFS (Breadth-First Search) to find connected components (sheets).
    log.info("--> Grouping ladders into sheets...")
    sheet_id_counter = 0
    unassigned_ladder_ids: Set[int] = set(lad.ladder_id for lad in ladders)
    ladder_map: Dict[int, Ladder] = {lad.ladder_id: lad for lad in ladders}

    while unassigned_ladder_ids:
        current_sheet_id = sheet_id_counter
        sheet_id_counter += 1
        queue = collections.deque()

        # Start BFS from an arbitrary unassigned ladder
        start_ladder_id = unassigned_ladder_ids.pop()
        ladder_map[start_ladder_id].sheet_id = current_sheet_id
        queue.append(start_ladder_id)

        while queue:
            current_ladder_id = queue.popleft()
            current_ladder = ladder_map[current_ladder_id]

            # Find neighbors among remaining unassigned ladders
            neighbors_to_check = list(unassigned_ladder_ids) # Avoid modifying set during iteration
            for neighbor_id in neighbors_to_check:
                 if neighbor_id in unassigned_ladder_ids: # Check if still unassigned
                    neighbor_ladder = ladder_map[neighbor_id]
                    if _check_ladders_linked(current_ladder, neighbor_ladder):
                        neighbor_ladder.sheet_id = current_sheet_id
                        unassigned_ladder_ids.remove(neighbor_id)
                        queue.append(neighbor_id)

    # log.info(f"  Grouped ladders into {sheet_id_counter} sheets.")

    # --- 3. Assign Partner/Ladder/Sheet Info to Residues --- #
    # Initialize assignment dictionary for each residue
    # Use lists for partners/ladders to store up to two
    residue_partners = [[] for _ in range(n_res)] # Stores (partner_idx, ladder_id, is_parallel)
    residue_sheet_id = [-1] * n_res

    log.info("--> Assigning partners and ladder/sheet IDs to residues...")
    for ladder in ladders:
        is_parallel = (ladder.type == BridgeType.PARALLEL)
        ladder_id = ladder.ladder_id
        sheet_id = ladder.sheet_id

        # Iterate through the pairs of residues forming the bridges in the ladder
        for res_i, res_j in zip(ladder.i_indices, ladder.j_indices):
            # Assign partner j to residue i
            if len(residue_partners[res_i]) < 2:
                residue_partners[res_i].append((res_j, ladder_id, is_parallel))
                residue_sheet_id[res_i] = sheet_id
            # Assign partner i to residue j
            if len(residue_partners[res_j]) < 2:
                residue_partners[res_j].append((res_i, ladder_id, is_parallel))
                residue_sheet_id[res_j] = sheet_id

    # --- 4. Update Pytree and Assign B/E Structure --- #
    log.info("--> Assigning B/E types and updating final Pytree...")
    updated_chain = []
    for i, res in enumerate(chain):
        updated_res = {**res} # Copy existing data
        partners = residue_partners[i]
        sheet_id = residue_sheet_id[i]

        # Update partner/ladder/sheet info
        updated_res['sheet_id'] = sheet_id
        if len(partners) > 0:
            updated_res['beta_partner_1'] = partners[0][0]
            updated_res['ladder_1'] = partners[0][1]
            updated_res['is_parallel_1'] = partners[0][2]
        if len(partners) > 1:
            updated_res['beta_partner_2'] = partners[1][0]
            updated_res['ladder_2'] = partners[1][1]
            updated_res['is_parallel_2'] = partners[1][2]

        # Assign B/E type - only if part of a sheet
        if sheet_id != -1:
            is_strand = False
            # Check if involved in any ladder longer than 1
            for _, ladder_id, _ in partners:
                if ladder_lengths.get(ladder_id, 0) > 1:
                    is_strand = True
                    break

            if is_strand:
                updated_res['secondary_structure'] = SecondaryStructureType.BETA_STRAND # E
            else:
                # If part of a sheet but only in ladders of length 1, it's a Bridge
                updated_res['secondary_structure'] = SecondaryStructureType.BETA_BRIDGE # B
        # else: Keep the initial SS type (usually LOOP)

        updated_chain.append(updated_res)

    log.info(f"--> Beta Structure Assembly Complete. Found [b]{sheet_id_counter}[/] sheets.", extra={"markup": True})
    return updated_chain 