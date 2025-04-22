# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2020 NKI/AVL, Netherlands Cancer Institute
# Substantial portions of this code are derived from the DSSP C++ implementation
# by Maarten L. Hekkelman and contributors, licensed under the BSD-2-Clause license.
# Full C++ source and license available at: https://github.com/PDB-REDO/dssp
"""Data structures, Enums, and Type hints for DSSP-JAX."""

import jax.numpy as jnp
from typing import List, Dict, Tuple, Any, NamedTuple
from enum import IntEnum

# --- Enums ---

class BridgeType(IntEnum):
    """Indicates the type of beta bridge between two residues."""
    NONE = 0
    PARALLEL = 1
    ANTIPARALLEL = 2

class SecondaryStructureType(IntEnum):
    """Represents the DSSP secondary structure classification."""
    LOOP = 0        # L / ' '
    HELIX_3_10 = 1  # G
    HELIX_ALPHA = 2 # H
    HELIX_PI = 3    # I
    HELIX_PP = 4    # P (Polyproline II)
    TURN = 5        # T
    BEND = 6        # S
    BETA_BRIDGE = 7 # B (Single residue beta bridge)
    BETA_STRAND = 8 # E (Extended strand in beta ladder)
    UNKNOWN = 9     # ? (Should not occur in final assignment)

    def to_char(self) -> str:
        """Convert the enum member to its single-character DSSP code."""
        mapping = {
            self.LOOP: ' ', self.HELIX_3_10: 'G', self.HELIX_ALPHA: 'H',
            self.HELIX_PI: 'I', self.HELIX_PP: 'P', self.TURN: 'T',
            self.BEND: 'S', self.BETA_BRIDGE: 'B', self.BETA_STRAND: 'E',
            self.UNKNOWN: '?'
        }
        return mapping.get(self, ' ') # Default to loop/space

class HelixFlagType(IntEnum):
    """Indicates the role of a residue within a helix (start, middle, end)."""
    NONE = 0
    START = 1
    MIDDLE = 2
    END = 3
    START_END = 4 # Single residue helix

# --- Pytree Data Structures ---

class AtomCoords(NamedTuple):
    """Represents coordinates for standard backbone atoms (N, CA, C, O, H)."""
    N: jnp.ndarray # Shape (3,)
    CA: jnp.ndarray # Shape (3,)
    C: jnp.ndarray # Shape (3,)
    O: jnp.ndarray # Shape (3,)
    H: jnp.ndarray # Shape (3,) - Calculated, can be NaN

# Type alias for the dictionary representing a single residue's data.
# Using TypedDict might offer better static analysis benefits later.
ResiduePytree = Dict[str, Any]
# Example Keys:
#   'res_index': int
#   'res_name': str
#   'seq_id': int
#   'chain_id': str
#   'pdb_ins_code': str
#   'auth_asym_id': str
#   'auth_seq_id': str
#   'is_proline': bool
#   'bb_coords': AtomCoords
#   'sidechain_coords': jnp.ndarray # Shape (N_sc_atoms, 3)
#   'sidechain_radii': jnp.ndarray # Shape (N_sc_atoms,)
#   'secondary_structure': SecondaryStructureType
#   'phi': float
#   'psi': float
#   'kappa': float
#   'accessibility': float
#   'hbond_acceptor_1/2_idx': int
#   'hbond_acceptor_1/2_nrg': float
#   'hbond_donor_1/2_idx': int
#   'hbond_donor_1/2_nrg': float
#   'beta_partner_1/2': int
#   'ladder_1/2': int
#   'is_parallel_1/2': bool
#   'sheet_id': int
#   'helix_3_10_flag': HelixFlagType
#   'helix_alpha_flag': HelixFlagType
#   'helix_pi_flag': HelixFlagType
#   'helix_pp_flag': HelixFlagType

# Type alias for a list of residue dictionaries, representing a protein chain.
ChainPytree = List[ResiduePytree] 