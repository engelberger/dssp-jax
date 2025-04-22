# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2020 NKI/AVL, Netherlands Cancer Institute
# Substantial portions of this code are derived from the DSSP C++ implementation
# by Maarten L. Hekkelman and contributors, licensed under the BSD-2-Clause license.
# Full C++ source and license available at: https://github.com/PDB-REDO/dssp
"""Physical and algorithmic constants used in DSSP-JAX."""

import jax.numpy as jnp

# --- Configuration ---
# Note: JAX configuration should ideally be handled by the user application,
# but setting it here ensures consistency if this module is imported early.
# Consider adding a note about this or moving it to a config function.
# import jax
# jax.config.update("jax_enable_x64", True)

# --- DSSP Algorithm Constants ---
kMinimalCADistanceSq = 9.0 * 9.0
kCouplingConstant = -27.888
kMinimalDistance = 0.5
kMinHBondEnergy = -9.9
kMaxHBondEnergy = -0.5
kPeptideBondLength = 1.33 # Max C(i)-N(i+1) distance for chain continuity
kCODistance = 1.23 # Typical C=O bond length

# --- Polyproline (PP) Helix Constants ---
kMinPPHelixLength = 3 # Default minimum length for PP helix assignment
kPPHelixPhiMin = -75.0 - 29.0
kPPHelixPhiMax = -75.0 + 29.0
kPPHelixPsiMin = 145.0 - 29.0
kPPHelixPsiMax = 145.0 + 29.0

# --- Bend Constant ---
kBendKappaThreshold = 70.0 # Minimum C-alpha angle for bend ('S') assignment

# --- Placeholder/Invalid Value Constants ---
kInvalidHBondEnergy = jnp.inf # Energy value for non-existent/invalid H-bonds

# --- Accessibility (SASA) Constants ---
kRadiusWater = 1.40 # Water probe radius

# Van der Waals radii (in Angstroms)
# Specific backbone radii potentially matching original DSSP values
kRadiusN = 1.65
kRadiusCA = 1.87 # C-alpha
kRadiusC = 1.76 # Backbone C=O Carbon
kRadiusO = 1.40 # Backbone C=O Oxygen

# General element radii for sidechains and other atoms
element_radii = {
    'H': 1.10, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'P': 1.80,
    'F': 1.47, 'CL': 1.75, 'BR': 1.85, 'I': 1.98,
    # Common ions/metals often found in PDB files
    'FE': 1.4, 'MG': 1.73, 'CA': 1.97, 'ZN': 1.39, 'MN': 1.4, 'CU': 1.4,
    'NA': 2.27, 'K': 2.75, 'SE': 1.90 # For Selenomethionine
}
# Fallback radius for elements not in the dictionary
kRadiusDefault = 1.80 