# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2020 NKI/AVL, Netherlands Cancer Institute
# Substantial portions of this code are derived from the DSSP C++ implementation
# by Maarten L. Hekkelman and contributors, licensed under the BSD-2-Clause license.
# Full C++ source and license available at: https://github.com/PDB-REDO/dssp
"""Main orchestration logic for the DSSP-JAX pipeline."""

import jax
import jax.numpy as jnp
import logging # Import logging
from typing import Optional, List # Added Optional, List

# Relative imports for different pipeline stages
from .io import load_cif_data, create_pytree_structure
from .types import ChainPytree, BridgeType # Import needed enums/types
from .hbonds import (
    calculate_hydrogen_positions,
    calculate_all_hbond_energies_jax,
    find_best_hbond_partners,
    update_pytree_hbond_partners,
    calculate_bridge_types
)
from .beta import assemble_ladders_sheets_and_assign_beta
from .secondary_structure import (
    assign_secondary_structure,
    update_pytree_final_ss_angles,
    refine_helix_flags
)
from .accessibility import calculate_accessibility
from .constants import kMinHBondEnergy, kMaxHBondEnergy # Import needed constants

# Get the logger instance configured in __init__
log = logging.getLogger("dsspjax")

def run_dssp(
    cif_url_or_file: str,
    model_num: int = 1,
    target_ligand_keys: Optional[List[str]] = None
) -> ChainPytree:
    """Runs the full DSSP-JAX calculation pipeline on a given CIF input.

    Args:
        cif_url_or_file: Path or URL to the input mmCIF file.
        model_num: The model number to process from the CIF file.
        target_ligand_keys: Optional list of ligand identifiers to include
                            in the environment for SASA calculation
                            (format: ["ChainID:ResNum:CompID"]).

    Returns:
        The final ChainPytree containing all calculated DSSP information.

    Raises:
        ValueError: If no residues are loaded or other processing errors occur.
        FileNotFoundError: If `cif_url_or_file` is a local path and not found.
        RuntimeError: If downloading from a URL fails.
    """
    # --- JAX Configuration --- #
    # Ensure JAX is using float64 if needed (can be set globally or locally)
    # Consider making this configurable or documenting the requirement.
    # log.debug("Attempting to enable float64") # Already done in __init__
    # jax.config.update("jax_enable_x64", True)

    # --- Pipeline Stages --- #

    # 1. Load Data (Handles download/local read internally now)
    # Load protein residues and optionally specified ligand atoms
    residue_info_list, all_atom_data_list, ligand_atoms_list = load_cif_data(
        cif_url_or_file, model_num, target_ligand_keys=target_ligand_keys
    )
    # Check if protein residues were found
    if not residue_info_list and not ligand_atoms_list:
        raise ValueError("No valid protein residues or target ligand atoms found in the input.")
    # Allow processing if only ligands were requested/found, although ChainPytree will be empty.
    elif not residue_info_list:
        log.warning("No valid protein residues found, processing may be limited.")
        # Handle case where only ligand SASA might be relevant (future step)

    # 2. Create Initial Pytree Structure
    protein_chain = create_pytree_structure(residue_info_list, all_atom_data_list)
    n_residues = len(protein_chain)
    if n_residues == 0:
        # This case might occur if create_pytree_structure filters all loaded residues
        log.warning("Pytree creation resulted in an empty chain.")
        return protein_chain # Return the empty list
    # log.info(f"\nCreated Pytree for {n_residues} residues.") # Covered in create_pytree_structure

    # 3. Calculate Amide Hydrogen Positions
    log.info("[bold cyan]Step 3: Geometry Refinement[/] - Calculating H positions...", extra={"markup": True})
    protein_chain = calculate_hydrogen_positions(protein_chain)
    log.info("--> Calculated approximate amide hydrogen positions.")

    # 4. Calculate H-bond Energies & Find Best Partners
    log.info("\n[bold cyan]Step 4: H-Bond Calculation[/]", extra={"markup": True})
    # Stack coordinates needed for H-bond calculation
    all_coords_n = jnp.stack([r['bb_coords'].N for r in protein_chain])
    all_coords_h = jnp.stack([r['bb_coords'].H for r in protein_chain])
    all_coords_c = jnp.stack([r['bb_coords'].C for r in protein_chain])
    all_coords_o = jnp.stack([r['bb_coords'].O for r in protein_chain])

    hbond_energy_matrix = calculate_all_hbond_energies_jax(
        all_coords_n, all_coords_h, all_coords_c, all_coords_o
    )
    # Define valid H-bonds based on energy range
    valid_hbonds = (hbond_energy_matrix < kMaxHBondEnergy) & \
                   (hbond_energy_matrix >= kMinHBondEnergy) # Note: >= for min energy
    num_valid_hbonds = jnp.sum(valid_hbonds).item()
    log.info(f"--> Calculated H-bond energy matrix. Found [b]{num_valid_hbonds}[/] potential H-bonds.", extra={"markup": True})

    hbond_partners = find_best_hbond_partners(hbond_energy_matrix, valid_hbonds)
    protein_chain = update_pytree_hbond_partners(protein_chain, hbond_partners)
    log.info("--> Identified and stored best H-bond partners for each residue.")

    # 5. Calculate Beta Bridge Types
    log.info("\n[bold cyan]Step 5: Beta Structure Assignment[/]", extra={"markup": True})
    bridge_type_matrix = calculate_bridge_types(n_residues, valid_hbonds)
    num_parallel = jnp.sum(bridge_type_matrix == BridgeType.PARALLEL.value).item()
    num_antiparallel = jnp.sum(bridge_type_matrix == BridgeType.ANTIPARALLEL.value).item()
    log.info(f"--> Calculated Bridge Types: Found [b]{num_parallel}[/] parallel, [b]{num_antiparallel}[/] antiparallel.", extra={"markup": True})

    # 6. Assemble Ladders, Sheets and Assign B/E Secondary Structure
    protein_chain_beta = assemble_ladders_sheets_and_assign_beta(
        bridge_type_matrix, protein_chain
    )

    # 7. Assign Helical/Turn/Bend Structures (Main Scan)
    log.info("\n[bold cyan]Step 6: Helical/Turn/Bend Assignment[/]", extra={"markup": True})
    # This function calculates angles internally and runs the scan
    ss_types, phi, psi, kappa = assign_secondary_structure(
        protein_chain_beta, valid_hbonds
    )
    log.info("--> Completed main secondary structure assignment scan.")

    # 8. Update Pytree with SS types and Angles from scan
    # This step merges the SS assignments from the scan and the calculated angles
    # into the Pytree that already contains beta structure info.
    protein_chain_ss = update_pytree_final_ss_angles(
        protein_chain_beta, ss_types, phi, psi, kappa
    )

    # 9. Refine Helix Flags (Start/Middle/End)
    # This needs the final SS assignments from the previous step.
    protein_chain_refined = refine_helix_flags(protein_chain_ss)
    log.info("--> Refined helix start/middle/end flags.")

    # 10. Calculate Accessibility (SASA)
    # Pass the processed protein chain and the extracted ligand atoms
    protein_chain_final = calculate_accessibility(
        protein_chain_refined, # Contains protein atoms
        ligand_atoms=ligand_atoms_list # Pass ligand atoms separately
    )
    # Accessibility calculation prints its own completion message.

    log.info("\n[bold green]:heavy_check_mark: DSSP-JAX Pipeline Complete[/]", extra={"markup": True})
    return protein_chain_final 