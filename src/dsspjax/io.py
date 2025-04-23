"""Input/Output operations: CIF parsing and DSSP/mmCIF output formatting."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import requests
import shlex
import collections
from datetime import date
import logging # Import logging
import os # Import os for path checking
import datetime # Add datetime for write_dssp_output function

# Relative imports for types and constants within the package
from .types import ChainPytree, ResiduePytree, AtomCoords, SecondaryStructureType, HelixFlagType
from .constants import element_radii, kRadiusDefault, kInvalidHBondEnergy

# Get the logger instance configured in __init__
log = logging.getLogger("dsspjax")

# --- CIF Parsing --- #

def load_cif_data(
    cif_url_or_file: str,
    model_num: int = 1,
    target_ligand_keys: Optional[List[str]] = None
) -> Tuple[List[Dict], List[List[Dict]], List[Dict]]:
    """Downloads (if URL) or reads (if local path) and parses a CIF file.

    Focuses on the _atom_site loop, parsing ATOM records and optionally
    specific HETATM records for target ligands.

    Args:
        cif_url_or_file: URL or local file path to the mmCIF file.
        model_num: The PDB model number to extract (default is 1).
        target_ligand_keys: Optional list of ligand identifiers in the format
                            "ChainID:ResNum:CompID" (e.g., ["B:301:TPA"]).
                            If provided, atoms matching these HETATM records
                            will be extracted.

    Returns:
        A tuple containing:
          - residue_info_list: List of dictionaries for protein residues.
          - all_atom_data_list: List of atom lists for protein residues.
          - ligand_atoms_list: Flat list of dictionaries for atoms belonging
                               to the target ligands.

    Raises:
        RuntimeError: If downloading the CIF file fails.
        ValueError: If a valid _atom_site loop cannot be found or parsed, or if
                    essential columns are missing.
        FileNotFoundError: If cif_url_or_file is a local path and not found.
        IOError: If reading a local file fails.
    """
    log.info(f"[bold cyan]Step 1: Loading Data[/] - Processing input: [i]{cif_url_or_file}[/] (Model {model_num})", extra={"markup": True})
    if target_ligand_keys:
        log.info(f"    Targeting HETATM ligands: {target_ligand_keys}")
    cif_text: str
    is_url = cif_url_or_file.lower().startswith(('http:', 'https:'))

    if is_url:
        log.debug(f"--> Input detected as URL. Downloading...")
        try:
            response = requests.get(cif_url_or_file)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            cif_text = response.text
            log.debug(f"--> Download successful ({len(cif_text)} bytes).")
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to download CIF file from URL: {cif_url_or_file}")
            raise RuntimeError(f"Failed to download CIF file: {e}") from e
    else:
        log.debug(f"--> Input detected as local path. Reading file...")
        if not os.path.exists(cif_url_or_file):
            log.error(f"Input file not found: {cif_url_or_file}")
            raise FileNotFoundError(f"Input file not found: {cif_url_or_file}")
        try:
            with open(cif_url_or_file, 'r') as f:
                cif_text = f.read()
            log.debug(f"--> File read successful ({len(cif_text)} bytes).")
        except IOError as e:
            log.error(f"Failed to read CIF file from path: {cif_url_or_file}")
            raise IOError(f"Failed to read CIF file: {e}") from e

    # --- Start Parsing --- #
    log.info("--> Parsing CIF data (_atom_site loop)...", extra={"markup": True})
    lines = cif_text.splitlines()
    atom_site_keys = []
    atom_data_lines = []
    in_atom_site_loop_data = False
    parsing_keys = False
    current_keys = []

    # Find the _atom_site loop and its keys
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'): continue
        if line.startswith("loop_"):
            # Reset when a new loop starts
            in_atom_site_loop_data = False; parsing_keys = True; current_keys = []
            continue
        if parsing_keys:
            if line.startswith("_"):
                current_keys.append(line)
            else:
                # End of keys, check if it was the _atom_site loop
                parsing_keys = False
                # Define required and optional keys
                required_keys_base = [
                    "_atom_site.group_PDB", "_atom_site.label_atom_id",
                    "_atom_site.label_comp_id", "_atom_site.label_asym_id",
                    "_atom_site.label_seq_id", "_atom_site.Cartn_x",
                    "_atom_site.Cartn_y", "_atom_site.Cartn_z",
                    "_atom_site.pdbx_PDB_model_num", "_atom_site.type_symbol"
                ]
                optional_keys = [
                    "_atom_site.pdbx_PDB_ins_code", "_atom_site.auth_asym_id",
                    "_atom_site.auth_seq_id", "_atom_site.auth_comp_id"
                ]
                required_keys_check = required_keys_base

                if any(key.startswith("_atom_site.") for key in current_keys):
                    key_map_check = {key: idx for idx, key in enumerate(current_keys)}
                    if all(rkey in key_map_check for rkey in required_keys_check):
                        log.debug("--- Found valid _atom_site loop ---")
                        atom_site_keys = current_keys
                        found_optional_keys = {k: key_map_check[k] for k in optional_keys if k in key_map_check}
                        if found_optional_keys:
                            log.debug(f"    Found optional keys: {list(found_optional_keys.keys())}")
                        in_atom_site_loop_data = True
                        atom_data_lines.append(line) # Current line is the first data line
                    else:
                        missing = [rkey for rkey in required_keys_check if rkey not in key_map_check]
                        log.warning(f"Found _atom_site loop but missing required keys: {missing}. Skipping.")
                        current_keys = [] # Reset if not a valid _atom_site loop
                else:
                    current_keys = [] # Reset if not an _atom_site loop
        elif in_atom_site_loop_data:
            # Check for end of loop data
            if line.startswith("loop_") or line.startswith("_") or line.startswith("data_") or line.startswith("stop_"):
                in_atom_site_loop_data = False
                break # Stop processing this loop
            else:
                atom_data_lines.append(line)

    if not atom_site_keys or not atom_data_lines:
        raise ValueError("Could not find or parse a valid _atom_site loop with required keys.")

    # Map required keys to their indices
    key_map = {key: i for i, key in enumerate(atom_site_keys)}
    try:
        idx_group, idx_atom, idx_res, idx_chain, idx_seq, idx_x, idx_y, idx_z, idx_model, idx_symbol = \
            [key_map[k] for k in required_keys_base]
        # Get optional indices safely
        idx_ins_code = key_map.get("_atom_site.pdbx_PDB_ins_code")
        idx_auth_asym = key_map.get("_atom_site.auth_asym_id")
        idx_auth_seq = key_map.get("_atom_site.auth_seq_id")
        idx_auth_comp = key_map.get("_atom_site.auth_comp_id")
    except KeyError as e:
        raise ValueError(f"Missing required key {e} in _atom_site loop.")

    residues_atoms_data = collections.OrderedDict() # Preserve insertion order
    ligand_atoms_list = [] # New list for ligand atoms
    parsed_target_ligand_keys = set(target_ligand_keys) if target_ligand_keys else set()

    # Process data lines
    for line_num, data_line_text in enumerate(atom_data_lines):
        try:
            data_line = shlex.split(data_line_text) # Handles quotes
            if len(data_line) != len(atom_site_keys):
                 # Shlex might merge empty fields if not quoted, try simple split as fallback
                 data_line_simple = data_line_text.split()
                 if len(data_line_simple) == len(atom_site_keys):
                     data_line = data_line_simple
                 else:
                     log.warning(f"Skipping line {line_num+1} due to column count mismatch ({len(data_line)} vs {len(atom_site_keys)}). Line: '{data_line_text}'")
                     continue
        except ValueError:
             log.warning(f"shlex parsing failed for line {line_num+1}, falling back to simple split. Line: '{data_line_text}'")
             data_line = data_line_text.split()
             if len(data_line) != len(atom_site_keys):
                 log.warning(f"Skipping line {line_num+1} after fallback split due to column count mismatch. Parsed: {data_line}")
                 continue

        # --- Process ATOM or HETATM --- #
        record_type = data_line[idx_group]
        if record_type not in ('ATOM', 'HETATM'): continue

        # --- Filter by Model --- #
        try:
            current_model = int(data_line[idx_model])
            if current_model != model_num: continue
        except (ValueError, IndexError):
            log.warning(f"Skipping line {line_num+1} due to model parsing error. Line: '{data_line_text}'")
            continue

        # --- Extract Common Fields --- #
        try:
            atom_name = data_line[idx_atom].strip('"')
            symbol = data_line[idx_symbol].strip('"').upper()
            coords_np = np.array([float(data_line[idx_x]), float(data_line[idx_y]), float(data_line[idx_z])], dtype=np.float64)
            # Get auth IDs safely, using label as fallback if necessary
            auth_asym_id = data_line[idx_auth_asym] if idx_auth_asym is not None else data_line[idx_chain]
            auth_seq_id_str = data_line[idx_auth_seq] if idx_auth_seq is not None else data_line[idx_seq]
            # Prioritize auth_comp_id, fallback to label_comp_id (res_name)
            auth_comp_id = data_line[idx_auth_comp] if idx_auth_comp is not None and data_line[idx_auth_comp] not in ('?', '.') else data_line[idx_res]

        except (ValueError, IndexError) as e:
            log.warning(f"Skipping line {line_num+1} due to basic field parsing error ({e}). Line: '{data_line_text}'")
            continue

        # --- Process ATOM Records (Protein Residues) --- #
        if record_type == 'ATOM':
            try:
                seq_id_str = data_line[idx_seq]
                if seq_id_str == '.': seq_id = -999
                elif not seq_id_str.lstrip('-').isdigit(): raise ValueError(f"non-integer seq_id '{seq_id_str}'")
                else: seq_id = int(seq_id_str)
            except (ValueError, IndexError) as e:
                log.warning(f"Skipping ATOM line {line_num+1} due to seq_id parsing error ({e}). Line: '{data_line_text}'")
                continue

            chain_id = data_line[idx_chain]
            res_name = data_line[idx_res]
            ins_code = data_line[idx_ins_code] if idx_ins_code is not None and data_line[idx_ins_code] not in ('?', '.') else ''
            residue_key = (chain_id, seq_id, ins_code) # Include ins_code in key

            # Use auth IDs from extracted common fields
            if residue_key not in residues_atoms_data:
                residues_atoms_data[residue_key] = {
                    'res_name': res_name, 'chain_id': chain_id, 'seq_id': seq_id,
                    'pdb_ins_code': ins_code, 'auth_asym_id': auth_asym_id, 'auth_seq_id': auth_seq_id_str,
                    'atoms': []
                }
            elif residues_atoms_data[residue_key]['res_name'] != res_name:
                log.warning(f"Inconsistent residue name ('{res_name}' vs '{residues_atoms_data[residue_key]['res_name']}') for key {residue_key}. Keeping first encountered.")

            # Store atom if name not already present (simple altLoc handling)
            if not any(a['name'] == atom_name for a in residues_atoms_data[residue_key]['atoms']):
                residues_atoms_data[residue_key]['atoms'].append({'name': atom_name, 'symbol': symbol, 'coords': coords_np})

        # --- Process HETATM Records (Potential Ligands) --- #
        elif record_type == 'HETATM' and parsed_target_ligand_keys:
            # --- Debug: Print raw values ---
            raw_auth_comp = data_line[idx_auth_comp] if idx_auth_comp is not None else "N/A"
            raw_label_comp = data_line[idx_res]
            #log.debug(f"    HETATM Raw: auth_comp='{raw_auth_comp}', label_comp='{raw_label_comp}'")
            # --- End Debug ---

            # Construct the key for matching: Chain:ResNum:CompID
            # Use the already extracted auth_asym_id, auth_seq_id_str, auth_comp_id
            # The 'auth_comp_id' variable here holds the result of the prioritized logic
            #log.debug(f"    Chosen auth_comp_id for key: '{auth_comp_id}'") # Debug chosen value
            ligand_key = f"{auth_asym_id}:{auth_seq_id_str}:{auth_comp_id}"
            #log.debug(f"    Checking HETATM: Key='{ligand_key}'")

            if ligand_key in parsed_target_ligand_keys:
                log.debug(f"      MATCH FOUND! Adding atom '{atom_name}'")
                # Store ligand atom info
                ligand_atoms_list.append({
                    'ligand_key': ligand_key,
                    'name': atom_name,
                    'symbol': symbol,
                    'coords': coords_np
                })
                # Simple altLoc handling for ligands: keep first encountered atom name per ligand
                # This might need refinement based on specific use cases
                # We can filter duplicates later if needed

    # --- Post-processing --- #
    # Convert protein residue data to final lists
    residue_info_list, all_atom_data_list = [], []
    required_backbone = {"N", "CA", "C"}
    skipped_residue_count = 0
    for key, res_data in residues_atoms_data.items():
        present_atoms = {a['name'] for a in res_data['atoms']}
        if required_backbone.issubset(present_atoms):
            # Reconstruct key components if needed (chain, seq, ins)
            chain_id, seq_id, ins_code = key
            residue_info_list.append({
                'res_name': res_data['res_name'], 'seq_id': seq_id, # Use parsed seq_id
                'chain_id': chain_id, 'pdb_ins_code': ins_code,
                'auth_asym_id': res_data['auth_asym_id'], 'auth_seq_id': res_data['auth_seq_id']
            })
            all_atom_data_list.append(res_data['atoms'])
        else:
            skipped_residue_count += 1
            missing_bb = required_backbone - present_atoms
            log.warning(f"Skipping residue {key} ({res_data['res_name']}) due to missing backbone atoms: {missing_bb}. Found: {present_atoms}")

    if skipped_residue_count > 0:
        log.warning(f"--- Skipped {skipped_residue_count} protein residues due to missing backbone atoms ---")

    log.info(f"--> Successfully parsed [b]{len(residue_info_list)}[/] protein residues.", extra={"markup": True})
    if ligand_atoms_list:
        log.info(f"--> Extracted [b]{len(ligand_atoms_list)}[/] atoms for target ligands: {list(parsed_target_ligand_keys)}.", extra={"markup": True})

    return residue_info_list, all_atom_data_list, ligand_atoms_list

def create_pytree_structure(residue_info: List[Dict], all_atom_data: List[List[Dict]]) -> ChainPytree:
    """Converts loaded atom data into the JAX-friendly Pytree structure.

    Initializes fields required for subsequent DSSP steps (e.g., SS type,
    H-bond partners, angles) to default/placeholder values.

    Args:
        residue_info: List of residue metadata dictionaries from `load_cif_data`.
        all_atom_data: List of atom lists from `load_cif_data`.

    Returns:
        A ChainPytree (list of ResiduePytree dictionaries).

    Raises:
        ValueError: If the lengths of residue_info and all_atom_data mismatch.
    """
    if len(residue_info) != len(all_atom_data):
        raise ValueError("Mismatch between residue info and atom data lists.")

    chain: ChainPytree = []
    nan_coord = jnp.full(3, jnp.nan, dtype=jnp.float64)
    skipped_residue_count = 0

    for i, (info, residue_atoms) in enumerate(zip(residue_info, all_atom_data)):
        # Extract backbone coords, find sidechain atoms, and get sidechain radii
        backbone_coords_dict = {'N': nan_coord, 'CA': nan_coord, 'C': nan_coord, 'O': nan_coord}
        sidechain_coords_list = []
        sidechain_radii_list = []
        atom_lookup = {a['name']: a for a in residue_atoms}

        for atom in residue_atoms:
            name = atom['name']
            symbol = atom['symbol'] # Already uppercased in load_cif_data
            coords = jnp.array(atom['coords'])

            if name in backbone_coords_dict:
                backbone_coords_dict[name] = coords
            else:
                sidechain_coords_list.append(coords)
                radius = element_radii.get(symbol, kRadiusDefault)
                sidechain_radii_list.append(radius)

        # Re-check for required backbone atoms (should be guaranteed by load_cif_data)
        # This check is slightly redundant but safe.
        if jnp.any(jnp.isnan(backbone_coords_dict['N'])) or \
           jnp.any(jnp.isnan(backbone_coords_dict['CA'])) or \
           jnp.any(jnp.isnan(backbone_coords_dict['C'])):
            skipped_residue_count += 1
            log.warning(f"Pytree creation skipping residue index {i} (SeqID: {info.get('seq_id', 'N/A')}) due to missing N/CA/C coordinates.")
            continue

        # Create AtomCoords tuple, handling potentially missing O
        bb_coords = AtomCoords(
            N=backbone_coords_dict['N'],
            CA=backbone_coords_dict['CA'],
            C=backbone_coords_dict['C'],
            O=backbone_coords_dict.get('O', nan_coord),
            H=nan_coord # H is calculated later
        )

        # Stack sidechain info into JAX arrays
        sc_coords_arr = jnp.stack(sidechain_coords_list) if sidechain_coords_list else jnp.empty((0, 3), dtype=jnp.float64)
        sc_radii_arr = jnp.array(sidechain_radii_list, dtype=jnp.float64) if sidechain_radii_list else jnp.empty((0,), dtype=jnp.float64)

        res_name = info.get('res_name', 'UNK')

        # Build the residue dictionary (ResiduePytree)
        residue: ResiduePytree = {
            # --- Identifying Info ---
            'res_index': i,
            'res_name': res_name,
            'seq_id': info.get('seq_id', -1),
            'chain_id': info.get('chain_id', '?'),
            'pdb_ins_code': info.get('pdb_ins_code', ''),
            'auth_asym_id': info.get('auth_asym_id', info.get('chain_id', '?')), # Fallback
            'auth_seq_id': info.get('auth_seq_id', info.get('seq_id', '?')),     # Fallback
            # --- Input Coordinates & Properties ---
            'is_proline': res_name == 'PRO',
            'bb_coords': bb_coords,
            'sidechain_coords': sc_coords_arr,
            'sidechain_radii': sc_radii_arr,
            # --- DSSP Calculated Fields (Initialized) ---
            'secondary_structure': SecondaryStructureType.LOOP,
            'phi': jnp.nan, 'psi': jnp.nan, 'kappa': jnp.nan,
            'accessibility': jnp.nan,
            'hbond_acceptor_1_idx': -1, 'hbond_acceptor_1_nrg': kInvalidHBondEnergy,
            'hbond_acceptor_2_idx': -1, 'hbond_acceptor_2_nrg': kInvalidHBondEnergy,
            'hbond_donor_1_idx': -1, 'hbond_donor_1_nrg': kInvalidHBondEnergy,
            'hbond_donor_2_idx': -1, 'hbond_donor_2_nrg': kInvalidHBondEnergy,
            'beta_partner_1': -1, 'beta_partner_2': -1,
            'ladder_1': -1, 'ladder_2': -1,
            'is_parallel_1': False, 'is_parallel_2': False,
            'sheet_id': -1,
            'helix_3_10_flag': HelixFlagType.NONE,
            'helix_alpha_flag': HelixFlagType.NONE,
            'helix_pi_flag': HelixFlagType.NONE,
            'helix_pp_flag': HelixFlagType.NONE,
        }
        chain.append(residue)

    if skipped_residue_count > 0:
         log.warning(f"--- Skipped {skipped_residue_count} residues during Pytree creation ---")
    log.info(f"[bold cyan]Step 2: Pytree Creation[/] - Created Pytree for [b]{len(chain)}[/] residues, including sidechain data.", extra={"markup": True})
    return chain

# --- Output Formatting --- #

def _format_residue_line(res: ResiduePytree, dssp_nr: int, res_to_nr_map: Dict, chain: ChainPytree) -> str:
    """Formats a single residue's data into the legacy DSSP fixed-width string."""

    # --- Extract Basic Info ---
    pdb_seq_id = res.get('seq_id', 0)
    # Legacy format expects PDB seq num; handle potential large numbers or non-integers gracefully
    try:
        pdb_seq_id_int = int(pdb_seq_id) if pdb_seq_id != '?' else 0
        pdb_seq_id_str = f"{pdb_seq_id_int:>5d}"
        if len(pdb_seq_id_str) > 5: # Handle overflow if PDB number > 99999
             pdb_seq_id_str = "*****"
    except ValueError:
        pdb_seq_id_str = "    ?" # Or handle differently if non-integer seq_id is possible

    pdb_ins_code = res.get('pdb_ins_code', ' ') or ' ' # Use space if None or empty
    pdb_chain_id = res.get('chain_id', ' ') or ' '
    if len(pdb_chain_id) > 1:
        # print(f"Warning: Chain ID '{pdb_chain_id}' longer than 1 char, using first char for legacy DSSP format.")
        pdb_chain_id = pdb_chain_id[0]

    # --- Amino Acid Code ---
    aa_code_3 = res.get('res_name', 'UNK')
    aa_map = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
              'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
              'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
              'TYR': 'Y', 'VAL': 'V'}
    aa_char = aa_map.get(aa_code_3, 'X')

    # --- SS Bridge Labeling (Placeholder) ---
    # This requires SS bridge partner info and numbering logic not present here.
    # ss_bridge_nr = res.get('ss_bridge_nr', 0)
    # if aa_char == 'C' and ss_bridge_nr > 0:
    #     aa_char = chr(ord('a') + (ss_bridge_nr - 1) % 26) # Example logic

    # --- Secondary Structure ---
    ss_type = res.get('secondary_structure', SecondaryStructureType.LOOP)
    ss_char = ss_type.to_char()

    # --- Helix Flags ---
    helix_flags = ""
    # Order: PP, 3_10, Alpha, Pi (matches C++ output order implicitly)
    hf_pp = res.get('helix_pp_flag', HelixFlagType.NONE)
    helix_flags += '>' if hf_pp == HelixFlagType.START else '<' if hf_pp == HelixFlagType.END else 'X' if hf_pp == HelixFlagType.START_END else 'P' if hf_pp == HelixFlagType.MIDDLE else ' '
    hf_310 = res.get('helix_3_10_flag', HelixFlagType.NONE)
    helix_flags += '>' if hf_310 == HelixFlagType.START else '<' if hf_310 == HelixFlagType.END else 'X' if hf_310 == HelixFlagType.START_END else '3' if hf_310 == HelixFlagType.MIDDLE else ' '
    hf_alpha = res.get('helix_alpha_flag', HelixFlagType.NONE)
    helix_flags += '>' if hf_alpha == HelixFlagType.START else '<' if hf_alpha == HelixFlagType.END else 'X' if hf_alpha == HelixFlagType.START_END else '4' if hf_alpha == HelixFlagType.MIDDLE else ' ' # Note: C++ uses '3'+type, legacy often used numbers
    hf_pi = res.get('helix_pi_flag', HelixFlagType.NONE)
    helix_flags += '>' if hf_pi == HelixFlagType.START else '<' if hf_pi == HelixFlagType.END else 'X' if hf_pi == HelixFlagType.START_END else '5' if hf_pi == HelixFlagType.MIDDLE else ' '

    # --- Bend ---
    bend_char = 'S' if ss_type == SecondaryStructureType.BEND else ' '

    # --- Chirality (Placeholder) ---
    # Requires DSSP alpha angle calculation (C-alpha trace dihedral)
    alpha_angle_placeholder = np.nan # Use NaN as placeholder
    chirality = '+' if alpha_angle_placeholder > 0 else '-' if alpha_angle_placeholder < 0 else ' '

    # --- Bridge Partners ---
    bp_indices = [res.get('beta_partner_1', -1), res.get('beta_partner_2', -1)]
    bp_ladders = [res.get('ladder_1', -1), res.get('ladder_2', -1)]
    bp_parallel = [res.get('is_parallel_1', False), res.get('is_parallel_2', False)]
    bp_rel_nr = [0, 0]
    bp_label = [' ', ' ']

    for i in range(2):
        partner_idx = bp_indices[i]
        if partner_idx != -1 and partner_idx in res_to_nr_map:
            partner_dssp_nr = res_to_nr_map[partner_idx]
            bp_rel_nr[i] = partner_dssp_nr - dssp_nr
            ladder_id = bp_ladders[i]
            if ladder_id != -1:
                label_char_code = ord('A') + ladder_id % 26
                bp_label[i] = chr(label_char_code).lower() if bp_parallel[i] else chr(label_char_code)
        else:
             bp_rel_nr[i] = 0 # Default if partner not found or invalid

    # --- Sheet Label ---
    sheet_id = res.get('sheet_id', -1)
    sheet_label = chr(ord('A') + (sheet_id - 1) % 26) if sheet_id > 0 else ' '

    # --- Accessibility ---
    acc = res.get('accessibility', np.nan)
    # Legacy format uses integer, rounding seems appropriate based on C++ code
    acc_int = int(np.round(acc)) if not np.isnan(acc) else 0 # Default to 0 if NaN

    # --- H-Bonds ---
    hb_acceptor_idx = [res.get('hbond_acceptor_1_idx', -1), res.get('hbond_acceptor_2_idx', -1)]
    hb_acceptor_nrg = [res.get('hbond_acceptor_1_nrg', np.inf), res.get('hbond_acceptor_2_nrg', np.inf)]
    hb_donor_idx = [res.get('hbond_donor_1_idx', -1), res.get('hbond_donor_2_idx', -1)]
    hb_donor_nrg = [res.get('hbond_donor_1_nrg', np.inf), res.get('hbond_donor_2_nrg', np.inf)]

    nho_str = ["     0, 0.0"] * 2
    onh_str = ["     0, 0.0"] * 2

    for i in range(2):
        # N-H-->O (Acceptor pairing)
        acc_idx = hb_acceptor_idx[i]
        acc_nrg = hb_acceptor_nrg[i]
        if acc_idx != -1 and acc_idx in res_to_nr_map and not np.isinf(acc_nrg):
            acc_dssp_nr = res_to_nr_map[acc_idx]
            rel_idx = acc_dssp_nr - dssp_nr
            nho_str[i] = f"{rel_idx:>6d},{acc_nrg:>4.1f}"

        # O-->H-N (Donor pairing)
        don_idx = hb_donor_idx[i]
        don_nrg = hb_donor_nrg[i]
        if don_idx != -1 and don_idx in res_to_nr_map and not np.isinf(don_nrg):
            don_dssp_nr = res_to_nr_map[don_idx]
            rel_idx = don_dssp_nr - dssp_nr
            onh_str[i] = f"{rel_idx:>6d},{don_nrg:>4.1f}"

    # --- Angles ---
    tco = res.get('tco', 0.0) # Placeholder if not calculated
    kappa = res.get('kappa', 360.0)
    alpha = 360.0 # Placeholder for C-alpha trace dihedral
    phi = res.get('phi', 360.0)
    psi = res.get('psi', 360.0)

    tco_str = f"{tco:>6.3f}" if not np.isnan(tco) else " 0.000" # Default TCO
    kappa_str = f"{kappa:>6.1f}" if not np.isnan(kappa) else " 360.0"
    alpha_str = f"{alpha:>6.1f}" if not np.isnan(alpha) else " 360.0"
    phi_str = f"{phi:>6.1f}" if not np.isnan(phi) else " 360.0"
    psi_str = f"{psi:>6.1f}" if not np.isnan(psi) else " 360.0"

    # --- Coordinates ---
    ca_coord = res.get('bb_coords').CA if res.get('bb_coords') else np.array([np.nan]*3)
    ca_x, ca_y, ca_z = ca_coord[0], ca_coord[1], ca_coord[2]
    ca_x_str = f"{ca_x:>7.1f}" if not np.isnan(ca_x) else "       "
    ca_y_str = f"{ca_y:>7.1f}" if not np.isnan(ca_y) else "       "
    ca_z_str = f"{ca_z:>7.1f}" if not np.isnan(ca_z) else "       "

    # --- Assemble the Line ---
    # Based on header:
    # #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA
    # Widths:
    # 5  # (DSSP Seq Num)
    # 5  RESIDUE (PDB Seq Num)
    # 1  Insertion Code
    # 1  Chain ID
    # 1  ' '
    # 1  AA
    # 2  '  '
    # 1  STRUCTURE (SS Code)
    # 4  STRUCTURE (Helix Flags)
    # 1  STRUCTURE (Bend)
    # 1  STRUCTURE (Chirality)
    # 1  BP1 (label)
    # 1  BP2 (label)
    # 4  BP1 (partner number - relative)
    # 4  BP2 (partner number - relative)
    # 1  SHEET
    # 4  ACC
    # 1  ' '
    # 11 N-H-->O 1
    # 11 O-->H-N 1
    # 11 N-H-->O 2
    # 11 O-->H-N 2
    # 2  '  '
    # 6  TCO
    # 6  KAPPA
    # 6  ALPHA
    # 6  PHI
    # 6  PSI
    # 1  ' '
    # 7  X-CA
    # 7  Y-CA
    # 7  Z-CA
    # = 136 columns approx

    line = (f"{dssp_nr:>5d}"
            f"{pdb_seq_id_str:5}" # Already formatted to 5 chars
            f"{pdb_ins_code:1}"
            f"{pdb_chain_id:1}"
            f" {aa_char:1}  " # Note spaces
            f"{ss_char:1}"
            f"{helix_flags:4}"
            f"{bend_char:1}"
            f"{chirality:1}"
            f"{bp_label[0]:1}"
            f"{bp_label[1]:1}"
            f"{bp_rel_nr[0]:>4d}"
            f"{bp_rel_nr[1]:>4d}"
            f"{sheet_label:1}"
            f"{acc_int:>4d} " # Note space after ACC
            f"{nho_str[0]:11}"
            f"{onh_str[0]:11}"
            f"{nho_str[1]:11}"
            f"{onh_str[1]:11}"
            f"  " # Note spaces
            f"{tco_str:>6}"
            f"{kappa_str:>6}"
            f"{alpha_str:>6}"
            f"{phi_str:>6}"
            f"{psi_str:>6} " # Note space after PSI
            f"{ca_x_str:>7}"
            f"{ca_y_str:>7}"
            f"{ca_z_str:>7}"
            )

    return line


# --- Main Legacy DSSP Output Function ---

def write_dssp_output(chain: ChainPytree, filename: str, stats: Dict = None, pdb_headers: Dict = None):
    """Writes output in legacy DSSP format, based on dssp-io.cpp.

    Args:
        chain: The ChainPytree containing DSSP results.
        filename: Path for the output legacy DSSP file.
        stats: Dictionary with pre-calculated statistics for the header.
        pdb_headers: Dictionary with pre-formatted PDB header lines.
    """
    log.info(f"[bold cyan]Step 7: Legacy DSSP Output[/] - Writing to [i]{filename}[/]", extra={"markup": True})

    if not chain:
        log.warning("Cannot write DSSP output: Chain is empty.")
        return

    # --- Use Defaults if Optional Args Missing ---
    if stats is None:
        stats = {}
        log.warning("Statistics dictionary not provided, using default values for header.")
    if pdb_headers is None:
        pdb_headers = {}
        log.warning("PDB headers dictionary not provided, using placeholders.")

    # --- Constants ---
    kHistogramSize = 30  # As defined in C++ code
    kMaxPeptideBondLength = 2.5  # Used for chain break detection heuristic

    # --- Default Values for Stats ---
    n_res = len(chain)
    n_chains = len(set(r.get('chain_id', '?') for r in chain)) # Simple chain count
    n_ss_bridges = stats.get('n_ss_bridges', 0)
    n_intra_ss_bridges = stats.get('n_intra_ss_bridges', 0)
    n_inter_ss_bridges = n_ss_bridges - n_intra_ss_bridges
    acc_surf = stats.get('acc_surf', 0.0) # Default to 0.0 if NaN or missing
    n_hbonds = stats.get('n_hbonds', 0)
    n_hbonds_par = stats.get('n_hbonds_par', 0)
    n_hbonds_anti = stats.get('n_hbonds_anti', 0)
    hbonds_dist = stats.get('hbonds_dist', [0]*11)
    hist_alpha = stats.get('hist_alpha', [0]*kHistogramSize)
    hist_par_bridge = stats.get('hist_par_bridge', [0]*kHistogramSize)
    hist_anti_bridge = stats.get('hist_anti_bridge', [0]*kHistogramSize)
    hist_ladder = stats.get('hist_ladder', [0]*kHistogramSize)

    # --- Default Values for PDB Headers ---
    pdb_header_line = pdb_headers.get('HEADER', "HEADER    (Not Available)")
    pdb_compnd_line = pdb_headers.get('COMPND', "COMPND    (Not Available)")
    pdb_source_line = pdb_headers.get('SOURCE', "SOURCE    (Not Available)")
    pdb_author_line = pdb_headers.get('AUTHOR', "AUTHOR    (Not Available)")

    try:
        with open(filename, 'w') as f:
            # --- Write Header Section ---
            today_str = datetime.date.today().strftime("%Y-%m-%d") # Use ISO format for consistency
            prog_version = "1.0.0-jax" # Placeholder version
            f.write(f"==== Secondary Structure Definition by the program DSSP (Python/JAX Version {prog_version:<10}) ==== DATE={today_str}        .\n")
            f.write("REFERENCE W. KABSCH AND C.SANDER, BIOPOLYMERS 22 (1983) 2577-2637                                                              .\n")
            # Write PDB headers, ensuring they end with '.' and fit legacy constraints
            f.write(f"{pdb_header_line:<80.80}.\n")
            f.write(f"{pdb_compnd_line:<80.80}.\n")
            f.write(f"{pdb_source_line:<80.80}.\n")
            f.write(f"{pdb_author_line:<80.80}.\n")

            # --- Write Statistics Section ---
            f.write(f"{n_res:>5d}{n_chains:>3d}{n_ss_bridges:>3d}{n_intra_ss_bridges:>3d}{n_inter_ss_bridges:>3d} TOTAL NUMBER OF RESIDUES, NUMBER OF CHAINS, NUMBER OF SS-BRIDGES(TOTAL,INTRACHAIN,INTERCHAIN)                .\n")
            f.write(f"{acc_surf:>8.1f}   ACCESSIBLE SURFACE OF PROTEIN (ANGSTROM**2)                                                                         .\n")
            hbond_per_100 = (n_hbonds * 100.0 / n_res) if n_res > 0 else 0.0
            f.write(f"{n_hbonds:>5d}{hbond_per_100:>5.1f}   TOTAL NUMBER OF HYDROGEN BONDS OF TYPE O(I)-->H-N(J)  , SAME NUMBER PER 100 RESIDUES                              .\n")
            hbond_par_per_100 = (n_hbonds_par * 100.0 / n_res) if n_res > 0 else 0.0
            f.write(f"{n_hbonds_par:>5d}{hbond_par_per_100:>5.1f}   TOTAL NUMBER OF HYDROGEN BONDS IN     PARALLEL BRIDGES, SAME NUMBER PER 100 RESIDUES                              .\n")
            hbond_anti_per_100 = (n_hbonds_anti * 100.0 / n_res) if n_res > 0 else 0.0
            f.write(f"{n_hbonds_anti:>5d}{hbond_anti_per_100:>5.1f}   TOTAL NUMBER OF HYDROGEN BONDS IN ANTIPARALLEL BRIDGES, SAME NUMBER PER 100 RESIDUES                              .\n")

            for k in range(11):
                count = hbonds_dist[k] if k < len(hbonds_dist) else 0
                per_100 = (count * 100.0 / n_res) if n_res > 0 else 0.0
                dist_val = k - 5
                dist_str = f"{dist_val:+d}" # Format with sign
                f.write(f"{count:>5d}{per_100:>5.1f}   TOTAL NUMBER OF HYDROGEN BONDS OF TYPE O(I)-->H-N(I{dist_str}), SAME NUMBER PER 100 RESIDUES                              .\n")

            # --- Write Histograms Section ---
            f.write("  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30     *** HISTOGRAMS OF *** .\n")
            f.write(''.join([f"{h:>3d}" for h in hist_alpha[:kHistogramSize]]) + "    RESIDUES PER ALPHA HELIX         .\n")
            f.write(''.join([f"{h:>3d}" for h in hist_par_bridge[:kHistogramSize]]) + "    PARALLEL BRIDGES PER LADDER      .\n")
            f.write(''.join([f"{h:>3d}" for h in hist_anti_bridge[:kHistogramSize]]) + "    ANTIPARALLEL BRIDGES PER LADDER  .\n")
            f.write(''.join([f"{h:>3d}" for h in hist_ladder[:kHistogramSize]]) + "    LADDERS PER SHEET                .\n")

            # --- Write Residue Data Header ---
            f.write("  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA\n")

            # --- Write Residue Data ---
            dssp_sequential_nr = 0
            prev_dssp_nr = 0
            prev_res_index = -2 # Initialize to ensure first residue doesn't trigger break check wrongly
            prev_chain_id = None
            residue_to_dssp_nr_map = {} # Map res_index to dssp_sequential_nr

            for i, res in enumerate(chain):
                dssp_sequential_nr += 1
                res_index = res.get('res_index', i) # Use res_index if available

                # --- Chain Break Detection ---
                is_break = False
                is_new_chain = False
                if i > 0:
                    current_chain_id = res.get('chain_id', '?')
                    if current_chain_id != prev_chain_id:
                        is_break = True
                        is_new_chain = True
                        dssp_sequential_nr += 1 # Increment DSSP number across chain break
                    # Check index continuity (gap detection)
                    elif res_index != prev_res_index + 1:
                        is_break = True
                        is_new_chain = False
                        dssp_sequential_nr += 1 # Increment DSSP number across gap break
                    # Optional: Add distance check if needed
                    # elif distance(prev_res['bb_coords'].C, res['bb_coords'].N) > kMaxPeptideBondLength * 1.5:
                    #     is_break = True
                    #     is_new_chain = False
                    #     dssp_sequential_nr += 1

                if is_break:
                    break_char = '*' if is_new_chain else ' '
                    f.write(f"{prev_dssp_nr + 1:>5d}        !{break_char}             0   0    0      0, 0.0     0, 0.0     0, 0.0     0, 0.0   0.000 360.0 360.0 360.0 360.0    0.0    0.0    0.0\n")

                # --- Store Mapping and Format Line ---
                residue_to_dssp_nr_map[res_index] = dssp_sequential_nr
                formatted_line = _format_residue_line(res, dssp_sequential_nr, residue_to_dssp_nr_map, chain)
                f.write(formatted_line + '\n')

                # --- Update State for Next Iteration ---
                prev_dssp_nr = dssp_sequential_nr
                prev_chain_id = res.get('chain_id', '?')
                prev_res_index = res_index

        log.info(f"--> Successfully wrote legacy DSSP output with {n_res} residues to {filename}")

    except IOError as e:
        log.error(f"Error writing output DSSP file {filename}: {e}")
    except KeyError as e:
        log.error(f"Error: Missing expected key {e} in ChainPytree or stats dictionary.")
    except Exception as e: # Catch any other unexpected errors
        log.error(f"An unexpected error occurred during DSSP file writing: {e}")
        import traceback
        traceback.print_exc()

def write_dssp_output_placeholder(chain: ChainPytree, filename: str):
    """Placeholder: Writes output in legacy DSSP format."""
    # Call the actual implementation, with default statistics
    write_dssp_output(chain, filename)

def write_mmcif_output(chain: ChainPytree, input_cif_url_or_file: str, filename: str, write_other: bool = False):
    """Writes DSSP results as an annotated mmCIF file.

    Reads the original mmCIF from a URL or file, replaces secondary structure
    annotations (_struct_conf, _struct_conf_type) based on the DSSP
    results in the ChainPytree, adds software/audit info, and writes
    to a new file. Uses the 'gemmi' library.

    Args:
        chain: The ChainPytree containing DSSP results.
        input_cif_url_or_file: URL or path of the original input mmCIF file.
        filename: Path for the output annotated mmCIF file.
        write_other: If True, include 'OTHER' type in _struct_conf_type
                     and corresponding segments in _struct_conf.
    """
    log.info(f"[bold cyan]Step 8: mmCIF Output[/] - Writing to [i]{filename}[/]", extra={"markup": True})

    if not chain:
        log.warning("Cannot write mmCIF output: Chain is empty.")
        return

    # --- 1. Dependency Check ---
    try:
        import gemmi
    except ImportError as e:
        log.error(f"Missing dependency 'gemmi'. Please install with 'pip install gemmi'.")
        return

    # --- 2. Fetch and Parse Input CIF ---
    try:
        cif_template_text: str
        is_url = input_cif_url_or_file.lower().startswith(('http:', 'https:'))
        if is_url:
            log.debug(f"    (Re-downloading {input_cif_url_or_file} for mmCIF template...)")
            response = requests.get(input_cif_url_or_file, timeout=30) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            cif_template_text = response.text
        else:
            if not os.path.exists(input_cif_url_or_file):
                log.error(f"    mmCIF template file not found: {input_cif_url_or_file}")
                return # Exit if template missing
            log.debug(f"    (Reading {input_cif_url_or_file} for mmCIF template...)")
            with open(input_cif_url_or_file, 'r') as f:
                cif_template_text = f.read()

        doc = gemmi.cif.read_string(cif_template_text)
        # Use the first data block, common practice for PDB mmCIF
        block = doc.sole_block()
    except Exception as e: # Catch potential gemmi parsing errors
        log.error(f"Error parsing input CIF file: {e}")
        return

    # --- 3. Process ChainPytree to Identify Contiguous SS Segments ---
    segments = []
    if chain:
        current_segment = None
        prev_res = None

        for i, res in enumerate(chain):
            ss_type = res['secondary_structure']
            # Check for segment start or continuation
            start_new_segment = False
            if current_segment is None:
                start_new_segment = True
            elif ss_type != current_segment['ss_type']:
                start_new_segment = True
            # Check for chain break or gap (using simple index check here,
            # could be refined if chain_break info is reliable in Pytree)
            elif prev_res and res['chain_id'] != prev_res['chain_id']:
                 start_new_segment = True
            elif prev_res and res['res_index'] != prev_res['res_index'] + 1:
                 start_new_segment = True


            if start_new_segment:
                # Finalize the previous segment if it exists
                if current_segment:
                    current_segment['end_res'] = prev_res
                    segments.append(current_segment)

                # Start a new segment
                current_segment = {
                    'ss_type': ss_type,
                    'start_res': res,
                    'end_res': res # Temporarily set end to start
                }
            else:
                # Continue current segment, just update the end residue
                current_segment['end_res'] = res

            prev_res = res

        # Add the last segment
        if current_segment:
            segments.append(current_segment)

    # --- 4. Map DSSP SS Types to mmCIF _struct_conf_type.id ---
    ss_type_to_mmcif = {
        SecondaryStructureType.HELIX_ALPHA: "HELX_RH_AL_P", # H
        SecondaryStructureType.BETA_BRIDGE: "STRN",        # B (Treat as strand)
        SecondaryStructureType.BETA_STRAND: "STRN",        # E
        SecondaryStructureType.HELIX_3_10: "HELX_RH_3T_P",  # G
        SecondaryStructureType.HELIX_PI: "HELX_RH_PI_P",    # I
        SecondaryStructureType.HELIX_PP: "HELX_LH_PP_P",    # P
        SecondaryStructureType.TURN: "TURN_TY1_P",          # T (Assuming type 1 turn)
        SecondaryStructureType.BEND: "BEND",                # S
        SecondaryStructureType.LOOP: "OTHER",               # ' '
        # Add UNKNOWN or map it to OTHER if necessary
    }
    # Get mmcif types present in the structure
    present_mmcif_types = set()
    for seg in segments:
         mmcif_type = ss_type_to_mmcif.get(seg['ss_type'])
         if mmcif_type:
             if mmcif_type == "OTHER" and not write_other:
                 continue
             present_mmcif_types.add(mmcif_type)


    # --- 5. Clear/Prepare mmCIF Categories ---
    # Use find_loop to avoid errors if loop doesn't exist
    struct_conf_tags = block.find_loop('_struct_conf.id')
    if struct_conf_tags:
        block.delete_loop('_struct_conf.')
    struct_conf_type_tags = block.find_loop('_struct_conf_type.id')
    if struct_conf_type_tags:
        block.delete_loop('_struct_conf_type.')

    # --- 6. Generate and Populate _struct_conf_type ---
    if present_mmcif_types:
        conf_type_loop = block.init_loop('_struct_conf_type.', ['id', 'criteria'])
        for type_id in sorted(list(present_mmcif_types)): # Sort for consistent output
            conf_type_loop.add_row([type_id, 'DSSP'])

    # --- 7. Generate and Populate _struct_conf ---
    if segments:
         conf_loop = block.init_loop('_struct_conf.', [
            'conf_type_id', 'id',
            'beg_label_asym_id', 'beg_label_seq_id', 'pdbx_beg_PDB_ins_code',
            'end_label_asym_id', 'end_label_seq_id', 'pdbx_end_PDB_ins_code',
            'beg_auth_asym_id', 'beg_auth_seq_id',
            'end_auth_asym_id', 'end_auth_seq_id'
            # Optional: Add helix class, details etc. if needed
        ])
         conf_id_counters = collections.defaultdict(int)

         for seg in segments:
            mmcif_type = ss_type_to_mmcif.get(seg['ss_type'])
            if not mmcif_type or (mmcif_type == "OTHER" and not write_other):
                continue

            start_res = seg['start_res']
            end_res = seg['end_res']

            # Generate unique ID like HELX_RH_AL_P_1, HELX_RH_AL_P_2, ...
            conf_id_counters[mmcif_type] += 1
            conf_instance_id = f"{mmcif_type}_{conf_id_counters[mmcif_type]}"

            # Safely get residue identifiers, using '.' for missing
            beg_ins_code = start_res.get('pdb_ins_code', '.') or '.'
            end_ins_code = end_res.get('pdb_ins_code', '.') or '.'
            beg_auth_asym = start_res.get('auth_asym_id', start_res.get('chain_id', '?')) # Fallback
            beg_auth_seq = start_res.get('auth_seq_id', start_res.get('seq_id', '?'))     # Fallback
            end_auth_asym = end_res.get('auth_asym_id', end_res.get('chain_id', '?'))     # Fallback
            end_auth_seq = end_res.get('auth_seq_id', end_res.get('seq_id', '?'))         # Fallback

            conf_loop.add_row([
                mmcif_type, conf_instance_id,
                start_res.get('chain_id', '?'), str(start_res.get('seq_id', '?')), beg_ins_code,
                end_res.get('chain_id', '?'), str(end_res.get('seq_id', '?')), end_ins_code,
                beg_auth_asym, str(beg_auth_seq),
                end_auth_asym, str(end_auth_seq)
            ])

    # --- 8. (Optional but Recommended) Add Custom _dssp_* Categories ---
    # Placeholder: Implement functions to generate these loops if needed,
    # similar to steps 6 & 7, extracting required data from ChainPytree.
    # Requires the corresponding dictionary extension (dssp-extension.dic).
    # write_dssp_struct_summary(block, chain)
    # write_dssp_struct_bridge_pairs(block, chain)
    # write_dssp_struct_ladders(block, chain)
    # write_dssp_statistics(block, chain) # Needs stats object

    # --- 9. Add/Update audit_conform ---
    # Assuming dssp-extension.dic defines the custom categories if used
    audit_conform_tags = block.find_loop('_audit_conform.dict_name')
    if audit_conform_tags:
        has_dssp_entry = False
        for i, name in enumerate(audit_conform_tags):
            if 'dssp' in name.lower():
                has_dssp_entry = True
                break
        if not has_dssp_entry:
            # Add DSSP entry if it doesn't exist
            block.init_loop('_audit_conform.', ['dict_name', 'dict_version'])
            block.add_to_loop('_audit_conform.', ['dict_name', 'dict_version'], 
                            ['dssp-extension.dic', '1.0'])
    else:
        # Create audit_conform loop if it doesn't exist
        audit_conform_loop = block.init_loop('_audit_conform.', ['dict_name', 'dict_version'])
        audit_conform_loop.add_row(['dssp-extension.dic', '1.0']) # Example version

    # --- 10. Add software Entry ---
    software_loop_tags = block.find_loop('_software.name')
    max_ordinal = 1
    if software_loop_tags:
        for i, row in enumerate(block.find([], '_software.pdbx_ordinal')):
            try:
                ordinal = int(row[0])
                max_ordinal = max(max_ordinal, ordinal + 1)
            except (ValueError, IndexError):
                pass
    else:
        software_loop = block.init_loop('_software.', ['pdbx_ordinal', 'name', 'version', 'date', 'classification'])

    # Get version/date from your package if possible
    today_str = date.today().isoformat()
    prog_version = "1.0.0-jax" # Placeholder

    block.add_to_loop('_software.', 
                    ['pdbx_ordinal', 'name', 'version', 'date', 'classification'],
                    [str(max_ordinal), 'dssp-jax', prog_version, today_str, 'secondary structure assignment'])

    # --- 11. Write Output File ---
    try:
        # Use style=gemmi.cif.Style.Indent to format loops nicely
        doc.write_file(filename, style=gemmi.cif.Style.Indent)
        log.info(f"--> Successfully wrote annotated mmCIF with {len(segments)} secondary structure segments to {filename}")
    except Exception as e:
        log.error(f"Error writing output mmCIF file {filename}: {e}")

def write_mmcif_output_placeholder(chain: ChainPytree, input_cif_url_or_file: str, filename: str):
    """Placeholder: Writes output as an annotated mmCIF file."""
    # Call the actual implementation
    write_mmcif_output(chain, input_cif_url_or_file, filename) 