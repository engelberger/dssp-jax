"""Example script demonstrating how to use the dsspjax package."""

import sys
import traceback
import argparse
import logging # Import logging
import numpy as np # Needed for isnan checks in table

# Rich imports for table output
import rich # Import the main rich package
from rich.console import Console
from rich.table import Table

# If installed via `pip install -e .`, you can import dsspjax directly
# Need to import types as well for the print loop
from dsspjax import run_dssp
from dsspjax.types import ChainPytree, SecondaryStructureType, HelixFlagType
from dsspjax.io import write_dssp_output_placeholder, write_mmcif_output_placeholder

# Get the logger configured in dsspjax package
log = logging.getLogger("dsspjax")

def print_summary(chain: ChainPytree, num_residues_to_print: int = 50):
    """Prints a summary of the DSSP results using a rich Table."""
    if not chain:
        log.warning("Attempted to print summary for an empty chain.")
        return

    console = Console()
    num_to_show = min(len(chain), num_residues_to_print)

    table = Table(
        title=f"DSSP-JAX Results Summary (First {num_to_show} Residues)",
        show_header=True,
        header_style="bold magenta",
        show_edge=True,
        box=rich.box.ROUNDED # Use rounded box style
    )

    # Define columns
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("RESIDUE", style="cyan", width=10) # Chain:SeqId Ins
    table.add_column("AA", style="green", width=3)
    table.add_column("STRUCTURE", style="bold yellow", width=1, justify="center") # Single char SS
    table.add_column("BP1", width=4, justify="right")
    table.add_column("BP2", width=4, justify="right")
    table.add_column("SHT", width=3, justify="right") # Sheet ID
    table.add_column("ACC", style="blue", width=6, justify="right") # Accessibility
    table.add_column("PHI", width=7, justify="right")
    table.add_column("PSI", width=7, justify="right")
    table.add_column("KAPPA", width=7, justify="right")
    table.add_column("HELIX FLAGS", width=4, justify="center") # 3AHPP

    for i, res in enumerate(chain):
        if i >= num_to_show:
            break

        ss_char = SecondaryStructureType(res['secondary_structure']).to_char()
        ss_char_styled = f"[bold yellow]{ss_char}[/]" if ss_char != ' ' else ' '

        # Format helix flags
        h3 = 'G' if res['helix_3_10_flag'] != HelixFlagType.NONE else '-'
        ha = 'H' if res['helix_alpha_flag'] != HelixFlagType.NONE else '-'
        hp = 'I' if res['helix_pi_flag'] != HelixFlagType.NONE else '-'
        hpp = 'P' if res['helix_pp_flag'] != HelixFlagType.NONE else '-'
        helix_flags = f"{h3}{ha}{hp}{hpp}"

        # Format beta partners and sheet
        bp1 = str(res['beta_partner_1']) if res['beta_partner_1'] != -1 else '.'
        bp2 = str(res['beta_partner_2']) if res['beta_partner_2'] != -1 else '.'
        sheet = str(res['sheet_id']) if res['sheet_id'] != -1 else '.'

        # Format residue identifier
        res_id = f"{res['chain_id']}:{res['seq_id']}{res['pdb_ins_code']}"

        # Add row to table
        table.add_row(
            str(res['res_index']),
            res_id,
            res['res_name'],
            ss_char_styled,
            bp1,
            bp2,
            sheet,
            f"{res['accessibility']:.1f}",
            f"{res['phi']:.1f}" if not np.isnan(res['phi']) else "nan",
            f"{res['psi']:.1f}" if not np.isnan(res['psi']) else "nan",
            f"{res['kappa']:.1f}" if not np.isnan(res['kappa']) else "nan",
            helix_flags
        )

    console.print(table)
    if len(chain) > num_to_show:
        console.print(f"(... and {len(chain) - num_to_show} more residues)", style="dim")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DSSP-JAX on a CIF file.")
    parser.add_argument("cif_input", help="URL or local path to the input mmCIF file.")
    parser.add_argument("-m", "--model", type=int, default=1, help="Model number to process (default: 1)")
    parser.add_argument("-dssp", "--dssp_out", help="Optional output file path for legacy DSSP format.")
    parser.add_argument("-cif", "--cif_out", help="Optional output file path for annotated mmCIF format.")
    parser.add_argument("--print_limit", type=int, default=50, help="Number of residues to print in summary (default: 50)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose DEBUG logging.")

    args = parser.parse_args()

    # Configure Logging Level Based on Verbosity
    if args.verbose:
        log.setLevel(logging.DEBUG)
        log.debug("Verbose logging enabled.")

    try:
        # Run the main DSSP pipeline
        log.info(f"Starting DSSP-JAX processing for: [bold]{args.cif_input}[/]", extra={"markup": True})
        final_chain = run_dssp(args.cif_input, model_num=args.model)

        # Print the summary table
        if final_chain:
            print_summary(final_chain, num_residues_to_print=args.print_limit)
        else:
            log.info("DSSP processing resulted in an empty chain.")

        # Write optional output files
        if args.dssp_out:
            log.info(f"Writing placeholder legacy DSSP output to: [i]{args.dssp_out}[/i]", extra={"markup": True})
            write_dssp_output_placeholder(final_chain, args.dssp_out)

        if args.cif_out:
            log.info(f"Writing placeholder annotated mmCIF output to: [i]{args.cif_out}[/i]", extra={"markup": True})
            # Pass the original input path/URL for potential template use
            write_mmcif_output_placeholder(final_chain, args.cif_input, args.cif_out)

        log.info("\nScript finished.")

    except FileNotFoundError:
        log.error(f"Input file not found at {args.cif_input}")
        sys.exit(1)
    except Exception as e:
        log.exception(f"An error occurred during DSSP calculation: {e}")
        sys.exit(1) 