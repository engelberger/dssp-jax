# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2020 NKI/AVL, Netherlands Cancer Institute
# Substantial portions of this code are derived from the DSSP C++ implementation
# by Maarten L. Hekkelman and contributors, licensed under the BSD-2-Clause license.
# Full C++ source and license available at: https://github.com/PDB-REDO/dssp
"""DSSP JAX Implementation Package.

Exposes the main run_dssp function and configures logging.
"""

import logging
from rich.logging import RichHandler
import jax # Import jax

# --- Configure Logging --- #
# Use RichHandler for beautiful console output.
# Set level to INFO by default. Can be overridden by user applications.
# Show only the message, no logger name or timestamp by default.
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]", # Used if format includes date/time
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, show_level=False, show_time=False)]
)

# Get the specific logger instance for this package
log = logging.getLogger("dsspjax")

# --- JAX Configuration --- #
# Enable float64 support globally for the package
# Note: User application might override this later.
jax.config.update("jax_enable_x64", True)
log.debug("JAX float64 support enabled.") # Log if needed for debugging

# --- Public API --- #
# Expose the main function
from .main import run_dssp 