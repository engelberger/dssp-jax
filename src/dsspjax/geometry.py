# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2020 NKI/AVL, Netherlands Cancer Institute
# Substantial portions of this code are derived from the DSSP C++ implementation
# by Maarten L. Hekkelman and contributors, licensed under the BSD-2-Clause license.
# Full C++ source and license available at: https://github.com/PDB-REDO/dssp

"""Core geometric calculations (distance, angle, dihedral)."""

import jax
import jax.numpy as jnp

@jax.jit
def _distance_sq_impl(p1: jnp.ndarray, p2: jnp.ndarray) -> float:
    """Calculates the squared Euclidean distance between two points."""
    # Ensure inputs are JAX arrays for tracing
    diff = jnp.asarray(p1) - jnp.asarray(p2)
    return jnp.sum(diff * diff)

# Expose the JITted implementation directly
distance_sq = _distance_sq_impl

@jax.jit
def _distance_impl(p1: jnp.ndarray, p2: jnp.ndarray) -> float:
    """Calculates the Euclidean distance between two points."""
    # Reuses the JITted squared distance for potential fusion
    return jnp.sqrt(_distance_sq_impl(p1, p2))

distance = _distance_impl

@jax.jit
def _angle_impl(p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray) -> float:
    """Calculates the angle (in degrees) formed by p1-p2-p3."""
    # Ensure inputs are JAX arrays
    p1, p2, p3 = map(jnp.asarray, [p1, p2, p3])
    v21 = p1 - p2 # Vector from p2 to p1
    v23 = p3 - p2 # Vector from p2 to p3

    n21 = jnp.linalg.norm(v21)
    n23 = jnp.linalg.norm(v23)

    # Handle zero-length vectors to avoid NaN/division by zero
    # Use a small epsilon for floating point comparisons
    is_zero_n21 = n21 < 1e-6
    is_zero_n23 = n23 < 1e-6

    safe_n21 = jnp.where(is_zero_n21, 1.0, n21)
    safe_n23 = jnp.where(is_zero_n23, 1.0, n23)

    norm_v21 = v21 / safe_n21
    norm_v23 = v23 / safe_n23

    # Calculate dot product, clipping for numerical stability with arccos
    dot_product = jnp.clip(jnp.dot(norm_v21, norm_v23), -1.0, 1.0)
    angle_rad = jnp.arccos(dot_product)

    # Return NaN if either input vector was effectively zero length
    result = jnp.where(is_zero_n21 | is_zero_n23, jnp.nan, jnp.degrees(angle_rad))
    return result

angle = _angle_impl

@jax.jit
def _dihedral_angle_impl(p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray, p4: jnp.ndarray) -> float:
    """Calculates the dihedral angle (in degrees) defined by p1-p2-p3-p4."""
    # References:
    # https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
    # https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
    p1, p2, p3, p4 = map(jnp.asarray, [p1, p2, p3, p4])
    v12 = p2 - p1 # Vector b1 in Wikipedia notation
    v23 = p3 - p2 # Vector b2 (central bond)
    v34 = p4 - p3 # Vector b3

    # Normals to the planes defined by (p1, p2, p3) and (p2, p3, p4)
    n1 = jnp.cross(v12, v23)
    n2 = jnp.cross(v23, v34)

    # Vector along the axis of rotation (p2-p3)
    n_v23 = jnp.linalg.norm(v23)
    # Handle zero-length axis vector
    is_zero_v23 = n_v23 < 1e-6
    safe_n_v23 = jnp.where(is_zero_v23, 1.0, n_v23)
    norm_v23 = v23 / safe_n_v23

    # Check for zero-length normal vectors (collinear points)
    n_n1 = jnp.linalg.norm(n1)
    n_n2 = jnp.linalg.norm(n2)
    is_zero_n1 = n_n1 < 1e-6
    is_zero_n2 = n_n2 < 1e-6

    # Calculate angle between the normals using atan2 for quadrant correctness
    # Project n1 onto the plane normal to v23 (creates vector m1)
    m1 = jnp.cross(n1, norm_v23)

    # We need the cosine and sine of the angle between n1 and n2.
    # cos(angle) = dot(n1, n2) / (|n1|*|n2|)
    # sin(angle) = dot(m1, n2) / (|m1|*|n2|) => |m1| = |n1 x norm_v23| = |n1|*|norm_v23|*sin(angle_n1_v23)
    # Alternatively, use the formula from Wikipedia: atan2( (v23 . (n1 x n2)), |v23| * (n1 . n2) )
    # Let's use the x, y approach with atan2, which is generally robust.
    x = jnp.dot(n1, n2)
    y = jnp.dot(m1, n2) # m1 is orthogonal to n1 and v23

    angle_rad = jnp.arctan2(y, x)

    # Return NaN if the central bond or either normal was zero length (collinear points)
    result = jnp.where(is_zero_v23 | is_zero_n1 | is_zero_n2, jnp.nan, jnp.degrees(angle_rad))
    return result

dihedral_angle = _dihedral_angle_impl 