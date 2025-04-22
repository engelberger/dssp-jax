from typing import Tuple
import jax.numpy as jnp

def generate_surface_dots(n_points: int) -> Tuple[jnp.ndarray, float]:
    """Generates evenly distributed points on a unit sphere using Fibonacci spiral.

    Args:
        n_points: The approximate number of points to generate.

    Returns:
        A tuple containing:
          - dots: A JAX array of shape (n_actual_points, 3) with unit vectors.
          - weight_per_dot: The surface area weight associated with each dot.
    """
    # Using the golden angle/Fibonacci lattice method
    # Adapted from https://stackoverflow.com/a/26127012
    indices = jnp.arange(0, n_points, dtype=jnp.float32) + 0.5
    phi = jnp.arccos(1 - 2 * indices / n_points) # Latitude
    theta = jnp.pi * (1 + jnp.sqrt(5.0)) * indices # Longitude (golden angle)

    x = jnp.cos(theta) * jnp.sin(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(phi)

    dots = jnp.stack([x, y, z], axis=-1)
    # The area weight should account for the sphere's surface area formula (4*pi*r^2)
    # Each dot represents an equal area patch, total area is 4*pi for unit sphere.
    weight_per_dot = 4.0 * jnp.pi / n_points # Area weight per dot on unit sphere

    return dots, float(weight_per_dot)
