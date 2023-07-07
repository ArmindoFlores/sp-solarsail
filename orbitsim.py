import numpy as np


def rotmat2d(angle):
    """Create a 2D rotation matrix by `angle` radians.

    Args:
        angle (float): The rotation angle in radians.

    Returns:
        np.ndarray: A 2x2 rotation matrix.
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def tangencial_vector(theta, a, b, angle=0):
    """Compute the vector tangential to an ellipse at an angle of `theta`.

    Args:
        theta (float): The true anomaly, in radians.
        a (float): The ellipse's semi-major axis.
        b (float): The ellipse's semi-minor axis.
        angle (float, optional): The angle of the ellipse's axis to the x axis. Defaults to 0.

    Returns:
        np.ndarray: A vector of length 2 tangential to the ellipse.
    """
    e = np.sqrt(1 - (b/a)**2)
    E = np.arccos((e + np.cos(theta)) / (1 + e * np.cos(theta)))
    if theta > np.pi:
        E = -E
    return rotmat2d(angle) @ np.array([
        -a * np.sin(E),
        b * np.cos(E)
    ]) / np.sqrt(b**2 * np.cos(E)**2 + a**2 * np.sin(E)**2)
    
def orbit_equation(theta, a, e):
    """Computes `r`, the distance to the central body.

    Args:
        theta (float): The true anomaly.
        a (float): The orbit's semi-major axis.
        e (float): The orbit's eccentricity.

    Returns:
        float: The distance to the central body.
    """
    return a * (1 - e**2) / (1 + e * np.cos(theta))

def compute_angle(vector1, vector2):
    """Compute the angle between two vectors in radians.

    Args:
        vector1 (np.ndarray): The first vector.
        vector2 (np.ndarray): The second vector.

    Returns:
        float: The angle between `vector1` and `vector2`.
    """
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cosine_angle = dot_product / norm_product
    angle = np.arccos(cosine_angle)
    return angle