import numpy as np
from typing import Tuple

def triad(r1, r2, b1, b2):
    """
    Compute the rotation matrix from ECI to body frame using the TRIAD algorithm.

    Parameters
    ----------
    r1 : np.ndarray
        First reference vector in ECI frame, shape (3,).
    r2 : np.ndarray
        Second reference vector in ECI frame, shape (3,).
    b1 : np.ndarray
        First measured vector in body frame, shape (3,).
    b2 : np.ndarray
        Second measured vector in body frame, shape (3,).

    Returns
    -------
    R_bi : np.ndarray
        The 3x3 rotation matrix from ECI to body frame.
    """
    # Normalize the first vectors
    R1 = r1 / np.linalg.norm(r1)
    B1 = b1 / np.linalg.norm(b1)

    # Create the perpendicular second vectors by removing components along the first
    r2_perp = r2 - np.dot(R1, r2)*R1
    b2_perp = b2 - np.dot(B1, b2)*B1

    # Normalize these perpendicular vectors
    R2 = r2_perp / np.linalg.norm(r2_perp)
    B2 = b2_perp / np.linalg.norm(b2_perp)

    # Create the third vectors using the cross products
    R3 = np.cross(R1, R2)
    B3 = np.cross(B1, B2)

    # Construct the rotation matrix
    R_eci = np.column_stack((R1, R2, R3))  # ECI triad
    R_body = np.column_stack((B1, B2, B3)) # Body triad

    # R_bi transforms a vector in ECI to body: v_body = R_bi * v_eci
    R_bi = R_body @ R_eci.T
    return R_bi