import numpy as np
from typing import Tuple

def get_attitude_matrix(R_1: np.ndarray, R_2: np.ndarray, r_1: np.ndarray, r_2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # docstring
    """
    This function calculates the attitude matrix A that rotates the reference frame R to the reference frame r.
    The attitude matrix A is calculated using the vectors R_1, R_2, r_1, and r_2.
    The vectors R_1, R_2, r_1, and r_2 are the vectors of the reference frames R and r respectively.
    
    Parameters:
    R_1 (np.ndarray): A 3x1 numpy array representing the first vector of the reference frame R.
    R_2 (np.ndarray): A 3x1 numpy array representing the second vector of the reference frame R.
    r_1 (np.ndarray): A 3x1 numpy array representing the first vector of the reference frame r.
    r_2 (np.ndarray): A 3x1 numpy array representing the second vector of the reference frame r.
    
    Returns:
    A (np.ndarray): A 3x3 numpy array representing the attitude matrix that rotates the reference frame R to the reference frame r.
    
    """

    R_1_norm = R_1 / np.linalg.norm(R_1)
    R_2_norm = R_2 / np.linalg.norm(R_2)
    r_1_norm = r_1 / np.linalg.norm(r_1)
    r_2_norm = r_2 / np.linalg.norm(r_2)

    # Concatenate the vectors in 2 matrices
    R = np.array([R_1_norm, R_2_norm, np.cross(R_1_norm, R_2_norm)])
    r = np.array([r_1_norm, r_2_norm, np.cross(r_1_norm, r_2_norm)])

    # calculate the rotation matrix
    A = R @ r.T

    return A



if __name__ == "__main__":
    R_1 = np.array([1, 2, 3])
    R_2 = np.array([4, 5, 6])
    r_1 = np.array([7, 8, 9])
    r_2 = np.array([10, 11, 12])

    A = get_attitude_matrix(R_1, R_2, r_1, r_2)
    print(A)
    