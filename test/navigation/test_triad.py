import numpy as np
from simwise.navigation.triad import triad
from simwise.math.quaternion import *
from simwise.math.dcm import passive_dcm

def test_triad_simple():
    # Define a known rotation matrix (e.g., 90-degree rotation about z-axis)
    R_true = passive_dcm("z", np.pi/2)
    
    # Generate synthetic ECI frame vectors
    R_1 = np.array([1, 0, 0])  # Along x-axis
    R_2 = np.array([0, 1, 0])  # Along y-axis

    # Transform to body frame using the known rotation matrix
    r_1 = R_true @ R_1
    r_2 = R_true @ R_2

    # Run TRIAD
    A = triad(R_1, R_2, r_1, r_2)

    # Verify results
    assert np.allclose(A, R_true)



def test_chatgpt():
    q = np.array([0.7071, 0, 0.7071, 0])  # 90 degrees about y-axis
    v = np.array([1, 0, 0])
    v_rotated = rotate_vector_by_quaternion(v, q)

    # Convert back using inverse rotation
    q_inv = quaternion_inverse(q)
    v_original = rotate_vector_by_quaternion(v_rotated, q_inv)

    print("Original vector:", v)
    print("Rotated vector:", v_rotated)
    print("Recovered vector:", v_original)

    # Simple ECI vectors
    v_sun_eci = np.array([1, 0, 0])
    v_mag_eci = np.array([0, 0, 1])

    # Define a known rotation quaternion
    q = np.array([0.7071, 0.7071, 0, 0])  # 90 degrees about x-axis

    # Rotate to body frame
    v_sun_body = rotate_vector_by_quaternion(v_sun_eci, q)
    v_mag_body = rotate_vector_by_quaternion(v_mag_eci, q)

    # TRIAD
    A = triad(v_sun_eci, v_mag_eci, v_sun_body, v_mag_body)
    q_triad = dcm_to_quaternion(A)
    q_triad = normalize_quaternion(q_triad)

    print("True quaternion:", q)
    print("TRIAD quaternion:", q_triad)

    # Check angle difference
    theta, axis = quaternions_to_axis_angle(q, q_triad)
    print(f"Angle difference: {theta} radians, Rotation axis: {axis}")
    
    # raise Exception


def random_quaternion():
    q = np.random.rand(4)
    q = q / np.linalg.norm(q)
    return q

def test_dcm_to_quaternion():
    for i in range(1000):
        q = random_quaternion()
        R = quaternion_to_dcm(q)
        q_recovered = dcm_to_quaternion(R)
        print(f"q: {q}\nq_recovered: {q_recovered}")
        assert np.allclose(normalize_quaternion(q), normalize_quaternion(q_recovered))

def test_triad_equatorial():
    npoints = 30
    for i in range(npoints):

        # Define a known rotation matrix (e.g., 90-degree rotation about z-axis)
        R_true = passive_dcm("z", np.deg2rad(360 * i / npoints))
        
        v_sun_eci = np.array([1.0, 0.0, 0.0])
        v_mag_eci = np.array([0.0, 0.0, 1.0])
        v_mag_body = R_true @ v_mag_eci
        v_sun_body = R_true @ v_sun_eci

        assert np.allclose(v_mag_body, v_mag_eci)

        # Generate synthetic ECI frame vectors
        A = triad(v_sun_eci, v_mag_eci, v_sun_body, v_mag_body)
        assert np.allclose(A, R_true)