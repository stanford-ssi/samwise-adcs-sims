import numpy as np

def passive_dcm(axis, angle):
    if axis == "x":
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), np.sin(angle)],
            [0, -np.sin(angle), np.cos(angle)]
        ])

    elif axis == "y":
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    
    elif axis == "z":
        return np.array([
            [np.cos(angle), np.sin(angle), 0],
            [-np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    
    else:
        raise ValueError("Invalid axis")