
import numpy as np

class SatelliteParams: 
    def __init__(self, I: np.array, m: float):
        self.I = I # [kg m^2]
        self.m = m # [kg]
    
    def __repr__(self):
        return f"SatelliteParams(I={self.I})"

    def __str__(self):
        return f"SatelliteParams(I={self.I})"

    def __add__(self, other):
        return SatelliteParams(self.I + other.I)
    
    def __sub__(self, other):
        return SatelliteParams(self.I - other.I)