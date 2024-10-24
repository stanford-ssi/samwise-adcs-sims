from dataclasses import dataclass

@dataclass
class ClassicalOrbitalElements:
    sma: float  # Semi-major axis (km)
    ecc: float  # Eccentricity
    inc: float  # Inclination (degrees)
    ta: float   # True anomaly (degrees)
    aop: float  # Argument of perigee (degrees)
    raan: float # Right ascension of ascending node (degrees)
