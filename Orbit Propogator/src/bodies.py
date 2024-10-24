import math

class Body:
    """
    A generic celestial body class, representing common characteristics of planets, moons, stars, etc.
    """

    def __init__(self, name: str, mass: float, radius: float, gravitational_parameter: float, rotation_rate: float):
        """
        Initialize a celestial body with basic properties.
        
        :param name: Name of the celestial body.
        :param mass: Mass of the body in kilograms.
        :param radius: Mean radius of the body in kilometers.
        :param gravitational_parameter: Gravitational parameter (mu) of the body in km^3/s^2.
        :param rotation_rate: Rotational rate of the body in rad/s.
        """
        self.name = name
        self.mass = mass
        self.radius = radius
        self.gravitational_parameter = gravitational_parameter
        self.rotation_rate = rotation_rate

    def gravity_at_altitude(self, altitude: float) -> float:
        """
        Calculate the gravitational acceleration at a specific altitude above the body's surface.
        
        :param altitude: Altitude in kilometers above the body's surface.
        :return: Gravitational acceleration at that altitude in m/s^2.
        """
        r = self.radius + altitude  # Distance from the center of the body
        gravity = self.gravitational_parameter * 1e9 / (r * 1000) ** 2  # Convert km to meters for the calculation
        return gravity

    def escape_velocity(self, altitude: float) -> float:
        """
        Compute the escape velocity at a specific altitude above the body's surface.
        
        :param altitude: Altitude above the body's surface in kilometers.
        :return: Escape velocity in kilometers per second.
        """
        r = self.radius + altitude  # Distance from the body's center in kilometers
        escape_velocity = math.sqrt(2 * self.gravitational_parameter / r)  # Escape velocity in km/s
        return escape_velocity

    def __str__(self):
        return f"{self.name}: Mass = {self.mass:.3e} kg, Radius = {self.radius:.3f} km"




Earth = Body(
    name="Earth",
    mass=5.97219e24,
    radius=6378.137,
    gravitational_parameter=398600.4418,
    rotation_rate=7.2921159e-5
)