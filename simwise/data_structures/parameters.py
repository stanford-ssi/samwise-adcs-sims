import numpy as np
from simwise.math.quaternion import quaternion_multiply, euler2quaternion

import numpy as np

class Parameter:
    def __init__(self, mean=None, variance=None):
        """
        Base class for parameters with common metadata.

        Args:
            mean: The mean for dispersion (default is the value itself).
            variance: The variance for dispersion (default is 0).
        """
        self.mean = mean
        self.variance = variance


class ScalarParameter(float):
    def __new__(cls, value, mean=None, variance=None):
        # Create the float part
        obj = super().__new__(cls, value)
        # Attach metadata explicitly
        obj._parameter = Parameter(mean, variance)
        return obj

    @property
    def mean(self):
        return self._parameter.mean

    @property
    def variance(self):
        return self._parameter.variance


class ArrayParameter(np.ndarray):
    def __new__(cls, value, mean=None, variance=None):
        # Create the ndarray part
        obj = np.asarray(value).view(cls)
        # Attach metadata explicitly
        obj._parameter = Parameter(mean, variance)
        return obj

    @property
    def mean(self):
        return self._parameter.mean

    @property
    def variance(self):
        return self._parameter.variance


class QuaternionParameter(ArrayParameter):
    """
    Specialized parameter for quaternions. Inherits behavior from ArrayParameter.
    """
    pass


class Parameters:
    def __init__(self, **overrides):
        """
        Initialize a Parameters object with default or overridden Parameter instances.

        Args:
            **overrides: Key-value pairs where keys are parameter names and values are Parameter instances.
        """
        # Time parameters
        self.dt_orbit = ScalarParameter(1)
        self.dt_attitude = ScalarParameter(0.1)
        self.t_start = ScalarParameter(0)
        self.t_end = ScalarParameter(60)

        # Inertia and controls
        self.inertia = ArrayParameter(
            [0.01461922201, 0.0412768466, 0.03235309961]
        )
        self.K_p = ScalarParameter(0.0005) #mean=0.0005, variance=1e-6)
        self.K_d = ScalarParameter(0.005)# mean=0.005, variance=1e-5)
        self.max_torque = ScalarParameter(0.0032)
        self.noise_torque = ScalarParameter(0.00000288)
        self.mu_max = ScalarParameter(0.03)

        # Initial orbit properties
        self.a = ScalarParameter(7000e3)
        self.e = ScalarParameter(0.001)
        self.i = ScalarParameter(0.1)
        self.Ω = ScalarParameter(0.1)
        self.ω = ScalarParameter(0.1)
        self.θ = ScalarParameter(0.1)

        # Attitude initial conditions
        self.q_initial = QuaternionParameter([1, 0, 0, 0]) #variance=(0.05, 0.05, 0.05))
        self.w_initial = ArrayParameter([0.0, 0.2, 0.1]) #mean=[0.0, 0.2, 0.1], variance=1e-4)
        self.q_desired = QuaternionParameter([0.5, 0.5, 0.5, 0.5])
        self.w_desired = ArrayParameter([0, 0, 0])

        # Satellite Cp and Cg
        self.Cp = ArrayParameter([0, 0, 0])
        self.Cg = ArrayParameter([20 / 4, 10 * np.sqrt(2) / 2, 10 * np.sqrt(2) / 2])

        # Apply overrides
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter '{key}' in overrides.")
            if not isinstance(value, (ScalarParameter, ArrayParameter, QuaternionParameter)):
                raise ValueError(f"Override for '{key}' must be an instance of ScalarParameter, ArrayParameter, or QuaternionParameter.")
            setattr(self, key, value)

    def generate_dispersions(self, N):
        """
        Generate N dispersed instances of Parameters.

        Args:
            N: Number of dispersed instances to generate.

        Returns:
            List of Parameters instances with dispersed values.
        """
        dispersed_instances = []

        for _ in range(N):
            dispersed_params = {}
            for attr in dir(self):
                if not attr.startswith("_") and isinstance(getattr(self, attr), (ScalarParameter, ArrayParameter, QuaternionParameter)):
                    param = getattr(self, attr)
                    if param.variance is None:
                        # No dispersion for parameters without mean and variance
                        dispersed_params[attr] = param
                        continue
                    print(f"Generating dispersion for {attr}")
                    if isinstance(param, QuaternionParameter):
                        # Generate random Euler angles for quaternion dispersion
                        e_angles = np.random.normal(0, np.sqrt(param.variance), size=3)
                        random_quaternion = euler2quaternion(e_angles)
                        dispersed_value = quaternion_multiply(param, random_quaternion)
                        dispersed_params[attr] = QuaternionParameter(
                            dispersed_value, mean=param.mean, variance=param.variance
                        )
                    elif isinstance(param, (ScalarParameter, ArrayParameter)):
                        dispersed_value = np.random.normal(
                            param.mean, np.sqrt(param.variance), size=None if isinstance(param, ScalarParameter) else param.shape
                        )
                        if isinstance(param, ScalarParameter):
                            dispersed_params[attr] = ScalarParameter(dispersed_value, mean=param.mean, variance=param.variance)
                        elif isinstance(param, ArrayParameter):
                            dispersed_params[attr] = ArrayParameter(dispersed_value, mean=param.mean, variance=param.variance)

            # Create a new Parameters instance with dispersed parameters
            dispersed_instances.append(Parameters(**dispersed_params))

        return dispersed_instances

    def __repr__(self):
        """Custom string representation for Parameters."""
        result = []
        for attr in dir(self):
            if not attr.startswith("_") and isinstance(getattr(self, attr), (ScalarParameter, ArrayParameter, QuaternionParameter)):
                param = getattr(self, attr)
                result.append(f"{attr}: {param}")
        return "\n".join(result)
