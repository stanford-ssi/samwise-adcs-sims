import numpy as np
from simwise.math.quaternion import quaternion_multiply, euler_to_quaternion
from simwise.utils.time import dt_utc_to_jd
from simwise.math.frame_transforms import generate_ecef_pn_table
from simwise import constants
import datetime
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
        # Simulation configuration
        self.num_dispersions = 100

        # Time parameters
        self.dt_orbit = 120
        self.dt_attitude = 0.1
        self.epoch_jd = dt_utc_to_jd(datetime.datetime(2024, 11, 29, 0, 0, 0))
        self.t_start = 0
        self.t_end = 90 * 60

        # Inertia and controls
        self.inertia = ArrayParameter(
            [0.01461922201, 0.0412768466, 0.03235309961]
        )
        self.K_p = 0.0005
        self.K_d = 0.005
        self.max_torque = ScalarParameter(0.0032)
        self.noise_torque = ScalarParameter(0.00000288)
        self.mu_max = ScalarParameter(0.03)
        self.motor_noise = 0.02 # 2% speed error

        # Control mode and allocation mode —— These are not dispersed
        self.control_mode = "PD"
        self.allocation_mode = "MagicActuators"

        # Attitude target
        self.pointing_mode = "NadirPointingVelocityConstrained"

        # Sensors
        self.magnetic_field_sensor_noise = 15 # 15 nT from RM3100 user manual at 200 counts
        self.photodiode_noise = 0.02 # 2% error based on 25k measurements (at low angle to sun)
                                     # TODO add dark current that follows a Poisson distribution
        self.photodiode_normals = np.array([
            [1, 0, 0],  # +X
            [-1, 0, 0], # -X
            [0, 1, 0],  # +Y
            [0, -1, 0], # -Y
            [0, 0, 1],  # +Z
            [0, 0, -1]  # -Z
        ])

        # Attitude Determination Method
        self.attitude_determination_mode = "EKF"

        # EKF parameters
        # Process noise
        self.Q = np.diag([
            1e-3, 1e-3, 1e-3, 1e-3,     # Quaternion
            1e-3, 1e-3, 1e-3            # Angular velocity
        ])
        # Measurement noise
        self.R = np.diag([
            2e-2, 2e-2, 2e-2,        # Sun sensor
            0.01, 0.01, 0.01            # Magnetometer
        ])
        # Covariance
        self.P = np.diag([
            3e-3, 3e-3, 3e-3, 3e-3,     # Quaternion
            1e-1, 1e-1, 1e-1            # Angular velocity
        ])

        # Initial orbit properties
        self.a = ScalarParameter(constants.EARTH_RADIUS_M + 590e3)
        self.e = ScalarParameter(0.005)
        self.i = ScalarParameter(np.deg2rad(97.5))
        self.Ω = ScalarParameter(0.1)
        self.ω = ScalarParameter(0.1)
        self.θ = ScalarParameter(0.1)
        self.orbit_period = 2 * np.pi * np.sqrt(self.a ** 3 / constants.MU_EARTH)
        self.use_J2 = True

        # Attitude initial conditions
        self.q_initial = QuaternionParameter([1, 0, 0, 0]) #variance=(0.05, 0.05, 0.05))
        self.w_initial = ArrayParameter([0.0, 0.2, 0.1]) #mean=[0.0, 0.2, 0.1], variance=1e-4)
        self.q_desired = [0.5, 0.5, 0.5, 0.5]
        self.w_desired = [0, 0, 0]

        # Satellite Cp and Cg
        self.Cp = ArrayParameter([0, 0, 0])
        self.Cg = ArrayParameter([20 / 4, 10 * np.sqrt(2) / 2, 10 * np.sqrt(2) / 2])
        
        # Apply overrides
        self.overrides = overrides
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter '{key}' in overrides.")
            setattr(self, key, value)
            
        # Generate ECEF to PN table
        # This is NOT a regular parameter of the table, and should not be dispersed
        self.ecef_pn_table = generate_ecef_pn_table(self.epoch_jd, self.t_end)


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
                    if isinstance(param, QuaternionParameter):
                        # Generate random Euler angles for quaternion dispersion
                        e_angles = np.random.normal(0, np.sqrt(param.variance), size=3)
                        random_quaternion = euler_to_quaternion(e_angles)
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

            # Merge dispersed parameters with overrides
            # Value from the right side (i.e. dipersion) takes precedence
            merged_overrides = self.overrides | dispersed_params

            # Create a new Parameters instance with dispersed parameters
            dispersed_instances.append(Parameters(**merged_overrides))

        return dispersed_instances

    def __repr__(self):
        """Custom string representation for Parameters."""
        result = []
        for attr in dir(self):
            if not attr.startswith("_") and isinstance(getattr(self, attr), (ScalarParameter, ArrayParameter, QuaternionParameter)):
                param = getattr(self, attr)
                result.append(f"{attr}: {param}")
        return "\n".join(result)
