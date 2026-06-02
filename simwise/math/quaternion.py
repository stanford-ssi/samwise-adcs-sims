import numpy as np

class Quaternion:
    """
    This class represents a quaternion (scalar-last convention).
    """
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    # Angle axis initialization
    @staticmethod
    def from_angle_axis(angle, axis):
        """
        Initialize a quaternion from an angle and axis.
        """
        return Quaternion(np.sin(angle/2) * axis[0], np.sin(angle/2) * axis[1], np.sin(angle/2) * axis[2], np.cos(angle/2))

    # String representation
    def __repr__(self):
        return f"Quaternion(x={self.x}, y={self.y}, z={self.z}, w={self.w})"

    def __str__(self):
        return f"Quaternion(x={self.x}, y={self.y}, z={self.z}, w={self.w})"

    # Arithmetic operations
    def __add__(self, other): 
        return Quaternion(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def __sub__(self, other):
        return Quaternion(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.x * other, self.y * other, self.z * other, self.w * other)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.x * other, self.y * other, self.z * other, self.w * other)
        return Quaternion(self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y, 
                          self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
                          self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
                          self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z)
    
    # Equality
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z and self.w == other.w

    def __ne__(self, other):
        return self.x != other.x or self.y != other.y or self.z != other.z or self.w != other.w

    # Conjugate
    def conj(self): 
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    # Magnitude and normalization
    def mag(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
    
    def normalize(self):
        return Quaternion(self.x / self.mag(), self.y / self.mag(), self.z / self.mag(), self.w / self.mag())
    
    # Rotation (passive)
    def rot(self, v):
        q_conj = self.conj()
        v_quat = Quaternion(v[0], v[1], v[2], 0)
        result = q_conj * v_quat * self
        return np.array([result.x, result.y, result.z])
    
    def rot_active(self, v): 
        q_conj = self.conj()
        v_quat = Quaternion(v[0], v[1], v[2], 0)
        result = self * v_quat * q_conj # order is reversed for active rotation
        return np.array([result.x, result.y, result.z])