import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ECEF_to_geodetic(r, ε = 1e-11):
    # WGS84 constants
    e_elipsiod = 0.08182
    r_e = 6378137.0  # Semi-major axis (meters)
    
    # calculate longitude
    λ = np.arctan2(r[1], r[0])
    
    # calculate initial guess of latitude based on circular earth
    ϕ_prev = np.arctan2(r[2], np.linalg.norm([r[0], r[1]]))
    N = r_e/(1 - (e_elipsiod**2 * np.sin(ϕ_prev)**2))**0.5
    ϕ = np.arctan2((r[2] + N * e_elipsiod**2 * np.sin(ϕ_prev)), 
                   np.linalg.norm([r[0], r[1]]))
    
    while (abs(ϕ_prev - ϕ) > ε):
        ϕ_prev = ϕ
        N = r_e/(1 - (e_elipsiod**2 * np.sin(ϕ_prev)**2))**0.5
        ϕ = np.arctan2((r[2] + N * e_elipsiod**2 * np.sin(ϕ_prev)), 
                       np.linalg.norm([r[0], r[1]]))
    
    h = ((np.linalg.norm([r[0], r[1]])/np.cos(ϕ))-N)
    return np.rad2deg(ϕ), np.rad2deg(λ), h

# Create visualization
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth as a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 6378137 * np.outer(np.cos(u), np.sin(v))
y = 6378137 * np.outer(np.sin(u), np.sin(v))
z = 6378137 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='blue', alpha=0.2)

# Example ECEF point (modify these coordinates as needed)
r = np.array([6378137.0, 0, 0])
lat, lon, alt = ECEF_to_geodetic(r)

# Plot the point
ax.scatter(r[0], r[1], r[2], color='red', s=100)
ax.text(r[0], r[1], r[2], 
        f'ECEF: [{r[0]}\nLat: {lat:.2f}°\nLon: {lon:.2f}°\nAlt: {alt:.2f}m', 
        fontsize=9)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('ECEF Point Visualization')

plt.show()
