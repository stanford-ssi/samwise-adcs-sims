import os
import matplotlib.pyplot as plt
import numpy as np

COASTLINES_COORDINATES_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.path.join('..', 'data', 'coastlines.csv')
)

EARTH_SURFACE_IMAGE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.path.join('..', 'data', 'earth_surface.png')
)

# CONSTANTS
COLORS = ['c', 'm', 'r', 'C3']

def plot_groundtracks(geocentric_states: np.ndarray[float]):
    plt.figure(figsize=(16, 8))
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.title('Aeolus Groundtrack')
    plt.grid(color='lightgray', alpha=0.25)

    # Plot the coastlines
    coast_cords = np.genfromtxt(COASTLINES_COORDINATES_FILE, delimiter=',')
    plt.plot(coast_cords[:, 0], coast_cords[:, 1], 'mo', markersize=0.3)

    # Plot the groundtracks
    i = 0
    for geoc in geocentric_states:
        plt.plot(
            geoc[:, 0],
            geoc[:, 1],
            color=COLORS[i],
            marker='.',
            markersize=0.5,
            linestyle='None',
        )
        i += 1

    # Plot the Earth surface image
    plt.imshow(
        plt.imread(EARTH_SURFACE_IMAGE),
        extent=[-180, 180, -90, 90])

# Function adpated from Alfonso Gonazelez YouTube channel
def plot_eci(rs, args):
    _args = {
        'figsize': (10, 8),
        'labels': [''] * len(rs),
        'colors': COLORS,
        'opacity': 1,
        'traj_lws': 3,
        'dist_unit': 'km',
        'cb_radius': 6378.0,
        'cb_axes': True,
        'cb_axes_mag': 2,
        'cb_cmap': 'Blues',
        'cb_axes_color': 'w',
        'axes_mag': 0.8,
        'axes_custom': None,
        'title': 'Trajectories',
        'legend': False,
        'hide_axes': False,
        'azimuth': False,
        'elevation': False,
        'draw_sun': False,
        'draw_moon': False,
    }
    for key in args.keys():
        _args[key] = args[key]

    fig = plt.figure(figsize=_args['figsize'])
    ax = fig.add_subplot(111, projection='3d')

    # Drawing trajectories
    max_val = 0
    n = 0
    for r in rs:
        r = np.array(r)
        ax.plot(r[:, 0], r[:, 1], r[:, 2],
                color=_args['colors'][n], label=_args['labels'][n],
                zorder=10, linewidth=_args['traj_lws'], alpha=_args['opacity'],)
        ax.plot([r[-1][0]], [r[-1][1]], [r[-1][2]], 'o',
                color=_args['colors'][n], alpha=_args['opacity'], zorder=10)

        max_val = max([r.max(), max_val])
        n += 1

    # Draw Earth
    _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
    _x = _args['cb_radius'] * np.cos(_u) * np.sin(_v)
    _y = _args['cb_radius'] * np.sin(_u) * np.sin(_v)
    _z = _args['cb_radius'] * np.cos(_v)
    ax.plot_surface(_x, _y, _z, cmap=_args['cb_cmap'], zorder=1)

    # Draw Sun
    if _args['draw_sun']:
        sun_x = rs[1][-1][0]
        sun_y = rs[1][-1][1]
        sun_z = rs[1][-1][2]
        _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
        _x = sun_x + 696000 * np.cos(_u) * np.sin(_v)
        _y = sun_y + 696000 * np.sin(_u) * np.sin(_v)
        _z = sun_z + 696000 * np.cos(_v)
        ax.plot_surface(_x, _y, _z, cmap='YlOrBr', zorder=1)

    # Draw Moon
    if _args['draw_moon']:
        moon_x = rs[2][-1][0]
        moon_y = rs[2][-1][1]
        moon_z = rs[2][-1][2]
        _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
        _x = moon_x + 1737.4 * np.cos(_u) * np.sin(_v)
        _y = moon_y + 1737.4 * np.sin(_u) * np.sin(_v)
        _z = moon_z + 1737.4 * np.cos(_v)
        ax.plot_surface(_x, _y, _z, cmap='gray', zorder=1)

    if _args['cb_axes']:
        l = _args['cb_radius'] * _args['cb_axes_mag']
        x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        u, v, w = [[l, 0, 0], [0, -l, 0], [0, 0, l]]
        ax.quiver(x, y, z, u, v, w, color=_args['cb_axes_color'])

    # Labeling axes
    xlabel = 'X (%s)' % _args['dist_unit']
    ylabel = 'Y (%s)' % _args['dist_unit']
    zlabel = 'Z (%s)' % _args['dist_unit']

    # Setting axes properties
    if _args['axes_custom'] is not None:
        max_val = _args['axes_custom']
    else:
        max_val *= _args['axes_mag']

    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_box_aspect([1, 1, 1])
    ax.set_aspect('auto')

    if _args['azimuth'] is not False:
        ax.view_init(elev=_args['elevation'],
                     azim=_args['azimuth'])

    if _args['hide_axes']:
        ax.set_axis_off()

    if _args['legend']:
        plt.legend()