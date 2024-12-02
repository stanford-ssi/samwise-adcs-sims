import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from simwise.constants import *
from simwise.math.quaternion import *
from simwise.forces.gravity_gradient import *

def analyze_gravity_gradient(I, altitude_km=450):
    R_EARTH = 6371000  # Earth radius in meters
    r_orbit = R_EARTH + altitude_km * 1000  # Convert to meters
    r_vector = np.array([r_orbit, 0.0, 0.0])

    print("\nGravity Gradient Analysis for {}km orbit".format(altitude_km))
    print("Orbital radius: {:.1f} km".format(r_orbit/1000))
    print("\nPrincipal moments of inertia:")
    print("Ixx = {:.2e} kg⋅m²".format(I[0]))
    print("Iyy = {:.2e} kg⋅m²".format(I[1]))
    print("Izz = {:.2e} kg⋅m²".format(I[2]))

    angles = np.linspace(0, np.pi, 181)  # 0 to 180 degrees
    torques_roll = []
    torques_pitch = []
    torques_yaw = []

    for angle in angles:
        q_roll = np.array([np.cos(angle/2), np.sin(angle/2), 0, 0])
        q_pitch = np.array([np.cos(angle/2), 0, np.sin(angle/2), 0])
        q_yaw = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])

        torques_roll.append(gravity_gradient_perturbation_torque(q_roll, r_vector, I))
        torques_pitch.append(gravity_gradient_perturbation_torque(q_pitch, r_vector, I))
        torques_yaw.append(gravity_gradient_perturbation_torque(q_yaw, r_vector, I))

    torques_roll = np.array(torques_roll)
    torques_pitch = np.array(torques_pitch)
    torques_yaw = np.array(torques_yaw)

    max_theoretical = (3 * MU_EARTH / (2 * r_orbit**3)) * max(abs(I[0]-I[1]), abs(I[1]-I[2]), abs(I[2]-I[0]))
    print("Theoretical maximum torque: {:.2e} Nm".format(max_theoretical))

    return angles, torques_roll, torques_pitch, torques_yaw, max_theoretical

def plot_gravity_gradient_results(angles, torques_roll, torques_pitch, torques_yaw, max_theoretical):
    angles_deg = np.degrees(angles)

    fig = make_subplots(rows=2, cols=2, 
                        shared_xaxes=True, 
                        vertical_spacing=0.1,
                        horizontal_spacing=0.1,
                        subplot_titles=("Roll", "Pitch", "Yaw", "Torque Magnitudes"))

    colors = {'X': '#4363d8', 'Y': '#3cb44b', 'Z': '#e6194B'}

    for i, (torques, axis) in enumerate([(torques_roll, 'Roll'), (torques_pitch, 'Pitch'), (torques_yaw, 'Yaw')]):
        row = i // 2 + 1
        col = i % 2 + 1
        for j, component in enumerate(['X', 'Y', 'Z']):
            fig.add_trace(go.Scatter(x=angles_deg, y=torques[:, j],
                                     mode='lines', name=f'{axis} - {component}',
                                     line=dict(color=colors[component], width=2)),
                          row=row, col=col)
        
        fig.add_trace(go.Scatter(x=angles_deg, y=[max_theoretical]*len(angles),
                                 mode='lines', name="Theoretical Maximum",
                                 line=dict(color='#000000', dash='dash', width=1)),
                      row=row, col=col)

    # Add torque magnitudes plot
    for i, (torques, axis) in enumerate([(torques_roll, 'Roll'), (torques_pitch, 'Pitch'), (torques_yaw, 'Yaw')]):
        magnitudes = np.linalg.norm(torques, axis=1)
        fig.add_trace(go.Scatter(x=angles_deg, y=magnitudes,
                                 mode='lines', name=f'{axis} Magnitude',
                                 line=dict(color=colors[list(colors.keys())[i]], width=2)),
                      row=2, col=2)

    fig.add_trace(go.Scatter(x=angles_deg, y=[max_theoretical]*len(angles),
                             mode='lines', name="Theoretical Maximum",
                             line=dict(color='#000000', dash='dash', width=1)),
                  row=2, col=2)

    for i in range(1, 5):
        row = (i-1) // 2 + 1
        col = (i-1) % 2 + 1
        fig.update_yaxes(
            title_text="Torque (Nm)",
            type='linear',
            tickformat='.2e',
            exponentformat='e',
            gridcolor='#eee',
            zeroline=True,
            zerolinecolor='#eee',
            showticklabels=True,
            row=row, col=col
        )
        fig.update_xaxes(
            title_text="Angle (degrees)",
            gridcolor='#eee',
            zeroline=True,
            zerolinecolor='#eee',
            showticklabels=True,
            row=row, col=col
        )

    fig.update_layout(
        title="Gravity Gradient Torque vs. Angle",
        height=800,
        width=1400,
        showlegend=True
    )

    fig.show()
    
def analyze_gravity_gradient_2d(I, altitude_km=450, num_points=50):
    R_EARTH = 6371000  # Earth radius in meters
    r_orbit = R_EARTH + altitude_km * 1000  # Convert to meters
    r_vector = np.array([r_orbit, 0.0, 0.0])

    pitch_angles = np.linspace(0, np.pi/2, num_points)
    yaw_angles = np.linspace(0, np.pi/2, num_points)
    yaw_mesh, pitch_mesh = np.meshgrid(yaw_angles, pitch_angles)  # Swapped order here
    
    torque_magnitudes = np.zeros_like(pitch_mesh)

    for i in range(num_points):
        for j in range(num_points):
            pitch = pitch_angles[i]
            yaw = yaw_angles[j]
            
            q_pitch = np.array([np.cos(pitch/2), 0, np.sin(pitch/2), 0])
            q_yaw = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
            q = quaternion_multiply(q_yaw, q_pitch)
            
            torque = gravity_gradient_perturbation_torque(q, r_vector, I)
            torque_magnitudes[i, j] = np.linalg.norm(torque)

    max_theoretical = (3 * MU_EARTH / (2 * r_orbit**3)) * max(abs(I[0]-I[1]), abs(I[1]-I[2]), abs(I[2]-I[0]))

    return yaw_mesh, pitch_mesh, torque_magnitudes, max_theoretical  # Swapped order here

def plot_gravity_gradient_2d(yaw_mesh, pitch_mesh, torque_magnitudes, max_theoretical):
    # Calculate the number of contour levels
    num_levels = 50  # Increase this for more level sets
    min_torque = np.min(torque_magnitudes)
    max_torque = np.max(torque_magnitudes)
    contour_levels = np.linspace(min_torque, max_torque, num_levels)

    fig = go.Figure(data=go.Contour(
        x=np.degrees(yaw_mesh[0]),  # Yaw on x-axis
        y=np.degrees(pitch_mesh[:,0]),  # Pitch on y-axis
        z=torque_magnitudes,
        colorbar=dict(title='Torque Magnitude (Nm)', tickformat='.2e'),
        contours=dict(
            start=min_torque,
            end=max_torque,
            size=(max_torque - min_torque) / num_levels,
            showlabels=False,
            labelfont=dict(size=8, color='white')
        ),
        line_smoothing=0.85,
    ))

    # Add contour lines
    fig.add_trace(go.Contour(
        x=np.degrees(yaw_mesh[0]),  # Yaw on x-axis
        y=np.degrees(pitch_mesh[:,0]),  # Pitch on y-axis
        z=torque_magnitudes,
        contours=dict(
            start=min_torque,
            end=max_torque,
            size=(max_torque - min_torque) / num_levels,
            showlabels=False,
            coloring='none'
        ),
        line=dict(width=0.3, color='white'),
        showscale=False
    ))

    fig.update_layout(
        title='Gravity Gradient Torque Magnitude',
        xaxis_title='Yaw Angle (degrees)',
        yaxis_title='Pitch Angle (degrees)',
        width=800,
        height=700
    )

    fig.show()

def analyze_gravity_gradient_3d(I, altitude_km=450, num_points=50):
    R_EARTH = 6371000  # Earth radius in meters
    r_orbit = R_EARTH + altitude_km * 1000  # Convert to meters
    r_vector = np.array([r_orbit, 0.0, 0.0])

    print("\nGravity Gradient Analysis for {}km orbit".format(altitude_km))
    print("Orbital radius: {:.1f} km".format(r_orbit/1000))
    print("\nPrincipal moments of inertia:")
    print("Ixx = {:.2e} kg⋅m²".format(I[0]))
    print("Iyy = {:.2e} kg⋅m²".format(I[1]))
    print("Izz = {:.2e} kg⋅m²".format(I[2]))

    yaw_angles = np.linspace(0, np.pi/2, num_points)
    pitch_angles = np.linspace(0, np.pi/2, num_points)
    yaw_mesh, pitch_mesh = np.meshgrid(yaw_angles, pitch_angles)  # Swapped order here
    
    torque_magnitudes = np.zeros_like(yaw_mesh)

    for i in range(num_points):
        for j in range(num_points):
            pitch = pitch_angles[i]
            yaw = yaw_angles[j]
            
            q_pitch = np.array([np.cos(pitch/2), 0, np.sin(pitch/2), 0])
            q_yaw = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
            q = quaternion_multiply(q_yaw, q_pitch)
            
            torque = gravity_gradient_perturbation_torque(q, r_vector, I)
            torque_magnitudes[i, j] = np.linalg.norm(torque)

    max_theoretical = (3 * MU_EARTH / (2 * r_orbit**3)) * max(abs(I[0]-I[1]), abs(I[1]-I[2]), abs(I[2]-I[0]))
    print("Theoretical maximum torque: {:.2e} Nm".format(max_theoretical))

    return yaw_mesh, pitch_mesh, torque_magnitudes, max_theoretical  # Swapped order here

def plot_gravity_gradient_3d(yaw_mesh, pitch_mesh, torque_magnitudes, max_theoretical):
    fig = go.Figure(data=[go.Surface(x=np.degrees(yaw_mesh), 
                                     y=np.degrees(pitch_mesh), 
                                     z=torque_magnitudes,
                                     colorbar=dict(tickformat='.2e', exponentformat='e', len=0.75))])

    fig.update_layout(
        title={
            'text': 'Gravity Gradient Torque Magnitude',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        scene = dict(
            xaxis_title='Yaw Angle (degrees)',
            yaxis_title='Pitch Angle (degrees)',
            zaxis_title='Torque Magnitude (Nm)',
            xaxis = dict(tickformat='.1f'),
            yaxis = dict(tickformat='.1f'),
            zaxis = dict(tickformat='.2e', exponentformat='e'),
            aspectmode='cube'
        ),
        width=1200,
        height=800,
        autosize=False,
        margin=dict(r=50, l=50, b=50, t=90),
    )

    fig.show()

def analyze_gravity_gradient_roll_animation(I, altitude_km=450, num_points=50, num_frames=180):
    R_EARTH = 6371000  # Earth radius in meters
    r_orbit = R_EARTH + altitude_km * 1000  # Convert to meters
    r_vector = np.array([r_orbit, 0.0, 0.0])

    yaw_angles = np.linspace(0, np.pi/2, num_points)
    pitch_angles = np.linspace(0, np.pi/2, num_points)
    yaw_mesh, pitch_mesh = np.meshgrid(yaw_angles, pitch_angles)
    
    roll_angles = np.linspace(0, np.pi, num_frames)
    
    torque_magnitudes_frames = []

    for roll in roll_angles:
        torque_magnitudes = np.zeros_like(yaw_mesh)
        for i in range(num_points):
            for j in range(num_points):
                pitch = pitch_angles[i]
                yaw = yaw_angles[j]
                
                q_roll = np.array([np.cos(roll/2), np.sin(roll/2), 0, 0])
                q_pitch = np.array([np.cos(pitch/2), 0, np.sin(pitch/2), 0])
                q_yaw = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
                q = quaternion_multiply(q_roll, quaternion_multiply(q_yaw, q_pitch))
                
                torque = gravity_gradient_perturbation_torque(q, r_vector, I)
                torque_magnitudes[i, j] = np.linalg.norm(torque)
        
        torque_magnitudes_frames.append(torque_magnitudes)

    max_theoretical = (3 * MU_EARTH / (2 * r_orbit**3)) * max(abs(I[0]-I[1]), abs(I[1]-I[2]), abs(I[2]-I[0]))

    return yaw_mesh, pitch_mesh, torque_magnitudes_frames, max_theoretical, roll_angles

def plot_gravity_gradient_roll_animation(yaw_mesh, pitch_mesh, torque_magnitudes_frames, max_theoretical, roll_angles):
    min_torque = np.min(torque_magnitudes_frames)
    max_torque = np.max(torque_magnitudes_frames)
    num_levels = 50
    contour_levels = np.linspace(min_torque, max_torque, num_levels)

    fig = go.Figure()

    for i, torque_magnitudes in enumerate(torque_magnitudes_frames):
        visible = (i == 0)
        
        contour = go.Contour(
            x=np.degrees(yaw_mesh[0]),
            y=np.degrees(pitch_mesh[:,0]),
            z=torque_magnitudes,
            colorbar=dict(title='Torque Magnitude (Nm)', tickformat='.2e'),
            contours=dict(
                start=min_torque,
                end=max_torque,
                size=(max_torque - min_torque) / num_levels,
                showlabels=False,
                labelfont=dict(size=8, color='white')
            ),
            line_smoothing=0.85,
            visible=visible
        )
        
        contour_lines = go.Contour(
            x=np.degrees(yaw_mesh[0]),
            y=np.degrees(pitch_mesh[:,0]),
            z=torque_magnitudes,
            contours=dict(
                start=min_torque,
                end=max_torque,
                size=(max_torque - min_torque) / num_levels,
                showlabels=False,
                coloring='none'
            ),
            line=dict(width=0.3, color='white'),
            showscale=False,
            visible=visible
        )
        
        fig.add_trace(contour)
        fig.add_trace(contour_lines)

    steps = []
    for i in range(len(roll_angles)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Gravity Gradient Torque Magnitude (Roll: {np.degrees(roll_angles[i]):.1f}°)"}],
            label=f"{np.degrees(roll_angles[i]):.1f}°"
        )
        step["args"][0]["visible"][2*i] = True
        step["args"][0]["visible"][2*i+1] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Roll Angle: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title='Gravity Gradient Torque Magnitude',
        xaxis_title='Yaw Angle (degrees)',
        yaxis_title='Pitch Angle (degrees)',
        width=800,
        height=700
    )

    fig.show()


def analyze_gravity_gradient_roll_animation_gif(I, altitude_km=450, num_points=50, num_frames=180):
    R_EARTH = 6371000  # Earth radius in meters
    r_orbit = R_EARTH + altitude_km * 1000  # Convert to meters
    r_vector = np.array([r_orbit, 0.0, 0.0])

    yaw_angles = np.linspace(0, np.pi/2, num_points)
    pitch_angles = np.linspace(0, np.pi/2, num_points)
    yaw_mesh, pitch_mesh = np.meshgrid(yaw_angles, pitch_angles)
    
    roll_angles = np.linspace(0, np.pi, num_frames)
    
    torque_magnitudes_frames = []

    for roll in roll_angles:
        torque_magnitudes = np.zeros_like(yaw_mesh)
        for i in range(num_points):
            for j in range(num_points):
                pitch = pitch_angles[i]
                yaw = yaw_angles[j]
                
                q_roll = np.array([np.cos(roll/2), np.sin(roll/2), 0, 0])
                q_pitch = np.array([np.cos(pitch/2), 0, np.sin(pitch/2), 0])
                q_yaw = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
                q = quaternion_multiply(q_roll, quaternion_multiply(q_yaw, q_pitch))
                
                torque = gravity_gradient_perturbation_torque(q, r_vector, I)
                torque_magnitudes[i, j] = np.linalg.norm(torque)
        
        torque_magnitudes_frames.append(torque_magnitudes)

    max_theoretical = (3 * MU_EARTH / (2 * r_orbit**3)) * max(abs(I[0]-I[1]), abs(I[1]-I[2]), abs(I[2]-I[0]))

    return yaw_mesh, pitch_mesh, torque_magnitudes_frames, max_theoretical, roll_angles

def plot_gravity_gradient_roll_animation_gif(yaw_mesh, pitch_mesh, torque_magnitudes_frames, max_theoretical, roll_angles):
    min_torque = np.min(torque_magnitudes_frames)
    max_torque = np.max(torque_magnitudes_frames)
    num_levels = 50
    contour_levels = np.linspace(min_torque, max_torque, num_levels)

    # Create a temporary directory to store individual frames
    if not os.path.exists('temp_frames'):
        os.makedirs('temp_frames')

    for i, (torque_magnitudes, roll_angle) in enumerate(zip(torque_magnitudes_frames, roll_angles)):
        fig = go.Figure()

        contour = go.Contour(
            x=np.degrees(yaw_mesh[0]),
            y=np.degrees(pitch_mesh[:,0]),
            z=torque_magnitudes,
            colorbar=dict(title='Torque Magnitude (Nm)', tickformat='.2e'),
            contours=dict(
                start=min_torque,
                end=max_torque,
                size=(max_torque - min_torque) / num_levels,
                showlabels=False,
                labelfont=dict(size=8, color='white')
            ),
            line_smoothing=0.85,
        )
        
        contour_lines = go.Contour(
            x=np.degrees(yaw_mesh[0]),
            y=np.degrees(pitch_mesh[:,0]),
            z=torque_magnitudes,
            contours=dict(
                start=min_torque,
                end=max_torque,
                size=(max_torque - min_torque) / num_levels,
                showlabels=False,
                coloring='none'
            ),
            line=dict(width=0.3, color='white'),
            showscale=False,
        )
        
        fig.add_trace(contour)
        fig.add_trace(contour_lines)

        fig.update_layout(
            title=f'Gravity Gradient Torque Magnitude (Roll: {np.degrees(roll_angle):.1f}°)',
            xaxis_title='Yaw Angle (degrees)',
            yaxis_title='Pitch Angle (degrees)',
            width=800,
            height=700
        )

        # Save each frame as an image
        fig.write_image(f'temp_frames/frame_{i:03d}.png')

    # Create a GIF from the saved frames
    with imageio.get_writer('gravity_gradient_animation.gif', mode='I', duration=0.1) as writer:
        for i in range(len(roll_angles)):
            image = imageio.imread(f'temp_frames/frame_{i:03d}.png')
            writer.append_data(image)

    # Clean up temporary files
    for i in range(len(roll_angles)):
        os.remove(f'temp_frames/frame_{i:03d}.png')
    os.rmdir('temp_frames')

    print("Animation saved as 'gravity_gradient_animation.gif'")


# CALL TESTS
I_sample = np.array([14.21389908e-9, 40.87154478e-9, 32.01974461e-9])  # kg⋅m²
angles, torques_roll, torques_pitch, torques_yaw, max_theoretical = analyze_gravity_gradient(I_sample)
plot_gravity_gradient_results(angles, torques_roll, torques_pitch, torques_yaw, max_theoretical)

yaw_mesh, pitch_mesh, torque_magnitudes, max_theoretical = analyze_gravity_gradient_3d(I_sample)
plot_gravity_gradient_3d(yaw_mesh, pitch_mesh, torque_magnitudes, max_theoretical)

yaw_mesh, pitch_mesh, torque_magnitudes, max_theoretical = analyze_gravity_gradient_2d(I_sample, num_points=50)
plot_gravity_gradient_2d(yaw_mesh, pitch_mesh, torque_magnitudes, max_theoretical)

yaw_mesh, pitch_mesh, torque_magnitudes_frames, max_theoretical, roll_angles = analyze_gravity_gradient_roll_animation(I_sample, num_points=50, num_frames=180)
plot_gravity_gradient_roll_animation(yaw_mesh, pitch_mesh, torque_magnitudes_frames, max_theoretical, roll_angles)

# MAKE ROLL ANIMATION GIF
# import imageio
# import os
# yaw_mesh, pitch_mesh, torque_magnitudes_frames, max_theoretical, roll_angles = analyze_gravity_gradient_roll_animation_gif(I_sample, num_points=50, num_frames=180)
# plot_gravity_gradient_roll_animation_gif(yaw_mesh, pitch_mesh, torque_magnitudes_frames, max_theoretical, roll_angles)