import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d

body_parts = [
    'right_wrist', 'left_wrist', 'right_shin', 'left_shin', 
    'right_thigh', 'left_thigh', 'right_arm', 'left_arm', 
    'right_shoulder', 'left_shoulder', 'forehead', 'right_foot', 
    'left_foot', 'back', 'right_shirt_pocket', 'left_shirt_pocket', 
    'chest', 'Necklace', 'belt', 'left_ear', 'right_ear'
]
# Define the path to the saved .npz file
body_part = "right_wrist"  # Replace with the actual body part name
save_dir = "Data/Defuse_Bomb"  # Replace with the actual save directory path
file_path = os.path.join(save_dir, f'{body_part}_v2.npz')

# Load the data from the .npz file
data = np.load(file_path)

# Function to convert Euler angles to direction vector
def euler_to_vector(euler_angles):
    rotation = R.from_euler('xyz', euler_angles)  # 'xyz' means rotations are applied around x, y, and z axes respectively
    return rotation.apply([1, 0, 0])  # Assuming a unit vector along the X-axis to show orientation

# Function to convert Euler angles (normalized between -1 and 1) to direction vector
def norm_to_vector(euler_angles_normalized):
    # Convert normalized angles (-1 to 1) into degrees (-180 to 180)
    euler_angles_degrees = euler_angles_normalized * 180
    
    # Create rotation object from Euler angles in degrees
    rotation = R.from_euler('xyz', euler_angles_degrees, degrees=True)  # 'xyz' is the sequence of rotations

    # Apply rotation to a unit vector along the X-axis to get the direction vector
    return rotation.apply([1, 0, 0])

# Extract data arrays
positions = data['positions']
orientations = data['orientations']
linear_acceleration = data['linear_acceleration']
linear_acceleration_g = data['linear_acceleration_with_gravity']
angular_velocity = data['angular_velocity']

# Define the original data shapes
target_length = angular_velocity.shape[0]

# Time indices for the original data
original_indices_pos = np.linspace(0, 1, positions.shape[0])
original_indices_orient = np.linspace(0, 1, orientations.shape[0])
original_indices_lin_acc = np.linspace(0, 1, linear_acceleration.shape[0])
original_indices_lin_acc_g = np.linspace(0, 1, linear_acceleration_g.shape[0])
original_indices_ang_vel = np.linspace(0, 1, angular_velocity.shape[0])

# New indices for interpolation
new_indices = np.linspace(0, 1, target_length)

# Interpolating positions
positions = interp1d(original_indices_pos, positions, axis=0, kind='linear')(new_indices)

# Interpolating orientations
orientations = interp1d(original_indices_orient, orientations, axis=0, kind='linear')(new_indices)

# Interpolating linear acceleration (without gravity)
linear_acceleration = interp1d(original_indices_lin_acc, linear_acceleration, axis=0, kind='linear')(new_indices)

# Interpolating linear acceleration (with gravity)
linear_acceleration_g = interp1d(original_indices_lin_acc_g, linear_acceleration_g, axis=0, kind='linear')(new_indices)

print(positions.shape,orientations.shape,linear_acceleration.shape, angular_velocity.shape)


fig = plt.figure(figsize=(14, 10))

# 1. 3D Plot of Positions and Orientations
ax1 = fig.add_subplot(221, projection='3d')

# Extract the x, y, z positions
x = positions[:, 0]
y = positions[:, 1]
z = positions[:, 2]

# Normalize orientation data for visualization
norm = Normalize()
orientation_norm = norm(np.linalg.norm(orientations, axis=1))

# Plot the positions in 3D
sc = ax1.scatter(x, y, z, c=orientation_norm, cmap='coolwarm', s=50, label='Position')

# Loop through the positions and orientations to plot direction vectors
for i in range(len(positions)):
    orientation_vector = norm_to_vector(orientations[i])
    
    # Define the start and end of the orientation arrow
    start = positions[i]
    end = start + orientation_vector * 0.1  # Scaling the orientation vector for visualization
    
    # Plot an arrow for the orientation
    ax1.quiver(start[0], start[1], start[2], 
              orientation_vector[0], orientation_vector[1], orientation_vector[2], 
              color='k', length=0.01, normalize=True)

# Labels and plot adjustments for 3D plot
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title(f'3D Positions and Orientations of {body_part.capitalize()}')

# Add colorbar for orientation magnitudes
cbar = plt.colorbar(sc, ax=ax1)
cbar.set_label('Orientation Magnitude')

# 2. Angular Velocity Plot
ax2 = fig.add_subplot(222)
time = np.arange(len(angular_velocity))

# Plot the angular velocity components
ax2.plot(time, angular_velocity[:, 0], label='Angular Velocity X', color='r')
ax2.plot(time, angular_velocity[:, 1], label='Angular Velocity Y', color='g')
ax2.plot(time, angular_velocity[:, 2], label='Angular Velocity Z', color='b')

ax2.set_xlabel('Time')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.set_title('Angular Velocity over Time')
ax2.legend()

# 3. Linear Acceleration without Gravity
ax3 = fig.add_subplot(223)

# Plot the linear acceleration components
ax3.plot(time, linear_acceleration[:, 0], label='Linear Acceleration X', color='r')
ax3.plot(time, linear_acceleration[:, 1], label='Linear Acceleration Y', color='g')
ax3.plot(time, linear_acceleration[:, 2], label='Linear Acceleration Z', color='b')

ax3.set_xlabel('Time')
ax3.set_ylabel('Linear Acceleration (m/s^2)')
ax3.set_title('Linear Acceleration (No Gravity) over Time')
ax3.legend()

# 4. Linear Acceleration with Gravity
ax4 = fig.add_subplot(224)

# Plot the linear acceleration with gravity components
ax4.plot(time, linear_acceleration_g[:, 0], label='Linear Acceleration (Gravity) X', color='r')
ax4.plot(time, linear_acceleration_g[:, 1], label='Linear Acceleration (Gravity) Y', color='g')
ax4.plot(time, linear_acceleration_g[:, 2], label='Linear Acceleration (Gravity) Z', color='b')

ax4.set_xlabel('Time')
ax4.set_ylabel('Linear Acceleration with Gravity (m/s^2)')
ax4.set_title('Linear Acceleration (With Gravity) over Time')
ax4.legend()

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()