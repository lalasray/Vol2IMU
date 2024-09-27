import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

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
file_path = os.path.join(save_dir, f'{body_part}_v3.npz')

# Load the data from the .npz file
data = np.load(file_path)

# Function to convert Euler angles to direction vector
def euler_to_vector(euler_angles):
    rotation = R.from_radians('xyz', euler_angles)  # 'xyz' means rotations are applied around x, y, and z axes respectively
    return rotation.apply([1, 0, 0])  # Assuming a unit vector along the X-axis to show orientation

# Extract data arrays
positions = data['positions']
orientations = data['orientations']
linear_acceleration = data['linear_acceleration']
linear_acceleration_g = data['linear_acceleration_with_gravity']
angular_velocity = data['angular_velocity']