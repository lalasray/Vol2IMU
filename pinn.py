import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load your data
def load_data(file_path):
    data = np.load(file_path)
    positions = data['positions']
    linear_acceleration = data['linear_acceleration']
    
    # Normalize positions and linear acceleration using min-max normalization
    positions_min = positions.min(axis=0)
    positions_max = positions.max(axis=0)
    linear_acceleration_min = linear_acceleration.min(axis=0)
    linear_acceleration_max = linear_acceleration.max(axis=0)
    
    positions_normalized = (positions - positions_min) / (positions_max - positions_min)
    linear_acceleration_normalized = (linear_acceleration - linear_acceleration_min) / (linear_acceleration_max - linear_acceleration_min)
    
    return positions_normalized, linear_acceleration_normalized, (positions_min, positions_max), (linear_acceleration_min, linear_acceleration_max)

# Interpolation function to align data lengths
def interpolate_data(original_data, target_length):
    original_indices = np.linspace(0, 1, original_data.shape[0])
    new_indices = np.linspace(0, 1, target_length)
    return interp1d(original_indices, original_data, axis=0, kind='linear')(new_indices)

# Create sliding windows
def create_windows(data, window_size=50, step_size=20):
    n_samples = data.shape[0]
    n_windows = (n_samples - window_size) // step_size + 1
    
    windows = np.array([data[i:i+window_size] for i in range(0, n_samples - window_size + 1, step_size)])
    return windows

# Define the PINN model with 1D CNN layers and residual connections
class CNNPINN(nn.Module):
    def __init__(self):
        super(CNNPINN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc1 = nn.Linear(128 * 50, 64)  # Adjust input dimension based on window size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 3)  # Output: Linear acceleration

        self.res_conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.res_conv2 = nn.Conv1d(64, 128, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, channels, sequence_length)
        
        # First convolutional block with residual connection
        identity = self.res_conv1(x).detach()  # Use detach() to avoid in-place modification
        out = torch.tanh(self.bn1(self.conv1(x)))
        out += identity
        
        # Second convolutional block with residual connection
        identity = self.res_conv2(out).detach()  # Use detach() to avoid in-place modification
        out = torch.tanh(self.bn2(self.conv2(out)))
        out += identity
        
        # Flatten the output
        out = out.view(out.size(0), -1)
        
        # Fully connected layers with dropout
        out = torch.tanh(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out



# Data Loss
def data_loss(y_true, y_pred):
    return nn.MSELoss()(y_pred, y_true)

# Residual Loss (Physics Loss)
def residual_loss(y_pred, sampling_rate):
    d2x_dt2 = (torch.roll(y_pred, -1, dims=0) - 2 * y_pred + torch.roll(y_pred, 1, dims=0)) / (1/sampling_rate)**2
    return nn.MSELoss()(d2x_dt2[1:-1], y_pred[1:-1])  # Avoid boundary issues

# Define limits for accelerometer measurements
min_acceleration = -10.0
max_acceleration = 10.0

# Modified Total Loss Function with Measurement Limits
def total_loss_with_limits(y_true, y_pred, sampling_rate, lambda1=1.0, lambda2=1.0):
    y_pred_clipped = torch.clamp(y_pred, min_acceleration, max_acceleration)
    data_loss_value = data_loss(y_true, y_pred_clipped)
    res_loss_value = residual_loss(y_pred_clipped, sampling_rate)
    penalty = torch.sum((y_pred - y_pred_clipped) ** 2)
    return lambda1 * data_loss_value + lambda2 * res_loss_value + penalty

# Main function to run the PINN
# Main function to run the PINN
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

    # Define parameters
    body_part = "right_wrist"
    save_dir = "Data/Defuse_Bomb"
    file_path = os.path.join(save_dir, f'{body_part}_v2.npz')

    # Load the data
    positions, linear_acceleration, (pos_min, pos_max), (acc_min, acc_max) = load_data(file_path)
    
    # Create sliding windows for batching
    positions_windows = create_windows(positions, window_size=50, step_size=1)
    linear_acceleration_windows = create_windows(linear_acceleration, window_size=50, step_size=1)

    # Convert numpy arrays to PyTorch tensors and move to device
    positions_tensor = torch.tensor(positions_windows, dtype=torch.float32).to(device)
    linear_acceleration_tensor = torch.tensor(linear_acceleration_windows, dtype=torch.float32).to(device)

    # Create the model and move it to the device
    model = CNNPINN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Training loop
    epochs = 100000
    batch_size = 32  # Define your batch size here
    sampling_rate = 50  # Hz
    n_samples = positions_tensor.size(0)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Shuffle indices for batching
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            # Create mini-batch
            batch_positions = positions_tensor[batch_indices]
            batch_linear_acceleration = linear_acceleration_tensor[batch_indices]

            # Forward pass
            outputs = model(batch_positions)

            # Compute total loss
            loss = total_loss_with_limits(batch_linear_acceleration, outputs, sampling_rate)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Model evaluation (Optional)
    model.eval()
    with torch.no_grad():
        predictions = model(positions_tensor)

    # Move predictions back to CPU for visualization
    predictions_cpu = predictions.cpu().numpy()
    linear_acceleration_cpu = linear_acceleration_tensor.cpu().numpy()

    # Reverse normalization for visualization
    linear_acceleration_cpu = linear_acceleration_cpu * (acc_max - acc_min) + acc_min
    predictions_cpu = predictions_cpu * (acc_max - acc_min) + acc_min

    # Visualization of predicted vs real for each component (x, y, z)
    time = np.arange(predictions_cpu.shape[0]) / sampling_rate
    plt.figure(figsize=(12, 8))

    for i, label in enumerate(['X', 'Y', 'Z']):
        plt.plot(time, linear_acceleration_cpu[:, i], label=f'Real Linear Acceleration {label}', color='b')
        plt.plot(time, predictions_cpu[:, i], label=f'Predicted Linear Acceleration {label}', linestyle='--', color='r')

    plt.title('Predicted vs Real Linear Acceleration (X, Y, Z)')
    plt.xlabel('Time (s)')
    plt.ylabel('Linear Acceleration (m/s²)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
