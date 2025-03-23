import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load('calibration_homography.npy')

# Print basic information
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Min value:", data.min())
print("Max value:", data.max())
print("Mean value:", data.mean())

# Display the data
print("\nMatrix content:")
print(data)

# If you want to visualize it (for matrices)
plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='viridis')
plt.colorbar(label='Value')
plt.title('Visualization of .npy file data')
plt.tight_layout()
plt.show()