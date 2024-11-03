import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import main
import visualization

X = np.array(main.x_p)
Y = np.array(main.y_p)
F = main.omegas.cpu().numpy()
imgn = main.full_known_disp_concatenate.cpu().numpy()
num_known_points = 22
x_t = main.x_t
y_t = main.x_t

shape = (35, 20, 10)
total_points = shape[0] * shape[1]
num_points_to_mask = total_points - num_known_points

X = X.reshape(shape[0], shape[1])  # Shape (35, 20)
Y = Y.reshape(shape[0], shape[1])  # Shape (35, 20)
imgn = imgn.reshape(shape)  # Shape (35, 20, 10)

mask = np.ones((shape[0], shape[1]), dtype=bool)
indices = np.random.choice(total_points, num_points_to_mask, replace=False)
mask.flat[indices] = False

known_x = X[mask]
known_y = Y[mask]
known_displacement = imgn[mask]

known_points = np.column_stack((known_x, known_y))

unknown_x = X[~mask]
unknown_y = Y[~mask]
unknown_points = np.column_stack((unknown_x, unknown_y))

predicted_displacement = np.zeros_like(imgn)
nmse_values = np.zeros(10)
print('Number of Known Points: ', len(known_points[:, 0]))
for i in range(10):
    rbfi = Rbf(known_points[:, 0], known_points[:, 1], known_displacement[:, i], function='thin_plate')
    predicted_displacement[:, :, i] = rbfi(X, Y)

    # NMSE
    nmse_values[i] = visualization.compute_nmse(imgn[:, :, i], predicted_displacement[:, :, i])

mean_nmse = np.mean(nmse_values)

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
for i in range(10):
    ax = axs[i // 5, i % 5]
    im = ax.imshow(predicted_displacement[:, :, i], cmap='viridis')
    ax.set_title(f"Predicted Mode {i + 1}\nNMSE: {nmse_values[i]:.4f}")
    ax.axis('off')
plt.suptitle(f"Predicted Displacement at Each Frequency, Mean NMSE: {mean_nmse:.4f}, Nkp: {len(known_points[:, 0])}")
plt.show()

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
for i in range(10):
    ax = axs[i // 5, i % 5]
    im = ax.imshow(imgn[:, :, i], cmap='viridis')
    ax.set_title(f"True Mode {i + 1}")
    ax.axis('off')
plt.suptitle("True Displacement at Each Frequency")
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(X, Y, color='lightgrey', s=5, label='All Points')
plt.scatter(known_x, known_y, color='red', s=20, label='Known Points')
plt.title(f"Known Points on Plate (Total Known Points: {num_known_points})")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.show()