
'''import matplotlib.pyplot as plt
import numpy as np

x = [18, 16, 14, 12, 10, 8, 6]

y1 = 20 * np.log10([0.062243, 0.093795, 0.113774, 0.217943, 0.368998, 0.7127060, 0.8026180])
y2 = 20 * np.log10([0.077352, 0.089527, 0.181085, 0.383054, 0.519841, 1.209315, 1.342902])

plt.plot(x, y1, label='w/ Governing Eq', color='blue', marker='o')
plt.plot(x, y2, label='w/o Governing Eq', color='red', marker='s')

plt.xlabel('Number of Known Points')
#plt.ylabel('NMSE')
plt.title('NMSE')
plt.legend()
plt.show()'''

import numpy as np
import matplotlib.pyplot as plt

# NMSE values for each set of known points
'''nmse_data = {
    18: [0.00495, 0.00706, 0.02004, 0.03844, 0.01887, 0.0347, 0.0532, 0.10724, 0.05231, 0.28562],
    16: [0.0141, 0.0088, 0.02056, 0.05158, 0.05423, 0.10906, 0.11794, 0.14503, 0.07353, 0.34313],
    14: [0.0035, 0.00885, 0.01938, 0.05524, 0.05817, 0.07771, 0.14306, 0.23865, 0.07024, 0.46294],
    12: [0.00647, 0.01177, 0.03261, 0.08865, 0.10286, 0.22379, 0.16924, 0.46311, 0.30191, 0.77902],
    10: [0.00532, 0.01174, 0.08915, 0.18698, 0.13307, 0.36582, 0.55241, 0.45761, 0.58347, 1.30442],
    8:  [0.02877, 0.03633, 0.27295, 0.63965, 0.58429, 1.05996, 1.23066, 0.72658, 0.95573, 1.59214],
    6:  [0.10854, 0.27982, 0.26637, 0.64939, 1.25145, 1.28623, 1.00847, 0.96159, 0.84314, 1.27118]
}'''

nmse_data = {
    18: [0.01028, 0.04128, 0.06381, 0.01479, 0.01769, 0.04999, 0.11665, 0.03281, 0.05038, 0.37584],
    16: [0.00394, 0.00243, 0.01493, 0.01327, 0.01209, 0.03112, 0.04012, 0.09502, 0.03911, 0.64324],
    14: [0.00942, 0.01652, 0.01216, 0.0185, 0.03575, 0.07722, 0.11993, 0.22697, 0.05747, 1.23691],
    12: [0.01049, 0.02143, 0.08859, 0.07015, 0.1038, 0.16366, 0.56626, 0.48411, 0.39038, 1.93167],
    10: [0.00753, 0.01027, 0.15828, 0.37999, 0.06135, 0.18202, 0.86284, 0.93952, 1.00162, 1.59499],
    8:  [0.01829, 0.07926, 0.12973, 1.84745, 0.154, 0.63983, 2.58554, 2.61701, 1.51657, 2.50547],
    6:  [0.08535, 0.36563, 0.78512, 1.91829, 1.08113, 2.08094, 1.82291, 2.09519, 1.07323, 2.12123]
}

# Calculate the standard deviation across modes (for each number of points known)
std_modes = {points: np.std(nmse) for points, nmse in nmse_data.items()}

# Now calculate the standard deviation across different numbers of points (for each mode)
nmse_by_mode = np.array([nmse for nmse in nmse_data.values()]).T  # Transpose to group by mode

std_points = {mode+1: np.std(nmse_by_mode[mode]) for mode in range(nmse_by_mode.shape[0])}

# Plot 1: Standard deviation across modes (for each number of points known)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(std_modes.keys(), std_modes.values(), color='skyblue')
plt.ylim(0, 1.1)
plt.xlabel('Number of Known Points')
plt.ylabel('Standard Deviation of NMSE')
plt.title('Standard Deviation Across Modes for Each Number of Known Points')

# Plot 2: Standard deviation across different numbers of points (for each mode)
plt.subplot(1, 2, 2)
plt.bar(std_points.keys(), std_points.values(), color='lightcoral')
plt.ylim(0, 1.1)
plt.xlabel('Mode')
plt.ylabel('Standard Deviation of NMSE')
plt.title('Standard Deviation Across Numbers of Known Points for Each Mode')

plt.tight_layout()
plt.show()
