import numpy as np

# Esempio di array unidimensionale
known_disp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Esempio di array bidimensionale
u_t = np.array([[11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]])

# Sottrazione con broadcasting
err_t = known_disp - u_t

print(err_t)