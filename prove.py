
import matplotlib.pyplot as plt
import numpy as np

x = [6, 8, 10, 12, 14, 16, 18]

y1 = 20 * np.log10([0.9, 0.6, 0.28, 0.19, 0.09, 0.04, 0.03])
y2 = 20 * np.log10([1.41, 1.11, 0.54, 0.35, 0.23, 0.14, 0.11])

plt.plot(x, y1, label='w/ Governing Eq', color='blue', marker='o')
plt.plot(x, y2, label='w/o Governing Eq', color='red', marker='s')

plt.xlabel('Number of Known Points')
#plt.ylabel('NMSE')
plt.title('NMSE')
plt.legend()
plt.show()