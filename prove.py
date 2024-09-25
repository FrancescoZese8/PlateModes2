
import matplotlib.pyplot as plt
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
plt.show()