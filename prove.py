import torch
import numpy as np

u = np.ones((10, 5))
u = torch.tensor(u)
print('u: ', u.shape)

omegas = [1, 2, 3, 4, 5]
omegas = torch.tensor(omegas)
print('omegas: ', omegas.shape)

#o = torch.sum(u * omegas, dim=1)
u[2:, :] = u[2:, :]/omegas

print('o: ', u)
