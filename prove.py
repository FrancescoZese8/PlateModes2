import torch

u = matrix = torch.tensor([[1, 1],
                       [2, 2],
                       [3, 3]])

omegas = torch.tensor([1, 2])
print('u: ', u.shape, 'omegas: ', omegas.shape)
f = torch.sum(omegas * u, dim=-1)
print('f: ', f)
