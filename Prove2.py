import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
multipliers = torch.tensor([2.0, 3.0]).unsqueeze(1)
print('multipliers: ', multipliers.shape)
out = x * multipliers

x_repeated = x.repeat_interleave(2).view(2, 2)

batched_grad = torch.ones_like(out)

grad = torch.autograd.grad(out, [x], grad_outputs=batched_grad)[0]

print('x: ', x)
print('x_repeated: ', x_repeated)
print('out: ', out)
print('grad: ', grad)


