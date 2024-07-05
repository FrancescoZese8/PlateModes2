import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
multipliers = torch.tensor([2.0, 3.0]).unsqueeze(1)
print('multipliers: ', multipliers.shape)
out = x * multipliers
#batched_grad = torch.arange(3)  # Size([3])
batched_grad = torch.ones_like(out)
x = x.repeat(2, 1)
grad = torch.autograd.grad(out, [x], grad_outputs=batched_grad)[0]
print('x: ', x)
print('out: ', out)
#print('grad_shape: ', grad.shape)
print('grad: ', grad)
