import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
#x = x.repeat(2, 1)
print('x: ', x)
multipliers = torch.tensor([2.0, 3.0]).unsqueeze(1)
out = x * multipliers
print('out: ', out)
#batched_grad = torch.arange(3)  # Size([3])
batched_grad = torch.eye(out.shape[0])
#batched_grad = torch.ones_like(out)
#batched_grad = batched_grad.unsqueeze(0)
print('batched_grad: ', batched_grad.shape)
grad = torch.autograd.grad(out, [x], grad_outputs=batched_grad)[0]
#print('grad_shape: ', grad.shape)
print('grad: ', grad)
