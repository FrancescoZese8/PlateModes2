import torch
from functorch import vmap, grad

x = torch.tensor([1.0, 2.0], requires_grad=True)
out = torch.stack([x * 2, x * 3], dim=0)

print('x:', x)
print('out:', out)


def single_gradient(out_row, x):
    grad_outputs = torch.ones_like(out_row)
    return torch.autograd.grad(out_row, [x], grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]


batched_grad = vmap(single_gradient, (0, None))(out, x)

print('Batched Grads:', batched_grad)
