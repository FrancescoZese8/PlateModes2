import torch
from functorch import vmap, grad

x = torch.tensor([1.0, 2.0], requires_grad=True)
out = torch.stack([x * 2, x * 3], dim=0)

print('x:', x)
print('out:', out)


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


dodx = gradient(out, x)
print('grad: ', dodx)
