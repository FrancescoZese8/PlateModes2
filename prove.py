import torch

x = torch.tensor([2.0], requires_grad=True)
print('x: ', x)

y = torch.arange(1, 11)
print('y: ', y)
u = x * y
print('u: ', u)

grad_outputs = torch.ones_like(u)
dudx = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]

print('dudx: ', dudx)
