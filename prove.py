import numpy as np
import torch
from functorch import vmap

x = torch.tensor([1.0, 2.0], requires_grad=True)

# Crea il tensore out come descritto
out = torch.stack([x * 2, x * 3], dim=0)

# Visualizza i tensori
print('x:', x)
print('out:', out)

# Funzione per calcolare il gradiente di una singola riga rispetto a x
def compute_grad(out_row, x):
    grad_outputs = torch.ones_like(out_row)
    grad = torch.autograd.grad(out_row, x, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
    return grad

# Calcola i gradienti per ciascuna riga di out rispetto a x
grad_outputs = torch.eye(out.size(0)).to(out)
grads = torch.autograd.grad(out, x, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]

# Visualizza i gradienti
print('grad:', grads)
