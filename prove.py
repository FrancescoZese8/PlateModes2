import torch

# Definisci il tensore iniziale
x = torch.tensor([2.0], requires_grad=True)
print('x: ', x)

# Definisci il vettore y
y = torch.arange(1, 11)
print('y: ', y)

# Calcola u = x * y
u = x * y
print('u: ', u)

# Definisci la funzione di cui calcolare il Jacobiano
def func(x):
    return x * y

# Calcola il Jacobiano di u rispetto a x
jacobian = torch.autograd.functional.jacobian(func, x)

# Mostra il risultato
print('jacobian: ', jacobian)

