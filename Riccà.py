import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Leggere i dati dal file CSV senza intestazioni
data = pd.read_csv('RicFunc.csv', delimiter=',', header=None)

# Trasporre il DataFrame e assegnare i nomi delle colonne
data.columns = ['w', 'sigma', 'rms_e', 'rms_r']

# Convertire i dati in tensori PyTorch
w = torch.tensor(data['w'].values, dtype=torch.float32)
sigma_vals = torch.tensor(data['sigma'].values, dtype=torch.float32)
rms_e_vals = torch.tensor(data['rms_e'].values, dtype=torch.float32)
rms_r_vals = torch.tensor(data['rms_r'].values, dtype=torch.float32)

# Plot delle funzioni iniziali
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(w.numpy(), sigma_vals.numpy(), label='σ(w)', color='b')
plt.xlabel('w')
plt.ylabel('σ(w)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(w.numpy(), rms_e_vals.numpy(), label='rms_e(w)', color='g')
plt.xlabel('w')
plt.ylabel('rms_e(w)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(w.numpy(), rms_r_vals.numpy(), label='rms_r(w)', color='r')
plt.xlabel('w')
plt.ylabel('rms_r(w)')
plt.legend()

plt.tight_layout()
plt.show()

# Definire la rete neurale
class CoefficientsNN(nn.Module):
    def __init__(self):
        super(CoefficientsNN, self).__init__()
        self.A1 = nn.Parameter(torch.randn(1))
        self.A2 = nn.Parameter(torch.randn(1))
        self.B1 = nn.Parameter(torch.randn(1))
        self.B2 = nn.Parameter(torch.randn(1))
        self.C1 = nn.Parameter(torch.randn(1))
        self.C2 = nn.Parameter(torch.randn(1))
        self.D1 = nn.Parameter(torch.randn(1))
        self.D2 = nn.Parameter(torch.randn(1))
        self.E = nn.Parameter(torch.randn(1))

    def forward(self, rms_e, rms_r):
        return (self.A1 * rms_e +
                self.A2 * rms_r +
                self.B1 * rms_e ** 2 +
                self.B2 * rms_r ** 2 +
                self.C1 * rms_e * rms_r +
                self.C2 * (rms_e ** 3) +
                self.D1 * (rms_r ** 3) +
                self.D2 * (rms_e ** 2) * rms_r +
                self.E)

# Inizializzare la rete neurale e l'ottimizzatore
model = CoefficientsNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Funzione di addestramento
def train_model(model, optimizer, criterion, epochs=50000):
    for epoch in range(epochs):
        model.train()

        # Reset del gradiente
        optimizer.zero_grad()

        # Passaggio in avanti
        output = model(rms_e_vals, rms_r_vals)
        loss = criterion(output, sigma_vals)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Addestrare il modello
train_model(model, optimizer, criterion)

# Stampare i coefficienti trovati
print(f'A1: {model.A1.item()}, A2: {model.A2.item()}, B1: {model.B1.item()}, B2: {model.B2.item()}, C1: {model.C1.item()}, C2: {model.C2.item()}, D1: {model.D1.item()}, D2: {model.D2.item()}, E: {model.E.item()}')

# Predire i valori utilizzando il modello addestrato
with torch.no_grad():
    predicted_sigma = model(rms_e_vals, rms_r_vals)

# Plot della funzione trovata e della funzione sigma(w)
plt.figure(figsize=(10, 6))
plt.plot(w.numpy(), sigma_vals.numpy(), label='σ(w) (reale)', linestyle='-', marker='o')
plt.plot(w.numpy(), predicted_sigma.numpy(), label='σ(w) (predetto)', linestyle='--', marker='x')
plt.xlabel('w')
plt.ylabel('σ(w)')
plt.title('Confronto tra σ(w) reale e predetta')
plt.legend()
plt.show()
