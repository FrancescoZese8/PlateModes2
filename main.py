from torch.utils.data import DataLoader
import dataSet
import torch
import loss
import modules
import training
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA
print('device: ', device)


num_epochs = 400
n_step = 10
batch_size = 1
total_length = 1
lr = 0.001
batch_size_domain = 800
batch_size_boundary = 100

steps_til_summary = 8
opt_model = 'silu'
mode = 'pinn'
clip_grad = 1.0
use_lbfgs = False
relo = True
max_epochs_without_improvement = 100

W = 10
H = 10
T = 0.2
E = 0.7e5
nue = 0.35
p0 = 0.15
den = 2700

eigen_mode = 10
free_edges = True

D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffnes of the plate


dispFile = pd.read_csv('DisplacementFieldCSV.csv')  # Read CSV and create eigenFreq dictionary
eigenFreq = {}
for index, row in dispFile.iterrows():
    if 'i' in str(row.iloc[0]):
        continue
    label = row.iloc[0]
    data = row.iloc[1:].astype(float)
    eigenFreq[label] = data

label = list(eigenFreq.keys())[eigen_mode-1]
print('EigenFrequency: ', label)
omega = float(label)
known_disp = eigenFreq[label]
known_disp = torch.tensor(known_disp)
known_disp = known_disp.to(device)  # CUDA
known_disp = known_disp * 1000
#known_disp = known_disp[::4]
print('Known disp: ', known_disp.shape)


plate = dataSet.KirchhoffDataset(T=T, nue=nue, E=E, W=W, H=H, total_length=total_length, den=den,
                                 omega=omega, batch_size_domain=batch_size_domain, batch_size_boundary=
                                 batch_size_boundary, known_disp=known_disp, free_edges=free_edges, device=device)
# plate.visualise()
'''M = [[0 for _ in range(W+1)] for _ in range(H+1)]
for i in range(W+1):
    for j in range(H+1):
        M[i][j] = known_disp[i + j*W]

M = torch.tensor(M)
# Genera le coordinate X e Y per la matrice M
X, Y = np.mgrid[0:W+1, 0:H+1]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotta solo i punti conosciuti
ax.plot_surface(X, Y, M, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('known_disp')

plt.show()'''

data_loader = DataLoader(plate, shuffle=True, batch_size=batch_size, pin_memory=False, num_workers=0)
model = modules.PINNet(out_features=1, type=opt_model, mode=mode)
model = model.to(device)  # CUDA

history_loss = {'L_f': [], 'L_b0': [], 'L_b2': [], 'L_u': [], 'L_t': []}
if not relo:
    loss_fn = loss.KirchhoffLoss(plate)
    kirchhoff_metric = loss.KirchhoffMetric(plate, free_edges=free_edges)
    history_lambda = None
    metric_lam = None
else:
    loss_fn = loss.ReLoBRaLoKirchhoffLoss(plate, temperature=0.1, rho=0.99, alpha=0.999)
    kirchhoff_metric = loss.KirchhoffMetric(plate, free_edges=free_edges)
    history_lambda = {'L_f_lambda': [], 'L_b0_lambda': [], 'L_b2_lambda': [], 'L_t_lambda': []}
    metric_lam = loss.ReLoBRaLoLambdaMetric(loss_fn, free_edges=free_edges)

training.train(model=model, train_dataloader=data_loader, epochs=num_epochs, n_step=n_step, lr=lr,
               steps_til_summary=steps_til_summary, loss_fn=loss_fn, history_loss=history_loss,
               history_lambda=history_lambda,
               metric=kirchhoff_metric, metric_lam=metric_lam, clip_grad=clip_grad,
               use_lbfgs=False, max_epochs_without_improvement=max_epochs_without_improvement)
model.eval()

plate.visualise(model)

fig = plt.figure(figsize=(6, 4.5), dpi=100)
plt.plot(torch.log(torch.tensor(history_loss['L_f'])), label='$L_f$ governing equation')
plt.plot(torch.log(torch.tensor(history_loss['L_t'])), label='$L_t$ Known points')
if not free_edges:
    plt.plot(torch.log(torch.tensor(history_loss['L_b0'])), label='$L_{b0}$ Dirichlet boundaries')
    plt.plot(torch.log(torch.tensor(history_loss['L_b2'])), label='$L_{b2}$ Moment boundaries')
    plt.plot(torch.log(torch.tensor(history_loss['L_u'])), label='$L_u$ analytical solution')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Log-loss')
plt.title('Loss evolution Kirchhoff PDE')
plt.savefig('kirchhoff_loss_unscaled')
plt.show()

if metric_lam is not None:
    fig2 = plt.figure(figsize=(6, 4.5), dpi=100)
    plt.plot(history_lambda['L_f_lambda'], label='$\lambda_f$ governing equation')
    plt.plot(history_lambda['L_t_lambda'], label='$\lambda_{t}$ Known points')
    if not free_edges:
        plt.plot(history_lambda['L_b0_lambda'], label='$\lambda_{b0}$ Dirichlet boundaries')
        plt.plot(history_lambda['L_b2_lambda'], label='$\lambda_{b2}$ Moment boundaries')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('scalings lambda')  # $\lambda$')
    plt.title('ReLoBRaLo weights on Kirchhoff PDE')
    plt.savefig('kirchhoff_lambdas_relobralo')
    plt.show()
