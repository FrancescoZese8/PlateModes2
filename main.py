from torch.utils.data import DataLoader
import dataSet
import torch
import loss
import modules
import training
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA
print('device: ', device)

num_epochs = 200    # 150 ep, 50 step, em 23, ds 256, temperature=0.00001, rho=0.99, alpha=0.9
n_step = 50
batch_size = 1
total_length = 1
lr = 0.001
batch_size_domain = 1000

steps_til_summary = 8
opt_model = 'silu'
mode = 'pinn'
clip_grad = 1.0
use_lbfgs = False
relo = True
max_epochs_without_improvement = 50

W = 10
H = 10
T = 0.2
E = 0.7e5
nue = 0.35
p0 = 0.15
den = 2700

eigen_mode = 23  # 8: 0.012216 / 11: 0.03027 / 13: 0.030792 / 18: 0.057536 / 23: 0.078311 / 32: 0.13573 / 36: 0.14482 / 39: 0.14971
omega = 0.078311
free_edges = True

D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffnes of the plate

df = pd.read_csv('FieldOfDisplacement.csv', sep=';')
df_numeric = df.apply(pd.to_numeric, errors='coerce')

x_p, y_p = [], []
for i in range(100):
    for j in range(100):
        x_p.append(round(j * 0.1 + 0.05, 2))
        y_p.append(round(i * 0.1 + 0.05, 2))

ds = 128
x_t = x_p[::ds]
y_t = y_p[::ds]
known_disp = torch.tensor(df_numeric.iloc[:, 2:].values)
known_disp = known_disp[:, eigen_mode]
full_known_disp = known_disp
known_disp = known_disp[::ds]
min_val = torch.min(known_disp)
max_val = torch.max(known_disp)
known_disp = -1 + 2 * (known_disp - min_val) / (max_val - min_val)
#known_disp = known_disp * 1000
#known_disp = known_disp.to(device)  # CUDA
known_disp_map = dict(zip(zip(x_t, y_t), known_disp))

plate = dataSet.KirchhoffDataset(T=T, nue=nue, E=E, W=W, H=H, total_length=total_length, den=den,
                                 omega=omega, batch_size_domain=batch_size_domain, known_disp=known_disp,
                                 known_disp_map=known_disp_map, x_t=x_t, y_t=y_t,
                                 free_edges=free_edges, device=device)
# plate.visualise()

known_disps = [known_disp_map.get((round(i, 2), round(j, 2)), 0) for index, (i, j) in
               enumerate(zip(x_p, y_p))]
kdp = np.reshape(known_disps, (100, 100)).astype(float)
fkdp = np.reshape(full_known_disp, (100, 100))
X, Y = np.meshgrid(np.arange(0.05, 10.05, 0.1), np.arange(0.05, 10.05, 0.1))

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

ax3d = fig.add_subplot(221, projection='3d')
surf1 = ax3d.plot_surface(X, Y, fkdp, cmap='viridis')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_title('Real Displacement mode: {}'.format(eigen_mode))

ax2d = fig.add_subplot(222)
im = ax2d.imshow(fkdp, extent=(0, 10, 0, 10), origin='lower', cmap='viridis')
ax2d.set_xlabel('X')
ax2d.set_ylabel('Y')
ax2d.set_title('Real Displacement mode: {}'.format(eigen_mode))

ax3d = fig.add_subplot(223, projection='3d')
surf2 = ax3d.plot_surface(X, Y, kdp, cmap='viridis')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_title('Known Points: {}'.format(len(known_disp)))

ax2d = fig.add_subplot(224)
im = ax2d.imshow(kdp, extent=(0, 10, 0, 10), origin='lower')
ax2d.set_xlabel('X')
ax2d.set_ylabel('Y')
ax2d.set_title('Known Points: {}'.format(len(known_disp)))

plt.tight_layout()
plt.show()


data_loader = DataLoader(plate, shuffle=True, batch_size=batch_size, pin_memory=False, num_workers=0)
model = modules.PINNet(known_disp_map, initial_conditions=False, out_features=1, type=opt_model, mode=mode)
model = model.to(device)  # CUDA

history_loss = {'L_f': [], 'L_b0': [], 'L_b2': [], 'L_u': [], 'L_t': []}
if not relo:
    loss_fn = loss.KirchhoffLoss(plate)
    kirchhoff_metric = loss.KirchhoffMetric(plate, free_edges=free_edges)
    history_lambda = None
    metric_lam = None
else:
    loss_fn = loss.ReLoBRaLoKirchhoffLoss(plate, temperature=10e-05, rho=0.99, alpha=0.9)
    kirchhoff_metric = loss.KirchhoffMetric(plate, free_edges=free_edges)
    history_lambda = {'L_f_lambda': [], 'L_b0_lambda': [], 'L_b2_lambda': [], 'L_t_lambda': []}
    metric_lam = loss.ReLoBRaLoLambdaMetric(loss_fn, free_edges=free_edges)

training.train(model=model, train_dataloader=data_loader, epochs=num_epochs, n_step=n_step, lr=lr,
               steps_til_summary=steps_til_summary, loss_fn=loss_fn, history_loss=history_loss,
               history_lambda=history_lambda,
               metric=kirchhoff_metric, metric_lam=metric_lam, clip_grad=clip_grad,
               use_lbfgs=False, max_epochs_without_improvement=max_epochs_without_improvement, relo=relo)
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
