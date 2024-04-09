from torch.utils.data import DataLoader
import dataSet
import torch
import numpy as np
import loss
import modules
import training
import matplotlib.pyplot as plt

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # TODO
#print(device)


# 1%/2-2:
# NMSE: 0.09404823830498811   loss: 0.00000090915
# NMSE: 0.09810258528924976   loss: 0.00000099067
# NMSE: 0.03157359262403152   loss: 1.311e-06
# NMSE: 0.20815187205348357
# NMSE: 0.08810789486876631
# 5%/2-2
# NMSE: 0.01767920614356985   loss: 0.00000106459
# NMSE: 0.02979781419086590   loss: 1.662e-06
# 10%/2-2
# NMSE: 0.01886094824469468   loss: 0.00000110469
# NMSE: 0.01374631926995931   loss: 0.00000117690
# NMSE: 0.01381509773969310   loss: loss 1.414e-06
# NMSE: 0.01838710502265463
# 20%/2-2
# NMSE:  0.0228620452786500   loss: 0.00000107551
# NMSE:  0.0133779539549649   loss: 0.00000123034
# NMSE:  0.0221876239077416   loss: 1.517e-06
# NMSE:  0.0192541305761053


num_epochs = 400
n_step = 50
batch_size = 32
lr = 0.001
batch_size_domain = 800
batch_size_boundary = 100

steps_til_summary = 10
opt_model = 'silu'
mode = 'pinn'
clip_grad = 1.0
use_lbfgs = False
relo = True
total_length = 1
max_epochs_without_improvement = 100

W = 10
H = 10
T = 0.2
E = 30000
nue = 0.2
p0 = 0.15
den = 1000

n = 2
m = 2
percentage_of_known_points = 1  # %

D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffnes of the plate
omega = ((n * np.pi / W) ** 2 + (m * np.pi / H) ** 2) * np.sqrt(D / (den * T))
print('omega:', omega)
nkp = percentage_of_known_points * batch_size_domain // 100
known_points_x = torch.rand((nkp, 1)) * W
known_points_y = torch.rand((nkp, 1)) * H


def u_val(x, y):
    return (torch.sin(n * np.pi * x / W) * torch.sin(m * np.pi * y / H)) / 100


plate = dataSet.KirchhoffDataset(u_val=u_val, T=T, nue=nue, E=E, W=W, H=H, total_length=total_length, den=den,
                                 omega=omega, batch_size_domain=batch_size_domain, batch_size_boundary=
                                 batch_size_boundary, known_points_x=known_points_x, known_points_y=known_points_y,
                                 nkp=nkp)
# plate.visualise()
data_loader = DataLoader(plate, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=0)
model = modules.PINNet(out_features=1, type=opt_model, mode=mode)
#model.to(device)  # TODO

history_loss = {'L_f': [], 'L_b0': [], 'L_b2': [], 'L_u': [], 'L_t': []}
if not relo:
    loss_fn = loss.KirchhoffLoss(plate)
    kirchhoff_metric = loss.KirchhoffMetric(plate)
    history_lambda = None
    metric_lam = None
else:
    loss_fn = loss.ReLoBRaLoKirchhoffLoss(plate, temperature=0.1, rho=0.99, alpha=0.999)
    kirchhoff_metric = loss.KirchhoffMetric(plate)
    history_lambda = {'L_f_lambda': [], 'L_b0_lambda': [], 'L_b2_lambda': [], 'L_t_lambda': []}
    metric_lam = loss.ReLoBRaLoLambdaMetric(loss_fn)

training.train(model=model, train_dataloader=data_loader, epochs=num_epochs, n_step=n_step, lr=lr,
               steps_til_summary=steps_til_summary, loss_fn=loss_fn, history_loss=history_loss,
               history_lambda=history_lambda,
               metric=kirchhoff_metric, metric_lam=metric_lam, clip_grad=clip_grad,
               use_lbfgs=False, max_epochs_without_improvement=max_epochs_without_improvement)
model.eval()

plate.visualise(model)

fig = plt.figure(figsize=(6, 4.5), dpi=100)
plt.plot(torch.log(torch.tensor(history_loss['L_f'])), label='$L_f$ governing equation')
plt.plot(torch.log(torch.tensor(history_loss['L_b0'])), label='$L_{b0}$ Dirichlet boundaries')
plt.plot(torch.log(torch.tensor(history_loss['L_b2'])), label='$L_{b2}$ Moment boundaries')
plt.plot(torch.log(torch.tensor(history_loss['L_t'])), label='$L_t$ Known points')
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
    plt.plot(history_lambda['L_b0_lambda'], label='$\lambda_{b0}$ Dirichlet boundaries')
    plt.plot(history_lambda['L_b2_lambda'], label='$\lambda_{b2}$ Moment boundaries')
    plt.plot(history_lambda['L_t_lambda'], label='$\lambda_{t}$ Known points')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('scalings lambda')  # $\lambda$')
    plt.title('ReLoBRaLo weights on Kirchhoff PDE')
    plt.savefig('kirchhoff_lambdas_relobralo')
    plt.show()
