from torch.utils.data import DataLoader
import dataSet
import torch
import loss
import modules
import training
import pandas as pd
import visualization



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA
print('device: ', device)

num_epochs = 150  # 150 ep, 50 step, em 23, ds 256, temperature=0.00001, rho=0.99, alpha=0.9
n_step = 10
batch_size = 1
total_length = 1
lr = 0.001
batch_size_domain = 1000
num_hidden_layers = 2  # 2
hidden_features = 128   # 64
temperature = 10e-05
rho = 0.9  # 0.9, 0.1
alpha = 0.9  # 0.9, 0.5

steps_til_summary = 10
opt_model = 'silu'
mode = 'pinn'
clip_grad = 1.0
use_lbfgs = False
relo = False
max_epochs_without_improvement = 50

W = 10
H = 10
T = 0.2
E = 0.7e5
nue = 0.35
den = 2700

eigen_mode = 23  # 8: 0.012216 / 11: 0.03027 / 13: 0.030792 / 18: 0.057536 / 23: 0.078311 / 25: 0.0975 32: 0.13573 / 36: 0.14482 / 39: 0.14971
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

ds = 4  # 430
full_known_disp = torch.tensor(df_numeric.iloc[:, 2:].values)
full_known_disp = full_known_disp[:, eigen_mode]
min_val = torch.min(full_known_disp)
max_val = torch.max(full_known_disp)
max_norm = 1
full_known_disp = (-1 + 2 * (full_known_disp - min_val) / (max_val - min_val)) * max_norm
# full_known_disp = known_disp

'''x_t, y_t = [], []
known_disp = []
for i in range(100):
    if (i+1) % ds == 0:
        for j in range(100):
            if (j+1) % ds == 0:
                known_disp.append(full_known_disp[(i*100)+j])
                x_t.append(round(j * 0.1 + 0.05, 2))
                y_t.append(round(i * 0.1 + 0.05, 2))
known_disp = torch.tensor(known_disp)'''
known_disp = full_known_disp[::ds]
x_t = x_p[::ds]
y_t = y_p[::ds]
# known_disp = known_disp * 1000
# known_disp = known_disp.to(device)  # CUDA
known_disp_map = dict(zip(zip(x_t, y_t), known_disp))

visualization.visualise_init(known_disp, known_disp_map, full_known_disp, x_p, y_p, eigen_mode, image_width=100,
                             image_height=100)

plate = dataSet.KirchhoffDataset(T=T, nue=nue, E=E, D=D, W=W, H=H, total_length=total_length, den=den,
                                 omega=omega, batch_size_domain=batch_size_domain, known_disp=known_disp,
                                 full_known_disp=full_known_disp, known_disp_map=known_disp_map, x_t=x_t, y_t=y_t, x_p=x_p, y_p=y_p,
                                 free_edges=free_edges, device=device)

data_loader = DataLoader(plate, shuffle=True, batch_size=batch_size, pin_memory=False, num_workers=0)
model = modules.PINNet(known_disp_map, num_hidden_layers=num_hidden_layers, hidden_features=hidden_features,
                       initial_conditions=False, out_features=1, type=opt_model, mode=mode)
model = model.to(device)  # CUDA

history_loss = {'L_f': [], 'L_b0': [], 'L_b2': [], 'L_u': [], 'L_t': []}
if not relo:
    loss_fn = loss.KirchhoffLoss(plate)
    kirchhoff_metric = loss.KirchhoffMetric(plate, free_edges=free_edges)
    history_lambda = None
    metric_lam = None
else:
    loss_fn = loss.ReLoBRaLoKirchhoffLoss(plate, temperature=temperature, rho=rho, alpha=alpha)
    kirchhoff_metric = loss.KirchhoffMetric(plate, free_edges=free_edges)
    history_lambda = {'L_f_lambda': [], 'L_b0_lambda': [], 'L_b2_lambda': [], 'L_t_lambda': []}
    metric_lam = loss.ReLoBRaLoLambdaMetric(loss_fn, free_edges=free_edges)

training.train(model=model, train_dataloader=data_loader, epochs=num_epochs, n_step=n_step, lr=lr,
               steps_til_summary=steps_til_summary, loss_fn=loss_fn, history_loss=history_loss,
               history_lambda=history_lambda,
               metric=kirchhoff_metric, metric_lam=metric_lam, clip_grad=clip_grad,
               use_lbfgs=use_lbfgs, max_epochs_without_improvement=max_epochs_without_improvement, relo=relo)
model.eval()

NMSE = visualization.visualise_prediction(x_p, y_p, full_known_disp, eigen_mode, device, model, image_width=100,
                                          image_height=100)
print('NMSE: ', NMSE)

visualization.visualise_loss(free_edges, metric_lam, history_loss, history_lambda)
