from torch.utils.data import DataLoader
import dataSet
import torch
import loss
import modules
import training
import pandas as pd
import visualization


#def main(rhos, alphas):
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA
print('device: ', device)

num_epochs = 200
n_step = 50
batch_size = 1
total_length = 1
lr = 0.001
batch_size_domain = 1000
num_hidden_layers = 2
hidden_features = 128
temperature = 10e-05
rho = 0.5  # 0.99, 0.5
alpha = 0.1  # 0.9, 0.1

steps_til_summary = 10
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
den = 2700

eigen_mode = 22    # 7: 0.0094905 / 8: 0.012216 / 9: 0.016826 / 11: 0.03027 / 13: 0.030792 / 14: 0.033438
# / 18: 0.057536 / 20: 0.06447 / 22: 0.074002 / 23: 0.078311 / 24: 0.082339 / 25: 0.0975 / 27: 0.099372 / 30: 0.10527
# / 32: 0.13573 / 33: 0.13573 / 34: 0.14103 / 36: 0.14482 / 39: 0.14971
omega = 0.074002 * 2 * torch.pi
free_edges = True

D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffnes of the plate

df = pd.read_csv('FieldOfDisplacement.csv', sep=';')
df_numeric = df.apply(pd.to_numeric, errors='coerce')

x_p, y_p = [], []
for i in range(100):
    for j in range(100):
        x_p.append(round(j * 0.1 + 0.05, 2))
        y_p.append(round(i * 0.1 + 0.05, 2))

sampled_points = [0.25, 2.55, 5.05, 7.45, 9.75]
x_t = []
y_t = []
for y in sampled_points:
    for x in sampled_points:
        x_t.append(x)
        y_t.append(y)

# ds = 4  # 430
full_known_disp = torch.tensor(df_numeric.iloc[:, 2:].values)
full_known_disp = full_known_disp[:, eigen_mode]
# known_disp = known_disp[:, eigen_mode]
min_val = torch.min(full_known_disp)
max_val = torch.max(full_known_disp)
max_norm = 1
full_known_disp = (-1 + 2 * (full_known_disp - min_val) / (max_val - min_val)) * max_norm
# known_disp = full_known_disp[::ds]
# x_t = x_p[::ds]
# y_t = y_p[::ds]
full_known_disp_map = dict(zip(zip(x_p, y_p), full_known_disp))
known_disp = [full_known_disp_map.get((round(i, 2), round(j, 2)), 0) for index, (i, j) in
              enumerate(zip(x_t, y_t))]
known_disp_map = dict(zip(zip(x_t, y_t), known_disp))
known_disp = torch.tensor(known_disp)

visualization.visualise_init(known_disp, known_disp_map, full_known_disp, x_p, y_p, eigen_mode, image_width=100,
                             image_height=100)

plate = dataSet.KirchhoffDataset(T=T, nue=nue, E=E, D=D, W=W, H=H, total_length=total_length, den=den,
                                 omega=omega, batch_size_domain=batch_size_domain, known_disp=known_disp,
                                 full_known_disp=full_known_disp, x_t=x_t, y_t=y_t, max_norm=max_norm,
                                 free_edges=free_edges, device=device)

data_loader = DataLoader(plate, shuffle=True, batch_size=batch_size, pin_memory=False, num_workers=0)
model = modules.PINNet(num_hidden_layers=num_hidden_layers, hidden_features=hidden_features,
                       out_features=1, type=opt_model, mode=mode)
model = model.to(device)  # CUDA

history_loss = {'L_f': [], 'L_b0': [], 'L_b2': [], 'L_u': [], 'L_t': [], 'L_m': []}
if not relo:
    loss_fn = loss.KirchhoffLoss(plate)
    kirchhoff_metric = loss.KirchhoffMetric(plate, free_edges=free_edges)
    history_lambda = None
    metric_lam = None
else:
    loss_fn = loss.ReLoBRaLoKirchhoffLoss(plate, temperature=temperature, rho=rho, alpha=alpha)
    kirchhoff_metric = loss.KirchhoffMetric(plate, free_edges=free_edges)
    history_lambda = {'L_f_lambda': [], 'L_b0_lambda': [], 'L_b2_lambda': [], 'L_t_lambda': [], 'L_m_lambda': []}
    metric_lam = loss.ReLoBRaLoLambdaMetric(loss_fn, free_edges=free_edges)

training.train(model=model, train_dataloader=data_loader, epochs=num_epochs, n_step=n_step, lr=lr,
               steps_til_summary=steps_til_summary, loss_fn=loss_fn, history_loss=history_loss,
               history_lambda=history_lambda,
               metric=kirchhoff_metric, metric_lam=metric_lam, clip_grad=clip_grad,
               use_lbfgs=use_lbfgs, max_epochs_without_improvement=max_epochs_without_improvement, relo=relo)
model.eval()

NMSE = visualization.visualise_prediction(x_p, y_p, full_known_disp, eigen_mode, max_norm, device, model, image_width=100,
                                          image_height=100)
print('NMSE: ', NMSE)

visualization.visualise_loss(free_edges, metric_lam, history_loss, history_lambda)
    #return NMSE
