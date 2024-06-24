from torch.utils.data import DataLoader
import dataSet
import torch
import loss
import modules
import training
import pandas as pd
import visualization
import numpy as np

# def main(rho, alpha, tmp):
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA
print('device: ', device)

num_epochs = 500
n_step = 100
num_known_points = 25
size_norm = 10
batch_size = 1
total_length = 1
lr = 0.001
batch_size_domain = 1000
num_hidden_layers = 2
hidden_features = 32
temperature = 0.001  # 10e-05
rho = 0.1  # 0.99, 0.5, 0.35, 0.1
alpha = 0.99  # 0.9, 0.1, 0.1, 0.99

steps_til_summary = 10
opt_model = 'silu'  # mish
mode = 'pinn'
clip_grad = 1.0
use_lbfgs = False
relo = True
max_epochs_without_improvement = 50
free_edges = True
color = 'viridis'  # bwr
n_d = 4  #

W, H, T, E, nue, den = 0.20, 0.35, 0.005, 10e6, 0.28, 420
W_p, H_p = W, H
W, H, scaling_factor = dataSet.scale_to_target(W, H, size_norm, n_d)
print('W, H, scaling_factor: ', W, H, scaling_factor)

#eigen_mode = [6, 22]
#eigen_mode = [11, 12, 13, 14, 15, 16, 17, 18]
eigen_mode = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

freqs = [None, None, None, None, None, None, 6.499, 7.0867, 15.854, 17.953, 20.396, 25.138, 28.221, 34.876,
         37.256, 45.472, 51.651, 56.464, 59.474, 59.625, 69.244, 71.409, 71.434, 88.497, 88.545, 95.667,
         97.758, 110.03, 110.36, 113.12, 122.91, 123.74, 126.64, 131.98, 136.81, 141.2, 152.5, 160.25, 162.56,
         165.3, ]  # ViolinPlateFOD3
omegas = []
for eig in eigen_mode:
    omegas.append(round(freqs[eig] * 2 * torch.pi / scaling_factor ** 2, 6))
print('omegas: ', omegas)

D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffnes of the plate

df = pd.read_csv('ViolinPlateFOD2.csv', sep=';')

df_numeric = df.apply(pd.to_numeric, errors='coerce')
n_samp_x, n_samp_y = 20, 35

sample_step = W / n_samp_x
if H / n_samp_y != sample_step:
    print('ERROR: sample step difference')
dist_bound = sample_step / 2
print('W / n_samp_x: ', W / n_samp_x, 'H / n_samp_y: ', H / n_samp_y)

x_p, y_p = [], []
for i in range(n_samp_y):
    for j in range(n_samp_x):
        x_p.append(round(j * sample_step + dist_bound, n_d))
        y_p.append(round(i * sample_step + dist_bound, n_d))

x_t = []
y_t = []

'''
for y in sampled_points_y:
    for x in sampled_points_x:
        x_t.append(x)
        y_t.append(y)'''

'''for y in sampled_points:
    for x in sampled_points:
        x_t.append(x)
        y_t.append(y)'''

min_distance = round(np.sqrt(H * W / num_known_points) - np.sqrt(H * W / num_known_points) / 2, n_d)


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


i = 0
while i < num_known_points:
    attempts = 0
    while True:
        rand_p = np.random.randint(0, n_samp_x * n_samp_y - 1)
        new_x, new_y = x_p[rand_p], y_p[rand_p]
        if all(euclidean_distance(new_x, new_y, x_t[j], y_t[j]) >= min_distance for j in range(len(x_t))):
            x_t.append(new_x)
            y_t.append(new_y)
            i += 1
            break
        attempts += 1
        if attempts >= 100:
            x_t, y_t = [], []
            i = 0
            break

# x_t = x_p
# y_t = y_p
max_norm = 1
full_known_disp, full_known_disp_map, known_disp, known_disp_map, known_disp_concatenate = [], [], [], [], []
known_disp_dict, full_known_disp_dict = {}, {}
full_known_disp_csv = torch.tensor(df_numeric.iloc[:, 2:].values)

for i in range(len(eigen_mode)):
    full_known_disp = full_known_disp_csv[:, eigen_mode[i]]
    min_val = torch.min(full_known_disp)
    max_val = torch.max(full_known_disp)
    full_known_disp = (-1 + 2 * (full_known_disp - min_val) / (max_val - min_val)) * max_norm
    full_known_disp_map = dict(zip(zip(x_p, y_p), full_known_disp))  # Mappa displcement e coordinate
    full_known_disp_dict[omegas[i]] = full_known_disp  # Mappa displacement e omega
    known_disp = [full_known_disp_map.get((round(i, n_d), round(j, n_d)), 0) for index, (i, j) in
                  enumerate(zip(x_t, y_t))]
    known_disp_map = dict(zip(zip(x_t, y_t), known_disp))
    known_disp = torch.tensor(known_disp)
    known_disp = known_disp.to(device)
    #known_disp_dict[omegas[i]] = known_disp
    known_disp_concatenate.append(known_disp)
    #print('Known disp: ', known_disp_dict[omegas[i]][0])


    '''visualization.visualise_init(known_disp, known_disp_map, full_known_disp, x_p, y_p, eigen_mode,
                                 image_width=n_samp_x,
                                 image_height=n_samp_y, H=H, W=W, H_p=H_p, W_p=W_p, sample_step=sample_step,
                                 dist_bound=dist_bound, n_d=n_d, size_norm=size_norm,
                                 color=color)'''
#print('OMEGAS_main: ', omegas)
known_disp_concatenate = torch.cat(known_disp_concatenate, dim=0)

plate = dataSet.KirchhoffDataset(T=T, nue=nue, E=E, D=D, W=W, H=H, total_length=total_length, den=den,
                                 omegas=omegas, batch_size_domain=batch_size_domain,
                                 known_disp_concatenate=known_disp_concatenate,
                                 x_t=x_t, y_t=y_t,
                                 max_norm=max_norm,
                                 free_edges=free_edges, device=device, sample_step=sample_step,
                                 dist_bound=dist_bound,
                                 n_samp_x=n_samp_x, n_samp_y=n_samp_y)

data_loader = DataLoader(plate, shuffle=True, batch_size=batch_size, pin_memory=False, num_workers=0)
model = modules.PINNet(num_hidden_layers=num_hidden_layers, hidden_features=hidden_features,
                       in_features=3, out_features=1, type=opt_model, mode=mode)
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
               metric=kirchhoff_metric, metric_lam=metric_lam, free_edges=free_edges, clip_grad=clip_grad,
               use_lbfgs=use_lbfgs, max_epochs_without_improvement=max_epochs_without_improvement, relo=relo)

model.eval()

omega_plot_ind = [0, 5, 10]
omegas_plot = [omegas[i] for i in omega_plot_ind]
print('op: ', omegas_plot)
NMSE = visualization.visualise_prediction(x_p, y_p, omegas_plot,  full_known_disp_dict, eigen_mode, max_norm, device,
                                          image_width=n_samp_x,
                                          image_height=n_samp_y, H=H, W=W, H_p=H_p, W_p=W_p, model=model,
                                          sample_step=sample_step,
                                          dist_bound=dist_bound, color=color)
print('NMSE: ', NMSE)

visualization.visualise_loss(free_edges, metric_lam, history_loss, history_lambda)