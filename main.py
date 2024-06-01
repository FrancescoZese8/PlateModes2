from torch.utils.data import DataLoader
import dataSet
import torch
import loss
import modules
import training
import pandas as pd
import visualization
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA
print('device: ', device)

num_epochs = 120
n_step = 50
batch_size = 1
total_length = 1
lr = 0.001
batch_size_domain = 1000
num_hidden_layers = 2
hidden_features = 32
temperature = 10e-05
rho = 0.5  # 0.99, 0.5, 0.35 da esplorare tra 0.35 e 0.5
alpha = 0.1  # 0.9, 0.1, 0.1 da esplorare tra 0.1 e 0.5

steps_til_summary = 10
opt_model = 'silu'  # mish
mode = 'pinn'
clip_grad = 1.0
use_lbfgs = False
relo = True
max_epochs_without_improvement = 50
free_edges = True
color = 'viridis'  # bwr

#W, H, T, E, nue, den = 0.6, 0.6, 0.005, 10e6, 0.28, 420
W, H, T, E, nue, den = 10, 10, 0.2, 0.7e5, 0.35, 2700
#W, H, scaling_factor = dataSet.scale_to_range(W, H, 1, 10)
scaling_factor = 1
print('W, H, scaling_factor: ', W, H, scaling_factor)

eigen_mode = 22
# omega = 0.078311 * 2 * torch.pi
omega = 0.074002 * 2 * torch.pi
omega = omega / scaling_factor ** 2

D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffnes of the plate

# df = pd.read_csv('ViolinPlateFOD.csv', sep=';')
#df = pd.read_csv('SquarePlateFOD.csv', sep=';')
df = pd.read_csv('FieldOfDisplacement.csv', sep=';')
df_numeric = df.apply(pd.to_numeric, errors='coerce')
# n_samp_x, n_samp_y = 35, 60
#n_samp_x, n_samp_y = 60, 60
n_samp_x, n_samp_y = 100, 100
n_d = 2  #

sample_step = W / n_samp_x
if H / n_samp_y != sample_step:
    print('ERROR: sample step difference')
dist_bound = sample_step / 2

x_p, y_p = [], []
for i in range(n_samp_y):
    for j in range(n_samp_x):
        x_p.append(round(j * sample_step + dist_bound, n_d))
        y_p.append(round(i * sample_step + dist_bound, n_d))

# x_p = [x + 5 for x in x_p]  # off
# y_p = [y + 5 for y in y_p]
#sampled_points = [0.25, 2.55, 5.05, 7.45, 9.75]
#sampled_points = [1.35, 3.65, 6.15, 8.55]
#sampled_points = [7.55]
# sampled_points = [0.25, 1.35, 2.45, 3.55, 4.65, 5.75]

# sampled_points_x = [0.075, 0.175, 0.275]
# sampled_points_y = [0.025, 0.135, 0.245, 0.355, 0.465, 0.575]
# sampled_points_x = [round(point * scaling_factor, n_d) for point in sampled_points_x]
# sampled_points_y = [round(point * scaling_factor, n_d) for point in sampled_points_y]

#sampled_points = [0.075, 0.185, 0.295, 0.405, 0.515]
#sampled_points = [round(point * scaling_factor, n_d) for point in sampled_points]

# sampled_points = [x + 5 for x in [0.025, 0.135, 0.245, 0.355, 0.465, 0.575]]  # off
# sampled_points = [x * 10 for x in [0.025, 0.135, 0.245, 0.355, 0.465, 0.575]]  # norm

'''x_t = []
y_t = []
for y in sampled_points_y:
    for x in sampled_points_x:
        x_t.append(x)
        y_t.append(y)'''

x_t = []
y_t = []

'''for y in sampled_points:
    for x in sampled_points:
        x_t.append(x)
        y_t.append(y)'''

num_known_points = 25
min_distance = 1.5#


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


for i in range(num_known_points):
    while True:
        rand_p = np.random.randint(0, n_samp_x * n_samp_y - 1)
        new_x, new_y = x_p[rand_p], y_p[rand_p]
        if all(euclidean_distance(new_x, new_y, x_t[j], y_t[j]) >= min_distance for j in range(len(x_t))):
            x_t.append(new_x)
            y_t.append(new_y)
            break

# x_t = x_p
# y_t = y_p


full_known_disp = torch.tensor(df_numeric.iloc[:, 2:].values)
full_known_disp = full_known_disp[:, eigen_mode]
min_val = torch.min(full_known_disp)
max_val = torch.max(full_known_disp)
max_norm = 1
full_known_disp = (-1 + 2 * (full_known_disp - min_val) / (max_val - min_val)) * max_norm
full_known_disp_map = dict(zip(zip(x_p, y_p), full_known_disp))
known_disp = [full_known_disp_map.get((round(i, n_d), round(j, n_d)), 0) for index, (i, j) in
              enumerate(zip(x_t, y_t))]
known_disp_map = dict(zip(zip(x_t, y_t), known_disp))
known_disp = torch.tensor(known_disp)

visualization.visualise_init(known_disp, known_disp_map, full_known_disp, x_p, y_p, eigen_mode, image_width=n_samp_x,
                             image_height=n_samp_y, H=H, W=W, sample_step=sample_step, dist_bound=dist_bound, n_d=n_d,
                             color=color)

plate = dataSet.KirchhoffDataset(T=T, nue=nue, E=E, D=D, W=W, H=H, total_length=total_length, den=den,
                                 omega=omega, batch_size_domain=batch_size_domain, known_disp=known_disp,
                                 full_known_disp=full_known_disp, x_t=x_t, y_t=y_t, max_norm=max_norm,
                                 free_edges=free_edges, device=device, sample_step=sample_step, dist_bound=dist_bound,
                                 n_samp_x=n_samp_x, n_samp_y=n_samp_y)

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
               metric=kirchhoff_metric, metric_lam=metric_lam, free_edges=free_edges, clip_grad=clip_grad,
               use_lbfgs=use_lbfgs, max_epochs_without_improvement=max_epochs_without_improvement, relo=relo)
model.eval()

NMSE = visualization.visualise_prediction(x_p, y_p, full_known_disp, eigen_mode, max_norm, device, image_width=n_samp_x,
                                          image_height=n_samp_y, H=H, W=W, model=model, sample_step=sample_step,
                                          dist_bound=dist_bound, color=color)
print('NMSE: ', NMSE)

visualization.visualise_loss(free_edges, metric_lam, history_loss, history_lambda)
