from torch.utils.data import DataLoader
import loss
import training
import config
import torch
import modules
import dataSet
import pandas as pd
import numpy as np
import visualization

# 18: mean_NMSE:  0.062243, NMSE_concatenate:  [0.00495, 0.00706, 0.02004, 0.03844, 0.01887, 0.0347, 0.0532, 0.10724, 0.05231, 0.28562]
# 16: mean_NMSE:  0.093795, NMSE_concatenate:  [0.0141, 0.0088, 0.02056, 0.05158, 0.05423, 0.10906, 0.11794, 0.14503, 0.07353, 0.34313]
# 14: mean_NMSE:  0.113774, NMSE_concatenate:  [0.0035, 0.00885, 0.01938, 0.05524, 0.05817, 0.07771, 0.14306, 0.23865, 0.07024, 0.46294]
# 12: mean_NMSE:  0.217943, NMSE_concatenate:  [0.00647, 0.01177, 0.03261, 0.08865, 0.10286, 0.22379, 0.16924, 0.46311, 0.30191, 0.77902]
# 10: mean_NMSE:  0.368998, NMSE_concatenate:  [0.00532, 0.01174, 0.08915, 0.18698, 0.13307, 0.36582, 0.55241, 0.45761, 0.58347, 1.30442]
# 8: mean_NMSE:  0.7127060, NMSE_concatenate:  [0.02877, 0.03633, 0.27295, 0.63965, 0.58429, 1.05996, 1.23066, 0.72658, 0.95573, 1.59214]
# 6: mean_NMSE:  0.8426180, NMSE_concatenate:  [0.10854, 0.27982, 0.26637, 0.64939, 1.25145, 1.78623, 1.00847, 0.96159, 0.84314, 1.27118]


# 18: mean_NMSE:  0.077352, NMSE_concatenate:  [0.01028, 0.04128, 0.06381, 0.01479, 0.01769, 0.04999, 0.11665, 0.03281, 0.05038, 0.37584]
# 16: mean_NMSE:  0.089527, NMSE_concatenate:  [0.00394, 0.00243, 0.01493, 0.01327, 0.01209, 0.03112, 0.04012, 0.09502, 0.03911, 0.64324]
# 14: mean_NMSE:  0.181085, NMSE_concatenate:  [0.00942, 0.01652, 0.01216, 0.0185, 0.03575, 0.07722, 0.11993, 0.22697, 0.05747, 1.23691]
# 12: mean_NMSE:  0.383054, NMSE_concatenate:  [0.01049, 0.02143, 0.08859, 0.07015, 0.1038, 0.16366, 0.56626, 0.48411, 0.39038, 1.93167]
# 10: mean_NMSE:  0.519841, NMSE_concatenate:  [0.00753, 0.01027, 0.15828, 0.37999, 0.06135, 0.18202, 0.86284, 0.93952, 1.00162, 1.59499]
# 8:  mean_NMSE:  1.209315, NMSE_concatenate:  [0.01829, 0.07926, 0.12973, 1.84745, 0.154, 0.63983, 2.58554, 2.61701, 1.51657, 2.50547]
#  6: mean_NMSE:  1.342902, NMSE_concatenate:  [0.08535, 0.36563, 0.78512, 1.91829, 1.08113, 2.08094, 1.82291, 2.09519, 1.07323, 2.12123]

#def main(n, l):
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA
modules.set_seed(3)  # 3 per 13, 4 per 15
print('device: ', device)
num_epochs = 400
n_step = 50
num_known_points = 10
size_norm = 10
batch_size = 1
total_length = 1
lr = 0.001
batch_size_domain = 500
num_hidden_layers = 2
hidden_features = 200

temperature = 10e-3
rho = 0.999
alpha = 0.99

lambda_f = 10  # 10e4 per singolo modo 15
lambda_t = 1
lambda_o = 1

#  [6, 11]: 2, 32
#  [6-12]: 2, 70
#  [6-13]: 2, 70 NMSE: 0.21

steps_til_summary = 10
opt_model = 'sine'  # mish
mode = 'pinn'
clip_grad = 1.0
use_lbfgs = False
relo = True
third_loss = True
adim = True
dynamic_CP = False
num_loss = 3 if third_loss else 2
max_epochs_without_improvement = 50
color = 'viridis'  # bwr

freqs = [None, None, None, None, None, None, 6.499, 7.0867, 15.854, 17.953, 20.396, 25.138, 28.221, 34.876,
         37.256, 45.472, 51.651, 56.464, 59.474, 59.625, 69.244, 71.409, 71.434, 88.497, 88.545, 95.667,
         97.758, 110.03, 110.36, 113.12, 122.91, 123.74, 126.64, 131.98, 136.81, 141.2, 152.5, 160.25, 162.56,
         165.3, ]  # ViolinPlateFOD3

#eigen_mode = [14]
#eigen_mode = [6, 7, 8, 9, 10, 11, 12, 13, 15]
eigen_mode = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

n_d = 6
W, H, T, E, nue, den = 0.20, 0.35, 0.005, 10e6, 0.28, 420
D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffnes of the plate
W_p, H_p = W, H

if not adim:
    W, H, scaling_factor = dataSet.scale_to_target(W, H, size_norm, n_d)
    print('W, H, scaling_factor: ', W, H, scaling_factor)  #
    omegas = [(freqs[i] * 2 * torch.pi / scaling_factor ** 2) for i in eigen_mode]
    adim_k = 1  #
else:
    omegas = [(freqs[i] * 2 * torch.pi) for i in eigen_mode]
    W_norm = max(omegas)
    omegas = [omegas[i] / W_norm for i in range(len(omegas))]
    L_norm = (D / (den * T * W_norm ** 2)) ** (1 / 4) / 1.4  # TODO
    print('L_norm: ', L_norm)
    W = W / L_norm
    H = H / L_norm
    adim_k = (den * T * W_norm ** 2 * L_norm ** 4) / D
    print('adim_k: ', adim_k)
    print('W: ', W, 'H: ', H)

df = pd.read_csv('ViolinPlateFOD2.csv', sep=';')
df_numeric = df.apply(pd.to_numeric, errors='coerce')
n_samp_x, n_samp_y = 20, 35

sample_step = W / n_samp_x
if round(H / n_samp_y, n_d) != sample_step:
    print('Sample step difference')
dist_bound = sample_step / 2
# print('W / n_samp_x: ', W / n_samp_x, 'H / n_samp_y: ', H / n_samp_y)

x_p, y_p = [], []
for i in range(n_samp_y):
    for j in range(n_samp_x):
        x_p.append(round(j * sample_step + dist_bound, n_d))
        y_p.append(round(i * sample_step + dist_bound, n_d))

x_t = []
y_t = []

'''for y in range(4):
    for x in range(3):
        x_t.append(round(x * 8*sample_step + 3*dist_bound, n_d))
        y_t.append(round(y * 11*sample_step + dist_bound, n_d))'''

min_distance = round(np.sqrt(H * W / num_known_points) - np.sqrt(H * W / num_known_points) / 20, n_d)


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

full_known_disp, full_known_disp_concatenate, full_known_disp_map, known_disp, known_disp_map, known_disp_concatenate = [], [], [], [], [], []
full_known_disp_csv = torch.tensor(df_numeric.iloc[:, 2:].values)
max_norm = 1
min_val = 1
max_val = 0
'''for i in range(len(eigen_mode)):
    full_known_disp = full_known_disp_csv[:, eigen_mode[i]]
    min_val_part = torch.min(full_known_disp)
    max_val_part = torch.max(full_known_disp)
    if min_val_part < min_val:
        min_val = min_val_part
    if max_val_part > max_val:
        max_val = max_val_part'''
for i in range(len(eigen_mode)):
    full_known_disp = full_known_disp_csv[:, eigen_mode[i]]
    min_val = torch.min(full_known_disp)
    max_val = torch.max(full_known_disp)
    full_known_disp = (-1 + 2 * (full_known_disp - min_val) / (max_val - min_val)) * max_norm
    full_known_disp_map = dict(zip(zip(x_p, y_p), full_known_disp))
    known_disp = [full_known_disp_map.get((round(i, n_d), round(j, n_d)), 0) for index, (i, j) in
                  enumerate(zip(x_t, y_t))]
    known_disp_map = dict(zip(zip(x_t, y_t), known_disp))
    known_disp = torch.tensor(known_disp).to(device)
    known_disp_concatenate.append(known_disp)
    full_known_disp = torch.tensor(full_known_disp)
    full_known_disp_concatenate.append(full_known_disp)
    if len(omegas) == 1:
        visualization.visualise_init(known_disp, known_disp_map, full_known_disp, x_p, y_p, eigen_mode,
                                     image_width=n_samp_x,
                                     image_height=n_samp_y, H=H, W=W, H_p=H_p, W_p=W_p, sample_step=sample_step,
                                     dist_bound=dist_bound, n_d=n_d, size_norm=size_norm,
                                     color=color)
known_disp_concatenate = torch.stack(known_disp_concatenate, dim=1)
full_known_disp_concatenate = torch.stack(full_known_disp_concatenate, dim=1)
omegas = torch.tensor(omegas).to(device)

'''for i in range(len(omegas)):  # MAC
    for j in range(len(omegas)):
        numerator = abs(torch.matmul(full_known_disp_concatenate[:, i].T,
                                     full_known_disp_concatenate[:, j])) ** 2

        denominator = (torch.matmul(full_known_disp_concatenate[:, i].T,z
                                    full_known_disp_concatenate[:, i])
                       * torch.matmul(full_known_disp_concatenate[:, j].T,
                                      full_known_disp_concatenate[:, j]))
        MAC = numerator / denominator
        off_diagonal_MAC = MAC
        loss_ortogonality = torch.sum(torch.abs(off_diagonal_MAC))
        print(f" {i + 6} e {j + 6}: {loss_ortogonality.item()}")'''
plate = config.dataSet.KirchhoffDataset(T=T, nue=nue, E=E, D=D, W=W, H=H,
                                        total_length=total_length, den=den,
                                        omegas=omegas, batch_size_domain=batch_size_domain,
                                        known_disp_concatenate=known_disp_concatenate, x_t=x_t,
                                        y_t=y_t, adim_k=adim_k,
                                        max_norm=max_norm,
                                        third_loss=third_loss, device=device,
                                        sample_step=sample_step,
                                        dist_bound=dist_bound,
                                        n_samp_x=n_samp_x, n_samp_y=n_samp_y,
                                        dynamic_CP=dynamic_CP, lambda_f=lambda_f,
                                        lambda_t=lambda_t, lambda_o=lambda_o)

data_loader = DataLoader(plate, shuffle=True, batch_size=batch_size, pin_memory=False, num_workers=0)
model = config.modules.PINNet(omegas=omegas, num_known_points=num_known_points,
                              num_hidden_layers=num_hidden_layers,
                              hidden_features=hidden_features,
                              out_features=len(eigen_mode), type=opt_model, mode=mode)
model = model.to(device)  # CUDA

history_loss = {'L_f': [], 'L_t': [], 'L_o': []}
if not relo:
    # loss _fn = loss.MultiTaskLossWrapper(plate, num_tasks=num_loss)
    loss_fn = loss.KirchhoffLoss(plate)
    # loss_fn = loss.IncrementalLoss(plate, config.num_epochs)
    # loss_fn = loss.DWALoss(plate, num_tasks=num_loss)
    kirchhoff_metric = loss.KirchhoffMetric(plate, third_loss=third_loss)
    history_lambda = None
    metric_lam = None
else:
    loss_fn = loss.ReLoBRaLoKirchhoffLoss(plate, num_loss=num_loss, temperature=temperature,
                                          rho=rho, alpha=alpha)
    kirchhoff_metric = loss.KirchhoffMetric(plate, third_loss=third_loss)
    history_lambda = {'L_f_lambda': [], 'L_t_lambda': [], 'L_o_lambda': []}
    metric_lam = loss.ReLoBRaLoLambdaMetric(loss_fn, third_loss=third_loss)

training.train(model=model, train_dataloader=data_loader, epochs=num_epochs, n_step=n_step,
               lr=lr,
               steps_til_summary=steps_til_summary, loss_fn=loss_fn, history_loss=history_loss,
               history_lambda=history_lambda,
               metric=kirchhoff_metric, metric_lam=metric_lam, third_loss=third_loss,
               clip_grad=clip_grad,
               use_lbfgs=use_lbfgs, max_epochs_without_improvement=max_epochs_without_improvement,
               relo=relo)
model.eval()

mean_NMSE, NMSE_concatenate = config.visualization.visualise_prediction(x_p, y_p, omegas,
                                                                        full_known_disp,
                                                                        full_known_disp_concatenate,
                                                                        eigen_mode, max_norm,
                                                                        device,
                                                                        image_width=n_samp_x,
                                                                        image_height=n_samp_y, H=H,
                                                                        W=W, H_p=H_p, W_p=W_p,
                                                                        model=model,
                                                                        sample_step=sample_step,
                                                                        dist_bound=dist_bound,
                                                                        color=color)
print('mean_NMSE: ', mean_NMSE)
print('NMSE_concatenate: ', NMSE_concatenate)

config.visualization.visualise_loss(third_loss, metric_lam, history_loss, history_lambda)
torch.save(model.state_dict(), '/nas/home/fzese/plateModes/model_weights_no_gov_18nkp.pth')

# Per ricaricare il modello in futuro
# model = PINNet(omegas, num_known_points, num_hidden_layers, hidden_features, ...)
# model.load_state_dict(torch.load('model_weights.pth'))

# Visualizzare i pesi finali
# state_dict = model.state_dict()
# for name, param in state_dict.items():
#    print(f"Layer: {name} | Shape: {param.shape}")
#    print(param)

#return mean_NMSE, NMSE_concatenate
