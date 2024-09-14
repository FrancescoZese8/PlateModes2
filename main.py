from torch.utils.data import DataLoader
import dataSet
import torch
import loss
import modules
import training
import pandas as pd
import visualization
import numpy as np


#def main(n, l):
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDA
print('device: ', device)
modules.set_seed(3)

num_epochs = 400
n_step = 50
num_known_points = 10
size_norm = 12
batch_size = 1
total_length = 1
lr = 0.001
batch_size_domain = 1000
num_hidden_layers = 2
hidden_features = 64

temperature = 0.01
rho = 0.9
alpha = 0.99
lambda_f = 1

#  [6, 11]: 2, 32

steps_til_summary = 10
opt_model = 'sine'  # mish
mode = 'pinn'
clip_grad = 1.0
use_lbfgs = False
relo = False
third_loss = False
num_loss = 3 if third_loss else 2
max_epochs_without_improvement = 10#
color = 'viridis'  # bwr
adim = True
dynamic_CP = False

# PROVO Xavier in init, Provo funzione di attivazione paper, provo mac
# omega_0 a 8, provare sine init con normal

freqs = [None, None, None, None, None, None, 6.499, 7.0867, 15.854, 17.953, 20.396, 25.138, 28.221, 34.876,
         37.256, 45.472, 51.651, 56.464, 59.474, 59.625, 69.244, 71.409, 71.434, 88.497, 88.545, 95.667,
         97.758, 110.03, 110.36, 113.12, 122.91, 123.74, 126.64, 131.98, 136.81, 141.2, 152.5, 160.25, 162.56,
         165.3, ]  # ViolinPlateFOD3

eigen_mode = [6, 7, 8, 9, 10, 11, 12]
# eigen_mode = [6, 7, 8, 14, 15]
#eigen_mode = [6, 7, 8, 9, 10, 11, 12, 13, 15, 16]

n_d = 6
W, H, T, E, nue, den = 0.20, 0.35, 0.005, 10e6, 0.28, 420
# W, H, T, E, nue, den = 10, 10, 0.05, 10e6, 0.28, 420
D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffnes of the plate
W_p, H_p = W, H

# omegas = [(freqs[i] * 2 * torch.pi) for i in eigen_mode]
# adim_k = 1

if not adim:
    W, H, scaling_factor = dataSet.scale_to_target(W, H, size_norm, n_d)  #
    print('W, H, scaling_factor: ', W, H, scaling_factor)  #
    omegas = [(freqs[i] * 2 * torch.pi / scaling_factor ** 2) for i in eigen_mode]  #
    adim_k = 1  #
else:
    omegas = [(freqs[i] * 2 * torch.pi) for i in eigen_mode]
    W_norm = max(omegas)
    omegas = [omegas[i] / W_norm for i in range(len(omegas))]
    L_norm = (D / (den * T * W_norm ** 2)) ** (1 / 4) / 1.3  # TODO
    print('L_norm: ', L_norm)
    #L_norm = 0.020566912384165324
    W = W / L_norm
    H = H / L_norm
    adim_k = (den * T * W_norm ** 2 * L_norm ** 4) / D
    print('adim_k: ', adim_k)
    print('W: ', W, 'H: ', H)

    # L_norm = 0.029, k = 0.35, omegas = 1 ---> NMSE = 0.56
    # L_norm = 0.021, k = 0.35, omegas = 0.48 ---> NMSE = 0.22
    # L_norm = 0.038, k = 1, o omegas = 1 ---> NMSE = 0.38
    # L_norm = 0.019, k = 0.062, o omegas = 1 ---> NMSE = 0.9

df = pd.read_csv('ViolinPlateFOD2.csv', sep=';')
df_numeric = df.apply(pd.to_numeric, errors='coerce')
n_samp_x, n_samp_y = 20, 35
# n_samp_x, n_samp_y = 50, 50

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

    '''visualization.visualise_init(known_disp, known_disp_map, full_known_disp, x_p, y_p, eigen_mode,
                                 image_width=n_samp_x,
                                 image_height=n_samp_y, H=H, W=W, H_p=H_p, W_p=W_p, sample_step=sample_step,
                                 dist_bound=dist_bound, n_d=n_d, size_norm=size_norm,
                                 color=color)'''
known_disp_concatenate = torch.stack(known_disp_concatenate, dim=1)
full_known_disp_concatenate = torch.stack(full_known_disp_concatenate, dim=1)
omegas = torch.tensor(omegas).to(device)

for i in range(len(omegas)):  # MAC
    for j in range(len(omegas)):
        numerator = abs(torch.matmul(full_known_disp_concatenate[:, i].T,
                                     full_known_disp_concatenate[:, j])) ** 2

        denominator = (torch.matmul(full_known_disp_concatenate[:, i].T,
                                    full_known_disp_concatenate[:, i])
                       * torch.matmul(full_known_disp_concatenate[:, j].T,
                                      full_known_disp_concatenate[:, j]))
        MAC = numerator / denominator
        off_diagonal_MAC = MAC
        loss_ortogonality = torch.sum(torch.abs(off_diagonal_MAC))
        print(f" {i + 6} e {j + 6}: {loss_ortogonality.item()}")

plate = dataSet.KirchhoffDataset(T=T, nue=nue, E=E, D=D, W=W, H=H, total_length=total_length, den=den,
                                 omegas=omegas, batch_size_domain=batch_size_domain,
                                 known_disp_concatenate=known_disp_concatenate, x_t=x_t, y_t=y_t, adim_k=adim_k,
                                 max_norm=max_norm,
                                 third_loss=third_loss, device=device, sample_step=sample_step,
                                 dist_bound=dist_bound,
                                 n_samp_x=n_samp_x, n_samp_y=n_samp_y, dynamic_CP=dynamic_CP, lambda_f=lambda_f)

data_loader = DataLoader(plate, shuffle=True, batch_size=batch_size, pin_memory=False, num_workers=0)
model = modules.PINNet(omegas=omegas, num_known_points=num_known_points, num_hidden_layers=num_hidden_layers,
                       hidden_features=hidden_features,
                       out_features=len(eigen_mode), type=opt_model, mode=mode)
model = model.to(device)  # CUDA

history_loss = {'L_f': [], 'L_t': [], 'L_o': []}
if not relo:
    # loss_fn = loss.MultiTaskLossWrapper(plate, num_tasks=num_loss)
    loss_fn = loss.KirchhoffLoss(plate)
    # loss_fn = loss.DWALoss(plate, num_tasks=num_loss)
    kirchhoff_metric = loss.KirchhoffMetric(plate, third_loss=third_loss)
    history_lambda = None
    metric_lam = None
else:
    loss_fn = loss.ReLoBRaLoKirchhoffLoss(plate, num_loss=num_loss, temperature=temperature, rho=rho, alpha=alpha)
    kirchhoff_metric = loss.KirchhoffMetric(plate, third_loss=third_loss)
    history_lambda = {'L_f_lambda': [], 'L_t_lambda': [], 'L_o_lambda': []}
    metric_lam = loss.ReLoBRaLoLambdaMetric(loss_fn, third_loss=third_loss)

training.train(model=model, train_dataloader=data_loader, epochs=num_epochs, n_step=n_step, lr=lr,
               steps_til_summary=steps_til_summary, loss_fn=loss_fn, history_loss=history_loss,
               history_lambda=history_lambda,
               metric=kirchhoff_metric, metric_lam=metric_lam, third_loss=third_loss, clip_grad=clip_grad,
               use_lbfgs=use_lbfgs, max_epochs_without_improvement=max_epochs_without_improvement, relo=relo)
model.eval()

mean_NMSE = visualization.visualise_prediction(x_p, y_p, omegas, full_known_disp, full_known_disp_concatenate,
                                               eigen_mode, max_norm, device,
                                               image_width=n_samp_x,
                                               image_height=n_samp_y, H=H, W=W, H_p=H_p, W_p=W_p, model=model,
                                               sample_step=sample_step,
                                               dist_bound=dist_bound, color=color)
print('mean_NMSE: ', mean_NMSE)

visualization.visualise_loss(third_loss, metric_lam, history_loss, history_lambda)
torch.save(model.state_dict(), '/nas/home/fzese/plateModes/model_weights.pth')

# Per ricaricare il modello in futuro
# model = PINNet(omegas, num_known_points, num_hidden_layers, hidden_features, ...)
# model.load_state_dict(torch.load('model_weights.pth'))

# Visualizzare i pesi finali
#state_dict = model.state_dict()
#for name, param in state_dict.items():
#    print(f"Layer: {name} | Shape: {param.shape}")
#    print(param)

    #return mean_NMSE
