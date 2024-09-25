from modules import PINNet
import config
import torch
import numpy as np
import matplotlib.pyplot as plt

model = config.modules.PINNet(omegas=config.omegas, num_known_points=config.num_known_points,
                              num_hidden_layers=config.num_hidden_layers,
                              hidden_features=config.hidden_features,
                              out_features=len(config.eigen_mode), type=config.opt_model, mode=config.mode)
model.load_state_dict(torch.load('/nas/home/fzese/plateModes/model_weights_no_gov_18nkp.pth'))  #model_weights1.pth
model = model.to(config.device)
model.eval()

n_d = 6
W, H = 0.23, 0.40
W_p, H_p = W, H
n_samp_x, n_samp_y = 23, 40

omegas = [(config.freqs[i] * 2 * torch.pi) for i in config.eigen_mode]
W_norm = max(omegas)
omegas = [omegas[i] / W_norm for i in range(len(omegas))]
L_norm = (config.D / (config.den * config.T * W_norm ** 2)) ** (1 / 4) / 1.4  # TODO
W = W / L_norm
H = H / L_norm
sample_step = W / n_samp_x
dist_bound = sample_step / 2

x_p, y_p = [], []
for i in range(n_samp_y):
    for j in range(n_samp_x):
        x_p.append(round(j * sample_step + dist_bound, n_d))
        y_p.append(round(i * sample_step + dist_bound, n_d))

x_p = torch.tensor(x_p, dtype=torch.float)
y_p = torch.tensor(y_p, dtype=torch.float)
x_p = x_p[..., None, None]
y_p = y_p[..., None, None]
x = x_p.to(config.device)  # CUDA
y = y_p.to(config.device)

c = {'coords': torch.cat([x, y], dim=-1).float()}
pred = model(c, training=False)['model_out']
no = len(config.omegas)
u_pred, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy = (
    pred[:, 0:no], pred[no:no + 1], pred[no + 1:no + 2], pred[no + 2:no + 3], pred[no + 3:no + 4],
    pred[no + 4:no + 5]
)
for i in range(len(config.omegas)):
    u_plot = u_pred[:, i:i + 1]
    u_plot = u_plot.cpu().detach().numpy().reshape(n_samp_y, n_samp_x)  # CUDA

    X, Y = np.meshgrid(np.arange(dist_bound, W + dist_bound, sample_step),
                       np.arange(dist_bound, H + dist_bound, sample_step))
 #W:  9.826033537603676 H:  17.19555869080643 Dist_bound:  0.005 sample_step:  0.01

    # Primo plot (plot 3D)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, u_plot, cmap=config.color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Predicted Displacement mode: {}'.format(config.eigen_mode[i]))

    plt.show()

'''config.visualization.visualise_prediction(config.x_p, config.y_p, config.omegas,
                                          config.full_known_disp,
                                          config.full_known_disp_concatenate,
                                          config.eigen_mode, config.max_norm,
                                          config.device,
                                          image_width=config.n_samp_x,
                                          image_height=config.n_samp_y, H=config.H,
                                          W=config.W, H_p=config.H_p, W_p=config.W_p,
                                          model=model,
                                          sample_step=config.sample_step,
                                          dist_bound=config.dist_bound,
                                          color=config.color)'''
