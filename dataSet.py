import torch
import numpy as np
from torch.utils.data import Dataset
from functorch import vmap

EPS = 1e-6


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=[grad_outputs], create_graph=True)[0]
    return grad


def compute_derivatives(x, y, u):
    # print('x: ', x)
    R_u = u[:, 0:1]
    I_u = u[:, 1:2]
    R_dudx = gradient(R_u, x)
    R_dudy = gradient(R_u, y)

    R_dudxx = gradient(R_dudx, x)
    R_dudyy = gradient(R_dudy, y)

    R_dudxxx = gradient(R_dudxx, x)
    R_dudxxy = gradient(R_dudxx, y)
    R_dudyyy = gradient(R_dudyy, y)

    R_dudxxxx = gradient(R_dudxxx, x)
    R_dudxxyy = gradient(R_dudxxy, y)
    R_dudyyyy = gradient(R_dudyyy, y)

    I_dudx = gradient(I_u, x)
    I_dudy = gradient(I_u, y)

    I_dudxx = gradient(I_dudx, x)
    I_dudyy = gradient(I_dudy, y)

    I_dudxxx = gradient(I_dudxx, x)
    I_dudxxy = gradient(I_dudxx, y)
    I_dudyyy = gradient(I_dudyy, y)

    I_dudxxxx = gradient(I_dudxxx, x)
    I_dudxxyy = gradient(I_dudxxy, y)
    I_dudyyyy = gradient(I_dudyyy, y)

    return R_dudxx, R_dudyy, R_dudxxxx, R_dudyyyy, R_dudxxyy, I_dudxx, I_dudyy, I_dudxxxx, I_dudyyyy, I_dudxxyy


def compute_moments(D, nue, dudxx, dudyy):
    mx = -D * (dudxx + nue * dudyy)
    my = -D * (nue * dudxx + dudyy)

    return mx, my


def scale_to_target(W, H, target, n_d):
    scaling_factor = round(target / max(W, H), 2)
    W = round(W * scaling_factor, n_d)
    H = round(H * scaling_factor, n_d)

    return W, H, scaling_factor


class KirchhoffDataset(Dataset):

    def __init__(self, T, nue, E, D, H, W, total_length, den: float, omega: float, batch_size_domain, known_disp,
                 full_known_disp, x_t, y_t, max_norm, free_edges, device, sample_step, dist_bound, n_samp_x,
                 n_samp_y):
        self.T = T
        self.nue = nue
        self.E = E
        self.D = D
        self.H = H
        self.W = W
        self.total_length = total_length
        self.den = den
        self.omega = omega
        self.batch_size_domain = batch_size_domain
        self.known_disp = known_disp.to(device)
        self.full_known_disp = full_known_disp
        self.x_t = torch.tensor(x_t, dtype=torch.float32)
        self.y_t = torch.tensor(y_t, dtype=torch.float32)
        self.max_norm = max_norm
        self.free_edges = free_edges
        self.device = device
        self.num_loss = 3
        self.sample_step = sample_step
        self.dist_bound = dist_bound
        self.n_samp_x = n_samp_x
        self.n_samp_y = n_samp_y

    def __getitem__(self, item):
        x, y = self.training_batch()
        x.requires_grad_(True)
        y.requires_grad_(True)
        xy = torch.cat([x, y], dim=-1)
        return {'coords': xy}

    def __len__(self):
        return self.total_length

    def training_batch(self):
        # x_p = np.arange(self.dist_bound, self.W + self.dist_bound, self.sample_step)
        # y_p = np.arange(self.dist_bound, self.H + self.dist_bound, self.sample_step)
        # x_index = np.random.randint(0, self.n_samp_x, size=self.batch_size_domain)
        # y_index = np.random.randint(0, self.n_samp_y, size=self.batch_size_domain)
        # x_random = torch.tensor(x_p[x_index], dtype=torch.float)
        # y_random = torch.tensor(y_p[y_index], dtype=torch.float)
        x_random = torch.rand((self.batch_size_domain,)) * self.W
        y_random = torch.rand((self.batch_size_domain,)) * self.H

        x = torch.cat((self.x_t, x_random), dim=0)
        y = torch.cat((self.y_t, y_random), dim=0)
        x = x[..., None]
        y = y[..., None]
        x = x.to(self.device)  # CUDA
        y = y.to(self.device)

        return x, y

    def compute_loss(self, x, y, preds, eval=False):

        R_u_t = np.squeeze(preds[:len(self.x_t), 0:1])
        I_u_t = np.squeeze(preds[:len(self.x_t), 1:2])
        # print('u_t ', u_t)
        R_u = np.squeeze(preds[:, 0:1])
        I_u = np.squeeze(preds[:, 1:2])
        R_dudxx = np.squeeze(preds[:, 2:3])
        R_dudyy = np.squeeze(preds[:, 3:4])
        R_dudxxxx = np.squeeze(preds[:, 4:5])
        R_dudyyyy = np.squeeze(preds[:, 5:6])
        R_dudxxyy = np.squeeze(preds[:, 6:7])
        I_dudxx = np.squeeze(preds[:, 7:8])
        I_dudyy = np.squeeze(preds[:, 8:9])
        I_dudxxxx = np.squeeze(preds[:, 9:10])
        I_dudyyyy = np.squeeze(preds[:, 10:11])
        I_dudxxyy = np.squeeze(preds[:, 11:12])

        err_t = (self.known_disp.real - R_u_t)# + (self.known_disp.imag - I_u_t)
        # print('u_t: ', u_t.shape, 'err_t: ', err_t.shape, 'kd: ', self.known_disp.shape)

        f = (R_dudxxxx + 2 * R_dudxxyy + R_dudyyyy - (self.den * self.T * (self.omega ** 2)) / self.D * R_u)
             # + (I_dudxxxx + 2 * I_dudxxyy + I_dudyyyy - (self.den * self.T * (self.omega ** 2)) / self.D * I_u)

        L_f = f ** 2*0
        L_t = err_t ** 2

        return {'L_f': L_f, 'L_t': L_t}
