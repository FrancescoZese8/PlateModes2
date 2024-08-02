import torch
import numpy as np
from torch.utils.data import Dataset
import random

EPS = 1e-6


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]
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


def scale_to_target(W, H, target, n_d):
    scaling_factor = round(target / max(W, H), 2)
    W = round(W * scaling_factor, n_d)
    H = round(H * scaling_factor, n_d)

    return W, H, scaling_factor


def gaussian_2d(x, y, mu_x, mu_y, sigma):
    return (1 / (2 * np.pi * sigma ** 2)) * torch.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))


class KirchhoffDataset(Dataset):

    def __init__(self, T, nue, E, D, H, W, total_length, den: float, omegas, batch_size_domain, known_disp_concatenate,
                 x_t, y_t, max_norm, free_edges, center_x, center_y, point_load_radius, num_points_load, load_vector,
                 device, sample_step, dist_bound, n_samp_x, n_samp_y, model):
        self.T = T
        self.nue = nue
        self.E = E
        self.D = D
        self.H = H
        self.W = W
        self.total_length = total_length
        self.den = den
        self.omegas = omegas
        self.batch_size_domain = batch_size_domain
        self.known_disp_concatenate = known_disp_concatenate
        self.x_t = x_t
        self.y_t = y_t
        self.max_norm = max_norm
        self.free_edges = free_edges
        self.center_x = center_x
        self.center_y = center_y
        self.point_load_radius = point_load_radius
        self.num_points_load = num_points_load
        self.load_vector = load_vector
        self.device = device
        self.num_loss = 3
        self.sample_step = sample_step
        self.dist_bound = dist_bound
        self.n_samp_x = n_samp_x
        self.n_samp_y = n_samp_y
        self.model = model
        self.gaussian_value_load = None

    def __getitem__(self, item):
        x, y, omega = self.training_batch()
        x.requires_grad_(True)
        y.requires_grad_(True)
        xy = torch.cat([x, y], dim=-1)
        return {'coords': xy, 'omega': omega}

    def __len__(self):
        return self.total_length

    def training_batch(self):
        x_random = torch.rand((self.batch_size_domain,)) * self.W
        y_random = torch.rand((self.batch_size_domain,)) * self.H

        # Generate random angles
        angles = np.random.uniform(0, 2 * np.pi, self.num_points_load)
        # Generate random radii
        radii = np.random.uniform(0, self.point_load_radius, self.num_points_load)
        # Calculate the random points
        x_l = self.center_x + radii * np.cos(angles)
        y_l = self.center_y + radii * np.sin(angles)
        x_l = torch.tensor(x_l, dtype=torch.float32)
        y_l = torch.tensor(y_l, dtype=torch.float32)

        gaussian_value = gaussian_2d(x_l, y_l, self.center_x, self.center_y, 0.01)
        gaussian_value = torch.tensor(gaussian_value, dtype=torch.complex32).to(self.device)
        g_v_l = gaussian_value / torch.sum(gaussian_value) * self.load_vector
        g_v_l = g_v_l.cpu().tolist()
        g_v_l = [0] * len(self.known_disp_concatenate) + g_v_l + [0] * self.batch_size_domain
        self.gaussian_value_load = torch.tensor(g_v_l, dtype=torch.complex32)

        x_t = np.tile(self.x_t, len(self.omegas))
        x_t = torch.tensor(x_t, dtype=torch.float32)
        y_t = np.tile(self.y_t, len(self.omegas))
        y_t = torch.tensor(y_t, dtype=torch.float32)
        x = torch.cat((x_t, x_l, x_random), dim=0)
        y = torch.cat((y_t, y_l, y_random), dim=0)
        x = x[..., None]
        y = y[..., None]
        x = x.to(self.device)  # CUDA
        y = y.to(self.device)

        omega_random = np.random.choice(self.omegas, size=self.batch_size_domain + self.num_points_load)
        omega_random = torch.tensor(omega_random, dtype=torch.float32)
        omegas = torch.tensor(self.omegas, dtype=torch.float32)
        omega_t = omegas.repeat_interleave(len(self.x_t))
        omega = torch.cat((omega_t, omega_random), dim=0)
        omega = omega[..., None]
        omega = omega.to(self.device)  # CUDA

        return x, y, omega

    def compute_loss(self, x, y, omega, preds, eval=False):
        # governing equation loss
        omega = np.squeeze(omega)
        R_u_t = np.squeeze(preds[:len(self.known_disp_concatenate), 0:1])
        I_u_t = np.squeeze(preds[:len(self.known_disp_concatenate), 1:2])
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

        # print('x: ', x.shape)
        # print('u: ', u.shape)
        # print('omega: ', omega.shape)
        # print('dudxxxx: ', dudxxxx.shape)
        # print('kdc: ', self.known_disp_concatenate.shape)
        L_t = (self.known_disp_concatenate.real - R_u_t) ** 2 + (self.known_disp_concatenate.imag - I_u_t) ** 2

        u = R_u + 1j * I_u
        dudxxxx = R_dudxxxx + 1j * I_dudxxxx
        dudxxyy = R_dudxxyy + 1j * I_dudxxyy
        dudyyyy = R_dudyyyy + 1j * I_dudyyyy

        mask = (self.gaussian_value_load == 0)
        L_tot = torch.abs((dudxxxx + 2 * dudxxyy + dudyyyy - (self.den * self.T * (omega ** 2)) / self.D * u
                           - self.gaussian_value_load / self.D))

        # print('1: ', (dudxxxx + 2 * dudxxyy + dudyyyy).mean())
        # print('2: ', ((self.den * self.T * (omega ** 2)) / self.D * u).mean())
        # print('3: ', abs(self.load_vector[~mask]).mean())
        L_f = L_tot[mask] ** 2
        L_l = L_tot[~mask] ** 2 / 10000

        return {'L_f': L_f, 'L_t': L_t, 'L_l': L_l}
