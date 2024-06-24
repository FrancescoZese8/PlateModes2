import torch
import numpy as np
from torch.utils.data import Dataset
import random

EPS = 1e-6


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def compute_derivatives(x, y, u):
    dudx = gradient(u, x)
    dudy = gradient(u, y)

    dudxx = gradient(dudx, x)
    dudyy = gradient(dudy, y)

    dudxxx = gradient(dudxx, x)
    dudxxy = gradient(dudxx, y)
    dudyyy = gradient(dudyy, y)

    dudxxxx = gradient(dudxxx, x)
    dudxxyy = gradient(dudxxy, y)
    dudyyyy = gradient(dudyyy, y)

    return dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy


def compute_moments(D, nue, dudxx, dudyy):
    mx = -D * (dudxx + nue * dudyy)
    my = -D * (nue * dudxx + dudyy)

    return mx, my


def scale_to_target(W, H, target, n_d):
    scaling_factor = round(target/max(W, H), 2)
    W = round(W * scaling_factor, n_d)
    H = round(H * scaling_factor, n_d)

    return W, H, scaling_factor


class KirchhoffDataset(Dataset):

    def __init__(self, T, nue, E, D, H, W, total_length, den: float, omegas, batch_size_domain, known_disp_concatenate,
                 x_t, y_t, max_norm, free_edges, device, sample_step, dist_bound, n_samp_x,
                 n_samp_y):
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
        self.device = device
        self.num_loss = 3
        self.sample_step = sample_step
        self.dist_bound = dist_bound
        self.n_samp_x = n_samp_x
        self.n_samp_y = n_samp_y

    def __getitem__(self, item):
        x, y, omega = self.training_batch()
        x.requires_grad_(True)
        y.requires_grad_(True)
        xy = torch.cat([x, y], dim=-1)
        return {'coords': xy, 'omega': omega}

    def __len__(self):
        return self.total_length

    def training_batch(self):

        x_p = np.arange(self.dist_bound, self.W + self.dist_bound, self.sample_step)
        y_p = np.arange(self.dist_bound, self.H + self.dist_bound, self.sample_step)
        x_index = np.random.randint(0, self.n_samp_x, size=self.batch_size_domain)
        y_index = np.random.randint(0, self.n_samp_y, size=self.batch_size_domain)
        x_random = torch.tensor(x_p[x_index], dtype=torch.float)
        y_random = torch.tensor(y_p[y_index], dtype=torch.float)
        # x_random = torch.rand((self.batch_size_domain,)) * self.W
        # y_random = torch.rand((self.batch_size_domain,)) * self.H

        x_t = np.tile(self.x_t, len(self.omegas))
        x_t = torch.tensor(x_t, dtype=torch.float32)
        y_t = np.tile(self.y_t, len(self.omegas))
        y_t = torch.tensor(y_t, dtype=torch.float32)
        x = torch.cat((x_t, x_random), dim=0)
        y = torch.cat((y_t, y_random), dim=0)
        x = x[..., None]
        y = y[..., None]
        x = x.to(self.device)  # CUDA
        y = y.to(self.device)
        #print('x_t: ', x_t)

        omega_random = np.random.choice(self.omegas)
        omega_random = np.tile(omega_random, self.batch_size_domain)
        omega_random = torch.tensor(omega_random, dtype=torch.float32)
        omegas = torch.tensor(self.omegas, dtype=torch.float32)
        omega_t = omegas.repeat_interleave(len(self.x_t))
        omega = torch.cat((omega_t, omega_random), dim=0)
        omega = omega[..., None]
        omega = omega.to(self.device)  # CUDA

        return x, y, omega

    def compute_loss(self, x, y, omega, preds, eval=False):
        # governing equation loss
        u_t = np.squeeze(preds[:len(self.known_disp_concatenate), 0:1])
        x = np.squeeze(x)
        y = np.squeeze(y)
        omega = np.squeeze(omega)
        #u = np.squeeze(preds[:, 0:1])
        u = np.squeeze(preds[len(self.known_disp_concatenate):, 0:1])
        omega = omega[len(self.known_disp_concatenate):]

        #dudxx = np.squeeze(preds[:, 1:2])
        #dudyy = np.squeeze(preds[:, 2:3])
        #dudxxxx = np.squeeze(preds[:, 3:4])
        #dudyyyy = np.squeeze(preds[:, 4:5])
        #dudxxyy = np.squeeze(preds[:, 5:6])
        dudxx = np.squeeze(preds[len(self.known_disp_concatenate):, 1:2])
        dudyy = np.squeeze(preds[len(self.known_disp_concatenate):, 2:3])
        dudxxxx = np.squeeze(preds[len(self.known_disp_concatenate):, 3:4])
        dudyyyy = np.squeeze(preds[len(self.known_disp_concatenate):, 4:5])
        dudxxyy = np.squeeze(preds[len(self.known_disp_concatenate):, 5:6])

        #print('x: ', x.shape)
        #print('u: ', u.shape)
        #print('omega: ', omega.shape)
        #print('dudxxxx: ', dudxxxx.shape)
        #print('kdc: ', self.known_disp_concatenate.shape)
        err_t = self.known_disp_concatenate - u_t

        f = (dudxxxx + 2 * dudxxyy + dudyyyy -
             (self.den * self.T * (omega ** 2)) / self.D * u)
        #print('f: ', f.shape)

        L_f = f ** 2
        L_t = err_t ** 2

        if not self.free_edges:
            # determine which points are on the boundaries of the domain
            # if a point is on either of the boundaries, its value is 1 and 0 otherwise
            x_lower = torch.where(x <= EPS, torch.tensor(1.0, device=self.device),
                                  torch.tensor(0.0, device=self.device))  # CUDA
            x_upper = torch.where(x >= self.W - EPS, torch.tensor(1.0, device=self.device),
                                  torch.tensor(0.0, device=self.device))
            y_lower = torch.where(y <= EPS, torch.tensor(1.0, device=self.device),
                                  torch.tensor(0.0, device=self.device))
            y_upper = torch.where(y >= self.H - EPS, torch.tensor(1.0, device=self.device),
                                  torch.tensor(0.0, device=self.device))
            # x_lower = torch.where(x <= EPS, torch.tensor(1.0), torch.tensor(0.0))
            # x_upper = torch.where(x >= self.W - EPS, torch.tensor(1.0), torch.tensor(0.0))
            # y_lower = torch.where(y <= EPS, torch.tensor(1.0), torch.tensor(0.0))
            # y_upper = torch.where(y >= self.H - EPS, torch.tensor(1.0), torch.tensor(0.0))

            L_b0 = torch.mul((x_lower + x_upper + y_lower + y_upper), u) ** 2

            # compute 2nd order boundary condition loss
            mx, my = compute_moments(self.D, self.nue, dudxx, dudyy)
            L_b2 = torch.mul((x_lower + x_upper), mx) ** 2 + torch.mul((y_lower + y_upper), my) ** 2

            if eval:
                L_u = torch.zeros(self.batch_size_domain + 4 * self.batch_size_boundary)  # TODO

                return {'L_f': L_f, 'L_b0': L_b0, 'L_b2': L_b2, 'L_u': L_u, 'L_t': L_t}
            return {'L_f': L_f, 'L_b0': L_b0, 'L_b2': L_b2, 'L_t': L_t}
        else:
            return {'L_f': L_f, 'L_t': L_t}
