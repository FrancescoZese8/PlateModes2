import torch
import numpy as np
from torch.utils.data import Dataset

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
    dudyyy = gradient(dudy, y)

    dudxxxx = gradient(dudxxx, x)
    dudxxyy = gradient(dudxxy, y)
    dudyyyy = gradient(dudyyy, y)

    return dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy


def compute_moments(D, nue, dudxx, dudyy):
    mx = -D * (dudxx + nue * dudyy)
    my = -D * (nue * dudxx + dudyy)

    return mx, my


def scale_to_range(W, H, min_value, max_value):
    scaling_factor = 1

    while max(W, H) < min_value or max(W, H) > max_value:
        if max(W, H) < min_value:
            W *= 10
            H *= 10
            scaling_factor *= 10
        elif max(W, H) > max_value:
            W /= 10
            H /= 10
            scaling_factor /= 10

    return W, H, scaling_factor


class KirchhoffDataset(Dataset):

    def __init__(self, T, nue, E, D, H, W, total_length, den: float, omega: float, batch_size_domain, known_disp,
                 full_known_disp, x_t, y_t, max_norm, free_edges, device, sample_step, dist_bound, n_samp_x, n_samp_y):
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

        #x_p = np.arange(self.dist_bound, self.W + self.dist_bound, self.sample_step)
        #y_p = np.arange(self.dist_bound, self.H + self.dist_bound, self.sample_step)
        #x_index = np.random.randint(0, self.n_samp_x, size=self.batch_size_domain)
        #y_index = np.random.randint(0, self.n_samp_y, size=self.batch_size_domain)
        #x_random = torch.tensor(x_p[x_index], dtype=torch.float)
        #y_random = torch.tensor(y_p[y_index], dtype=torch.float)
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
        # governing equation loss
        # u = np.squeeze(preds[:, len(self.x_t):, 0:1])
        u_t = np.squeeze(preds[:len(self.x_t), 0:1])
        x = np.squeeze(x)
        y = np.squeeze(y)
        u = np.squeeze(preds[:, 0:1])
        # dudxx = np.squeeze(preds[:, len(self.x_t):, 1:2])
        # dudyy = np.squeeze(preds[:, len(self.x_t):, 2:3])
        # dudxxxx = np.squeeze(preds[:, len(self.x_t):, 3:4])
        # dudyyyy = np.squeeze(preds[:, len(self.x_t):, 4:5])
        # dudxxyy = np.squeeze(preds[:, len(self.x_t):, 5:6])
        dudxx = np.squeeze(preds[:, 1:2])
        dudyy = np.squeeze(preds[:, 2:3])
        dudxxxx = np.squeeze(preds[:, 3:4])
        dudyyyy = np.squeeze(preds[:, 4:5])
        dudxxyy = np.squeeze(preds[:, 5:6])

        err_t = self.known_disp - u_t
        # print('u_t: ', u_t.shape, 'err_t: ', err_t.shape, 'kd: ', self.known_disp.shape)
        max_u = abs(u.max().item())
        if max_u > self.max_norm:
            err_m = max_u - self.max_norm
        else:
            err_m = 0
        err_m = torch.tensor(err_m, dtype=torch.float)
        # known_disps = [self.known_disp_map.get((round(i, 2), round(j, 2)), u[index]) for index, (i, j) in
        # enumerate(zip(x.tolist(), y.tolist()))]
        # known_disps = torch.tensor(known_disps).to(self.device)
        f = (dudxxxx + 2 * dudxxyy + dudyyyy -
             (self.den * self.T * (self.omega ** 2)) / self.D * u)

        L_f = f ** 2
        L_t = err_t ** 2
        L_m = err_m ** 2 * 0

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
            return {'L_f': L_f, 'L_t': L_t, 'L_m': L_m}
