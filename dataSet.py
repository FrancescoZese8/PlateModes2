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
    # Inizializza le liste per accumulare i risultati delle derivate
    dudx_list = []
    dudy_list = []

    dudxx_list = []
    dudyy_list = []

    dudxxx_list = []
    dudxxy_list = []
    dudyyy_list = []

    dudxxxx_list = []
    dudxxyy_list = []
    dudyyyy_list = []

    # Cicla su ciascuna colonna di u (per ogni frequenza)
    for i in range(u.shape[1]):
        u_i = u[:, i]  # Estrai la colonna i-esima di u

        # Calcola le derivate per la colonna corrente
        dudx = gradient(u_i, x)
        dudy = gradient(u_i, y)

        dudxx = gradient(dudx, x)
        dudyy = gradient(dudy, y)

        dudxxx = gradient(dudxx, x)
        dudxxy = gradient(dudxx, y)
        dudyyy = gradient(dudyy, y)

        dudxxxx = gradient(dudxxx, x)
        dudxxyy = gradient(dudxxy, y)
        dudyyyy = gradient(dudyyy, y)

        # Aggiungi i risultati alla lista corrispondente
        dudx_list.append(dudx)
        dudy_list.append(dudy)

        dudxx_list.append(dudxx)
        dudyy_list.append(dudyy)

        dudxxx_list.append(dudxxx)
        dudxxy_list.append(dudxxy)
        dudyyy_list.append(dudyyy)

        dudxxxx_list.append(dudxxxx)
        dudxxyy_list.append(dudxxyy)
        dudyyyy_list.append(dudyyyy)

    # Converti le liste in tensori con shape [1000, n]
    dudx = torch.stack(dudx_list, dim=1)
    dudy = torch.stack(dudy_list, dim=1)

    dudxx = torch.stack(dudxx_list, dim=1).squeeze(-1)
    dudyy = torch.stack(dudyy_list, dim=1).squeeze(-1)

    dudxxx = torch.stack(dudxxx_list, dim=1)
    dudxxy = torch.stack(dudxxy_list, dim=1)
    dudyyy = torch.stack(dudyyy_list, dim=1)

    dudxxxx = torch.stack(dudxxxx_list, dim=1).squeeze(-1)
    dudxxyy = torch.stack(dudxxyy_list, dim=1).squeeze(-1)
    dudyyyy = torch.stack(dudyyyy_list, dim=1).squeeze(-1)

    return dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy


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

    def __init__(self, T, nue, E, D, H, W, total_length, den, omegas, batch_size_domain, known_disp_concatenate,
                 x_t, y_t, adim_k, max_norm, third_loss, device, sample_step, dist_bound, n_samp_x,
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
        self.known_disp_concatenate = known_disp_concatenate.to(device)
        self.x_t = x_t
        self.y_t = y_t
        self.adim_k = adim_k
        self.max_norm = max_norm
        self.third_loss = third_loss
        self.device = device
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

        # x_t = np.tile(self.x_t, len(self.omegas))
        x_t = torch.tensor(self.x_t, dtype=torch.float32)
        # y_t = np.tile(self.y_t, len(self.omegas))
        y_t = torch.tensor(self.y_t, dtype=torch.float32)
        x = torch.cat((x_t, x_random), dim=0)
        y = torch.cat((y_t, y_random), dim=0)
        x = x[..., None]
        y = y[..., None]
        x = x.to(self.device)  # CUDA
        y = y.to(self.device)

        return x, y

    def compute_loss(self, x, y, preds, eval=False):
        # governing equation loss
        no = len(self.omegas)
        u_t = np.squeeze(preds[:len(self.x_t), 0:no])
        # print('preds ', preds.shape)
        x = np.squeeze(x)
        y = np.squeeze(y)
        omegas = self.omegas.unsqueeze(0)
        u = np.squeeze(preds[len(self.x_t):, 0:no])
        # print('u ', u.shape)
        dudxx = np.squeeze(preds[len(self.x_t):, no:no + no])
        dudyy = np.squeeze(preds[len(self.x_t):, no + no:no + no * 2])
        dudxxxx = np.squeeze(preds[len(self.x_t):, no + no * 2:no + no * 3])
        dudyyyy = np.squeeze(preds[len(self.x_t):, no + no * 3:no + no * 4])
        dudxxyy = np.squeeze(preds[len(self.x_t):, no + no * 4:no + no * 5])

        #  Per singola omega
        omegas = omegas.squeeze(-1)
        u_t = u_t.squeeze(-1)
        self.known_disp_concatenate = self.known_disp_concatenate.squeeze(-1)
        dudxxxx = dudxxxx.squeeze(-1)
        dudxxyy = dudxxyy.squeeze(-1)
        dudyyyy = dudyyyy.squeeze(-1)

        err_t = self.known_disp_concatenate - u_t
        # print('kdc: ', self.known_disp_concatenate.shape)
        # print('u_t: ', u_t.shape)
        # print('u_t: ', u_t.shape)
        # print('dudxxxx: ', dudxxxx.shape, 'omegas: ', omegas.shape, 'u: ', u.shape, 'err_t: ', err_t.shape)

        f = (dudxxxx + 2 * dudxxyy + dudyyyy) - (self.adim_k * (omegas ** 2) * u)
        f = f / (omegas ** 2)

        # f = (dudxxxx + 2 * dudxxyy + dudyyyy - (self.den * self.T * (omegas ** 2)) / self.D * u)
        # print('omegas: ', omegas.shape)
        # print('1: ,', (dudxxxx + 2 * dudxxyy + dudyyyy).shape)
        # print('2: ,', (self.adim_k * (omegas ** 2) * u).shape)
        # print('f: ', f.shape)

        L_f = f ** 2
        L_t = err_t ** 2

        if self.third_loss:
            dot_products = torch.matmul(u.T, u)
            off_diagonal_dot_products = dot_products - torch.diag(torch.diag(dot_products))
            loss_ortogonality = torch.sum(torch.abs(off_diagonal_dot_products))
            L_o = loss_ortogonality ** 2

            return {'L_f': L_f, 'L_t': L_t, 'L_o': L_o}
        else:
            return {'L_f': L_f, 'L_t': L_t}
