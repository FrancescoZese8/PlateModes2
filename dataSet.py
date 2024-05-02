import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

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


class KirchhoffDataset(Dataset):

    def __init__(self, T, nue, E, D, H, W, total_length, den: float, omega: float, batch_size_domain, known_disp,
                 full_known_disp, x_t, y_t, x_p, y_p, free_edges, device):
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
        self.x_t = torch.tensor(x_t, dtype=torch.float)
        self.y_t = torch.tensor(y_t, dtype=torch.float)
        self.x_p = torch.tensor(x_p, dtype=torch.float)
        self.y_p = torch.tensor(y_p, dtype=torch.float)
        self.free_edges = free_edges
        self.device = device
        self.num_loss = 2

    def __getitem__(self, item):
        x, y = self.training_batch()
        x.requires_grad_(True)
        y.requires_grad_(True)
        xy = torch.cat([x, y], dim=-1)
        return {'coords': xy}

    def __len__(self):
        return self.total_length

    def training_batch(self):

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
        ## PREDS.SHAPE = [1,1231,6], X.SHAPE = [1,1231]
        ## Il dataLoader aggiunge la prima dimensione = al batchsize
        u = np.squeeze(preds[:, len(self.x_t):, 0:1])
        u_t = np.squeeze(preds[:, :len(self.x_t), 0:1])
        x = np.squeeze(x)
        y = np.squeeze(y)
        dudxx = np.squeeze(preds[:, len(self.x_t):, 1:2])
        dudyy = np.squeeze(preds[:, len(self.x_t):, 2:3])
        dudxxxx = np.squeeze(preds[:, len(self.x_t):, 3:4])
        dudyyyy = np.squeeze(preds[:, len(self.x_t):, 4:5])
        dudxxyy = np.squeeze(preds[:, len(self.x_t):, 5:6])

        # known_disps = [self.known_disp_map.get((round(i, 2), round(j, 2)), u[index]) for index, (i, j) in
        # enumerate(zip(x.tolist(), y.tolist()))]
        # print('k: ', len(known_disps_filtered))
        # print('u: ', len(u_cleaned))
        err_t = self.known_disp - u_t
        f = dudxxxx + 2 * dudxxyy + dudyyyy - (
                self.den * self.T * (self.omega ** 2)) / self.D * u

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

    def __validation_results(self, pinn, image_width=64, image_height=64):
        # x, y, u_real = self.validation_batch(image_width, image_height)
        self.x_p = self.x_p[..., None, None]
        self.y_p = self.y_p[..., None, None]
        x = self.x_p.to(self.device)  # CUDA
        y = self.y_p.to(self.device)
        # x, y = np.mgrid[0:self.W:complex(0, image_width), 0:self.H:complex(0, image_height)]
        # x = torch.tensor(x.reshape(image_width * image_height, 1), dtype=torch.float32)
        # y = torch.tensor(y.reshape(image_width * image_height, 1), dtype=torch.float32)
        # x = x.unsqueeze(1)
        # y = y.unsqueeze(1)
        # x = x.to(self.device)  # CUDA
        # y = y.to(self.device)
        c = {'coords': torch.cat([x, y], dim=-1).float()}
        pred = pinn(c)['model_out']
        u_pred, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy = (
            pred[:, :, 0:1], pred[:, :, 1:2], pred[:, :, 2:3], pred[:, :, 3:4], pred[:, :, 4:5], pred[:, :, 5:6]
        )
        mx, my = compute_moments(self.D, self.nue, dudxx, dudyy)
        f = dudxxxx + 2 * dudxxyy + dudyyyy
        return u_pred, mx, my, f

    def visualise(self, pinn=None, image_width=100, image_height=100):  # QUI

        u_pred, mx, my, f = self.__validation_results(pinn, image_width, image_height)
        u_real = self.full_known_disp.numpy().reshape(image_width, image_height)
        u_pred = u_pred.cpu().detach().numpy().reshape(image_width, image_height)  # CUDA
        NMSE = (np.linalg.norm(u_real - u_pred) ** 2) / (np.linalg.norm(u_real) ** 2)

        X, Y = np.meshgrid(np.arange(0.05, 10.05, 0.1), np.arange(0.05, 10.05, 0.1))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, u_pred, cmap='inferno')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Predicted Displacement')

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        axes[0].imshow(u_pred, extent=(0, 10, 0, 10), origin='lower', cmap='viridis')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_title('Predicted Displacement')

        axes[1].imshow((u_pred - u_real) ** 2, extent=(0, 10, 0, 10), origin='lower', cmap='viridis')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].set_title('Squared Error Displacement: {}'.format(NMSE))

        plt.tight_layout()
        plt.show()

        return NMSE
