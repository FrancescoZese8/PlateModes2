import torch
import numpy as np
from typing import Tuple
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

    def __init__(self, T, nue, E, H, W, total_length, den: float, omega: float, batch_size_domain, known_disp,
                 known_disp_map, x_t, y_t, coords, free_edges, device):
        self.T = T
        self.nue = nue
        self.E = E
        self.D = (E * T ** 3) / (12 * (1 - nue ** 2))  # flexural stiffness of the plate
        self.H = H
        self.W = W
        self.num_terms = 4
        self.total_length = total_length
        self.den = den
        self.omega = omega
        self.batch_size_domain = batch_size_domain
        self.known_disp = known_disp.to(device)
        self.known_disp_map = known_disp_map
        self.x_t = torch.tensor(x_t, dtype=torch.float)
        self.y_t = torch.tensor(y_t, dtype=torch.float)
        self.coords = coords
        self.free_edges = free_edges
        self.device = device

    def __getitem__(self, item):
        x, y = self.training_batch()
        x.requires_grad_(True)
        y.requires_grad_(True)
        xy = torch.cat([x, y], dim=-1)
        ## XY.SHAPE = [1231,2]
        return {'coords': xy}

    def __len__(self):
        return self.total_length

    def training_batch(self):

        #x_random = torch.rand((self.batch_size_domain,)) * self.W
        #y_random = torch.rand((self.batch_size_domain,)) * self.H
        y_random = []
        x_random = []

        for i in range(self.batch_size_domain):
            index_x = torch.randint(0, len(self.coords), (1,)).item()
            index_y = torch.randint(0, len(self.coords), (1,)).item()
            x_random.append(self.coords[index_x])
            y_random.append(self.coords[index_y])

        x_random = torch.tensor(x_random, dtype=torch.float)
        y_random = torch.tensor(y_random, dtype=torch.float)

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

        #known_disps = [self.known_disp_map.get((round(i, 2), round(j, 2)), u[index]) for index, (i, j) in
                       #enumerate(zip(x.tolist(), y.tolist()))]
        #print('k: ', len(known_disps_filtered))
        #print('u: ', len(u_cleaned))
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
        x, y = np.mgrid[0:self.W:complex(0, image_width), 0:self.H:complex(0, image_height)]
        x = torch.tensor(x.reshape(image_width * image_height, 1), dtype=torch.float32)
        y = torch.tensor(y.reshape(image_width * image_height, 1), dtype=torch.float32)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        x = x.to(self.device)  # CUDA
        y = y.to(self.device)
        c = {'coords': torch.cat([x, y], dim=-1).float()}
        pred = pinn(c)['model_out']
        u_pred, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy = (
            pred[:, :, 0:1], pred[:, :, 1:2], pred[:, :, 2:3], pred[:, :, 3:4], pred[:, :, 4:5], pred[:, :, 5:6]
        )
        mx, my = compute_moments(self.D, self.nue, dudxx, dudyy)
        f = dudxxxx + 2 * dudxxyy + dudyyyy
        return u_pred, mx, my, f

    def visualise(self, pinn=None, image_width=64, image_height=64):

        u_pred, mx, my, f = self.__validation_results(pinn, image_width, image_height)
        # u_real = u_real.cpu().detach().numpy().reshape(image_width, image_height)  # CUDA
        u_pred = u_pred.cpu().detach().numpy().reshape(image_width, image_height)  # CUDA

        # self.__plot_3d(u_real, 'Real Displacement (m)')

        # Plot 3D for u_pred
        self.__plot_3d(u_pred, 'Predicted Displacement (m)')

        fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
        self.__show_image(u_pred, axs[0], 'Predicted Displacement (m)')
        # self.__show_image((u_pred - u_real) ** 2, axs[1], 'Squared Error Displacement')
        ##NMSE = (np.linalg.norm(u_real - u_pred) ** 2) / (np.linalg.norm(u_real) ** 2)
        ##print('NMSE: ', NMSE)

        for ax in axs.flat:
            ax.label_outer()

        plt.tight_layout()
        plt.show()

    def __show_image(self, img, axis=None, title='', x_label='x [m]', y_label='y [m]', z_label=''):
        if axis is None:
            _, axis = plt.subplots(1, 1, figsize=(4, 3.2), dpi=100)
        im = axis.imshow(np.rot90(img, k=3), origin='lower', aspect='auto', cmap='viridis')
        cb = plt.colorbar(im, label=z_label, ax=axis)
        axis.set_xticks([0, img.shape[0] - 1])
        axis.set_xticklabels([0, self.W])
        axis.set_yticks([0, img.shape[1] - 1])
        axis.set_yticklabels([0, self.H])
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title)
        return im

    def __plot_3d(self, data, title=''):

        X, Y = np.mgrid[0:self.W:complex(0, 64), 0:self.H:complex(0, 64)]
        Z = data

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        # ax.invert_xaxis()
        plt.show()
