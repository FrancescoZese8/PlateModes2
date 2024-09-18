import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import math

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


def scale_to_target(W, H, target, n_d):
    scaling_factor = round(target / max(W, H), 2)
    W = round(W * scaling_factor, n_d)
    H = round(H * scaling_factor, n_d)

    return W, H, scaling_factor


class KirchhoffDataset(Dataset):

    def __init__(self, T, nue, E, D, H, W, total_length, den, omegas, batch_size_domain, known_disp_concatenate,
                 x_t, y_t, adim_k, max_norm, third_loss, device, sample_step, dist_bound, n_samp_x,
                 n_samp_y, dynamic_CP, lambda_f):
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
        self.x_t = torch.tensor(x_t, dtype=torch.float32).to(device)
        self.y_t = torch.tensor(y_t, dtype=torch.float32).to(device)
        self.x_refined = None
        self.y_refined = None
        self.adim_k = adim_k
        self.max_norm = max_norm
        self.third_loss = third_loss
        self.device = device
        self.sample_step = sample_step
        self.dist_bound = dist_bound
        self.n_samp_x = n_samp_x
        self.n_samp_y = n_samp_y
        self.residuals = None
        self.x = None
        self.y = None
        self.x_grid = None
        self.y_grid = None
        self.counter = 0
        self.dynamic_CP = dynamic_CP
        self.lambda_f = lambda_f
        # if dynamic_CP:
        self.initialize_grid()
        self.fft = False

    def initialize_grid(self):

        # Calcolo del numero di punti lungo ogni dimensione della griglia
        grid_size = int(math.sqrt(1000))

        x_grid = torch.linspace(0, self.W, grid_size)
        y_grid = torch.linspace(0, self.H, grid_size)

        # Creazione della griglia 2D
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid)

        # Appiattimento della griglia 2D in un vettore 1D
        self.x_grid = x_grid.flatten().to(self.device)
        self.y_grid = y_grid.flatten().to(self.device)

    def __getitem__(self, item):
        x, y = self.training_batch()
        x.requires_grad_(True)
        y.requires_grad_(True)
        xy = torch.cat([x, y], dim=-1)
        return {'coords': xy}

    def __len__(self):
        return self.total_length

    def training_batch(self):

        if not self.dynamic_CP:
            if self.counter % 1000 == 0:
                self.fft = True
                x = self.x_grid
                y = self.y_grid
            else:
                x_random = (torch.rand((self.batch_size_domain,)) * self.W).to(self.device)
                y_random = (torch.rand((self.batch_size_domain,)) * self.H).to(self.device)

                x = torch.cat((self.x_t, x_random), dim=0)
                y = torch.cat((self.y_t, y_random), dim=0)
            x = x[..., None]
            y = y[..., None]
            self.counter = self.counter + 1
        else:
            added_points = self.batch_size_domain
            refining_step = 0.5
            if self.residuals is not None and (self.counter % 2500 == 0):  # TODO

                prob_dist = abs(self.residuals) / abs(self.residuals).sum()

                # Calcola la distribuzione cumulativa
                cumsum_probs = torch.cumsum(prob_dist, dim=0)
                random_values = torch.rand(added_points).to(self.device)
                # Trova gli indici corrispondenti ai numeri casuali nella distribuzione cumulativa
                sampled_indices = torch.searchsorted(cumsum_probs, random_values)
                x_sampled = self.x[sampled_indices]
                y_sampled = self.y[sampled_indices]

                # Step 3: Refine the grid around sampled points
                lambda_x = (torch.rand(added_points, device=self.device) * 2 - 1)  # random between -1 and 1
                lambda_y = (torch.rand(added_points, device=self.device) * 2 - 1)

                #old_x_refined, old_y_refined = (self.x_refined, self.y_refined) if self.x_refined is not None else (None, None)
                x_refined = x_sampled + lambda_x * refining_step
                y_refined = y_sampled + lambda_y * refining_step
                self.x_refined = torch.clamp(x_refined, min=0, max=self.W)
                self.y_refined = torch.clamp(y_refined, min=0, max=self.H)
                #if old_x_refined is not None:
                    #self.x_refined = torch.cat((self.x_refined, old_x_refined), dim=0)
                    #self.y_refined = torch.cat((self.y_refined, old_y_refined), dim=0)

                if self.counter % 2500 == 0:
                    x_t_plot = self.x_t.detach().cpu().numpy()
                    y_t_plot = self.y_t.detach().cpu().numpy()
                    x_refined_plot = self.x_refined.detach().cpu().numpy()
                    y_refined_plot = self.y_refined.detach().cpu().numpy()
                    plt.figure(figsize=(6, 10))
                    plt.scatter(x_t_plot, y_t_plot, c='black', label='Known Points', alpha=1, s=100)
                    plt.scatter(x_refined_plot, y_refined_plot, c='red', label='Refined Points', alpha=0.7)
                    plt.xlabel('x')
                    plt.ylabel('y')
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.title('Refined Collocation Points %s' % len(self.x_refined))
                    plt.legend()
                    plt.grid(True)
                    plt.show()

            self.counter = self.counter + 1
            x_random = (torch.rand((self.batch_size_domain,)) * self.W).to(self.device)
            y_random = (torch.rand((self.batch_size_domain,)) * self.H).to(self.device)
            if self.x_refined is None:
                self.x = torch.cat((self.x_t, x_random), dim=0)
                self.y = torch.cat((self.y_t, y_random), dim=0)
            else:
                self.x = torch.cat((self.x_t, x_random, self.x_refined), dim=0)
                self.y = torch.cat((self.y_t, y_random, self.y_refined), dim=0)
            x = self.x[..., None]
            y = self.y[..., None]
        return x, y

    def compute_loss(self, x, y, preds, eval=False):
        # governing equation loss
        no = len(self.omegas)
        u_t = np.squeeze(preds[:len(self.x_t), 0:no])
        # print('preds ', preds.shape)
        u = np.squeeze(preds[:, 0:no])
        # print('u ', u.shape)
        dudxx = np.squeeze(preds[:, no:no + 1])
        dudyy = np.squeeze(preds[:, no + 1:no + 2])
        dudxxxx = np.squeeze(preds[:, no + 2:no + 3])
        dudyyyy = np.squeeze(preds[:, no + 3:no + 4])
        dudxxyy = np.squeeze(preds[:, no + 4:no + 5])

        if len(self.omegas) == 1:
            u_t = u_t.unsqueeze(-1)
        # print('u_t: ', u_t.shape)
        # print('kdc: ', self.known_disp_concatenate.shape)
        err_t = self.known_disp_concatenate - u_t
        # print('dudxxxx: ', dudxxxx.shape, 'omegas: ', self.omegas.shape, 'u: ', u.shape, 'err_t: ', err_t.shape)

        if len(self.omegas) == 1:
            f = (dudxxxx + 2 * dudxxyy + dudyyyy) - (self.adim_k * (self.omegas ** 2) * u)
            #f = (dudxxxx + 2 * dudxxyy + dudyyyy) - (0.3501277966457757 * (0.1258 ** 2) * u)
        else:
            f = (dudxxxx + 2 * dudxxyy + dudyyyy) - (self.adim_k * torch.sum((self.omegas ** 2) * u, dim=-1))
            #print('omegas: ', self.omegas)

        # print('AAA: ', self.den * self.T / self.D * self.omegas)
        # print('AAA: ', self.adim_k * self.omegas)
        # print('1: ,', (dudxxxx + 2 * dudxxyy + dudyyyy).shape)
        # print('2: ,', (self.adim_k * torch.sum((self.omegas ** 2) * u, dim=-1)).shape)
        # print('f: ', f.shape)

        self.residuals = f

        L_f = f ** 2 * self.lambda_f
        L_t = err_t ** 2

        if self.fft:
            self.fft = False
            for i in range(u.shape[1]):
                u_i = u[:, i].cpu().detach().numpy()
                fourier_transform = np.fft.fft(u_i)
                frequencies = np.fft.fftfreq(len(u_i)) * 2 * np.pi
                amplitude = np.abs(fourier_transform)

                plt.figure()
                plt.plot(frequencies, amplitude)
                plt.xlabel('Frequenza (rad/s)')
                plt.ylabel('Ampiezza')
                plt.title(f'Spettro di Frequenze del Modeshape {i + 1}')
                plt.grid(True)
                plt.show()

        if self.third_loss:
            dot_products = torch.matmul(u.T, u)
            MAC_matrix = torch.zeros_like(dot_products)

            for i in range(u.shape[1]):
                for j in range(u.shape[1]):
                    numerator = (torch.matmul(u[:, i].T, u[:, j])) ** 2
                    denominator = torch.matmul(u[:, i].T, u[:, i]) * torch.matmul(u[:, j].T, u[:, j])
                    MAC_matrix[i, j] = numerator / denominator

            off_diagonal_MAC = MAC_matrix - torch.diag(torch.diag(MAC_matrix))
            loss_ortogonality = torch.sum(torch.abs(off_diagonal_MAC))
            L_o = loss_ortogonality ** 2

            return {'L_f': L_f, 'L_t': L_t, 'L_o': L_o}
        else:
            return {'L_f': L_f, 'L_t': L_t}
