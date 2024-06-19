import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib import cm
from matplotlib.colors import Normalize


def visualise_init(known_disp, known_disp_map, full_known_disp, x_p, y_p, eigen_mode, image_width,
                   image_height, H, W, H_p, W_p, sample_step, dist_bound, n_d, size_norm, color):
    known_disps = [known_disp_map.get((round(i, n_d), round(j, n_d)), 0) for index, (i, j) in
                   enumerate(zip(x_p, y_p))]
    kdp = np.reshape(known_disps, (image_height, image_width))
    fkdp = np.reshape(full_known_disp, (image_height, image_width))
    X, Y = np.meshgrid(np.arange(dist_bound, W + dist_bound, sample_step),
                       np.arange(dist_bound, H + dist_bound, sample_step))

    fig = plt.figure(figsize=(12, 10))

    # Primo subplot
    ax3d_1 = fig.add_subplot(221, projection='3d')
    surf1 = ax3d_1.plot_surface(X, Y, fkdp, cmap=color)
    ax3d_1.set_xlabel('X')
    ax3d_1.set_ylabel('Y')
    ax3d_1.set_title('Real Displacement mode: {}'.format(eigen_mode))

    ax3d_1.text2D(0.05, 0.95, 'Size_norm: {}'.format(size_norm), transform=ax3d_1.transAxes)

    # Secondo subplot
    ax2d_1 = fig.add_subplot(222)
    im1 = ax2d_1.imshow(fkdp, extent=(0, W_p, 0, H_p), origin='lower', cmap=color)
    ax2d_1.set_xlabel('X')
    ax2d_1.set_ylabel('Y')
    ax2d_1.set_title('Real Displacement mode: {}'.format(eigen_mode))

    # Terzo subplot
    ax3d_2 = fig.add_subplot(223, projection='3d')
    surf2 = ax3d_2.plot_surface(X, Y, kdp, cmap=color)
    ax3d_2.set_xlabel('X')
    ax3d_2.set_ylabel('Y')
    ax3d_2.set_title('Known Points: {}'.format(len(known_disp)))

    # Quarto subplot
    ax2d_2 = fig.add_subplot(224)
    im2 = ax2d_2.imshow(kdp, extent=(0, W_p, 0, H_p), origin='lower', cmap=color)
    ax2d_2.set_xlabel('X')
    ax2d_2.set_ylabel('Y')
    ax2d_2.set_title('Known Points: {}'.format(len(known_disp)))

    plt.tight_layout()
    plt.show()


def visualise_prediction(x_p, y_p, omegas_plot, full_known_disp_dict, eigen_mode, max_norm, device, image_width,
                         image_height, H, W, H_p,
                         W_p, model, sample_step, dist_bound, color):

    x_p = torch.tensor(x_p, dtype=torch.float)
    y_p = torch.tensor(y_p, dtype=torch.float)
    omegas_plot = torch.tensor(omegas_plot, dtype=torch.float)
    x_p = x_p[..., None, None]
    y_p = y_p[..., None, None]
    x = x_p.to(device)  # CUDA
    y = y_p.to(device)
    for i in range(len(omegas_plot)):
        omega_plot = omegas_plot[i]
        omega_plot = omega_plot[None]
        print('omega_plot: ', omega_plot)
        #print('x_p: ', x_p.shape)
        omega_plot = omega_plot.to(device)

        c = {'coords': torch.cat([x, y], dim=-1).float(), 'omega': omega_plot}
        pred = model(c)['model_out']
        u_pred, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy = (
            pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4], pred[:, 4:5], pred[:, 5:6]
        )
        u_real = full_known_disp_dict[round(omega_plot.item(), 6)].numpy().reshape(image_height, image_width)
        u_pred = u_pred.cpu().detach().numpy().reshape(image_height, image_width)  # CUDA
        NMSE = round((np.linalg.norm(u_real - u_pred) ** 2) / (np.linalg.norm(u_real) ** 2), 5)

        # dudy = dudyyyy.cpu().detach().numpy().reshape(image_height, image_width)
        # dudx = dudxxxx.cpu().detach().numpy().reshape(image_height, image_width)

        X, Y = np.meshgrid(np.arange(dist_bound, W + dist_bound, sample_step),
                           np.arange(dist_bound, H + dist_bound, sample_step))

        # Primo plot (plot 3D)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, u_pred, cmap=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Predicted Displacement mode: {}'.format(eigen_mode))

        # Mostra il primo plot
        plt.show()

        # Secondo plot (subplot con due immagini)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

        im1 = axes[0].imshow(u_pred, extent=(0, W_p, 0, H_p), origin='lower', cmap=color)
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_title('Predicted Displacement mode: {}'.format(eigen_mode))

        im2 = axes[1].imshow((u_pred - u_real) ** 2, extent=(0, W_p, 0, H_p), origin='lower', cmap=color)
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        axes[1].set_title('Squared Error Displacement: {}'.format(NMSE))

        plt.show()

        fig.colorbar(im1, ax=axes[0])
        fig.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()

        '''# Plot di du
        plt.figure(figsize=(8, 6))
        plt.imshow(dudy, extent=(0, W_p, 0, H_p), origin='lower', cmap=color)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('dudyyyy')
        plt.colorbar(label='Increment')
        plt.show()

        # Plot di du
        plt.figure(figsize=(8, 6))
        plt.imshow(dudx, extent=(0, W_p, 0, H_p), origin='lower', cmap=color)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('dudxxxx')
        plt.colorbar(label='Increment')
        plt.show()'''

    return NMSE


def visualise_loss(free_edges, metric_lam, history_loss, history_lambda):
    fig = plt.figure(figsize=(6, 4.5), dpi=100)
    plt.plot(torch.log(torch.tensor(history_loss['L_f'])), label='$L_f$ governing equation')
    plt.plot(torch.log(torch.tensor(history_loss['L_t'])), label='$L_t$ Known points')
    # plt.plot(torch.log(torch.tensor(history_loss['L_m'])), label='$L_m$ Simmetry points')
    if not free_edges:
        plt.plot(torch.log(torch.tensor(history_loss['L_b0'])), label='$L_{b0}$ Dirichlet boundaries')
        plt.plot(torch.log(torch.tensor(history_loss['L_b2'])), label='$L_{b2}$ Moment boundaries')
        plt.plot(torch.log(torch.tensor(history_loss['L_u'])), label='$L_u$ analytical solution')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Log-loss')
    plt.title('Loss evolution Kirchhoff PDE')
    plt.savefig('kirchhoff_loss_unscaled')
    plt.show()

    if metric_lam is not None:
        fig2 = plt.figure(figsize=(6, 4.5), dpi=100)
        plt.plot(history_lambda['L_f_lambda'], label='$\lambda_f$ governing equation')
        plt.plot(history_lambda['L_t_lambda'], label='$\lambda_{t}$ Known points')
        # plt.plot(history_lambda['L_m_lambda'], label='$\lambda_{m}$ Simmetry points')
        if not free_edges:
            plt.plot(history_lambda['L_b0_lambda'], label='$\lambda_{b0}$ Dirichlet boundaries')
            plt.plot(history_lambda['L_b2_lambda'], label='$\lambda_{b2}$ Moment boundaries')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('scalings lambda')  # $\lambda$')
        plt.title('ReLoBRaLo weights on Kirchhoff PDE')
        plt.savefig('kirchhoff_lambdas_relobralo')
        plt.show()
