import numpy as np
import matplotlib.pyplot as plt
import torch


def visualise_init(known_disp, known_disp_map, full_known_disp, x_p, y_p, eigen_mode, image_width=100,
                   image_height=100):
    known_disps = [known_disp_map.get((round(i, 2), round(j, 2)), 0) for index, (i, j) in
                   enumerate(zip(x_p, y_p))]
    kdp = np.reshape(known_disps, (image_width, image_height)).astype(float)
    fkdp = np.reshape(full_known_disp, (image_width, image_height))
    X, Y = np.meshgrid(np.arange(0.05, 10.05, 0.1), np.arange(0.05, 10.05, 0.1))

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    ax3d = fig.add_subplot(221, projection='3d')
    surf1 = ax3d.plot_surface(X, Y, fkdp, cmap='viridis')  # viridis, plasma, inferno, magma, cividis
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_title('Real Displacement mode: {}'.format(eigen_mode))

    ax2d = fig.add_subplot(222)
    im = ax2d.imshow(fkdp, extent=(0, 10, 0, 10), origin='lower', cmap='viridis')
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_title('Real Displacement mode: {}'.format(eigen_mode))

    ax3d = fig.add_subplot(223, projection='3d')
    surf2 = ax3d.plot_surface(X, Y, kdp, cmap='viridis')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_title('Known Points: {}'.format(len(known_disp)))

    ax2d = fig.add_subplot(224)
    im = ax2d.imshow(kdp, extent=(0, 10, 0, 10), origin='lower', cmap='viridis')
    ax2d.set_xlabel('X')
    ax2d.set_ylabel('Y')
    ax2d.set_title('Known Points: {}'.format(len(known_disp)))

    plt.tight_layout()
    plt.show()


def visualise_prediction(x_p, y_p, full_known_disp, eigen_mode, device, pinn=None, image_width=100, image_height=100):
    x_p = torch.tensor(x_p, dtype=torch.float)
    y_p = torch.tensor(y_p, dtype=torch.float)
    x_p = x_p[..., None, None]
    y_p = y_p[..., None, None]
    x = x_p.to(device)  # CUDA
    y = y_p.to(device)

    c = {'coords': torch.cat([x, y], dim=-1).float()}
    pred = pinn(c)['model_out']
    u_pred, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy = (
        pred[:, :, 0:1], pred[:, :, 1:2], pred[:, :, 2:3], pred[:, :, 3:4], pred[:, :, 4:5], pred[:, :, 5:6]
    )
    u_real = full_known_disp.numpy().reshape(image_width, image_height)
    u_pred = u_pred.cpu().detach().numpy().reshape(image_width, image_height)  # CUDA
    NMSE = (np.linalg.norm(u_real - u_pred) ** 2) / (np.linalg.norm(u_real) ** 2)

    X, Y = np.meshgrid(np.arange(0.05, 10.05, 0.1), np.arange(0.05, 10.05, 0.1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, u_pred, cmap='inferno')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Predicted Displacement mode: {}'.format(eigen_mode))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    im1 = axes[0].imshow(u_pred, extent=(0, 10, 0, 10), origin='lower', cmap='viridis')  # Aggiunta della color map
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Predicted Displacement mode: {}'.format(eigen_mode))

    im2 = axes[1].imshow((u_pred - u_real) ** 2, extent=(0, 10, 0, 10), origin='lower',
                         cmap='viridis')  # Aggiunta della color map
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('Squared Error Displacement: {}'.format(NMSE))

    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()

    return NMSE


def visualise_loss(free_edges, metric_lam, history_loss, history_lambda):
    fig = plt.figure(figsize=(6, 4.5), dpi=100)
    plt.plot(torch.log(torch.tensor(history_loss['L_f'])), label='$L_f$ governing equation')
    plt.plot(torch.log(torch.tensor(history_loss['L_t'])), label='$L_t$ Known points')
    plt.plot(torch.log(torch.tensor(history_loss['L_m'])), label='$L_m$')
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
        plt.plot(history_lambda['L_m_lambda'], label='$\lambda_{m}$')
        if not free_edges:
            plt.plot(history_lambda['L_b0_lambda'], label='$\lambda_{b0}$ Dirichlet boundaries')
            plt.plot(history_lambda['L_b2_lambda'], label='$\lambda_{b2}$ Moment boundaries')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('scalings lambda')  # $\lambda$')
        plt.title('ReLoBRaLo weights on Kirchhoff PDE')
        plt.savefig('kirchhoff_lambdas_relobralo')
        plt.show()