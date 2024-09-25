from torch.utils.data import DataLoader
import loss
import training
import config
import torch


# 18: mean_NMSE:  0.062243, NMSE_concatenate:  [0.00495, 0.00706, 0.02004, 0.03844, 0.01887, 0.0347, 0.0532, 0.10724, 0.05231, 0.28562]
# 16: mean_NMSE:  0.093795, NMSE_concatenate:  [0.0141, 0.0088, 0.02056, 0.05158, 0.05423, 0.10906, 0.11794, 0.14503, 0.07353, 0.34313]
# 14: mean_NMSE:  0.113774, NMSE_concatenate:  [0.0035, 0.00885, 0.01938, 0.05524, 0.05817, 0.07771, 0.14306, 0.23865, 0.07024, 0.46294]
# 12: mean_NMSE:  0.217943, NMSE_concatenate:  [0.00647, 0.01177, 0.03261, 0.08865, 0.10286, 0.22379, 0.16924, 0.46311, 0.30191, 0.77902]
# 10: mean_NMSE:  0.368998, NMSE_concatenate:  [0.00532, 0.01174, 0.08915, 0.18698, 0.13307, 0.36582, 0.55241, 0.45761, 0.58347, 1.30442]
# 8: mean_NMSE:  0.7127060, NMSE_concatenate:  [0.02877, 0.03633, 0.27295, 0.63965, 0.58429, 1.05996, 1.23066, 0.72658, 0.95573, 1.59214]
# 6: mean_NMSE:  0.8426180, NMSE_concatenate:  [0.10854, 0.27982, 0.26637, 0.64939, 1.25145, 1.78623, 1.00847, 0.96159, 0.84314, 1.27118]


# 18: mean_NMSE:  0.077352, NMSE_concatenate:  [0.01028, 0.04128, 0.06381, 0.01479, 0.01769, 0.04999, 0.11665, 0.03281, 0.05038, 0.37584]
# 16: mean_NMSE:  0.089527, NMSE_concatenate:  [0.00394, 0.00243, 0.01493, 0.01327, 0.01209, 0.03112, 0.04012, 0.09502, 0.03911, 0.64324]
# 14: mean_NMSE:  0.181085, NMSE_concatenate:  [0.00942, 0.01652, 0.01216, 0.0185, 0.03575, 0.07722, 0.11993, 0.22697, 0.05747, 1.23691]
# 12: mean_NMSE:  0.383054, NMSE_concatenate:  [0.01049, 0.02143, 0.08859, 0.07015, 0.1038, 0.16366, 0.56626, 0.48411, 0.39038, 1.93167]
# 10: mean_NMSE:  0.519841, NMSE_concatenate:  [0.00753, 0.01027, 0.15828, 0.37999, 0.06135, 0.18202, 0.86284, 0.93952, 1.00162, 1.59499]
# 8:  mean_NMSE:  1.209315, NMSE_concatenate:  [0.01829, 0.07926, 0.12973, 1.84745, 0.154, 0.63983, 2.58554, 2.61701, 1.51657, 2.50547]
#  6: mean_NMSE:  1.342902, NMSE_concatenate:  [0.08535, 0.36563, 0.78512, 1.91829, 1.08113, 2.08094, 1.82291, 2.09519, 1.07323, 2.12123]

#def main(n):
plate = config.dataSet.KirchhoffDataset(T=config.T, nue=config.nue, E=config.E, D=config.D, W=config.W, H=config.H,
                                        total_length=config.total_length, den=config.den,
                                        omegas=config.omegas, batch_size_domain=config.batch_size_domain,
                                        known_disp_concatenate=config.known_disp_concatenate, x_t=config.x_t,
                                        y_t=config.y_t, adim_k=config.adim_k,
                                        max_norm=config.max_norm,
                                        third_loss=config.third_loss, device=config.device,
                                        sample_step=config.sample_step,
                                        dist_bound=config.dist_bound,
                                        n_samp_x=config.n_samp_x, n_samp_y=config.n_samp_y,
                                        dynamic_CP=config.dynamic_CP, lambda_f=config.lambda_f)

data_loader = DataLoader(plate, shuffle=True, batch_size=config.batch_size, pin_memory=False, num_workers=0)
model = config.modules.PINNet(omegas=config.omegas, num_known_points=config.num_known_points,
                              num_hidden_layers=config.num_hidden_layers,
                              hidden_features=config.hidden_features,
                              out_features=len(config.eigen_mode), type=config.opt_model, mode=config.mode)
model = model.to(config.device)  # CUDA

history_loss = {'L_f': [], 'L_t': [], 'L_o': []}
if not config.relo:
    # loss_fn = loss.MultiTaskLossWrapper(plate, num_tasks=num_loss)
    loss_fn = loss.KirchhoffLoss(plate)
    # loss_fn = loss.DWALoss(plate, num_tasks=num_loss)
    kirchhoff_metric = loss.KirchhoffMetric(plate, third_loss=config.third_loss)
    history_lambda = None
    metric_lam = None
else:
    loss_fn = loss.ReLoBRaLoKirchhoffLoss(plate, num_loss=config.num_loss, temperature=config.temperature,
                                          rho=config.rho, alpha=config.alpha)
    kirchhoff_metric = loss.KirchhoffMetric(plate, third_loss=config.third_loss)
    history_lambda = {'L_f_lambda': [], 'L_t_lambda': [], 'L_o_lambda': []}
    metric_lam = loss.ReLoBRaLoLambdaMetric(loss_fn, third_loss=config.third_loss)

training.train(model=model, train_dataloader=data_loader, epochs=config.num_epochs, n_step=config.n_step,
               lr=config.lr,
               steps_til_summary=config.steps_til_summary, loss_fn=loss_fn, history_loss=history_loss,
               history_lambda=history_lambda,
               metric=kirchhoff_metric, metric_lam=metric_lam, third_loss=config.third_loss,
               clip_grad=config.clip_grad,
               use_lbfgs=config.use_lbfgs, max_epochs_without_improvement=config.max_epochs_without_improvement,
               relo=config.relo)
model.eval()

mean_NMSE, NMSE_concatenate = config.visualization.visualise_prediction(config.x_p, config.y_p, config.omegas,
                                                                        config.full_known_disp,
                                                                        config.full_known_disp_concatenate,
                                                                        config.eigen_mode, config.max_norm,
                                                                        config.device,
                                                                        image_width=config.n_samp_x,
                                                                        image_height=config.n_samp_y, H=config.H,
                                                                        W=config.W, H_p=config.H_p, W_p=config.W_p,
                                                                        model=model,
                                                                        sample_step=config.sample_step,
                                                                        dist_bound=config.dist_bound,
                                                                        color=config.color)
print('mean_NMSE: ', mean_NMSE)
print('NMSE_concatenate: ', NMSE_concatenate)

config.visualization.visualise_loss(config.third_loss, metric_lam, history_loss, history_lambda)
torch.save(model.state_dict(), '/nas/home/fzese/plateModes/model_weights_no_gov_18nkp.pth')

# Per ricaricare il modello in futuro
# model = PINNet(omegas, num_known_points, num_hidden_layers, hidden_features, ...)
# model.load_state_dict(torch.load('model_weights.pth'))

# Visualizzare i pesi finali
# state_dict = model.state_dict()
# for name, param in state_dict.items():
#    print(f"Layer: {name} | Shape: {param.shape}")
#    print(param)

    #return mean_NMSE, NMSE_concatenate
