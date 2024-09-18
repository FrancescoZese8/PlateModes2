from dataSet import KirchhoffDataset
import torch
import torch.nn as nn


class CustomVariable:
    def __init__(self, initial_value, trainable=True, dtype=torch.float32):
        self.data = torch.nn.Parameter(torch.tensor(initial_value, dtype=dtype), requires_grad=trainable)

    def assign(self, new_value):
        self.data.data = torch.tensor(new_value, dtype=self.data.dtype)


class DWALoss(torch.nn.Module):
    def __init__(self, plate, num_tasks):
        super(DWALoss, self).__init__()
        self.plate = plate
        self.num_tasks = num_tasks
        self.prev_losses = [1.0 for _ in range(num_tasks)]  # Inizializza con 1.0
        self.T = 2  # Fattore di temperatura per il calcolo dei pesi dinamici

    def call(self, preds, xy, epoch):
        xy = xy['coords']
        x, y = xy[:, :, 0], xy[:, :, 1]
        preds = preds['model_out']

        losses = self.plate.compute_loss(x, y, preds)
        weighted_losses = {}

        if epoch > 1:
            ratios = [loss.mean().item() / self.prev_losses[i] for i, loss in enumerate(losses.values())]
            weights = [self.num_tasks * torch.exp(torch.tensor(ratio / self.T)) for ratio in ratios]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            for i, (name, loss) in enumerate(losses.items()):
                weighted_losses[name] = weights[i] * loss.mean()
        else:
            for name, loss in losses.items():
                weighted_losses[name] = loss.mean()

        self.prev_losses = [loss.mean().item() for loss in losses.values()]

        return weighted_losses


class MultiTaskLossWrapper(torch.nn.Module):
    def __init__(self, plate, num_tasks, epsilon=1e-6):
        super(MultiTaskLossWrapper, self).__init__()
        self.plate = plate
        self.epsilon = torch.tensor(epsilon)
        self.log_vars = torch.nn.Parameter(torch.zeros(num_tasks))  # w_i^2 sotto log

    def call(self, preds, xy):
        xy = xy['coords']
        x, y = xy[:, :, 0], xy[:, :, 1]
        preds = preds['model_out']

        # Calcola le perdite individuali
        losses = self.plate.compute_loss(x, y, preds)
        weighted_losses = {}

        # Pondera le perdite utilizzando log_vars e epsilon
        for i, (name, loss) in enumerate(losses.items()):
            uncertainty = self.epsilon + torch.exp(self.log_vars[i])  # ϵ² + w_i²
            precision = 1 / (2 * uncertainty)  # 1 / (2(ϵ² + w_i²))
            weighted_loss = precision * loss.mean() + torch.log(uncertainty)  # Li(θ) + log(ϵ² + w_i²)
            weighted_losses[name] = weighted_loss

        # Somma ponderata delle perdite
        return weighted_losses


class KirchhoffLoss(torch.nn.Module):
    def __init__(self, plate: KirchhoffDataset):
        super(KirchhoffLoss, self).__init__()
        self.plate = plate

    def call(self, preds, xy):
        xy = xy['coords']
        x, y = xy[:, :, 0], xy[:, :, 1]
        preds = preds['model_out']
        return self.plate.compute_loss(x, y, preds)


class ReLoBRaLoKirchhoffLoss(KirchhoffLoss):

    def __init__(self, plate: KirchhoffDataset, num_loss, alpha: float = 0.999, temperature: float = 1.,
                 rho: float = 0.9999):
        super().__init__(plate)
        self.plate = plate
        self.num_loss = num_loss
        self.alpha = torch.tensor(alpha)
        self.temperature = temperature
        self.rho = rho
        self.call_count = CustomVariable(0., trainable=False, dtype=torch.float32)
        self.lambdas = [CustomVariable(1., trainable=False) for _ in range(self.num_loss)]
        self.last_losses = [CustomVariable(1., trainable=False) for _ in range(self.num_loss)]
        self.init_losses = [CustomVariable(1., trainable=False) for _ in range(self.num_loss)]

    def call(self, preds, xy):
        xy = xy['coords']
        x, y = xy[:, :, 0], xy[:, :, 1]
        preds = preds['model_out']
        EPS = 1e-7

        losses = {key: torch.mean(loss) for key, loss in self.plate.compute_loss(x, y, preds).items()}

        cond1 = torch.tensor(self.call_count.data.item() == 0, dtype=torch.bool)
        cond2 = torch.tensor(self.call_count.data.item() == 1, dtype=torch.bool)

        alpha = torch.where(cond1, torch.tensor(1.0),
                            torch.where(cond2, torch.tensor(0.0),
                                        self.alpha))
        cond3 = torch.rand(1).item() < self.rho
        rho = torch.where(cond1, torch.tensor(1.0),
                          torch.where(cond2, torch.tensor(1.0),
                                      torch.tensor(cond3, dtype=torch.float32)))

        # Calcola nuove lambdas w.r.t. le losses nella precedente iterazione
        lambdas_hat = [list(losses.values())[i].item() / (self.last_losses[i].data.item() * self.temperature + EPS)
                       for i in range(len(losses))]

        lambdas_hat = torch.tensor(lambdas_hat)
        lambdas_hat = (torch.nn.functional.softmax(lambdas_hat - torch.max(lambdas_hat), dim=-1)
                       * torch.tensor(len(losses), dtype=torch.float32))

        # Calcola nuove lambdas w.r.t. le losses nella prima iterazione
        init_lambdas_hat = [list(losses.values())[i].item() / (self.init_losses[i].data.item() * self.temperature + EPS)
                            for i in range(len(losses))]

        init_lambdas_hat = torch.tensor(init_lambdas_hat)
        init_lambdas_hat = (torch.nn.functional.softmax(init_lambdas_hat - torch.max(init_lambdas_hat), dim=-1)
                            * torch.tensor(len(losses), dtype=torch.float32))
        # Usa rho per decidere se eseguire uno sguardo casuale all'indietro
        new_lambdas = [
            (rho * alpha * self.lambdas[i].data + (1 - rho) * alpha * init_lambdas_hat[i] + (1 - alpha)
             * lambdas_hat[i]) for i in range(len(losses))]
        self.lambdas = [var.detach().requires_grad_(False) for var in new_lambdas]
        # Calcola la loss ponderata
        l = {key: lam * loss for lam, (key, loss) in zip(self.lambdas, losses.items())}
        #  loss = torch.sum(torch.stack(l))
        # Memorizza le losses correnti in self.last_losses per essere accedute nella prossima iterazione
        self.last_losses = [loss.clone().detach() for loss in losses.values()]

        # Nella prima iterazione, memorizza le losses in self.init_losses per essere accedute nelle iterazioni successive
        first_iteration = torch.tensor(self.call_count.data.item() < 1, dtype=torch.float32)

        for i, (var, loss) in enumerate(zip(self.init_losses, losses.values())):
            self.init_losses[i].data = (loss.data * first_iteration + var.data * (1 - first_iteration)).detach()
        self.call_count.data += 1
        # Restituisci un dizionario contenente le losses distinte
        return l


class KirchhoffMetric(nn.Module):
    def __init__(self, plate, third_loss):
        super(KirchhoffMetric, self).__init__()
        self.plate = plate
        self.third_loss = third_loss
        self.L_f_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.L_t_mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.third_loss:
            self.L_o_mean = nn.Parameter(torch.zeros(1), requires_grad=False)

    def update_state(self, xy, y_pred, losses=None, sample_weight=None):
        xy = xy['coords']
        y_pred = y_pred['model_out']
        x, y = xy[:, :, 0], xy[:, :, 1]

        compute_loss_dic = self.plate.compute_loss(x, y, y_pred, eval=True)
        self.L_f_mean.data = torch.mean(compute_loss_dic['L_f'])
        self.L_t_mean.data = torch.mean(compute_loss_dic['L_t'])
        if self.third_loss:
            self.L_o_mean.data = torch.mean(compute_loss_dic['L_o'])

    def reset_state(self):
        self.L_f_mean.data = torch.zeros(1)
        self.L_t_mean.data = torch.zeros(1)
        if self.third_loss:
            self.L_o_mean.data = torch.zeros(1)

    def result(self):
        return {'L_f': self.L_f_mean.item(),
                'L_t': self.L_t_mean.item(),
                'L_o': self.L_o_mean.item() if self.third_loss else None}


class ReLoBRaLoLambdaMetric(nn.Module):
    def __init__(self, loss, third_loss, name='relobralo_lambda_metric'):
        super(ReLoBRaLoLambdaMetric, self).__init__()
        self.loss = loss
        self.third_loss = third_loss
        self.L_f_lambda_mean = CustomVariable(0.0, trainable=False)
        self.L_t_lambda_mean = CustomVariable(0.0, trainable=False)
        if self.third_loss:
            self.L_o_lambda_mean = CustomVariable(0.0, trainable=False)

    def update_state(self, xy, y_pred, sample_weight=None):
        if self.third_loss:
            L_f_lambda, L_t_lambda, L_o_lambda = self.loss.lambdas
            self.L_f_lambda_mean.assign(L_f_lambda.data.data.item())
            self.L_t_lambda_mean.assign(L_t_lambda.data.item())
            self.L_o_lambda_mean.assign(L_o_lambda.data.item())
        else:
            L_f_lambda, L_t_lambda = self.loss.lambdas
            self.L_f_lambda_mean.assign(L_f_lambda.data.data.item())
            self.L_t_lambda_mean.assign(L_t_lambda.data.item())

    def reset_state(self):
        self.L_f_lambda_mean.assign(0.0)
        self.L_t_lambda_mean.assign(0.0)
        if self.third_loss:
            self.L_o_lambda_mean.assign(0.0)

    def result(self):
        return {'L_f': self.L_f_lambda_mean.data.data,
                'L_t': self.L_t_lambda_mean.data.data,
                'L_o': self.L_o_lambda_mean.data.data if self.third_loss else None}
