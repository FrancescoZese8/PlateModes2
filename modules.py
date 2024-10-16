import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
from dataSet import compute_derivatives
import matplotlib.pyplot as plt

omega_zero = 5


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input_net, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        # print('bias: ', bias.shape)
        weight = params['weight']
        # print('weight: ', weight.shape)
        # print('input: ', input_net.shape)
        output = input_net.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        # print('output: ', output.shape)
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(omega_zero * input)


class Rowdy(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_1 = nn.Parameter(torch.tensor(0.1))
        self.alpha_2 = nn.Parameter(torch.tensor(0.1))
        self.alpha_3 = nn.Parameter(torch.tensor(0.1))
        self.alpha_4 = nn.Parameter(torch.tensor(0.1))
        self.alpha_5 = nn.Parameter(torch.tensor(0.1))
        self.cc = 0

    def forward(self, input):
        self.cc += 1
        output = (torch.sin(omega_zero * input) +
                  self.alpha_1 * torch.sin(2 * omega_zero * input) +
                  self.alpha_2 * torch.sin(3 * omega_zero * input) +
                  self.alpha_3 * torch.sin(4 * omega_zero * input) +
                  self.alpha_4 * torch.sin(5 * omega_zero * input) +
                  self.alpha_5 * torch.sin(6 * omega_zero * input))

        if self.cc % 1000 == 0:
            print(f"Iteration {self.cc}: alpha_1 = {self.alpha_1.item()}, "
                  f"alpha_2 = {self.alpha_2.item()}, "
                  f"alpha_3 = {self.alpha_3.item()}, "
                  f"alpha_4 = {self.alpha_4.item()}, "
                  f"alpha_5 = {self.alpha_5.item()}")
            '''input_np = input.cpu().detach().numpy()
            output_np = output.cpu().detach().numpy()

            plt.figure()
            plt.plot(input_np, output_np)
            plt.title('Activation Function Plot')
            plt.xlabel('Input')
            plt.ylabel('Output')
            plt.grid(True)
            plt.show()'''
        return output


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=True, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Rowdy(), first_layer_sine_init, sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'silu': (nn.SiLU(), init_weights_xavier, None),  # first_layer_silu_init
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None),
                         'mish': (nn.Mish(), init_weights_xavier, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        '''self.net.append(MetaSequential(
            FourierLayer(in_features, hidden_features, mapping_size=16, scale=0.5), nl
        ))'''
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features, bias=True), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features, bias=True), nl  # nn.BatchNorm1d(hidden_features),
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features, bias=True)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations


class PINNet(nn.Module):
    '''Architecture used by Raissi et al. 2019.'''

    def __init__(self, omegas, num_known_points, num_hidden_layers, hidden_features, initial_conditions=True,
                 out_features=1, type='tanh',
                 in_features=2, mode='mlp'):
        super().__init__()
        self.omegas = omegas
        self.num_known_points = num_known_points
        self.mode = mode
        self.num_hidden_layers = num_hidden_layers
        self.hidden_features = hidden_features
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type,
                           weight_init=None)
        print(self)

    def forward(self, model_input, training=True):
        # Enables us to compute gradients w.r.t. input
        coords = model_input['coords']
        x, y = coords[:, :, 0], coords[:, :, 1]
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        x = x[..., None]
        y = y[..., None]
        x.requires_grad_(True)
        y.requires_grad_(True)
        o = self.net(torch.cat((x, y), dim=-1))
        if training:
            o[self.num_known_points:, :] = o[self.num_known_points:, :] / self.omegas ** 2  # TODO
        dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy = compute_derivatives(x, y, o)
        output = torch.cat((o, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy), dim=-1)
        return {'model_in': coords, 'model_out': output}


########################
# Initialization methods

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight, 1.0)
            nn.init.zeros_(m.bias)


# Plotting helper function
def plot_weights(weights, title):
    plt.hist(weights.cpu().numpy().flatten(), bins=30, alpha=0.75, color='b', edgecolor='black')
    plt.title(f'Weight Distribution: {title}')
    plt.xlabel('Weight values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def sine_init(m, omega_zero=5):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # Initialize weights based on the sine method
            m.weight.uniform_(-np.sqrt(6 / num_input) / omega_zero, np.sqrt(6 / num_input) / omega_zero)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # Initialize weights for the first layer
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def first_layer_silu_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight, 1.0)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def he_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))  # Kaiming He uniform initialization
        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(m.bias, -bound, bound)


def unique_uniform_init(m, min_value=-0.0001, max_value=0.0001):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            # Creare valori uniformi unici per il numero di pesi
            num_weights = m.weight.numel()  # Numero totale di pesi
            values = torch.linspace(min_value, max_value, num_weights)
            m.weight.copy_(values.view_as(m.weight))  # Copia i valori nei pesi

            # Opzionalmente, inizializza i bias a zero o un altro valore
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.fill_(0)


class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, mapping_size, scale):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)
        self.linear = nn.Linear(2 * mapping_size, out_features)

    def forward(self, x):
        # Fourier feature mapping
        # print('x: ', x.shape)
        # print('B: ', self.B.shape)
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        # print('x_proj: ', x_proj)
        x_fourier = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        # print('x_fourier: ', x_fourier.shape)
        return self.linear(x_fourier)


class SinusoidalLayer(nn.Module):
    def __init__(self, in_features, out_features, mapping_size, scale, sigma):
        super(SinusoidalLayer, self).__init__()

        # Trainable weight matrix W1
        self.W1 = nn.Parameter(torch.randn(in_features, mapping_size) * sigma)  # initialize from N(0, sigma^2)

        # Trainable bias vector b1
        self.b1 = nn.Parameter(torch.zeros(mapping_size))  # initialize bias to zero

        # Linear layer to be applied after sinusoidal mapping
        self.linear = nn.Linear(2 * mapping_size, out_features)

        self.scale = scale

    def forward(self, x):
        # Sinusoidal feature mapping: sin(2π(W1 * x + b1))
        x_proj = 2 * torch.pi * (torch.matmul(x, self.W1) + self.b1)

        # Apply sin and cos for the projection
        x_sin = torch.sin(x_proj)
        x_cos = torch.cos(x_proj)

        # Concatenate sin and cos results
        x_fourier = torch.cat([x_sin, x_cos], dim=-1)

        # Apply the linear layer on the mapped inputs
        return self.linear(x_fourier)
