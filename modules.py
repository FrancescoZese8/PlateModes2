import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
from dataSet import compute_derivatives


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input_net, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        # MODIFICATO: Gestione dei pesi complessi
        weight_real = params['real_weight']
        weight_imag = params['imag_weight']
        weight = torch.complex(weight_real, weight_imag)

        output = input_net.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        if bias is not None:
            output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(5 * input)

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(5 * input)


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
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

    def __init__(self, num_hidden_layers, hidden_features, initial_conditions=True, out_features=1, type='tanh',
                 in_features=2, mode='mlp'):
        super().__init__()
        self.mode = mode
        self.num_hidden_layers = num_hidden_layers
        self.hidden_features = hidden_features
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type,
                           weight_init=None)
        print(self)

    def forward(self, model_input):
        coords = model_input['coords']
        x, y = coords[:, :, 0], coords[:, :, 1]
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        x = x[..., None]
        y = y[..., None]
        x.requires_grad_(True)
        y.requires_grad_(True)
        o = self.net(torch.cat((x, y), dim=-1))
        dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy = compute_derivatives(x, y, o)
        output = torch.cat((o, dudxx, dudyy, dudxxxx, dudyyyy, dudxxyy), dim=-1)
        return {'model_in': coords, 'model_out': output}


########################
# Initialization methods

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'real_weight'):
            fan_in = m.real_weight.size(1)
            fan_out = m.real_weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            _no_grad_trunc_normal_(m.real_weight, mean, std, -2 * std, 2 * std)
            _no_grad_trunc_normal_(m.imag_weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'real_weight'):
            nn.init.kaiming_normal_(m.real_weight, a=0.0, nonlinearity='relu', mode='fan_in')
            nn.init.kaiming_normal_(m.imag_weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'real_weight'):
            num_input = m.real_weight.size(-1)
            nn.init.normal_(m.real_weight, std=1 / math.sqrt(num_input))
            nn.init.normal_(m.imag_weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'real_weight'):
            num_input = m.real_weight.size(-1)
            nn.init.normal_(m.real_weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))
            nn.init.normal_(m.imag_weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'real_weight'):
            nn.init.xavier_normal_(m.real_weight, 1.0)
            nn.init.xavier_normal_(m.imag_weight, 1.0)
            nn.init.zeros_(m.real_bias)
            nn.init.zeros_(m.imag_bias)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'real_weight'):
            num_input = m.real_weight.size(-1)
            m.real_weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
            m.imag_weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'real_weight'):
            num_input = m.real_weight.size(-1)
            m.real_weight.uniform_(-1 / num_input, 1 / num_input)
            m.imag_weight.uniform_(-1 / num_input, 1 / num_input)


def first_layer_silu_init(m):
    with torch.no_grad():
        if hasattr(m, 'real_weight'):
            nn.init.xavier_normal_(m.real_weight, 1.0)
            nn.init.xavier_normal_(m.imag_weight, 1.0)