from token import EQUAL

import numpy as np
from torch import nn, Tensor, tanh

from Utils.utils import device, len_1hot


class LeCunTanh(nn.Module):
    def __init__(self):
        super(LeCunTanh, self).__init__()

        self.adjustedTanh = TanhAdjusted(outer=1.7159, inner=2 / 3, specified=True)

    def forward(self, input: Tensor) -> Tensor:
        return self.adjustedTanh(input)


class TanhAdjusted(nn.Module):
    def __init__(self, outer: float = 1., inner: float = 1., specified=False):
        super(TanhAdjusted, self).__init__()

        self.a = outer
        self.b = inner

        if not specified:
            if EQUAL(self.a, 1.) and not EQUAL(self.b, 1.):
                self.a = 1. / np.tanh(self.b)
            elif not EQUAL(self.a, 1.) and EQUAL(self.b, 1.):
                self.b = np.log((self.a + 1.) / (self.a - 1.)) / 2.

    def forward(self, input: Tensor) -> Tensor:
        return self.a * tanh(self.b * input)


class Configuration():
    def __init__(self, path: str = '', dump: bool = False, existing: bool = False):
        self.defaults = {
            'activation_conv': 'leakyrelu',

            'activation_linear': 'lecuntanh',

            'inception_bottleneck_channels': 32,

            # 'inception_kernel_sizes': [1, 10, 20, 40],
            'inception_kernel_sizes': [1, 9, 17, 33],

            'num_inceptionen_blocks': 6,

            'num_inceptionde_blocks': 5,

            'dim_en_latent': 256,

            'num_en_channels': 256,

            'size_batch': 32,

            'num_epoch': 100,

            'relu_slope': 1e-2,

            'optim_type': 'sgd',

            'momentum': 0.9,

            'lr_mode': 'linear',

            'lr_cons': 1e-3,

            'lr_max': 1e-3,

            'lr_min': 1e-5,

            'lr_everyk': 2,

            'lr_ebase': 0.9,

            'wd_mode': 'fix',

            'wd_cons': 1e-4,

            'wd_max': 1e-4,

            'wd_min': 1e-8,

            'len_1hot': len_1hot,

            'device': device,

            'checkpoint_folder_path': './results/checkpoint',

            'early_stop_tracebacks': 10,

            'patience': 5,

            'verbose': True,

            'factor': 0.5,

            'step_size': 25,

            'gamma': 0.1
        }

        self.settings = {}

    def getHP(self, name: str):
        if name in self.settings:
            return self.settings[name]

        if name in self.defaults:
            return self.defaults[name]

        raise ValueError('hyperparmeter {} doesn\'t exist'.format(name))

    def setHP(self, key: str, value):
        self.settings[key] = value

    # TODO this design (of getActivation in conf) is a little tricky
    def getActivation(self, name: str) -> nn.Module:
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'lecuntanh':
            return LeCunTanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'leakyrelu':
            return nn.LeakyReLU(self.getHP('relu_slope'))

        return nn.Identity()