import argparse
import os
from token import EQUAL

import numpy as np
import torch
from torch import nn, Tensor, tanh

# len_1hot = len(bonn_labels['Z'])
len_1hot = 10
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

data_index = {
    'Air_Compressor': 0,
    '1400Ripples': 1,
    '1080Lines': 2,
    'Blip': 3,
    'Extremely_Loud': 4,
    'Koi_Fish': 5,
    'Chirp': 6,
    'Light_Modulation': 7,
    'Low_Frequency_Burst': 8,
    'Low_Frequency_Lines': 9
}


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-u", "--update_data", type=bool, default=False)
    parser.add_argument("-n", "--num_runs", type=int, default=10)
    parser.add_argument("-k", "--num_kernels", type=int, default=10_000)
    return parser.parse_args()


def check_if_file_exits(file_name):
    return os.path.exists(file_name)


def read_ucr(filename, delimiter=','):
    data = np.loadtxt(filename)
    Y, X = data[:, 0].astype(np.int), data[:, 1:]
    return X, Y


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def read_dataset(root_dir, archive_name, dataset_name):
    datasets_dict = {}

    file_name = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/' + dataset_name
    x_train, y_train = read_ucr(file_name + '_TRAIN')
    x_test, y_test = read_ucr(file_name + '_TEST')
    datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                   y_test.copy())

    return datasets_dict


def z_normalize(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values)
    std = np.std(values)
    epsilon = 1e-20

    return (values - mean) / (std + epsilon)


def read_txt(file_path: str) -> np.ndarray:
    assert os.path.isfile(file_path)

    buffer = []

    with open(file_path, 'r') as fin:
        for line in fin:
            buffer.append(np.float32(line))

    return np.array(buffer)


class Configuration():
    def __init__(self, path: str = '', dump: bool = False, existing: bool = False):
        self.defaults = {
            'activation_conv': 'leakyrelu',
            'activation_linear': 'lecuntanh',

            'dim_en_latent': 256,

            'num_en_channels': 256,

            'size_batch': 16,
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

            'inception_bottleneck_channels': 32,
            #             'inception_kernel_sizes': [1, 10, 20, 40],
            'inception_kernel_sizes': [1, 9, 17, 33],
            'num_inceptionen_blocks': 6,
            'num_inceptionde_blocks': 5,

            'len_1hot': len_1hot,

            'device': device
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


def create_1hot(index: int, length: int) -> np.array:
    """

    :param index: one hot position in the array
    :param length: the length of the array
    :return: 1D array
    """
    temp = np.zeros(length)
    temp[index] = 1
    return temp


def create_1hot_2d(input_array: np.array, length: int) -> np.array:
    """

    :param input_array: 2D array
    :param length: the length of the hot vector
    :return:
    """
    output_array = np.zeros((len(input_array), length))
    print(f"input for create_1hot_2d: {input_array}")
    for i in range(len(input_array)):
        output_array[i] = create_1hot(input_array[i], length)
    return output_array


def one_hot2array(input_array: np.array) -> np.array:
    """

    :param input_array: 1D array
    :return: the index of the one hot in the array
    """
    # np.squeeze(np.where(input_array == 1))
    return np.argmax(input_array)


def one_hot2d2array(input_array: np.array) -> np.array:
    """

    :param input_array: 2D array of the one hot
    :return:
    """
    rows, cols = input_array.shape
    output_array = np.zeros(rows)
    print(f"input for one_hot2d2array: {input_array.shape}")
    for i in range(rows):
        if type(input_array[i]) is np.ndarray:
            # print(input_array[i])
            # print(input_array[i].dtype)
            output_array[i] = one_hot2array(input_array[i])
        else:
            output_array[i] = one_hot2array(np.array(input_array[i]))
    return output_array


def one_hot_3d2array(input_array: np.array) -> np.array:
    """

    :param input_array: 3D the array of the one hot vector
    :return:
    """
    dim1, dim2, dim3 = input_array.shape
    output_array = np.zeros((dim1, dim2))
    print(f"one_hot_3d2array".center(60, '*'))
    print(input_array[0].shape)
    for i in range(dim1):
        output_array[i] = one_hot2d2array(input_array[i])
    return output_array.astype(np.int64)


def merge_vote(one_hot_array: np.array) -> np.array:
    """

    :param one_hot_array: 3d array of the hot vector
    :return:
    """
    list_ensemble = []
    input_array = one_hot_3d2array(one_hot_array)
    rows, cols = input_array.shape
    for row in range(cols):
        d = np.argmax(np.bincount(input_array[:, row].T))
        list_ensemble.append(d)
    return create_1hot_2d(np.array(list_ensemble), one_hot_array.shape[2])