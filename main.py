# -*- coding: utf-8 -*-
# @Time    : 4/22/21 11:09 AM
# @Author  : Yan
# @Site    : 
# @File    : main.py
# @Software: PyCharm


# %%
# import ray
# import ray.tune as tune

import shutil
import os
from token import EQUAL
from typing import List, Tuple
from timeit import default_timer as timer
from itertools import chain
from scipy import stats

from inception import InceptionClassifier
from utils import create_parser, Configuration, merge_vote
import numpy as np
import pandas as pd

# import mne
import torch
from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class LabeledSeries(Dataset):
    def __init__(self, series, labels):
        super(LabeledSeries, self).__init__()

        assert len(series) == len(labels)

        self.series = series
        self.labels = labels

    def __len__(self):
        return self.series.shape[0]

    def __getitem__(self, indices):
        return self.series[indices], self.labels[indices]


class Fitter:
    def __init__(self, conf: Configuration, train_set, val_set):
        self.__conf = conf
        self.__batch_size = conf.getHP('size_batch')

        self.train_dataloader = DataLoader(LabeledSeries(train_set[0], train_set[1]), batch_size=self.__batch_size,
                                           shuffle=True)
        self.val_dataloader = DataLoader(LabeledSeries(val_set[0], val_set[1]), batch_size=self.__batch_size,
                                         shuffle=True)

        self.epoch = 0
        self.max_epoch = conf.getHP('num_epoch')

        self.model = InceptionClassifier(conf).to(device)
        self.optimizer = self.__getOptimizer()
        self.lossf = nn.CrossEntropyLoss().to(device)

        self.train_losses = []
        self.val_losses = []

    def fit(self):
        while self.epoch < self.max_epoch:
            start = timer()

            self.__adjust_lr()
            self.__adjust_wd()
            self.epoch += 1

            train_loss, val_loss = self.__train()

            duration = timer() - start
            print('train {:d} in {:.3f}s = {:.4f}'.format(self.epoch, duration, train_loss))
            print('val {:d} in {:.3f}s = {:.4f}'.format(self.epoch, duration, val_loss))

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if len(self.train_losses) > early_stop_tracebacks \
                    and train_loss > np.mean(self.train_losses[-1 - early_stop_tracebacks: -1]) + 1e-4:
                break

        checkpoint_filename = '-'.join([
            'FIT',
            str(self.epoch)
        ]) + '.pickle'

        torch.save(self.model.state_dict(), os.path.join(checkpoint_folderpath, checkpoint_filename))

    def __train(self):
        local_losses = []
        for batch, truths in self.train_dataloader:
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            loss = self.lossf(predictions, truths)
            loss.backward()
            self.optimizer.step()
            local_losses.append(loss.detach().item())

        train_loss = np.mean(local_losses)

        local_losses = []
        with torch.no_grad():
            for batch, truths in self.val_dataloader:
                predictions = self.model(batch)
                loss = self.lossf(predictions, truths)
                local_losses.append(loss.detach().item())

        val_loss = np.mean(local_losses)

        return train_loss, val_loss

    def __getOptimizer(self) -> optim.Optimizer:
        if self.__conf.getHP('optim_type') == 'sgd':
            if self.__conf.getHP('lr_mode') == 'fix':
                initial_lr = self.__conf.getHP('lr_cons')
            else:
                initial_lr = self.__conf.getHP('lr_max')

            if self.__conf.getHP('wd_mode') == 'fix':
                initial_wd = self.__conf.getHP('wd_cons')
            else:
                initial_wd = self.__conf.getHP('wd_min')

            momentum = self.__conf.getHP('momentum')

            return optim.SGD(self.model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=initial_wd)

        raise ValueError('cannot obtain optimizer')

    def __adjust_lr(self) -> None:
        # should be based on self.epoch and hyperparameters ONLY for easily resumming

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
            break

        new_lr = current_lr

        if self.__conf.getHP('lr_mode') == 'linear':
            lr_max = self.__conf.getHP('lr_max')
            lr_min = self.__conf.getHP('lr_min')

            new_lr = lr_max - self.epoch * (lr_max - lr_min) / self.max_epoch
        elif self.__conf.getHP('lr_mode') == 'exponentiallyhalve':
            lr_max = self.__conf.getHP('lr_max')
            lr_min = self.__conf.getHP('lr_min')

            for i in range(1, 11):
                if (self.max_epoch - self.epoch) * (2 ** i) == self.max_epoch:
                    new_lr = lr_max / (10 ** i)
                    break

            if new_lr < lr_min:
                new_lr = lr_min
        elif self.__conf.getHP('lr_mode') == 'exponentially':
            lr_max = self.__conf.getHP('lr_max')
            lr_min = self.__conf.getHP('lr_min')
            lr_k = self.__conf.getHP('lr_everyk')
            lr_ebase = self.__conf.getHP('lr_ebase')

            lr_e = int(np.floor(self.epoch / lr_k))
            new_lr = lr_max * (lr_ebase ** lr_e)

            if new_lr < lr_min:
                new_lr = lr_min
        elif self.__conf.getHP('lr_mode') == 'plateauhalve':
            raise ValueError('plateauhalve is not yet supported')

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def __adjust_wd(self):
        # should be based on self.epoch and hyperparameters ONLY for easily resumming

        for param_group in self.optimizer.param_groups:
            current_wd = param_group['weight_decay']
            break

        new_wd = current_wd

        if self.__conf.getHP('wd_mode') == 'linear':
            wd_max = self.__conf.getHP('wd_max')
            wd_min = self.__conf.getHP('wd_min')

            new_wd = wd_min + self.epoch * (wd_max - wd_min) / self.max_epoch

        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = new_wd

# debug with 'cpu' to show verbose messages
# device = 'cpu'


# load dataset
dataset_range = \
    (
        1024,
        2048,
        3072,
        4096,
        0
        # 100,
        # 200,
        # 300,
        # 400,
        # 500
    )

# len_1hot = len(bonn_labels['Z'])
len_1hot = 3

# %%


def composition_list(train_series):
    empty_list = []
    for i in train_series:
        for j in i:
            empty_list += [[j]]

    return np.array(empty_list)


def flatten_list(labels):
    empty_list = []
    for i in labels:
        if isinstance(i, int):
            empty_list += [i]
        else:
            empty_list += i
    return np.array(empty_list)


def random_sample(series, labels, current_series, current_label, sample_length, max_overlapping, range_value, offset):
    counter = 0
    series.append([])
    labels.append([])
    for i in range(offset, offset + range_value):
        length = len(current_series[i])
        start_pos = 0
        end_pos = start_pos + sample_length

        while end_pos <= length:
            series[current_label].append(current_series[i][start_pos: end_pos])
            counter += 1

            start_pos += np.random.randint(sample_length - max_overlapping, sample_length)
            end_pos = start_pos + sample_length
    labels[current_label].extend([current_label] * counter)
    return series, labels


def samples2tensor(series: List, labels: List) -> Tuple[Tensor, Tensor]:
    return torch.FloatTensor(composition_list(series)).to(device), \
           torch.from_numpy(flatten_list(labels)).to(device)


# evaluate
def precision(predictions_1hot, truths):
    print(len(predictions_1hot))
    print(len(truths))
    assert len(predictions_1hot) == len(truths)
    predictions = np.argmax(predictions_1hot, axis=-1)
    return np.sum(predictions == truths) / len(predictions)


def load_split_data(dataset_name):
    series_dataset = {}
    # series_dataset[dataset_name] = {}
    print(f"{dataset_name}".center(80, "-"))
    print(f"Loading data".ljust(80 - 5, "."), end="", flush=True)
    data = np.genfromtxt(f"{arguments.input_path}/0_{dataset_name}.csv", delimiter=',')
    data = np.delete(data, 0, axis=0)
    for i in range(10):
        temp = stats.zscore(data[data[:, 0] == i, :])[:, 1:]
        series_dataset[i] = temp.tolist()
    print("Done.")
    split = np.array([0.6, 0.2, 0.2])
    train_series, train_labels = [], []
    val_series, val_labels = [], []
    test_series, test_labels = [], []
    len_series = len(series_dataset[0][0])
    for i in range(10):
        current_label = i
        current_series = series_dataset[i]
        num_current_series = len(series_dataset[i])

        split_num = (np.floor(split * num_current_series)).astype(int)

        offset = 0
        train_series.append(current_series[offset: offset + split_num[0]])
        train_labels.append([current_label] * split_num[0])

        offset += split_num[0]
        val_series.append(current_series[offset: offset + split_num[1]])
        val_labels.append([current_label] * split_num[1])

        offset += split_num[1]
        test_series.append(current_series[offset: offset + split_num[2]])
        test_labels.append([current_label] * split_num[2])
    train_series, train_labels = samples2tensor(train_series, train_labels)
    val_series, val_labels = samples2tensor(val_series, val_labels)
    test_series, test_labels = samples2tensor(test_series, test_labels)
    return test_labels, test_series, train_labels, train_series, val_labels, val_series


def ensemble_initialize_1hot(data: List, label: Tensor,
                             curr_data: List, curr_label: Tensor) -> Tuple[List, Tensor]:
    if data:
        if label.equal(curr_label):
            pass
        else:
            print(f"error.".center(90, '*'))
            exit()
    else:
        label = curr_label
    data.append(curr_data)
    return data, label


def get_prediction_1hot(data: Tensor, labels: Tensor) -> List:
    predictions_1hot = []
    with torch.no_grad():
        for batch, truths in DataLoader(LabeledSeries(data, labels), batch_size=256):
            predictions_1hot += fitter.model(batch).detach().cpu().tolist()
    return predictions_1hot


if __name__ == '__main__':
    # ray.init()

    arguments = create_parser()

    results_dataset = pd.DataFrame(index=dataset_range,
                                   columns=["accuracy_train",
                                            "accuracy_valide",
                                            "accuracy_test"],
                                   data=0)
    results_dataset.index.name = "dataset"

    train1hot = []
    test1hot = []
    for dataset_name in dataset_range:

        if dataset_name != 0:

            test_labels, test_series, train_labels, train_series, val_labels, val_series = load_split_data(dataset_name)

            # train InceptionTime
            checkpoint_folderpath = './results'
            early_stop_tracebacks = 10

            conf = Configuration()
            fitter = Fitter(conf, (train_series, train_labels), (val_series, val_labels))
            fitter.fit()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            train_predictions_1hot = get_prediction_1hot(train_series, train_labels)

            test_predictions_1hot = get_prediction_1hot(val_series, val_labels)

            test_predictions_1hot += get_prediction_1hot(test_series, test_labels)

            trainlabel1hot = train_labels
            train1hot, trainlabel1hot = ensemble_initialize_1hot(train1hot, trainlabel1hot, train_predictions_1hot, train_labels)

            print(precision(train_predictions_1hot, train_labels.cpu().numpy()))
            results_dataset.loc[dataset_name, "accuracy_train"] = precision(train_predictions_1hot, train_labels.cpu().numpy())

            testlabel1hot = torch.cat((val_labels, test_labels), 0)
            test1hot, testlabel1hot = ensemble_initialize_1hot(test1hot, testlabel1hot, test_predictions_1hot, torch.cat((val_labels, test_labels), 0))

            print(precision(test_predictions_1hot, np.array(val_labels.cpu().tolist() + test_labels.cpu().tolist())))
            results_dataset.loc[dataset_name, "accuracy_test"] = precision(test_predictions_1hot, np.array(val_labels.cpu().tolist() + test_labels.cpu().tolist()))
        else:
            final_1hot = merge_vote(np.array(train1hot))
            results_dataset.loc[dataset_name, "accuracy_train"] = precision(final_1hot, trainlabel1hot.cpu().numpy())

            final_test_1hot = merge_vote(np.array(test1hot))
            results_dataset.loc[dataset_name, "accuracy_test"] = precision(final_test_1hot, testlabel1hot.cpu().numpy())

    print(f"FINISHED".center(80, "="))
    results_dataset.to_csv(f"{arguments.output_path}/results_dataset.csv")

        # # split
        # sampling_rate = int(173.61)
        #
        # # subsequence_length = int(sampling_rate)
        # subsequence_length = int(sampling_rate * 2)
        # # subsequence_length = int(sampling_rate * 4)
        # print(f"subsequence_length: - {subsequence_length}")
        # max_overlapping_length = int(subsequence_length / 2)
        #
        # train_series, train_labels = [], []
        # val_series, val_labels = [], []
        # test_series, test_labels = [], []
        #
        # split = np.array([0.6, 0.2, 0.2])
        #
        # for class_name in range(10):
        #     current_label = class_name
        #     current_series = series_dataset[class_name]
        #     split_num = (np.floor(split*len(current_series))).astype(int)
        #     # print(f"split_num[0]: - {split_num[0]}")
        #     offset = 0
        #     train_series, train_labels = random_sample(train_series, train_labels, current_series, current_label, subsequence_length,
        #                                            max_overlapping_length, split_num[0], offset)
        #
        #     offset += split_num[0]
        #     val_series, val_labels = random_sample(val_series, val_labels, current_series, current_label, subsequence_length,
        #                                            max_overlapping_length, split_num[1], offset)
        #
        #     offset += split_num[1]
        #     test_series, test_labels = random_sample(test_series, test_labels, current_series, current_label, subsequence_length,
        #                                            max_overlapping_length, split_num[2], offset)
        #
        # train_series, train_labels = samples2tensor(train_series, train_labels)
        # val_series, val_labels = samples2tensor(val_series, val_labels)
        # test_series, test_labels = samples2tensor(test_series, test_labels)
        #
        # ## train InceptionTime
        # conf = Configuration()
        # conf.setHP('inception_kernel_sizes', [1, 5, 9, 17])
        #
        # fitter = Fitter(conf, (train_series, train_labels), (val_series, val_labels))
        # fitter.fit()
        #
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #
        # # evaluate
        # train_predictions_1hot = []
        #
        # with torch.no_grad():
        #     for batch, truths in DataLoader(LabeledSeries(train_series, train_labels), batch_size=256):
        #         train_predictions_1hot += fitter.model(batch).detach().cpu().tolist()
        #
        # print(precision(train_predictions_1hot, train_labels.cpu().numpy()))
        #
        # # %%
        #
        # test_predictions_1hot = []
        # with torch.no_grad():
        #     for batch, truths in DataLoader(LabeledSeries(val_series, val_labels), batch_size=256):
        #         test_predictions_1hot += fitter.model(batch).detach().cpu().tolist()
        #
        #     for batch, truths in DataLoader(LabeledSeries(test_series, test_labels), batch_size=256):
        #         test_predictions_1hot += fitter.model(batch).detach().cpu().tolist()
        #
        # print(precision(test_predictions_1hot, np.array(val_labels.cpu().tolist() + test_labels.cpu().tolist())))