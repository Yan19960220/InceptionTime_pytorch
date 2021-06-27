# -*- coding: utf-8 -*-
# @Time    : 4/22/21 11:09 AM
# @Author  : Yan
# @Site    : 
# @File    : main.py
# @Software: PyCharm


# %%
# import ray
# import ray.tune as tune

from typing import List, Tuple
from scipy import stats

from Utils.configuration import Configuration
from Utils.fitter import Fitter
from Utils.labeledSeries import LabeledSeries
from Utils.utils import create_parser, merge_vote, samples2tensor
import numpy as np
import pandas as pd

# import mne
import torch
from torch import Tensor
from torch.utils.data import DataLoader

dataset_range = \
    (
        1024,
        2048,
        3072,
        4096,
        0
    )


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


def precision(predictions_1hot: List,
              truths: Tensor) -> float:
    assert len(predictions_1hot) == len(truths)
    predictions = np.argmax(predictions_1hot, axis=-1)
    return np.sum(predictions == truths) / len(predictions)


def load_split_data(dataset_name: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    series_dataset = {}
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
    # len_series = len(series_dataset[0][0])
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


def get_prediction_1hot(data: Tensor,
                        labels: Tensor) -> List:
    predictions_1hot = []
    with torch.no_grad():
        for batch, truths in DataLoader(LabeledSeries(data, labels), batch_size=256):
            predictions_1hot += fitter.model(batch).detach().cpu().tolist()
    return predictions_1hot


if __name__ == '__main__':

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
            checkpoint_folderpath = arguments.output_path
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