import os
from typing import List

import numpy as np
from torch import optim, nn

from Utils.Configuration import Configuration
from Utils.LabeledSeries import LabeledSeries
from Utils.utils import device, get_bandpass_width
from classifiers.inception import InceptionClassifier
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import torch


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
        self.scheduler = self.__getScheduler()
        self.scheduler_exp = self.__getExpScheduler()

        self.lossf = nn.CrossEntropyLoss().to(device)
        self.train_losses = []
        self.val_losses = []

        self.delta = 1e-4

    def fit(self):

        while self.epoch < self.max_epoch:
            start = timer()

            curr_lr = self.__adjust_lr()
            self.__adjust_wd()
            self.epoch += 1

            train_loss, val_loss = self.__train()

            duration = timer() - start
            print(f"train {self.epoch} in {duration}s = {train_loss}")
            print(f"test {self.epoch} in {duration}s = {val_loss}")
            print(f"learning rate: {curr_lr}")

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # self.scheduler_exp.step()
            # self.scheduler.step()

            early_stop_tracebacks = self.__conf.getHP('early_stop_tracebacks')
            if len(self.train_losses) > early_stop_tracebacks:
                def get_window(losses_list: List,
                               width: int) -> List:
                    return losses_list[-1 - width: -1]
                loss_window = get_window(self.train_losses, early_stop_tracebacks)
                fault_tolerant = get_bandpass_width(loss_window)
                print(f"Epoch {self.epoch}: fault_tolerant - {fault_tolerant}")
                if fault_tolerant < self.delta*50:
                    if train_loss > np.mean(loss_window) + self.delta:
                        print(f"Early stop at the epoch: {self.epoch}".center(50, '*'))
                        break
                    else:
                        print(f"It's a pity. not the concave".ljust(40, '*'))

        checkpoint_folder_path = self.__conf.getHP('checkpoint_folder_path')
        checkpoint_filename = '-'.join([
            'FIT',
            str(self.epoch)
        ]) + '.pickle'
        torch.save(self.model.state_dict(), os.path.join(checkpoint_folder_path, checkpoint_filename))

    def __train(self):
        ###################
        # train the model #
        ###################
        local_losses = []
        for batch, truths in self.train_dataloader:
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            loss = self.lossf(predictions, truths)
            loss.backward()
            self.optimizer.step()
            local_losses.append(loss.detach().item())

        train_loss = np.mean(local_losses)

        ######################
        # validate the model #
        ######################
        local_losses = []
        with torch.no_grad():
            for batch, truths in self.val_dataloader:
                predictions = self.model(batch)
                loss = self.lossf(predictions, truths)
                local_losses.append(loss.detach().item())

        val_loss = np.mean(local_losses)

        return train_loss, val_loss

    def __getScheduler(self) -> optim.lr_scheduler:
        # return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.__conf.getHP('factor'),
        #                                             patience=self.__conf.getHP('patience'),
        #                                             verbose=self.__conf.getHP('verbose'))
        # lr_max = self.__conf.getHP('lr_max')
        # lr_min = self.__conf.getHP('lr_min')
        # lambda1 = lambda epoch: lr_max - epoch * (lr_max - lr_min) / self.max_epoch
        # return optim.lr_scheduler.LambdaLR(self.optimizer, lambda1)
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.__conf.getHP('step_size'),
                                               gamma=self.__conf.getHP('gamma'))

    def __getExpScheduler(self) -> optim.lr_scheduler:
        return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

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

            return optim.SGD(self.model.parameters(), lr=initial_lr,
                             momentum=momentum, weight_decay=initial_wd, nesterov=True)

        raise ValueError('cannot obtain optimizer')

    def __adjust_lr(self) -> np.float64:
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

        return new_lr

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
