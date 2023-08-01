import os
os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
import torch
import numpy as np
from model import FourierDeepONet
from data import *


class Dataset(dde.data.Data):

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.train_sampler = dde.data.BatchSampler(len(X_train[0]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (self.train_x[0][indices], self.train_x[1][indices],), self.train_y[indices]

    def test(self):
        return self.test_x, self.test_y


def main(dataset, task):
    if dataset == 'fvb':
        X_train, y_train, X_test, y_test = data_fvb_train(task=task)
    elif dataset == 'cva':
        X_train, y_train, X_test, y_test = data_cva_train(task=task)
    elif dataset == 'cfa':
        X_train, y_train, X_test, y_test = data_cfa_train(task=task)
    elif dataset == 'sta':
        X_train, y_train, X_test, y_test = data_sta_train(task=task)
    else:
        raise NotImplementedError(f"dataset name should be 'fvb', 'cva', 'cfa', or 'sta'")
    data = Dataset(X_train, y_train, X_test, y_test)

    net = FourierDeepONet(num_parameter=X_train[1].shape[1], width=64, modes1=20, modes2=20, regularization=["l2", 3e-6])
    model = dde.Model(data, net)

    path = f'./model_{dataset}_{task}'
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    def loss_func_L1(y_true, y_pred):
        return torch.nn.L1Loss()(y_pred, y_true)

    def loss_func_L2(y_true, y_pred):
        return torch.nn.MSELoss()(y_pred, y_true)

    model.compile("adam", lr=1e-3, loss=loss_func_L1, decay=("step", 5000, 0.9),
                  metrics=[lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)),
                           lambda y_true, y_pred: np.sqrt(np.mean(((y_true - y_pred) ** 2)))], )
    checker = dde.callbacks.ModelCheckpoint(f"{path}/model", save_better_only=False, period=10000)
    losshistory, train_state = model.train(iterations=100000, batch_size=32, display_every=100, callbacks=[checker])
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)


if __name__ == "__main__":
    main(dataset='cva', task='loc')
