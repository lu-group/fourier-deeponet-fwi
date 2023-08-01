import os
os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
import torch
import numpy as np
from model import FourierDeepONet
from pytorch_ssim import ssim
from data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(dataset, task, iterations):
    if dataset == 'fvb':
        X_test, y_true = data_fvb_test(task)
    elif dataset == 'cva':
        X_test, y_true = data_cva_test(task)
    elif dataset == 'cfa':
        X_test, y_true = data_cfa_test(task)
    elif dataset == 'sta':
        X_test, y_true = data_sta_test(task)
    else:
        raise NotImplementedError(f"dataset name should be 'fvb', 'cva', 'cfa', or 'sta'")

    length = len(y_true)//100
    net = FourierDeepONet(num_parameter=X_test[1].shape[1], width=64, modes1=20, modes2=20)

    y_pred = []
    save_path = f'./model_{dataset}_{task}/model-{iterations}.pt'
    checkpoint = torch.load(save_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    for i in range(length):
        inputs = tuple(map(lambda x: torch.as_tensor(x).to(device).requires_grad_(),
                           (X_test[0][100*i:100*(i+1)], X_test[1][100*i:100*(i+1)])))
        outputs = net(inputs)
        dde.gradients.clear()
        outputs = dde.utils.to_numpy(outputs)
        print(np.mean(np.abs(outputs - y_true[100*i:100*(i+1)])))
        y_pred.append(outputs)
        torch.cuda.empty_cache()
    y_pred = np.concatenate(y_pred, axis=0)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(((y_true-y_pred)**2).mean())
    ssim_value = ssim(torch.from_numpy(y_true) / 2 + 0.5,
                      torch.from_numpy(y_pred) / 2 + 0.5).numpy().tolist()
    l2 = dde.metrics.mean_l2_relative_error(y_true.reshape(-1, 4900), y_pred.reshape(-1, 4900))
    print(f'mae: {mae}, rmse {rmse}, ssim_value: {ssim_value}, l2: {l2}')
    y_pred = minmax_denormalize(y_pred, 1500, 4500)
    np.save(f"y_pred_{dataset}_{task}_{iterations}.npy", y_pred)


if __name__ == "__main__":
    # dataset name should be 'fvb', 'cva', 'cfa', or 'sta'
    # task name should be 'loc', 'f', or 'loc_f'
    main(dataset='fvb', task='loc', iterations=100000)
