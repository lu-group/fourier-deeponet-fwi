import numpy as np


def log_transform(data, k=1, c=0):
    return (np.log1p(np.abs(k * data) + c)) * np.sign(data)


def minmax_normalize(vid, vmin, vmax, scale=2):
    vid -= vmin
    vid /= (vmax - vmin)
    return (vid - 0.5) * 2 if scale == 2 else vid


def minmax_denormalize(vid, vmin, vmax, scale=2):
    if scale == 2:
        vid = vid / 2 + 0.5
    return vid * (vmax - vmin) + vmin


def data_fvb_train(task):
    num_dataset = 48
    y_train = []
    for i in range(1, num_dataset + 1, 1):
        y_train.append(np.load(f"../data/fvb/velocity/model{i}.npy"))
    y_train = np.concatenate(y_train).astype(np.float32)
    y_train = minmax_normalize(y_train, 1500, 4500)

    y_test = np.load(f"../data/fvb/velocity/model60.npy")[:50].astype(np.float32)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_train_branch = []
    for i in range(1, num_dataset + 1, 1):
        X_train_branch.append(np.load(f"../data/fvb/seismic/{task}/data{i}.npy"))
    X_train_branch = np.concatenate(X_train_branch).astype(np.float32)
    X_train_branch = log_transform(X_train_branch)
    X_train_branch = minmax_normalize(X_train_branch, log_transform(-30), log_transform(60))

    X_test_branch = np.load(f"../data/fvb/seismic/{task}/data60.npy")[:50].astype(np.float32)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_train_trunk = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk.append(np.load(f"../data/fvb/{task}/{task}{i}.npy"))
        X_train_trunk = np.concatenate(X_train_trunk).astype(np.float32)
        X_train_trunk = log_transform(X_train_trunk)
        if task == 'loc':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(5), log_transform(25))
        X_test_trunk = np.load(f"../data/fvb/{task}/{task}60.npy")[:50].astype(np.float32)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_train_trunk1 = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk1.append(np.load(f"../data/fvb/loc/loc{i}.npy"))
        X_train_trunk1 = np.concatenate(X_train_trunk1).astype(np.float32)
        X_train_trunk1 = log_transform(X_train_trunk1)
        X_train_trunk1 = minmax_normalize(X_train_trunk1, log_transform(0), log_transform(690))

        X_train_trunk2 = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk2.append(np.load(f"../data/fvb/f/f{i}.npy"))
        X_train_trunk2 = np.concatenate(X_train_trunk2).astype(np.float32)
        X_train_trunk2 = log_transform(X_train_trunk2)
        X_train_trunk2 = minmax_normalize(X_train_trunk2, log_transform(5), log_transform(25))

        X_train_trunk = np.concatenate([X_train_trunk1, X_train_trunk2], axis=1)

        X_test_trunk1 = np.load(f"../data/fvb/loc/loc60.npy")[:50].astype(np.float32)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = np.load(f"../data/fvb/f/f60.npy")[:50].astype(np.float32)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_train = (X_train_branch, X_train_trunk)
    X_test = (X_test_branch, X_test_trunk)

    return X_train, y_train, X_test, y_test


def data_cva_train(task):
    num_dataset = 48
    y_train = []
    for i in range(1, num_dataset + 1, 1):
        y_train.append(np.load(f"../data/cva/velocity/model{i}.npy"))
    y_train = np.concatenate(y_train).astype(np.float32)
    y_train = minmax_normalize(y_train, 1500, 4500)

    y_test = np.load(f"../data/cva/velocity/model60.npy")[:50].astype(np.float32)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_train_branch = []
    for i in range(1, num_dataset + 1, 1):
        X_train_branch.append(np.load(f"../data/cva/seismic/{task}/data{i}.npy"))
    X_train_branch = np.concatenate(X_train_branch).astype(np.float32)
    X_train_branch = log_transform(X_train_branch)
    X_train_branch = minmax_normalize(X_train_branch, log_transform(-30), log_transform(60))

    X_test_branch = np.load(f"../data/cva/seismic/{task}/data60.npy")[:50].astype(np.float32)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_train_trunk = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk.append(np.load(f"../data/cva/{task}/{task}{i}.npy"))
        X_train_trunk = np.concatenate(X_train_trunk).astype(np.float32)
        X_train_trunk = log_transform(X_train_trunk)
        if task == 'loc':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(5), log_transform(25))
        X_test_trunk = np.load(f"../data/cva/{task}/{task}60.npy")[:50].astype(np.float32)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))
        
    elif task == 'loc_f':
        X_train_trunk1 = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk1.append(np.load(f"../data/cva/loc/loc{i}.npy"))
        X_train_trunk1 = np.concatenate(X_train_trunk1).astype(np.float32)
        X_train_trunk1 = log_transform(X_train_trunk1)
        X_train_trunk1 = minmax_normalize(X_train_trunk1, log_transform(0), log_transform(690))
        
        X_train_trunk2 = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk2.append(np.load(f"../data/cva/f/f{i}.npy"))
        X_train_trunk2 = np.concatenate(X_train_trunk2).astype(np.float32)
        X_train_trunk2 = log_transform(X_train_trunk2)
        X_train_trunk2 = minmax_normalize(X_train_trunk2, log_transform(5), log_transform(25))

        X_train_trunk = np.concatenate([X_train_trunk1, X_train_trunk2], axis=1)

        X_test_trunk1 = np.load(f"../data/cva/loc/loc60.npy")[:50].astype(np.float32)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))
        
        X_test_trunk2 = np.load(f"../data/cva/f/f60.npy")[:50].astype(np.float32)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_train = (X_train_branch, X_train_trunk)
    X_test = (X_test_branch, X_test_trunk)

    return X_train, y_train, X_test, y_test


def data_cfa_train(task):
    num_dataset = 96
    y_train = []
    for i in [2, 3, 4]:
        for j in range(num_dataset//3):
            y_train.append(np.load(f"../data/cfa/velocity/vel{i}_1_{j}.npy"))
    y_train = np.concatenate(y_train).astype(np.float32)
    y_train = minmax_normalize(y_train, 1500, 4500)

    y_test = np.load(f"../data/cfa/velocity/vel3_1_35.npy")[:50].astype(np.float32)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_train_branch = []
    for i in [2, 3, 4]:
        for j in range(num_dataset//3):
            X_train_branch.append(np.load(f"../data/cfa/seismic/{task}/seis{i}_1_{j}.npy"))
    X_train_branch = np.concatenate(X_train_branch).astype(np.float32)
    X_train_branch = log_transform(X_train_branch)
    X_train_branch = minmax_normalize(X_train_branch, log_transform(-30), log_transform(60))

    X_test_branch = np.load(f"../data/cfa/seismic/{task}/seis3_1_35.npy")[:50].astype(np.float32)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_train_trunk = []
        for i in [2, 3, 4]:
            for j in range(num_dataset // 3):
                X_train_trunk.append(np.load(f"../data/cfa/{task}/{task}{i}_1_{j}.npy"))
        X_train_trunk = np.concatenate(X_train_trunk).astype(np.float32)
        X_train_trunk = log_transform(X_train_trunk)
        if task == 'loc':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(5), log_transform(25))
        X_test_trunk = np.load(f"../data/cfa/{task}/{task}3_1_35.npy")[:50].astype(np.float32)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_train_trunk1 = []
        for i in [2, 3, 4]:
            for j in range(num_dataset // 3):
                X_train_trunk1.append(np.load(f"../data/cfa/loc/loc{i}_1_{j}.npy"))
        X_train_trunk1 = np.concatenate(X_train_trunk1).astype(np.float32)
        X_train_trunk1 = log_transform(X_train_trunk1)
        X_train_trunk1 = minmax_normalize(X_train_trunk1, log_transform(0), log_transform(690))

        X_train_trunk2 = []
        for i in [2, 3, 4]:
            for j in range(num_dataset // 3):
                X_train_trunk2.append(np.load(f"../data/cfa/f/f{i}_1_{j}.npy"))
        X_train_trunk2 = np.concatenate(X_train_trunk2).astype(np.float32)
        X_train_trunk2 = log_transform(X_train_trunk2)
        X_train_trunk2 = minmax_normalize(X_train_trunk2, log_transform(5), log_transform(25))

        X_train_trunk = np.concatenate([X_train_trunk1, X_train_trunk2], axis=1)

        X_test_trunk1 = np.load(f"../data/cfa/loc/loc3_1_35.npy")[:50].astype(np.float32)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = np.load(f"../data/cfa/f/f3_1_35.npy")[:50].astype(np.float32)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_train = (X_train_branch, X_train_trunk)
    X_test = (X_test_branch, X_test_trunk)

    return X_train, y_train, X_test, y_test


def data_sta_train(task):
    num_dataset = 120
    y_train = []
    for i in range(1, num_dataset + 1, 1):
        y_train.append(np.load(f"../data/sta/velocity/model{i}.npy"))
    y_train = np.concatenate(y_train).astype(np.float32)
    y_train = minmax_normalize(y_train, 1500, 4500)

    y_test = np.load(f"../data/sta/velocity/model134.npy")[:50].astype(np.float32)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_train_branch = []
    for i in range(1, num_dataset + 1, 1):
        X_train_branch.append(np.load(f"../data/sta/seismic/{task}/data{i}.npy"))
    X_train_branch = np.concatenate(X_train_branch).astype(np.float32)
    X_train_branch = log_transform(X_train_branch)
    X_train_branch = minmax_normalize(X_train_branch, log_transform(-30), log_transform(60))

    X_test_branch = np.load(f"../data/sta/seismic/{task}/data134.npy")[:50].astype(np.float32)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_train_trunk = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk.append(np.load(f"../data/sta/{task}/{task}{i}.npy"))
        X_train_trunk = np.concatenate(X_train_trunk).astype(np.float32)
        X_train_trunk = log_transform(X_train_trunk)
        if task == 'loc':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_train_trunk = minmax_normalize(X_train_trunk, log_transform(5), log_transform(25))
        X_test_trunk = np.load(f"../data/sta/{task}/{task}134.npy")[:50].astype(np.float32)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_train_trunk1 = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk1.append(np.load(f"../data/sta/loc/loc{i}.npy"))
        X_train_trunk1 = np.concatenate(X_train_trunk1).astype(np.float32)
        X_train_trunk1 = log_transform(X_train_trunk1)
        X_train_trunk1 = minmax_normalize(X_train_trunk1, log_transform(0), log_transform(690))

        X_train_trunk2 = []
        for i in range(1, num_dataset + 1, 1):
            X_train_trunk2.append(np.load(f"../data/sta/f/f{i}.npy"))
        X_train_trunk2 = np.concatenate(X_train_trunk2).astype(np.float32)
        X_train_trunk2 = log_transform(X_train_trunk2)
        X_train_trunk2 = minmax_normalize(X_train_trunk2, log_transform(5), log_transform(25))

        X_train_trunk = np.concatenate([X_train_trunk1, X_train_trunk2], axis=1)

        X_test_trunk1 = np.load(f"../data/sta/loc/loc134.npy")[:50].astype(np.float32)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = np.load(f"../data/sta/f/f134.npy")[:50].astype(np.float32)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_train = (X_train_branch, X_train_trunk)
    X_test = (X_test_branch, X_test_trunk)

    return X_train, y_train, X_test, y_test


def data_fvb_test(task):
    num_dataset = 12
    y_test = []
    for i in range(49, num_dataset + 49, 1):
        y_test.append(np.load(f"../data/fvb/velocity/model{i}.npy"))
    y_test = np.concatenate(y_test).astype(np.float32)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_test_branch = []
    for i in range(49, num_dataset + 49, 1):
        X_test_branch.append(np.load(f"../data/fvb/seismic/{task}/data{i}.npy"))
    X_test_branch = np.concatenate(X_test_branch).astype(np.float32)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_test_trunk = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk.append(np.load(f"../data/fvb/{task}/{task}{i}.npy"))
        X_test_trunk = np.concatenate(X_test_trunk).astype(np.float32)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_test_trunk1 = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk1.append(np.load(f"../data/fvb/loc/loc{i}.npy"))
        X_test_trunk1 = np.concatenate(X_test_trunk1).astype(np.float32)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk2.append(np.load(f"../data/fvb/f/f{i}.npy"))
        X_test_trunk2 = np.concatenate(X_test_trunk2).astype(np.float32)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_test = (X_test_branch, X_test_trunk)

    return X_test, y_test


def data_cva_test(task):
    num_dataset = 12
    y_test = []
    for i in range(49, num_dataset + 49, 1):
        y_test.append(np.load(f"../data/cva/velocity/model{i}.npy"))
    y_test = np.concatenate(y_test).astype(np.float32)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_test_branch = []
    for i in range(49, num_dataset + 49, 1):
        X_test_branch.append(np.load(f"../data/cva/seismic/{task}/data{i}.npy"))
    X_test_branch = np.concatenate(X_test_branch).astype(np.float32)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_test_trunk = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk.append(np.load(f"../data/cva/{task}/{task}{i}.npy"))
        X_test_trunk = np.concatenate(X_test_trunk).astype(np.float32)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_test_trunk1 = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk1.append(np.load(f"../data/cva/loc/loc{i}.npy"))
        X_test_trunk1 = np.concatenate(X_test_trunk1).astype(np.float32)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = []
        for i in range(49, num_dataset + 49, 1):
            X_test_trunk2.append(np.load(f"../data/cva/f/f{i}.npy"))
        X_test_trunk2 = np.concatenate(X_test_trunk2).astype(np.float32)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_test = (X_test_branch, X_test_trunk)

    return X_test, y_test


def data_cfa_test(task):
    num_dataset = 12
    y_test = []
    for i in [2, 3, 4]:
        for j in range(32, num_dataset//3 + 32, 1):
            y_test.append(np.load(f"../data/cfa/velocity/vel{i}_1_{j}.npy"))
    y_test = np.concatenate(y_test).astype(np.float32)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_test_branch = []
    for i in [2, 3, 4]:
        for j in range(32, num_dataset//3 + 32, 1):
            X_test_branch.append(np.load(f"../data/cfa/seismic/{task}/seis{i}_1_{j}.npy"))
    X_test_branch = np.concatenate(X_test_branch).astype(np.float32)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_test_trunk = []
        for i in [2, 3, 4]:
            for j in range(32, num_dataset // 3 + 32, 1):
                X_test_trunk.append(np.load(f"../data/cfa/{task}/{task}{i}_1_{j}.npy"))
        X_test_trunk = np.concatenate(X_test_trunk).astype(np.float32)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_test_trunk1 = []
        for i in [2, 3, 4]:
            for j in range(32, num_dataset // 3 + 32, 1):
                X_test_trunk1.append(np.load(f"../data/cfa/loc/loc{i}_1_{j}.npy"))
        X_test_trunk1 = np.concatenate(X_test_trunk1).astype(np.float32)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = []
        for i in [2, 3, 4]:
            for j in range(32, num_dataset // 3 + 32, 1):
                X_test_trunk2.append(np.load(f"../data/cfa/f/f{i}_1_{j}.npy"))
        X_test_trunk2 = np.concatenate(X_test_trunk2).astype(np.float32)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_test = (X_test_branch, X_test_trunk)

    return X_test, y_test


def data_sta_test(task):
    num_dataset = 14
    y_test = []
    for i in range(121, num_dataset + 121, 1):
        y_test.append(np.load(f"../data/fvb/velocity/model{i}.npy"))
    y_test = np.concatenate(y_test).astype(np.float32)
    y_test = minmax_normalize(y_test, 1500, 4500)

    X_test_branch = []
    for i in range(121, num_dataset + 121, 1):
        X_test_branch.append(np.load(f"../data/fvb/seismic/{task}/data{i}.npy"))
    X_test_branch = np.concatenate(X_test_branch).astype(np.float32)
    X_test_branch = log_transform(X_test_branch)
    X_test_branch = minmax_normalize(X_test_branch, log_transform(-30), log_transform(60))

    if task == 'loc' or task == 'f':
        X_test_trunk = []
        for i in range(121, num_dataset + 121, 1):
            X_test_trunk.append(np.load(f"../data/fvb/{task}/{task}{i}.npy"))
        X_test_trunk = np.concatenate(X_test_trunk).astype(np.float32)
        X_test_trunk = log_transform(X_test_trunk)
        if task == 'loc':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(0), log_transform(690))
        if task == 'f':
            X_test_trunk = minmax_normalize(X_test_trunk, log_transform(5), log_transform(25))

    elif task == 'loc_f':
        X_test_trunk1 = []
        for i in range(121, num_dataset + 121, 1):
            X_test_trunk1.append(np.load(f"../data/fvb/loc/loc{i}.npy"))
        X_test_trunk1 = np.concatenate(X_test_trunk1).astype(np.float32)
        X_test_trunk1 = log_transform(X_test_trunk1)
        X_test_trunk1 = minmax_normalize(X_test_trunk1, log_transform(0), log_transform(690))

        X_test_trunk2 = []
        for i in range(121, num_dataset + 121, 1):
            X_test_trunk2.append(np.load(f"../data/fvb/f/f{i}.npy"))
        X_test_trunk2 = np.concatenate(X_test_trunk2).astype(np.float32)
        X_test_trunk2 = log_transform(X_test_trunk2)
        X_test_trunk2 = minmax_normalize(X_test_trunk2, log_transform(5), log_transform(25))

        X_test_trunk = np.concatenate([X_test_trunk1, X_test_trunk2], axis=1)

    else:
        raise NotImplementedError(f"task name should be 'loc', 'f', or 'loc_f'")

    X_test = (X_test_branch, X_test_trunk)

    return X_test, y_test
