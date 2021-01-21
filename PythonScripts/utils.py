import numpy as np
import torch

def cal_DAD(modi_adj):
    # calculate D*A*D matrix with modified adjacent matrix (adj+eye)
    madj = np.array(modi_adj)
    Dsqrt = np.sqrt(np.diag(madj.sum(axis=1)))
    DAD = np.matmul(np.matmul(Dsqrt, madj),Dsqrt)
    return torch.Tensor(DAD)

def gen_batches(features, labels, batch_size, is_rand = True):

    n_samples = len(features)
    if is_rand:
        rand_index = np.random.permutation(n_samples)
        features[rand_index] = features
        labels[rand_index] = labels

    features_batches = list()
    for ibatch in range(int(n_samples/batch_size)):
        batch = features[ibatch*batch_size : (ibatch+1)*batch_size]
        features_batches.append(batch)

    labels_batches = list()
    for ibatch in range(int(n_samples / batch_size)):
        batch = labels[ibatch * batch_size: (ibatch + 1) * batch_size]
        labels_batches.append(batch)

    return torch.Tensor(features_batches), torch.Tensor(labels_batches)

