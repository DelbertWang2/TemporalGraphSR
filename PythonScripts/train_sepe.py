import scipy.io as scio
from models import PF_GCN, VM_GCN
from utils import cal_DAD, gen_batches
from genSamples import LoadMatSamples, loadADJ
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim

# This script is for SR in the base case (chapter 4.1)

# This script do several things:
# 1. construct the mask M, the LTR features and the HTR labels with rawdata.mat
#     so, before run this script, make sure the /Data/rawdata.mat are correctly generated
# 2.  separate training set and test set, construct the training batches.
# 2. implement the graph convolution layer and the 6 layer GCN model
# 3. train the GCN model for Vm and Plf separately
# 4. write the LTR features, HTR labels, GCN results and other details into file /Data/trained_data.mat


device = torch.device('cuda:0')
# device = torch.device('cpu:0')


SAVE = True
LOAD = not SAVE
pwd = os.getcwd()

##############################################################################################
# -------------------------------------for PF--------------------------------------------------
##############################################################################################

# ---------------------use 6 layers of GC, 200 epoch training, no fc layer, batchsize 32------------------
SAVE_PATH = os.path.join(pwd, 'save', 'pf', '8.pt')
LOAD_PATH = SAVE_PATH
BATCH_SIZE = 32
TEST_SIZE = 1000
EPOCHES = 200
SAMPLE_LEN = 64
modi_adj = loadADJ()
DAD = cal_DAD(modi_adj).to(device)
load_mat_samples = LoadMatSamples()
feature, label, SCADA = load_mat_samples.load('pf')
ava_idx = load_mat_samples.ava_idx



if SAVE:

    feature_train = feature[:-TEST_SIZE]
    feature_test = feature[-TEST_SIZE:]
    label_train = label[:-TEST_SIZE]
    label_test = label[-TEST_SIZE:]

    features_batches_train, labels_batches_train = gen_batches(feature_train, label_train, BATCH_SIZE)
    gcn = PF_GCN(DAD).to(device)  # get object of the GCN with matrix DAD.
    criterion = nn.MSELoss()

    # print(features_batches_train.shape)
    loss = 0.1
    pf_loss = []
    for iepoch in range(EPOCHES):
        for ibatch in range(len(features_batches_train)):
            step = float(loss / 500)
            optimizer = optim.Adam(gcn.parameters(), lr=step)
            optimizer.zero_grad()
            output = gcn(features_batches_train[ibatch].to(device))
            loss = criterion(output, labels_batches_train[ibatch].to(device))
            loss.backward()
            optimizer.step()
            print('PF: epoch:%d, batch: %d, loss:%.7f' % (iepoch, ibatch, loss))
        pf_loss.append(float(loss))

    torch.save(gcn.state_dict(), SAVE_PATH)

if LOAD:
    model = PF_GCN(DAD).to(device)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

output_test = gcn(torch.Tensor(feature_test).to(device))
output_test_np = output_test.detach().cpu().numpy()
trained_deno_pi, uls = load_mat_samples.denormalize(output_test_np, np.array([]))  # uls mean useless
feature_deno_pi, uls = load_mat_samples.denormalize(feature_test, np.array([]))
label_deno_pi, uls = load_mat_samples.denormalize(label_test, np.array([]))


# replace the trained data of the parent node with labels (because the parent node do not need SR)
trained_deno_pi[:, 0, :] = SCADA[-TEST_SIZE:]
feature_deno_pi[:, 0, :] = SCADA[-TEST_SIZE:]
label_deno_pi[:, 0, :] = SCADA[-TEST_SIZE:]
##############################################################################################
# -------------------------------------for PF--------------------------------------------------
##############################################################################################



##############################################################################################
# -------------------------------------for VM-------------------------------------------------
##############################################################################################

# ---------------------use 6 layers of GC, 200 epoch training, no fc layer, batchsize 32------------------
SAVE_PATH = os.path.join(pwd, 'save', 'vm', '8.pt')
LOAD_PATH = SAVE_PATH
BATCH_SIZE = 32
TEST_SIZE = 1000
EPOCHES = 200
SAMPLE_LEN = 64
modi_adj = loadADJ()
DAD = cal_DAD(modi_adj).to(device)
load_mat_samples = LoadMatSamples()
feature, label, SCADA = load_mat_samples.load('vm')
ava_idx = load_mat_samples.ava_idx


if SAVE:

    feature_train = feature[:-TEST_SIZE]
    feature_test = feature[-TEST_SIZE:]
    label_train = label[:-TEST_SIZE]
    label_test = label[-TEST_SIZE:]

    features_batches_train, labels_batches_train = gen_batches(feature_train, label_train, BATCH_SIZE)
    gcn = VM_GCN(DAD).to(device)  # get object of the GCN with matrix DAD.
    criterion = nn.MSELoss()

    # print(features_batches_train.shape)
    loss = 0.1
    vm_loss = []
    for iepoch in range(EPOCHES):
        for ibatch in range(len(features_batches_train)):
            step = float(loss / 500)
            optimizer = optim.Adam(gcn.parameters(), lr=step)
            optimizer.zero_grad()
            output = gcn(features_batches_train[ibatch].to(device))
            loss = criterion(output, labels_batches_train[ibatch].to(device))
            loss.backward()
            optimizer.step()
            print('VM: epoch:%d, batch: %d, loss:%.7f' % (iepoch, ibatch, loss))
        vm_loss.append(float(loss))

    torch.save(gcn.state_dict(), SAVE_PATH)

if LOAD:
    model = VM_GCN(DAD).to(device)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

output_test = gcn(torch.Tensor(feature_test).to(device))
output_test_np = output_test.detach().cpu().numpy()



uls, trained_deno_vm = load_mat_samples.denormalize(np.array([]), output_test_np)
uls, feature_deno_vm = load_mat_samples.denormalize(np.array([]), feature_test)
uls, label_deno_vm = load_mat_samples.denormalize(np.array([]), label_test)


# replace the trained data of the parent node with labels (because the parent node do not need SR)
trained_deno_vm[:, 0, :] = SCADA[-TEST_SIZE:]
feature_deno_vm[:, 0, :] = SCADA[-TEST_SIZE:]
label_deno_vm[:, 0, :] = SCADA[-TEST_SIZE:]

##############################################################################################
# -------------------------------------for VM--------------------------------------------------
##############################################################################################


scio.savemat('..//Data//trained_data.mat', {'feature_test': {'pi': feature_deno_pi, 'vm': feature_deno_vm},
                                            'label_test': {'pi': label_deno_pi, 'vm': label_deno_vm},
                                            'trained': {'pi': trained_deno_pi, 'vm': trained_deno_vm},
                                            'ava_idx': load_mat_samples.ava_idx,
                                            'pf_loss': pf_loss,
                                            'vm_loss': vm_loss
                                            })


