import numpy as np
import scipy.io as scio

class LoadMatSamples:
    def __init__(self):
        self.dataFile = '..//Data//rawdata.mat'
        self.sample_length = 64
        self.vm_range = (0.88, 1.01)
        self.pi_range = (-0.5, 0.3) # normalize the data

    def normalize(self, pi, vm):
        normalized_pi = (pi - self.pi_range[0])/(self.pi_range[1] - self.pi_range[0])
        normalized_vm = (vm - self.vm_range[0])/(self.vm_range[1] - self.vm_range[0])
        normalized_pi[normalized_pi>1] = 1
        normalized_pi[normalized_pi<0] = 0
        return normalized_pi, normalized_vm

    def denormalize(self, normalized_pi, normalized_vm):
        denormalize_pi = normalized_pi*(self.pi_range[1] - self.pi_range[0]) + self.pi_range[0]
        denormalize_vm = normalized_vm * (self.vm_range[1] - self.vm_range[0]) + self.vm_range[0]
        return denormalize_pi, denormalize_vm

    def load(self, type, omitted_nodes=[]):
        dataFile = self.dataFile
        sample_len = self.sample_length

        mat = scio.loadmat(dataFile)

        # print(dataset)
        dataset = mat['raw'][0][0]
        # self.pi = dataset['pi']
        # self.vm = dataset['vm']
        self.pi = dataset['plineF_pi']
        self.vm = dataset['vm']

        self.normalized_pi, self.normalized_vm = self.normalize(self.pi, self.vm)

        t = dataset['t']

        feature = []
        label = []
        SCADA = []

        self.ava_idx = [[] for i in range(33)]

        feature_mask = np.zeros((self.normalized_pi.shape[0], sample_len))
        feature_mask[0, :] = 1

        # generate the feature mask M:
        for ibus in range(1):
            if ibus not in omitted_nodes:
                tmp_idx = []
                for i in range(feature_mask.shape[1]):
                    feature_mask[ibus, i] = 1
                    tmp_idx.append(i)
                self.ava_idx[ibus] = tmp_idx



        for ibus in range(1, 17):
            if ibus not in omitted_nodes:
                tmp_idx = []
                for i in range(0, feature_mask.shape[1], 2):
                    feature_mask[ibus, i] = 1
                    tmp_idx.append(i)
                self.ava_idx[ibus] = tmp_idx



        for ibus in range(17, 33):
            if ibus not in omitted_nodes:
                tmp_idx = []
                for i in range(0, feature_mask.shape[1], 4):
                    feature_mask[ibus, i] = 1
                    tmp_idx.append(i)
                self.ava_idx[ibus] = tmp_idx

        # construct features and labels for both Vm and Plf
        if type == 'all':
            for it in range(0, t.shape[1] - sample_len, 1):
                tmp_label = np.hstack((self.normalized_pi[:, it:it + sample_len], self.normalized_vm[:, it:it + sample_len]))
                label.append(tmp_label)

            for it in range(0, t.shape[1] - sample_len, 1):
                tmp_pi = self.normalized_pi[:, it:it + sample_len] * feature_mask
                tmp_vm = self.normalized_vm[:, it:it + sample_len] * feature_mask
                tmp_feature = np.hstack((tmp_pi, tmp_vm))
                feature.append(tmp_feature)
                SCADA.append(np.hstack(self.pi[0, it:it + sample_len], self.vm[0, it:it + sample_len]))

        # construct features and labels for Plf
        if type == 'pf':
            for it in range(0, t.shape[1] - sample_len, 1):
                tmp_label = self.normalized_pi[:, it:it + sample_len]
                label.append(tmp_label)

            for it in range(0, t.shape[1] - sample_len, 1):
                tmp_feature = self.normalized_pi[:, it:it + sample_len] * feature_mask
                feature.append(tmp_feature)
                SCADA.append(self.pi[0, it:it + sample_len])

        # construct features and labels for both Vm
        if type == 'vm':
            for it in range(0, t.shape[1] - sample_len, 1):
                tmp_label = self.normalized_vm[:, it:it + sample_len]
                label.append(tmp_label)

            for it in range(0, t.shape[1] - sample_len, 1):
                tmp_feature = self.normalized_vm[:, it:it + sample_len] * feature_mask
                feature.append(tmp_feature)
                SCADA.append(self.vm[0, it:it + sample_len])

        self.feature = np.array(feature)
        self.label = np.array(label)
        self.SCADA = SCADA



        return self.feature, self.label, self.SCADA


def loadADJ():
    file_path = '..//Data//modi_adj.mat'
    mat = scio.loadmat(file_path)
    modi_adj = mat['modi_adj']
    return modi_adj

























