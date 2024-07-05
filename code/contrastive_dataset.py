import numpy as np
import mat73
import scipy.io as io
import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils import data

from torch_geometric.data import Data
from torch_geometric.data import Dataset

class mri_dataset(data.Dataset):
    def __init__(self, args):
        super(mri_dataset, self).__init__()

        self.args = args
        self.fmri = io.loadmat(os.path.join(args.data, 'FCNetworks.mat'))['FCNetworks']
        self.dmri = mat73.loadmat(os.path.join(args.data, 'SCNetworks.mat'))['SCNetworks']['FN']
    
    def __getitem__(self, index):
        fmri = torch.Tensor(self.fmri[:,index][0]).float()
        dmri = torch.from_numpy(self.dmri[index]).float()
        return fmri, dmri

    def __len__(self):
        return len(self.dmri)
class fmri_dataset_connectome(Dataset):
    def __init__(self, args, index_list, fmri, label, mode='train'):
        super(fmri_dataset_connectome, self).__init__()

        self.args = args
        self.fmri = fmri
        self.label = label
        self.mode = mode
        self.index_list = index_list
        
        self.fmri_use = self.fmri[index_list]
        self.label_use = self.label['age'][index_list].tolist()
        
    def get(self, index):
        fmri = torch.Tensor(self.fmri_use[index]).float()
        age = torch.Tensor([self.label_use[index]]).float()
        # if self.mode == 'train':
        #     fmri += torch.Tensor(np.random.normal(0, 0.2, fmri.shape)).float()
        # print(fmri.shape)
        # print(label_ind.shape)
        # if label_ind.shape[0] == 0 or label_ind.shape[0] == 3:
        #     print(label)
        return (fmri, age)

    def len(self):
        return len(self.index_list)

class NKI_dataset(data.Dataset):
    def __init__(self, args, index_list, fmri, label, mode='train'):
        super(NKI_dataset, self).__init__()

        self.args = args
        self.fmri = fmri
        self.label = label
        self.mode = mode
        self.index_list = index_list
        # self.dmri = io.loadmat(os.path.join(args.data, 'normal_dmri_matrix.mat'))['dmri']
        # self.fmri = io.loadmat(os.path.join(args.data, 'normal_fmri_matrix.mat'))['fmri']
        self.fmri_use = self.fmri[index_list]
        self.label_use = self.label['age'][index_list].tolist()
    
    def __getitem__(self, index):
        fmri = torch.Tensor(self.fmri_use[index]).float()
        age = torch.Tensor([self.label_use[index]]).float()
        return fmri, age

    def __len__(self):
        return len(self.index_list)

class fmri_dataset(data.Dataset):
    def __init__(self, args, index_list, fmri, label, mode='train'):
        super(fmri_dataset, self).__init__()

        self.args = args
        self.fmri = fmri
        self.label = label
        self.mode = mode
        self.index_list = index_list
        
        self.fmri_use = self.fmri[index_list]
        self.label_use = self.label['label'][index_list].tolist()
        
    def __getitem__(self, index):
        fmri = torch.Tensor(self.fmri_use[index]).float()
        label = self.label_use[index]
        if label == 'ADHD-Combined':
            label_ind = [0]
        elif label == 'Typically Developing':
            label_ind = [1]
        elif label == 'ADHD-Inattentive':
            label_ind = [0]
        elif label == 'ADHD-Hyperactive/Impulsive':
            label_ind = [0]
        label_ind = torch.Tensor(label_ind).long()
        if self.mode == 'train':
            fmri += torch.Tensor(np.random.normal(0, 0.05, fmri.shape)).float()
        # print(fmri.shape)
        # print(label_ind.shape)
        # if label_ind.shape[0] == 0 or label_ind.shape[0] == 3:
        #     print(label)
        return fmri, label_ind

    def __len__(self):
        return len(self.index_list)


class graph_dataset(data.Dataset):
    def __init__(self, args, index_list, fmri, label, mode='train'):
        super(graph_dataset, self).__init__()

        self.args = args
        self.fmri = fmri
        self.label = label
        self.mode = mode
        self.index_list = index_list
        
        self.fmri_use = self.fmri[index_list]
        self.label_use = self.label['age'][index_list].tolist()
        
    def __getitem__(self, index):
        fmri = torch.Tensor(self.fmri_use[index]).float()
        age = torch.Tensor([self.label_use[index]]).float()
        fmri_index = (fmri > 0).nonzero().t()
        row, col = fmri_index
        fmri_weight = fmri[row, col]
        # if self.mode == 'train':
        #     fmri += torch.Tensor(np.random.normal(0, 0.2, fmri.shape)).float()
        fmri_data = Data(x=fmri, edge_index=fmri_index, edge_attr=fmri_weight, y=age)
        # print(fmri.shape)
        # print(label_ind.shape)
        # if label_ind.shape[0] == 0 or label_ind.shape[0] == 3:
        #     print(label)
        return fmri_data

    def __len__(self):
        return len(self.index_list)
