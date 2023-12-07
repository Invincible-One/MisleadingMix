import os
from PIL import Image

import numpy as np

import torch
from torch.utils.data import Dataset



class Corr_Dataset(Dataset):
    def __init__(self, root, mode, split, transform=None):
        self.x_shape = (32, 32, 3)
        self.data_dir = os.path.join(root, mode, split)

        self.path_list = os.listdir(self.data_dir)
        self.path_list = sorted(self.path_list)

        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        filename = self.path_list[idx]
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            x = np.fromfile(f, dtype=np.uint8)
        x = x.reshape(self.x_shape)

        clabel = int(filename.split('_')[3])
        mlabel = int(filename.split('_')[5].split('.')[0])

        x = Image.fromarray(x)
        if self.transform:
            x = self.transform(x)

        return x, clabel, mlabel



class Type1_Dataset(Dataset):   # smooth hard samples
    def __init__(self, root, mode, split, margin, transform=None):
        self.x_shape = (32, 32, 3)
        self.num_classes = 10
        self.data_dir = os.path.join(root, mode, split)
        self.margin = margin

        self.path_list = os.listdir(self.data_dir)
        self.path_list = sorted(self.path_list)

        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        filename = self.path_list[idx]
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            x = np.fromfile(f, dtype=np.uint8)
        x = x.reshape(self.x_shape)

        clabel = int(filename.split('_')[3])
        mlabel = int(filename.split('_')[5].split('.')[0])
        smooth_label = self._smoothen(clabel, mlabel)

        x = Image.fromarray(x)
        if self.transform:
            x = self.transform(x)
        
        return x, smooth_label

    def _smoothen(self, clabel, mlabel):
        identity_matrix = torch.eye(self.num_classes)
        if clabel == mlabel:
            return identity_matrix[clabel]
        else:
            return (0.5 + 0.5 * self.margin) * identity_matrix[clabel] + (0.5 - 0.5 * self.margin) * identity_matrix[mlabel]



class Type2_Dataset(Dataset):   # smooth easy samples
    def __init__(self, root, mode, split, margin, transform=None):
        self.x_shape = (32, 32, 3)
        self.num_classes = 10
        self.data_dir = os.path.join(root, mode, split)
        self.margin = margin

        self.path_list = os.listdir(self.data_dir)
        self.path_list = sorted(self.path_list)

        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        filename = self.path_list[idx]
        with open(os.path.join(self.data_dir, filename), 'rb') as f:
            x = np.fromfile(f, dtype=np.uint8)
        x = x.reshape(self.x_shape)

        clabel = int(filename.split('_')[3])
        mlabel = int(filename.split('_')[5].split('.')[0])
        smooth_label = self._smoothen(clabel, mlabel)

        x = Image.fromarray(x)
        if self.transform:
            x = self.transform(x)
    
        return x, smooth_label

    def _smoothen(self, clabel, mlabel):
        identity_matrix = torch.eye(self.num_classes)
        if clabel == mlabel:
            return self.margin * identity_matrix[clabel] + sum([(0.1 - 0.1 * self.margin) * identity_matrix[idx] for idx in range(self.num_classes)])
        else:
            return identity_matrix[clabel]
