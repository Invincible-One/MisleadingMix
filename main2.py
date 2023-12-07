import os
import sys
import time
import argparse
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import model
from utils import progress_bar
from utils import parse_list_arg
import data



def get_data(training_mode, test_modes, training_transform, test_transform, batch_size, requires_cring=False, **kwargs):   # {"fully_correlated", "partially_correlated", "conditionally_correlated", "not_correlated", "cifar10"}
    if requires_cring:
        print('''
        I have the ability to generate the data here, but I don't want to. The data is not generated. If you wish to
        generate the data, please refer to the 'data_generation.ipynb' file. Now, I kindly request that you bounce!
        ''')
    if training_mode == "cifar10":
        training_set = datasets.CIFAR10(cifar10_dir, download=True, train=True, transform=training_transform)
    else:
        training_set = data.Type2_Dataset(data_root, mode=training_mode, split='train', margin=margin, transform=training_transform)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

    test_loaders = dict()
    for mode in test_modes:
        if mode == "cifar10":
            temp_test_set = datasets.CIFAR10(cifar10_dir, download=True, train=False, transform=test_transform)
        else:
            temp_test_set = data.Corr_Dataset(data_root, mode=mode, split='val', transform=test_transform)
        temp_test_loader = DataLoader(temp_test_set, batch_size=batch_size, shuffle=False)
        test_loaders[mode] = temp_test_loader
    return training_loader, test_loaders



def train():
    network.train()
    total = 0
    correct = 0

    if training_mode == "cifar10":
        for X, y in training_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            output = network(X)
            loss_v = loss_f(output, y)
            loss_v.backward()
            optimizer.step()

            correct += y.eq(torch.max(output.detach(), 1)[1]).sum().item()
            total += y.numel()
    else:
        for X, y in training_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            output = network(X)
            lsm = nn.LogSoftmax(dim=1).to(device)
            output = lsm(output)
            loss_v = loss_f(output, y.softmax(dim=1))
            loss_v.backward()
            optimizer.step()

            correct += torch.max(y, 1)[1].eq(torch.max(output.detach(), 1)[1]).sum().item()
            total += y.size(0)
    acc = correct / total
    return acc, f"Train {acc:.5f}"

def test(mode, mode_test_loader):
    network.eval()
    with torch.no_grad():
        if mode == "cifar10":
            correct = 0
            total = 0
            for X, y in mode_test_loader:
                X, y = X.to(device), y.to(device)
                output = network(X)
                correct += y.eq(torch.max(output.detach(), 1)[1]).sum().item()
                total += y.numel()
            acc = correct / total
            return acc, f"{mode} {acc:.5f}"
        elif mode == "fully_correlated":
            correct = 0
            total = 0
            for X, y, _ in mode_test_loader:
                X, y = X.to(device), y.to(device)
                output = network(X)
                correct += y.eq(torch.max(output.detach(), 1)[1]).sum().item()
                total += y.numel()
            acc = correct / total
            return acc, f"{mode} {acc:.5f}"
        else:
            ccorrect = 0
            mcorrect = 0
            total = 0
            for X, cy, my in mode_test_loader:
                X, cy, my = X.to(device), cy.to(device), my.to(device)
                output = network(X)
                ccorrect += cy.eq(torch.max(output.detach(), 1)[1]).sum().item()
                mcorrect += my.eq(torch.max(output.detach(), 1)[1]).sum().item()
                total += cy.numel()
            cacc = ccorrect / total
            macc = mcorrect / total
            return (cacc, macc), f"{mode} c. vs. m. {cacc:.5f} vs. {macc:.5f}"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=3.14)
    parser.add_argument("--epochs", nargs='*', type=int, default=[15, 15, 15, 15, 10, 10, 10, 10])
    parser.add_argument("--network_type", type=str, default="Special_VGG")
    parser.add_argument("--network_cfg", nargs='+', type=parse_list_arg)
    parser.add_argument("--data_root", type=str, default="corr_exp1/")
    parser.add_argument("--training_mode", type=str, default="fully_correlated")
    parser.add_argument("--test_modes", nargs='*', type=str, default=["fully_correlated", "not_correlated", "cifar10"])
    parser.add_argument("--margin", type=float, default=0.6)
    params = parser.parse_args()

    batch_size = params.batch_size
    lr = params.lr
    lr_decay = params.lr_decay
    epochs = params.epochs
    network_type = params.network_type
    network_cfg = params.network_cfg[0]
    data_root = params.data_root
    training_mode = params.training_mode
    test_modes = params.test_modes
    margin = params.margin

    print(f'''
    *********************************************INFO*********************************************
    This is one of the experiment series that study data correlation and deep learning.
    The training mode is {training_mode},
    The test modes are {test_modes},
    The data root is {data_root}
    The epochs, batch_size, lr, and lr_decay: {epochs}, {batch_size}, {lr}, {lr_decay}.
    The margin is {margin}.
    Good luck!
    **********************************************************************************************
    ''')

    device = torch.device('cuda')
    
    parent_dir = "/scratch/ym2380/data/"
    data_root = os.path.join(parent_dir, data_root)
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    cifar10_path = "cifar10/"
    cifar10_dir = os.path.join(parent_dir, cifar10_path)
    if not os.path.exists(cifar10_dir):
        os.makedirs(cifar10_dir)

    transform = transforms.ToTensor()
    training_loader, test_loaders = get_data(
            training_mode=training_mode, 
            test_modes=test_modes, 
            training_transform=transform, 
            test_transform=transform, 
            batch_size=batch_size)

    if network_type == "ResNet":
        network = model.ResNet(network_cfg).to(device)
    elif network_type == "VGG":
        network = model.VGG(network_cfg).to(device)
    elif network_type == "Special_VGG":
        network = model.Special_VGG(network_cfg).to(device)
    else:
        raise ValueError
    loss_f = nn.KLDivLoss(reduction="batchmean").to(device)

    count = 0
    num_epochs = sum(epochs)
    for epoch in epochs:
        optimizer = optim.Adam(network.parameters(), lr=lr)
        for _ in range(epoch):
            beg = time.time()
            count += 1
            test_logs = list()
            _, training_log = train()
            for k, v in test_loaders.items():
                _, temp_log = test(k, v)
                test_logs.append(temp_log)
            test_log = "Test " + ", ".join(test_logs)
            runtime = time.time() - beg
            epoch_log = f"Epoch [{count:02d}/{num_epochs}] {progress_bar(count, num_epochs)}, Time {runtime:.1f}, {training_log}, {test_log}."
            print(epoch_log)
            sys.stdout.flush()
        lr /= lr_decay
