import collections
import glob
import os

import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import pdb

VGGFACE_MEAN = [0.5, 0.5, 0.5]
VGGFACE_STD  = [0.5, 0.5, 0.5]


def train_loader(path, train_batch_size, num_workers=4, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=VGGFACE_MEAN, std=VGGFACE_STD)

    train_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(path, train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def val_loader(path, val_batch_size, num_workers=4, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=VGGFACE_MEAN, std=VGGFACE_STD)

    val_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.ImageFolder(path, val_transform)

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)
