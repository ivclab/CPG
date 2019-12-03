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
from UTILS.cifar100_config import *

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

def cifar100_train_loader_two_class(dataset_name, train_batch_size, num_workers=4, pin_memory=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4866, 0.4409], std=[0.26, 0.2517, 0.268])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder('datasets/'
        'cifar100_2class/train/{}'.format(dataset_name),
            train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)

def cifar100_val_loader_two_class(dataset_name, val_batch_size, num_workers=4, pin_memory=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4866, 0.4409], std=[0.26, 0.2517, 0.268])

    val_dataset = \
        datasets.ImageFolder('datasets/'
            'cifar100_2class/test/{}'.format(
                dataset_name),
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def cifar100_train_loader(dataset_name, train_batch_size, num_workers=4, pin_memory=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=mean[dataset_name], std=std[dataset_name])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder('datasets/'
        'cifar100_org/train/{}'.format(dataset_name),
            train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)

def cifar100_val_loader(dataset_name, val_batch_size, num_workers=4, pin_memory=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=mean[dataset_name], std=std[dataset_name])

    val_dataset = \
        datasets.ImageFolder('datasets/'
           'cifar100_org/test/{}'.format(
                dataset_name),
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def cifar10_train_loader(path, train_batch_size, num_workers=4, pin_memory=False):
    normalize = transforms.Normalize(
        mean=CIFAR_MEAN, std=CIFAR_STD)

    train_transform = transforms.Compose([
        # transforms.Scale(224),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # train_transform.transforms.append(Cutout(16))

    train_dataset = datasets.CIFAR10(root=path, train=True, download=True, transform=train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)

def cifar10_val_loader(path, val_batch_size, num_workers=4, pin_memory=False):
    normalize = transforms.Normalize(
        mean=CIFAR_MEAN, std=CIFAR_STD)
    valid_transform = transforms.Compose([
        # transforms.Scale(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.CIFAR10(root=path, train=False, download=True, transform=valid_transform)

    return torch.utils.data.DataLoader(val_dataset,
          batch_size=val_batch_size, shuffle=False, sampler=None,
          num_workers=num_workers, pin_memory=pin_memory) 

def train_loader(path, train_batch_size, num_workers=4, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_transform.transforms.append(Cutout(16))

    train_dataset = datasets.ImageFolder(path, train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)

def val_loader(path, val_batch_size, num_workers=4, pin_memory=False, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=IMAGENET_MEAN, std=IMAGENET_STD)

    val_dataset = \
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def val_loader_caffe(path, batch_size, num_workers=4, pin_memory=False):
    """Legacy loader for caffe. Used with models loaded from caffe."""
    # Returns images in 256 x 256 to subtract given mean of same size.
    return torch.utils.data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 Scale((256, 256)),
                                 # transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)


def train_loader_cropped(path, train_batch_size, num_workers=4, pin_memory=False):
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD)

    train_transform = transforms.Compose([
        # transforms.Scale(224),
        Scale((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,        
    ])

    train_transform.transforms.append(Cutout(16))

    train_dataset = datasets.ImageFolder(path, train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def val_loader_cropped(path, val_batch_size, num_workers=4, pin_memory=False):
    normalize = transforms.Normalize(
        mean=IMAGENET_MEAN, std=IMAGENET_STD)

    val_dataset = \
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 # transforms.Scale(224),
                                 Scale((224, 224)),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


# Note: This might not be needed anymore given that this functionality exists in
# the newer PyTorch versions.
class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)
