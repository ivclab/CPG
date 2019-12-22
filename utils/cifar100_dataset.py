import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .cifar100_config import *


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

    train_dataset = datasets.ImageFolder('data/cifar100_org/train/{}'.format(dataset_name),
            train_transform)

    return torch.utils.data.DataLoader(train_dataset,
        batch_size=train_batch_size, shuffle=True, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)


def cifar100_val_loader(dataset_name, val_batch_size, num_workers=4, pin_memory=True, normalize=None):
    if normalize is None:
        normalize = transforms.Normalize(
            mean=mean[dataset_name], std=std[dataset_name])

    val_dataset = \
        datasets.ImageFolder('data/cifar100_org/test/{}'.format(
                dataset_name),
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]))

    return torch.utils.data.DataLoader(val_dataset,
        batch_size=val_batch_size, shuffle=False, sampler=None,
        num_workers=num_workers, pin_memory=pin_memory)
