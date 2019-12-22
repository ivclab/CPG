"""Main entry point for doing all stuff."""
import argparse
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn.parameter import Parameter

import logging
import os
import pdb
import math
from tqdm import tqdm
import sys
import numpy as np

import utils
from utils import Optimizers
from utils.packnet_manager import Manager
import utils.cifar100_dataset as dataset
import packnet_models


# To prevent PIL warnings.
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='vgg16_bn_cifar100',
                   help='Architectures')
parser.add_argument('--num_classes', type=int, default=-1,
                   help='Num outputs for dataset')

# Optimization options.
parser.add_argument('--lr', type=float, default=0.1,
                   help='Learning rate for parameters, used for baselines')

parser.add_argument('--batch_size', type=int, default=32,
                   help='input batch size for training')
parser.add_argument('--val_batch_size', type=int, default=100,
                   help='input batch size for validation')
parser.add_argument('--workers', type=int, default=24, help='')
parser.add_argument('--weight_decay', type=float, default=4e-5,
                   help='Weight decay')

# Paths.
parser.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')
parser.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
parser.add_argument('--val_path', type=str, default='',
                   help='Location of test data')

# Other.
parser.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')

parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--checkpoint_format', type=str,
                    default='./{save_folder}/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--epochs', type=int, default=160,
                    help='number of epochs to train')
parser.add_argument('--restore_epoch', type=int, default=0, help='')
parser.add_argument('--save_folder', type=str,
                    help='folder name inside one_check folder')
parser.add_argument('--load_folder', default='', help='')
parser.add_argument('--one_shot_prune_perc', type=float, default=0.5,
                   help='% of neurons to prune per layer')
parser.add_argument('--mode',
                   choices=['finetune', 'prune', 'inference'],
                   help='Run mode')
parser.add_argument('--logfile', type=str, help='file to save baseline accuracy')
parser.add_argument('--initial_from_task', type=str, help="")


def main():
    """Do stuff."""
    args = parser.parse_args()
    if args.save_folder and not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        args.cuda = False

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    resume_folder = args.load_folder
    for try_epoch in range(200, 0, -1):
        if os.path.exists(args.checkpoint_format.format(
            save_folder=resume_folder, epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    if args.restore_epoch:
        resume_from_epoch = args.restore_epoch

    # Set default train and test path if not provided as input.
    utils.set_dataset_paths(args)

    if resume_from_epoch:
        filepath = args.checkpoint_format.format(save_folder=resume_folder, epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        checkpoint_keys = checkpoint.keys()
        dataset_history = checkpoint['dataset_history']
        dataset2num_classes = checkpoint['dataset2num_classes']
        masks = checkpoint['masks']
        if 'shared_layer_info' in checkpoint_keys:
            shared_layer_info = checkpoint['shared_layer_info']
        else:
            shared_layer_info = {}

        if 'num_for_construct' in checkpoint_keys:
            num_for_construct = checkpoint['num_for_construct']
    else:
        dataset_history = []
        dataset2num_classes = {}
        masks = {}
        shared_layer_info = {}

    if args.arch == 'vgg16_bn_cifar100':
        model = packnet_models.__dict__[args.arch](pretrained=False, dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
    elif args.arch == 'resnet18':
        model = packnet_models.__dict__[args.arch](dataset_history=dataset_history, dataset2num_classes=dataset2num_classes)
    else:
        print('Error!')
        sys.exit(0)

    # Add and set the model dataset
    model.add_dataset(args.dataset, args.num_classes)
    model.set_dataset(args.dataset)

    if args.dataset not in shared_layer_info:
        shared_layer_info[args.dataset] = {
            'conv_bias': {},
            'bn_layer_running_mean': {},
            'bn_layer_running_var': {},
            'bn_layer_weight': {},
            'bn_layer_bias': {},
            'fc_bias': {}
        }

    model = nn.DataParallel(model)
    model = model.cuda()
    if args.initial_from_task and 'None' not in args.initial_from_task:
        filepath = ''
        for try_epoch in range(200, 0, -1):
            if os.path.exists(args.checkpoint_format.format(
                save_folder=args.initial_from_task, epoch=try_epoch)):
                filepath = args.checkpoint_format.format(save_folder=args.initial_from_task, epoch=try_epoch)
                break
        if filepath == '':
            pdb.set_trace()
            print('Something is wrong')
        checkpoint = torch.load(filepath)
        state_dict = checkpoint['model_state_dict']
        curr_model_state_dict = model.module.state_dict()

        for name, param in state_dict.items():
            if 'num_batches_tracked' in name:
                continue
            try:
                curr_model_state_dict[name][:].copy_(param)
            except:
                pdb.set_trace()
                print('here')

    if not masks:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if 'classifiers' in name:
                    continue
                mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                if 'cuda' in module.weight.data.type():
                    mask = mask.cuda()
                masks[name] = mask

    if args.num_classes == 5:
        train_loader = dataset.cifar100_train_loader(args.dataset, args.batch_size)
        val_loader = dataset.cifar100_val_loader(args.dataset, args.val_batch_size)
    else:
        print("num_classes should be 5")
        sys.exit(1)

    # if we are going to save checkpoint in other folder, then we recalculate the starting epoch
    if args.save_folder != args.load_folder:
        start_epoch = 0
    else:
        start_epoch = resume_from_epoch

    manager = Manager(args, model, shared_layer_info, masks, train_loader, val_loader)

    if args.mode == 'inference':
        manager.load_checkpoint_for_inference(resume_from_epoch, resume_folder)
        manager.validate(resume_from_epoch-1)
        return

    lr = args.lr
    # update all layers
    named_params = dict(model.named_parameters())
    params_to_optimize_via_SGD = []
    named_params_to_optimize_via_SGD = []
    masks_to_optimize_via_SGD = []
    named_masks_to_optimize_via_SGD = []

    for tuple_ in named_params.items():
        if 'classifiers' in tuple_[0]:
            if '.{}.'.format(model.module.datasets.index(args.dataset)) in tuple_[0]:
                params_to_optimize_via_SGD.append(tuple_[1])
                named_params_to_optimize_via_SGD.append(tuple_)
            continue
        else:
            params_to_optimize_via_SGD.append(tuple_[1])
            named_params_to_optimize_via_SGD.append(tuple_)

    # here we must set weight decay to 0.0,
    # because the weight decay strategy in build-in step() function will change every weight elem in the tensor,
    # which will hurt previous tasks' accuracy. (Instead, we do weight decay ourself in the `prune.py`)
    optimizer_network = optim.SGD(params_to_optimize_via_SGD, lr=lr,
                          weight_decay=0.0, momentum=0.9, nesterov=True)

    optimizers = Optimizers()
    optimizers.add(optimizer_network, lr)

    manager.load_checkpoint(optimizers, resume_from_epoch, resume_folder)

    """Performs training."""
    curr_lrs = []
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            curr_lrs.append(param_group['lr'])
            break

    if args.mode == 'prune':
        print()
        print('Sparsity ratio: {}'.format(args.one_shot_prune_perc))
        print('Before pruning: ')
        baseline_acc = manager.validate(start_epoch-1)
        print('Execute one shot pruning ...')
        manager.one_shot_prune(args.one_shot_prune_perc)
    elif args.mode == 'finetune':
        manager.pruner.make_finetuning_mask()

    for epoch_idx in range(start_epoch, args.epochs):
        avg_train_acc = manager.train(optimizers, epoch_idx, curr_lrs)
        avg_val_acc = manager.validate(epoch_idx)

        if args.mode == 'finetune':
            if epoch_idx + 1 == 50 or epoch_idx + 1 == 80:
                for param_group in optimizers[0].param_groups:
                    param_group['lr'] *= 0.1
                curr_lrs[0] = param_group['lr']

        if args.mode == 'prune':
            if epoch_idx + 1 == 25:
                for param_group in optimizers[0].param_groups:
                    param_group['lr'] *= 0.1
                curr_lrs[0] = param_group['lr']


    if args.save_folder is not None:
    #     paths = os.listdir(args.save_folder)
    #     if paths and '.pth.tar' in paths[0]:
    #         for checkpoint_file in paths:
    #             os.remove(os.path.join(args.save_folder, checkpoint_file))
        pass
    else:
        print('Something is wrong! Block the program with pdb')
        pdb.set_trace()

    if args.mode == 'finetune':
        manager.save_checkpoint(optimizers, epoch_idx, args.save_folder)
        if args.logfile:
            json_data = {}
            if os.path.isfile(args.logfile):
                with open(args.logfile) as json_file:
                    json_data = json.load(json_file)
            json_data[args.dataset] = '{:.4f}'.format(avg_val_acc)
            with open(args.logfile, 'w') as json_file:
                json.dump(json_data, json_file)

        if avg_train_acc < 0.97:
            print('Cannot prune any more!')

    elif args.mode == 'prune':
        #if avg_train_acc > 0.97 and (avg_val_acc - baseline_acc) >= -0.01:
        if avg_train_acc > 0.97:
            manager.save_checkpoint(optimizers, epoch_idx, args.save_folder)
        else:
            print('Pruning too much!')

    print('-' * 16)

if __name__ == '__main__':
    main()
